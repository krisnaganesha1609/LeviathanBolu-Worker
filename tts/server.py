"""
FastAPI app for the TTS worker.

Go's TTSWorkerClient.SynthesizeStream dials once, sends exactly one JSON
request, reads binary frames until it sees {"event":"done"} (or the socket
closes), then closes the connection itself — so this endpoint only needs
to handle a single request/response cycle per connection, not a
long-lived multi-turn session like STT.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from common.config import get_audio_settings, get_general_settings, get_tts_settings
from common.errors import TTSErrorCode, tts_error_event
from common.logger import configure_logging, get_logger
from common.metrics import tts_metrics
from common.protocol import HealthResponse, TTSDoneEvent, TTSRequest
from common.websocket import send_json
from tts.kokoro_engine import TTSEngine, build_engine
from tts.personalities import PersonalityRegistry
from tts.worker import TTSSession

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_tts_settings()
    audio_settings = get_audio_settings()
    engine: TTSEngine = build_engine(settings, audio_settings)
    personalities = PersonalityRegistry(settings.personalities_path, settings.default_personality)
    log.info(
        "tts.server.starting",
        engine=settings.engine,
        host=settings.host,
        port=settings.port,
        personalities=personalities.names(),
    )
    start = time.monotonic()
    await engine.warm_up()  # model lifecycle: load once at startup, not per-request
    log.info("tts.server.ready", warm_up_seconds=round(time.monotonic() - start, 2))
    app.state.engine = engine
    app.state.personalities = personalities
    app.state.started_at = time.monotonic()
    yield
    log.info("tts.server.shutdown")


app = FastAPI(title="LEVIATHAN TTS Worker", lifespan=lifespan)


@app.get(get_general_settings().health_path, response_model=HealthResponse)
async def health() -> HealthResponse:
    engine: TTSEngine = app.state.engine
    engine_health = engine.health()
    status = "ok" if engine_health.get("ready", True) else "degraded"
    if engine_health.get("load_error"):
        status = "down"
    return HealthResponse(
        status=status,  # type: ignore[arg-type]
        service="tts",
        engine=engine_health.get("engine", "unknown"),
        active_sessions=tts_metrics.active_sessions,
        uptime_seconds=round(time.monotonic() - app.state.started_at, 1),
        detail={**engine_health, "personalities": app.state.personalities.names()},
    )


@app.get(get_general_settings().metrics_path)
async def metrics() -> dict:
    return tts_metrics.snapshot()


@app.websocket("/tts")
async def tts_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    audio_settings = get_audio_settings()
    tts_settings = get_tts_settings()
    engine: TTSEngine = app.state.engine
    personalities: PersonalityRegistry = app.state.personalities

    session_id = websocket.query_params.get("session_id")
    conversation_id = websocket.query_params.get("conversation_id")

    try:
        raw = await websocket.receive_text()
    except WebSocketDisconnect:
        return

    try:
        request = TTSRequest.model_validate_json(raw)
    except ValidationError as exc:
        log.warning("tts.endpoint.invalid_request", error=str(exc), raw=raw)
        await send_json(websocket, tts_error_event(TTSErrorCode.INVALID_TEXT, str(exc)))
        await websocket.close()
        return

    session = TTSSession(
        engine, personalities, audio_settings, tts_settings, tts_metrics,
        session_id=session_id, conversation_id=conversation_id,
    )

    try:
        async for frame in session.stream_reply(request.text, request.personality):
            if websocket.application_state != WebSocketState.CONNECTED:
                break
            await websocket.send_bytes(frame)
        if websocket.application_state == WebSocketState.CONNECTED:
            await send_json(websocket, TTSDoneEvent())
    except Exception as exc:  # pragma: no cover - defensive
        log.error("tts.endpoint.unhandled_error", session_id=session.session_id, error=str(exc))
        if websocket.application_state == WebSocketState.CONNECTED:
            await send_json(websocket, tts_error_event(TTSErrorCode.ENGINE_FAILURE, str(exc)))
    finally:
        session.close()
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.close()
