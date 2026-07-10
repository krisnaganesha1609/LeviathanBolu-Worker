"""
FastAPI app for the STT worker.

Pipeline shape requested: WebSocket -> async Queue -> Worker -> Model -> WebSocket.
A dedicated reader task drains the socket into an asyncio.Queue; the main
session loop consumes that queue (with a short timeout so it can also flush
completed partial-transcript background tasks) and is the only coroutine
that writes back to the socket, so writes are never interleaved.
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from common.config import get_audio_settings, get_general_settings, get_stt_settings
from common.errors import STTErrorCode, WorkerError, stt_error_event
from common.logger import configure_logging, get_logger
from common.metrics import stt_metrics
from common.protocol import HealthResponse, STTEndAction, STTFinalTranscript, STTPartialTranscript
from common.websocket import send_json
from stt.sensevoice_engine import STTEngine, build_engine
from stt.worker import STTSession

log = get_logger(__name__)

_QUEUE_POLL_TIMEOUT = 0.05  # seconds — governs partial-flush responsiveness


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_stt_settings()
    engine: STTEngine = build_engine(settings)
    log.info("stt.server.starting", engine=settings.engine, host=settings.host, port=settings.port)
    start = time.monotonic()
    await engine.warm_up()  # model lifecycle: load once at startup, not per-request
    log.info("stt.server.ready", warm_up_seconds=round(time.monotonic() - start, 2))
    app.state.engine = engine
    app.state.started_at = time.monotonic()
    yield
    log.info("stt.server.shutdown")


app = FastAPI(title="LEVIATHAN STT Worker", lifespan=lifespan)


@app.get(get_general_settings().health_path, response_model=HealthResponse)
async def health() -> HealthResponse:
    engine: STTEngine = app.state.engine
    engine_health = engine.health()
    status = "ok" if engine_health.get("ready", True) else "degraded"
    if engine_health.get("load_error"):
        status = "down"
    return HealthResponse(
        status=status,  # type: ignore[arg-type]
        service="stt",
        engine=engine_health.get("engine", "unknown"),
        active_sessions=stt_metrics.active_sessions,
        uptime_seconds=round(time.monotonic() - app.state.started_at, 1),
        detail=engine_health,
    )


@app.get(get_general_settings().metrics_path)
async def metrics() -> dict:
    return stt_metrics.snapshot()


async def _reader(ws: WebSocket, queue: asyncio.Queue) -> None:
    """Drains the socket into `queue`. Puts (kind, payload) tuples; kind in
    {"binary", "text", "disconnect"}."""
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                await queue.put(("disconnect", None))
                return
            if (data := message.get("bytes")) is not None:
                await queue.put(("binary", data))
            elif (text := message.get("text")) is not None:
                await queue.put(("text", text))
    except WebSocketDisconnect:
        await queue.put(("disconnect", None))
    except Exception as exc:  # pragma: no cover - defensive
        log.error("stt.reader.error", error=str(exc))
        await queue.put(("disconnect", None))


@app.websocket("/stt")
async def stt_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    audio_settings = get_audio_settings()
    stt_settings = get_stt_settings()
    engine: STTEngine = app.state.engine

    session_id = websocket.query_params.get("session_id")
    conversation_id = websocket.query_params.get("conversation_id")
    session = STTSession(
        engine, audio_settings, stt_settings, stt_metrics,
        session_id=session_id, conversation_id=conversation_id,
    )

    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    reader_task = asyncio.create_task(_reader(websocket, queue))

    try:
        while True:
            if session.expired:
                log.warning("stt.session.expired", session_id=session.session_id)
                await send_json(websocket, stt_error_event(STTErrorCode.TIMEOUT, "max_session_seconds exceeded"))
                break
            if session.stalled:
                log.warning("stt.session.stalled", session_id=session.session_id)

            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=_QUEUE_POLL_TIMEOUT)
            except asyncio.TimeoutError:
                kind, payload = None, None

            if kind == "disconnect":
                break

            if kind == "binary":
                session.ingest_frame(payload)
                session.maybe_start_partial()

            elif kind == "text":
                try:
                    obj = json.loads(payload)
                    STTEndAction.model_validate(obj)
                except (json.JSONDecodeError, ValidationError):
                    log.debug("stt.endpoint.ignored_text", session_id=session.session_id, raw=payload)
                else:
                    result = await session.finalize()
                    await send_json(websocket, STTFinalTranscript(text=result.text))
                    break

            # Opportunistically flush a completed partial regardless of what
            # (if anything) arrived on the queue this iteration.
            partial = session.pending_partial_result()
            if partial is not None and partial.text:
                await send_json(websocket, STTPartialTranscript(text=partial.text))

    except WorkerError as exc:
        log.error("stt.endpoint.worker_error", session_id=session.session_id, code=exc.code.value, message=exc.message)
        if websocket.application_state == WebSocketState.CONNECTED:
            await send_json(websocket, stt_error_event(exc.code, exc.message))
    except Exception as exc:  # pragma: no cover - defensive
        log.error("stt.endpoint.unhandled_error", session_id=session.session_id, error=str(exc))
    finally:
        reader_task.cancel()
        session.close()
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.close()
