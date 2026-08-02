"""
Wire protocol for both workers ("Worker Wire v1.1"). These models are the
single source of truth for what the Go orchestrator
(Orchestrator/internal/assistant/python_workers.go) sends/expects — keep
them in lockstep with that file.

This is the Go<->Python hop only. The runtime protocol envelope
(Documentation 05/25: voice.stt.partial, voice.tts.chunk, ...) lives on
the Flutter<->Orchestrator hop; the Go gateway translates between the two
(partial_transcript -> voice.stt.partial, done -> voice.tts.completed, ...).

STT  ws://<host>:9001/stt
  IN  (binary)          raw PCM16LE mono frames (already Opus-decoded by Go)
  IN  (text/json)       {"action": "end"}     finalize -> final_transcript
  IN  (text/json)       {"action": "cancel"}  abort session, no final emitted
  OUT (text/json)       {"event": "partial_transcript", "text": "..."}
  OUT (text/json)       {"event": "final_transcript", "text": "..."}
  OUT (text/json)       {"event": "error", "code": "STT_00x",
                         "recoverable": bool, "retryable": bool, "message": "..."}

TTS  ws://<host>:9002/tts
  IN  (text/json)       {"text": "...", "voice": "...", "speed": 1.0,
                         "pitch": 0, "lang": "id-ID", "personality": "..."}
  OUT (binary)          raw PCM16LE mono frames (Go re-encodes to Opus)
  OUT (text/json)       {"event": "done"}
  OUT (text/json)       {"event": "error", ...same shape as STT error}
  Cancellation: Go simply closes the socket mid-stream.

Notes:
- Go streams PCM frames to STT *while the user is speaking* and relays
  partial_transcript upstream in real time; "end" asks for the final pass.
- Go's read loops branch on "final_transcript" / "done" / "error" and
  ignore unknown events, so new events stay additive.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ── STT inbound ──────────────────────────────────────────────────────────


class STTEndAction(BaseModel):
    action: Literal["end"]


class STTCancelAction(BaseModel):
    """Abort the session: pending inference is cancelled and no
    final_transcript is emitted (voice barge-in / turn interruption)."""

    action: Literal["cancel"]


# ── STT outbound ─────────────────────────────────────────────────────────


class STTPartialTranscript(BaseModel):
    event: Literal["partial_transcript"] = "partial_transcript"
    text: str


class STTFinalTranscript(BaseModel):
    event: Literal["final_transcript"] = "final_transcript"
    text: str


class STTErrorEvent(BaseModel):
    """Structured error contract (03_SYSTEM_CONTRACTS.md §Error Contract,
    carried over the worker wire so the Go gateway can map it into a
    runtime error event without parsing message strings)."""

    event: Literal["error"] = "error"
    code: str = ""
    recoverable: bool = True
    retryable: bool = False
    message: str


# ── TTS inbound ──────────────────────────────────────────────────────────


class TTSRequest(BaseModel):
    """
    Matches Go's user_settings.domain.go WakeWordConfig field-for-field
    (voice/speed/pitch/lang json tags) — Go resolves the full voice config
    from its own data and sends it directly. The worker holds no static
    personality->voice mapping; `voice` is validated (not just parsed) in
    tts/voice_config.py, not here — so every invalid-voice case (missing
    OR blank) returns the same TTS_006 error code.

    `speed`/`pitch` arrive as JSON *numbers* from Go
    (TTSWorkerClient.SynthesizeStream sends float32/int8), but strings are
    also accepted for manual testing/other clients — both are normalized
    in tts/voice_config.py. `personality` is optional and used only as a
    label in logs/metrics.
    """

    text: str = Field(min_length=1, max_length=8000)
    voice: str | None = None
    speed: float | int | str | None = None
    pitch: float | int | str | None = None
    lang: str | None = None
    personality: str | None = None

    @field_validator("text")
    @classmethod
    def _strip_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be blank")
        return v

# ── TTS outbound ─────────────────────────────────────────────────────────


class TTSDoneEvent(BaseModel):
    event: Literal["done"] = "done"


class TTSErrorEvent(BaseModel):
    event: Literal["error"] = "error"
    code: str = ""
    recoverable: bool = True
    retryable: bool = False
    message: str


HEALTH_OK = "ok"
HEALTH_DEGRADED = "degraded"
HEALTH_DOWN = "down"


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    service: Literal["stt", "tts"]
    engine: str
    active_sessions: int
    uptime_seconds: float
    detail: dict[str, object] = Field(default_factory=dict)
