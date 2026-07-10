"""
Wire protocol for both workers. These models are the single source of
truth for what the Go orchestrator (assistant/stt_tts.go) sends/expects —
keep them in lockstep with that file.

STT  ws://<host>:9001/stt
  IN  (binary)         raw PCM16LE mono frames (already Opus-decoded by Go)
  IN  (text/json)       {"action": "end"}
  OUT (text/json)       {"event": "partial_transcript", "text": "..."}
  OUT (text/json)       {"event": "final_transcript", "text": "..."}

TTS  ws://<host>:9002/tts
  IN  (text/json)       {"text": "...", "personality": "LEVIATHAN"}
  OUT (binary)          raw PCM16LE mono frames (Go re-encodes to Opus)
  OUT (text/json)       {"event": "done"}

Notes:
- Go's read loops only special-case specific event names ("final_transcript",
  "done") and silently ignore/continue on anything else, so we can add
  additive events (e.g. "error", "partial_transcript") without breaking it.
- Go NEVER sends anything other than raw PCM binary + the "end" action to
  STT, and NEVER sends anything other than the initial JSON request to TTS.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ── STT inbound ──────────────────────────────────────────────────────────


class STTEndAction(BaseModel):
    action: Literal["end"]


# ── STT outbound ─────────────────────────────────────────────────────────


class STTPartialTranscript(BaseModel):
    event: Literal["partial_transcript"] = "partial_transcript"
    text: str


class STTFinalTranscript(BaseModel):
    event: Literal["final_transcript"] = "final_transcript"
    text: str


class STTErrorEvent(BaseModel):
    """Additive — safe for Go to ignore, useful for debugging/monitoring."""

    event: Literal["error"] = "error"
    message: str


# ── TTS inbound ──────────────────────────────────────────────────────────


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    personality: str = "LEVIATHAN"

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
