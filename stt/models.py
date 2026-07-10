from __future__ import annotations

from dataclasses import dataclass, field

from common.audio import AudioSession


@dataclass
class TranscriptResult:
    text: str
    is_final: bool
    language: str | None = None
    confidence: float | None = None


@dataclass
class STTSessionState:
    """Per-connection state, distinct from AudioSession (which is generic)."""

    session: AudioSession = field(default_factory=lambda: AudioSession("stt"))
    ended: bool = False
    final_emitted: bool = False
    partials_emitted: int = 0
