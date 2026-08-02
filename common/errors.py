"""
Structured error codes instead of bare `raise Exception(...)`.

Error events carry the structured error contract (`code`, `recoverable`,
`retryable`, `message`) so the Go gateway can map worker failures into
runtime error events (03_SYSTEM_CONTRACTS.md §Error Contract) without
parsing message strings. `message` still embeds the code prefix for
human-readable logs.
"""
from __future__ import annotations

from enum import Enum

from common.protocol import STTErrorEvent, TTSErrorEvent


class STTErrorCode(str, Enum):
    MODEL_NOT_LOADED = "STT_001"
    INVALID_PCM = "STT_002"
    SESSION_CLOSED = "STT_003"
    TIMEOUT = "STT_004"
    ENGINE_FAILURE = "STT_005"


class TTSErrorCode(str, Enum):
    MODEL_NOT_LOADED = "TTS_001"
    INVALID_TEXT = "TTS_002"
    SESSION_CLOSED = "TTS_003"
    TIMEOUT = "TTS_004"
    ENGINE_FAILURE = "TTS_005"
    # Missing/empty `voice` (or unparseable speed/pitch beyond graceful
    # fallback) — fatal now: the worker has no static voice table to fall
    # back to, Go must always send a complete, valid voice config.
    INVALID_VOICE_CONFIG = "TTS_006"


# retryable = the exact same request may succeed on retry (transient);
# non-retryable = the request itself is wrong and must change first.
_RETRYABLE: frozenset[str] = frozenset(
    {"STT_001", "STT_004", "STT_005", "TTS_001", "TTS_004", "TTS_005"}
)


class WorkerError(Exception):
    def __init__(self, code: STTErrorCode | TTSErrorCode, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code.value}: {message}")


def stt_error_event(code: STTErrorCode, message: str) -> STTErrorEvent:
    return STTErrorEvent(
        code=code.value,
        recoverable=True,
        retryable=code.value in _RETRYABLE,
        message=f"{code.value}: {message}",
    )


def tts_error_event(code: TTSErrorCode, message: str) -> TTSErrorEvent:
    return TTSErrorEvent(
        code=code.value,
        recoverable=True,
        retryable=code.value in _RETRYABLE,
        message=f"{code.value}: {message}",
    )
