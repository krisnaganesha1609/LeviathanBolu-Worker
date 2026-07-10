"""
Structured error codes instead of bare `raise Exception(...)`.

Go only ever branches on the JSON `event` field ("final_transcript" /
"done"); any other event, including "error", is safely ignored by its
current read loop. So emitting `{"event":"error","code":"STT_002",...}`
is a pure *addition* to the wire contract — safe today, and gives Go
something explicit to branch on if/when it's updated to handle errors.
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
    UNKNOWN_PERSONALITY = "TTS_006"  # non-fatal: we fall back to default, still logged


class WorkerError(Exception):
    def __init__(self, code: STTErrorCode | TTSErrorCode, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code.value}: {message}")


def stt_error_event(code: STTErrorCode, message: str) -> STTErrorEvent:
    return STTErrorEvent(message=f"{code.value}: {message}")


def tts_error_event(code: TTSErrorCode, message: str) -> TTSErrorEvent:
    return TTSErrorEvent(message=f"{code.value}: {message}")
