"""
Explicit state machines for both workers, per ADR-003.

Illegal transitions raise immediately rather than being silently allowed —
during development that catches pipeline bugs (e.g. trying to stream TTS
audio before Synthesizing) as loudly as possible instead of producing
subtly wrong wire output.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from common.logger import get_logger

log = get_logger(__name__)


class STTState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    RECEIVING_AUDIO = "receiving_audio"
    PROCESSING = "processing"
    FINISHED = "finished"
    IDLE = "idle"


class TTSState(str, Enum):
    IDLE = "idle"
    SYNTHESIZING = "synthesizing"
    STREAMING = "streaming"
    FINISHED = "finished"


_STT_TRANSITIONS: dict[STTState, set[STTState]] = {
    STTState.DISCONNECTED: {STTState.CONNECTED},
    STTState.CONNECTED: {STTState.RECEIVING_AUDIO, STTState.DISCONNECTED},
    STTState.RECEIVING_AUDIO: {STTState.RECEIVING_AUDIO, STTState.PROCESSING, STTState.DISCONNECTED},
    STTState.PROCESSING: {STTState.RECEIVING_AUDIO, STTState.FINISHED, STTState.DISCONNECTED},
    STTState.FINISHED: {STTState.IDLE, STTState.DISCONNECTED},
    STTState.IDLE: {STTState.DISCONNECTED},
}

_TTS_TRANSITIONS: dict[TTSState, set[TTSState]] = {
    TTSState.IDLE: {TTSState.SYNTHESIZING},
    TTSState.SYNTHESIZING: {TTSState.STREAMING, TTSState.FINISHED},
    TTSState.STREAMING: {TTSState.FINISHED},
    TTSState.FINISHED: {TTSState.IDLE},
}


class InvalidTransition(RuntimeError):
    pass


class _StateMachine:
    _transitions: dict[Any, set[Any]]

    def __init__(self, initial: Any, *, session_id: str, kind: str) -> None:
        self.state = initial
        self._session_id = session_id
        self._kind = kind

    def transition(self, new_state: Any) -> None:
        allowed = self._transitions.get(self.state, set())
        if new_state not in allowed:
            raise InvalidTransition(
                f"{self._kind} session {self._session_id}: illegal transition "
                f"{self.state.value} -> {new_state.value}"
            )
        log.debug(
            f"{self._kind}.state_transition",
            session_id=self._session_id,
            from_state=self.state.value,
            to_state=new_state.value,
        )
        self.state = new_state


class STTStateMachine(_StateMachine):
    _transitions = _STT_TRANSITIONS

    def __init__(self, session_id: str) -> None:
        super().__init__(STTState.DISCONNECTED, session_id=session_id, kind="stt")


class TTSStateMachine(_StateMachine):
    _transitions = _TTS_TRANSITIONS

    def __init__(self, session_id: str) -> None:
        super().__init__(TTSState.IDLE, session_id=session_id, kind="tts")
