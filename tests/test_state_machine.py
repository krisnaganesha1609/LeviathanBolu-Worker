import pytest

from common.state_machine import InvalidTransition, STTState, STTStateMachine, TTSState, TTSStateMachine


def test_stt_happy_path():
    fsm = STTStateMachine("s1")
    fsm.transition(STTState.CONNECTED)
    fsm.transition(STTState.RECEIVING_AUDIO)
    fsm.transition(STTState.PROCESSING)
    fsm.transition(STTState.RECEIVING_AUDIO)
    fsm.transition(STTState.PROCESSING)
    fsm.transition(STTState.FINISHED)
    fsm.transition(STTState.IDLE)
    assert fsm.state == STTState.IDLE


def test_stt_illegal_transition_raises():
    fsm = STTStateMachine("s1")
    with pytest.raises(InvalidTransition):
        fsm.transition(STTState.FINISHED)  # can't jump straight from DISCONNECTED


def test_tts_happy_path():
    fsm = TTSStateMachine("t1")
    fsm.transition(TTSState.SYNTHESIZING)
    fsm.transition(TTSState.STREAMING)
    fsm.transition(TTSState.FINISHED)
    fsm.transition(TTSState.IDLE)
    assert fsm.state == TTSState.IDLE


def test_tts_illegal_transition_raises():
    fsm = TTSStateMachine("t1")
    with pytest.raises(InvalidTransition):
        fsm.transition(TTSState.STREAMING)  # must go through SYNTHESIZING first
