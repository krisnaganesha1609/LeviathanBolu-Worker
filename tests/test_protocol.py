import json

import pytest
from pydantic import ValidationError

from common.errors import STTErrorCode, TTSErrorCode, stt_error_event, tts_error_event
from common.protocol import (
    STTCancelAction,
    STTEndAction,
    STTFinalTranscript,
    STTPartialTranscript,
    TTSDoneEvent,
    TTSRequest,
)


def test_stt_end_action_matches_go_wire_format():
    obj = STTEndAction.model_validate(json.loads('{"action":"end"}'))
    assert obj.action == "end"


def test_stt_end_action_rejects_other_actions():
    with pytest.raises(ValidationError):
        STTEndAction.model_validate({"action": "start"})


def test_stt_cancel_action_matches_go_wire_format():
    obj = STTCancelAction.model_validate(json.loads('{"action":"cancel"}'))
    assert obj.action == "cancel"


def test_stt_cancel_action_rejects_other_actions():
    with pytest.raises(ValidationError):
        STTCancelAction.model_validate({"action": "end"})


def test_stt_partial_transcript_serializes_exact_shape():
    payload = STTPartialTranscript(text="Halo").model_dump()
    assert payload == {"event": "partial_transcript", "text": "Halo"}


def test_stt_final_transcript_serializes_exact_shape():
    payload = STTFinalTranscript(text="Halo dunia").model_dump()
    assert payload == {"event": "final_transcript", "text": "Halo dunia"}


def test_tts_request_matches_go_wire_format():
    # Exactly what TTSWorkerClient.SynthesizeStream sends: speed/pitch as
    # JSON numbers (float32/int8), not strings.
    obj = TTSRequest.model_validate_json(
        '{"text":"Hi","personality":"LEVIATHAN","voice":"am_onyx",'
        '"speed":0.85,"pitch":-3,"lang":"id-ID"}'
    )
    assert obj.text == "Hi"
    assert obj.personality == "LEVIATHAN"
    assert obj.voice == "am_onyx"
    assert obj.speed == 0.85
    assert obj.pitch == -3
    assert obj.lang == "id-ID"


def test_tts_request_accepts_string_speed_and_pitch():
    obj = TTSRequest.model_validate({"text": "Hi", "voice": "af_bella", "speed": "1.15", "pitch": "2"})
    assert obj.speed == "1.15"
    assert obj.pitch == "2"


def test_tts_request_personality_is_optional():
    obj = TTSRequest.model_validate({"text": "Hi"})
    assert obj.personality is None


def test_tts_request_rejects_blank_text():
    with pytest.raises(ValidationError):
        TTSRequest.model_validate({"text": "   "})


def test_tts_done_event_serializes_exact_shape():
    assert TTSDoneEvent().model_dump() == {"event": "done"}


def test_stt_error_event_carries_structured_contract():
    payload = stt_error_event(STTErrorCode.TIMEOUT, "session too long").model_dump()
    assert payload["event"] == "error"
    assert payload["code"] == "STT_004"
    assert payload["recoverable"] is True
    assert payload["retryable"] is True
    assert payload["message"].startswith("STT_004: ")


def test_tts_invalid_voice_config_error_is_not_retryable():
    payload = tts_error_event(TTSErrorCode.INVALID_VOICE_CONFIG, "voice missing").model_dump()
    assert payload["code"] == "TTS_006"
    assert payload["retryable"] is False
