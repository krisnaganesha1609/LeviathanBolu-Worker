import json

import pytest
from pydantic import ValidationError

from common.protocol import (
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


def test_stt_partial_transcript_serializes_exact_shape():
    payload = STTPartialTranscript(text="Halo").model_dump()
    assert payload == {"event": "partial_transcript", "text": "Halo"}


def test_stt_final_transcript_serializes_exact_shape():
    payload = STTFinalTranscript(text="Halo dunia").model_dump()
    assert payload == {"event": "final_transcript", "text": "Halo dunia"}


def test_tts_request_matches_go_wire_format():
    obj = TTSRequest.model_validate_json('{"text":"Hi","personality":"LEVIATHAN"}')
    assert obj.text == "Hi"
    assert obj.personality == "LEVIATHAN"


def test_tts_request_defaults_personality():
    obj = TTSRequest.model_validate({"text": "Hi"})
    assert obj.personality == "LEVIATHAN"


def test_tts_request_rejects_blank_text():
    with pytest.raises(ValidationError):
        TTSRequest.model_validate({"text": "   "})


def test_tts_done_event_serializes_exact_shape():
    assert TTSDoneEvent().model_dump() == {"event": "done"}
