"""
Voice parameters for one TTS request — no static personality/voice
mapping lives in this worker anymore. Go owns that data entirely (see
`user_settings.domain.go`'s `WakeWordConfig`: Word, Personality, Voice,
Speed, Pitch, Language) and sends the fully-resolved voice, speed, pitch,
and language on every request. The worker just validates and uses them.

`personality` is carried through purely as a label for logs/metrics — it
is never used to look anything up here.
"""
from __future__ import annotations

from pydantic import BaseModel

from common.logger import get_logger

log = get_logger(__name__)

# Fallbacks below are numeric identity values (1.0 speed = unchanged,
# 0.0 pitch = unchanged, "en-us" = a language tag, not a voice choice) —
# they exist purely so a missing/malformed field degrades gracefully
# instead of crashing the request. They are not personality data.
_DEFAULT_SPEED = 1.0
_DEFAULT_PITCH = 0.0
_DEFAULT_LANG = "en-us"


class VoiceConfig(BaseModel):
    voice: str
    speed: float = _DEFAULT_SPEED
    pitch: float = _DEFAULT_PITCH
    lang: str = _DEFAULT_LANG


class InvalidVoiceConfig(ValueError):
    pass


def _parse_float(raw: str | float | None, *, default: float, field: str, session_id: str) -> float:
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        log.warning(
            "tts.voice_config.unparseable_field",
            session_id=session_id,
            field=field,
            raw=raw,
            using_default=default,
        )
        return default


def parse_voice_config(
    *,
    voice: str | None,
    speed: str | float | None,
    pitch: str | float | None,
    lang: str | None,
    session_id: str,
) -> VoiceConfig:
    """
    Builds a VoiceConfig straight from the request Go sent. `voice` is the
    one field with no sensible fallback (there's no default voice baked
    into this worker) — an empty/missing voice is a hard error, since
    without it there is nothing to synthesize with.
    """
    voice = (voice or "").strip()
    if not voice:
        raise InvalidVoiceConfig("voice is required and was empty/missing")

    return VoiceConfig(
        voice=voice,
        speed=_parse_float(speed, default=_DEFAULT_SPEED, field="speed", session_id=session_id),
        pitch=_parse_float(pitch, default=_DEFAULT_PITCH, field="pitch", session_id=session_id),
        lang=(lang or _DEFAULT_LANG).strip() or _DEFAULT_LANG,
    )