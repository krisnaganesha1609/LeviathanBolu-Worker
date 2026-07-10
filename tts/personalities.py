"""
Loads config/personalities.yaml into typed PersonalityVoiceConfig objects.
Unknown personality names fall back to the configured default (never a
hard failure — a typo'd personality name shouldn't break a voice reply).
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from common.errors import TTSErrorCode
from common.logger import get_logger

log = get_logger(__name__)


class PersonalityVoiceConfig(BaseModel):
    voice: str
    speed: float = 1.0
    pitch: float = 0.0
    lang: str = "en-us"


class PersonalityRegistry:
    def __init__(self, path: str, default: str = "LEVIATHAN") -> None:
        self._path = path
        self._default_name = default
        self._map: dict[str, PersonalityVoiceConfig] = {}
        self.reload()

    def reload(self) -> None:
        p = Path(self._path)
        if not p.exists():
            log.warning("tts.personalities.file_missing", path=self._path)
            self._map = {
                self._default_name: PersonalityVoiceConfig(voice="am_adam", speed=0.9, pitch=-1)
            }
            return
        raw = yaml.safe_load(p.read_text()) or {}
        self._default_name = raw.get("default", self._default_name)
        entries = raw.get("personalities", {})
        self._map = {
            name.upper(): PersonalityVoiceConfig.model_validate(cfg)
            for name, cfg in entries.items()
        }
        if self._default_name.upper() not in self._map:
            raise ValueError(
                f"personalities.yaml default '{self._default_name}' has no matching entry"
            )
        log.info("tts.personalities.loaded", count=len(self._map), path=self._path)

    def resolve(self, name: str) -> tuple[PersonalityVoiceConfig, bool]:
        """Returns (config, matched). matched=False means we fell back to default."""
        key = (name or "").strip().upper()
        if key in self._map:
            return self._map[key], True
        log.warning(
            "tts.personalities.unknown",
            requested=name,
            code=TTSErrorCode.UNKNOWN_PERSONALITY.value,
            fallback=self._default_name,
        )
        return self._map[self._default_name.upper()], False

    @property
    def default_name(self) -> str:
        return self._default_name

    def names(self) -> list[str]:
        return list(self._map.keys())
