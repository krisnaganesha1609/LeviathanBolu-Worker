"""
TTS inference engines.

`TTSEngine.synthesize()` is an async generator: it splits `text` into
sentences, synthesizes each one (in a thread executor — Kokoro/ONNX
inference is blocking/CPU-bound), resamples to the pipeline's 16kHz mono
contract, and yields raw PCM16LE bytes per sentence. `tts/streamer.py`
re-chunks whatever comes out of here into exact 20ms frames — this module
does NOT need to worry about frame alignment.

  - DummyEngine    No ML deps required. Synthesizes a simple sine-wave
                    tone scaled by sentence length so the full WS/protocol
                    pipeline is testable without downloading Kokoro
                    weights. Selected via TTS_ENGINE=dummy (default).

  - KokoroEngine    Wraps kokoro_onnx.Kokoro(model_path, voices_path).
                    Selected via TTS_ENGINE=kokoro. Requires
                    requirements-tts.txt (kokoro-onnx, soundfile, scipy)
                    and the two Kokoro model files
                    (kokoro-v1.0.onnx, voices-v1.0.bin) at TTS_MODEL_PATH /
                    TTS_VOICES_PATH.
"""
from __future__ import annotations

import abc
import asyncio
import re
import time
from typing import Any, AsyncIterator

import numpy as np

from common.audio import int16_to_pcm16
from common.config import AudioSettings, TTSSettings
from common.logger import get_logger
from tts.personalities import PersonalityVoiceConfig

log = get_logger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    return parts or [text.strip()]


class TTSEngine(abc.ABC):
    @abc.abstractmethod
    async def warm_up(self) -> None: ...

    @abc.abstractmethod
    def synthesize(
        self, text: str, voice_config: PersonalityVoiceConfig
    ) -> AsyncIterator[bytes]:
        """Async-generator: yields raw PCM16LE mono bytes at the pipeline's
        configured sample rate, one chunk per sentence."""

    @abc.abstractmethod
    def health(self) -> dict[str, Any]: ...


class DummyEngine(TTSEngine):
    name = "dummy"

    def __init__(self, audio_settings: AudioSettings) -> None:
        self.audio_settings = audio_settings
        self._ready = False

    async def warm_up(self) -> None:
        await asyncio.sleep(0)
        self._ready = True
        log.info("tts.engine.dummy.ready")

    async def synthesize(
        self, text: str, voice_config: PersonalityVoiceConfig
    ) -> AsyncIterator[bytes]:
        sr = self.audio_settings.sample_rate
        for sentence in split_sentences(text):
            # Deterministic filler tone: duration scales with sentence
            # length, pitch nudged by voice_config.pitch just so DummyEngine
            # output visibly reacts to personality selection in tests/demos.
            duration_s = max(0.2, min(3.0, len(sentence) * 0.05))
            freq = 220.0 * (2 ** (voice_config.pitch / 12.0))
            n = int(sr * duration_s)
            t = np.linspace(0, duration_s, n, endpoint=False)
            wave = (0.2 * np.sin(2 * np.pi * freq * t * voice_config.speed)).astype(np.float32)
            samples = (wave * 32767).astype(np.int16)
            await asyncio.sleep(0)  # yield control
            yield int16_to_pcm16(samples)

    def health(self) -> dict[str, Any]:
        return {"engine": self.name, "ready": self._ready}


class KokoroEngine(TTSEngine):
    name = "kokoro"

    def __init__(self, settings: TTSSettings, audio_settings: AudioSettings) -> None:
        self.settings = settings
        self.audio_settings = audio_settings
        self._kokoro: Any = None
        self._native_sample_rate: int | None = None
        self._load_lock = asyncio.Lock()
        self._load_error: str | None = None

    async def _ensure_loaded(self) -> None:
        if self._kokoro is not None or self._load_error is not None:
            return
        async with self._load_lock:
            if self._kokoro is not None or self._load_error is not None:
                return
            loop = asyncio.get_running_loop()
            try:
                self._kokoro = await loop.run_in_executor(None, self._load_blocking)
            except Exception as exc:  # pragma: no cover - depends on env/weights
                self._load_error = str(exc)
                log.error("tts.engine.kokoro.load_failed", error=str(exc))
                raise

    def _load_blocking(self) -> Any:
        from kokoro_onnx import Kokoro  # type: ignore[import-not-found]

        start = time.perf_counter()
        kokoro = Kokoro(self.settings.model_path, self.settings.voices_path)
        log.info(
            "tts.engine.kokoro.loaded",
            model_path=self.settings.model_path,
            elapsed_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        return kokoro

    async def warm_up(self) -> None:
        try:
            await self._ensure_loaded()
        except Exception:
            pass  # health endpoint reports degraded; retried lazily on first request

    async def synthesize(
        self, text: str, voice_config: PersonalityVoiceConfig
    ) -> AsyncIterator[bytes]:
        await self._ensure_loaded()
        loop = asyncio.get_running_loop()
        for sentence in split_sentences(text):
            samples, native_sr = await loop.run_in_executor(
                None, self._synth_blocking, sentence, voice_config
            )
            pcm16 = self._postprocess(samples, native_sr, voice_config)
            yield int16_to_pcm16(pcm16)

    def _synth_blocking(
        self, sentence: str, voice_config: PersonalityVoiceConfig
    ) -> tuple[np.ndarray, int]:
        samples, sample_rate = self._kokoro.create(
            sentence,
            voice=voice_config.voice,
            speed=voice_config.speed,
            lang=voice_config.lang,
        )
        return np.asarray(samples, dtype=np.float32), int(sample_rate)

    def _postprocess(
        self, samples: np.ndarray, native_sr: int, voice_config: PersonalityVoiceConfig
    ) -> np.ndarray:
        target_sr = self.audio_settings.sample_rate
        if native_sr != target_sr:
            samples = _resample(samples, native_sr, target_sr)
        if voice_config.pitch:
            samples = _pitch_shift(samples, target_sr, voice_config.pitch)
        clipped = np.clip(samples, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16)

    def health(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "ready": self._kokoro is not None,
            "load_error": self._load_error,
            "model_path": self.settings.model_path,
        }


def _resample(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    try:
        from scipy.signal import resample_poly  # type: ignore[import-not-found]
        from math import gcd

        g = gcd(src_sr, dst_sr)
        return resample_poly(samples, dst_sr // g, src_sr // g).astype(np.float32)
    except ImportError:  # pragma: no cover - scipy is in requirements-tts.txt
        log.warning("tts.resample.scipy_missing_using_naive")
        n_dst = int(len(samples) * dst_sr / src_sr)
        idx = np.linspace(0, len(samples) - 1, n_dst)
        return np.interp(idx, np.arange(len(samples)), samples).astype(np.float32)


def _pitch_shift(samples: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    """Best-effort pitch shift. No-op (with a one-time warning) if the
    optional `librosa` extra (requirements-tts.txt) isn't installed."""
    try:
        import librosa  # type: ignore[import-not-found]

        return librosa.effects.pitch_shift(
            samples, sr=sample_rate, n_steps=semitones
        ).astype(np.float32)
    except ImportError:
        log.warning("tts.pitch_shift.librosa_missing_skipping", semitones=semitones)
        return samples


def build_engine(settings: TTSSettings, audio_settings: AudioSettings) -> TTSEngine:
    if settings.engine == "kokoro":
        return KokoroEngine(settings, audio_settings)
    return DummyEngine(audio_settings)
