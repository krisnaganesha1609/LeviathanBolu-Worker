"""
STT inference engines.

`STTEngine` is the abstract interface the worker pipeline talks to.
Two implementations:

  - DummyEngine     No ML deps required. Produces a deterministic,
                    audio-derived placeholder transcript. Lets you run
                    the entire WS/ring-buffer/VAD/protocol pipeline in CI
                    or local dev without downloading any model weights.
                    Selected via STT_ENGINE=dummy (default).

  - WhisperEngine   Wraps faster-whisper (CTranslate2 backend — no torch
                    dependency at all, which matters a lot on a
                    RAM-constrained VPS). Model load + inference are both
                    blocking/CPU-bound, so they're always run in a thread
                    executor and never block the event loop. Selected via
                    STT_ENGINE=whisper.

Swap engines purely via config — the worker (worker.py) never imports a
concrete engine class directly.

Why faster-whisper over SenseVoice (see docs/ADR-004): SenseVoiceSmall's
language-ID mechanism only covers zh/en/yue/ja/ko — there is no
Indonesian token, so short/ambiguous audio would get mis-tagged into
Mandarin/Japanese and the model would hallucinate plausible-looking CJK
text rather than emit nothing. faster-whisper's underlying Whisper models
have an explicit `id` (Indonesian) language code, and forcing a language
(STT_LANGUAGE=en/id/...) instead of "auto" avoids language-ID guessing
entirely for short utterances like wake-word checks.
"""
from __future__ import annotations

import abc
import asyncio
import time
from typing import Any

import numpy as np

from common.audio import normalize_gain
from common.config import STTSettings
from common.logger import get_logger
from stt.models import TranscriptResult

log = get_logger(__name__)


class STTEngine(abc.ABC):
    @abc.abstractmethod
    async def warm_up(self) -> None:
        """Called once at process startup so the first real request isn't slow."""

    @abc.abstractmethod
    async def transcribe(
        self, pcm_int16: np.ndarray, *, sample_rate: int, is_final: bool, language: str | None = None
    ) -> TranscriptResult:
        """Transcribe the audio buffered so far. `is_final` hints at effort budget
        (a partial can afford to be cheaper/lower-quality than a final).
        `language` overrides the engine's configured STT_LANGUAGE for this
        one call — used to honor a per-user language preference (e.g. Go's
        user_settings.Language) without needing a server-wide restart.
        None falls back to the engine's configured default."""

    @abc.abstractmethod
    def health(self) -> dict[str, Any]:
        """Lightweight status dict for the /health endpoint."""


class DummyEngine(STTEngine):
    """Deterministic placeholder engine — no model weights required."""

    name = "dummy"

    def __init__(self, settings: STTSettings) -> None:
        self.settings = settings
        self._ready = False

    async def warm_up(self) -> None:
        await asyncio.sleep(0)  # nothing to load
        self._ready = True
        log.info("stt.engine.dummy.ready")

    async def transcribe(
        self, pcm_int16: np.ndarray, *, sample_rate: int, is_final: bool, language: str | None = None
    ) -> TranscriptResult:
        duration_s = len(pcm_int16) / float(sample_rate) if sample_rate else 0.0
        kind = "final" if is_final else "partial"
        lang_tag = f" lang={language}" if language else ""
        text = f"[dummy-stt {kind} transcript: {duration_s:.2f}s audio, {len(pcm_int16)} samples{lang_tag}]"
        return TranscriptResult(text=text, is_final=is_final, language=language or "und", confidence=1.0)

    def health(self) -> dict[str, Any]:
        return {"engine": self.name, "ready": self._ready}


class WhisperEngine(STTEngine):
    """
    faster-whisper (CTranslate2) engine — no torch dependency.

    pip install -r requirements-stt.txt (faster-whisper + onnxruntime)
    before using STT_ENGINE=whisper. Model weights download automatically
    from Hugging Face on first load (cached under ~/.cache/huggingface).
    """

    name = "whisper"

    def __init__(self, settings: STTSettings) -> None:
        self.settings = settings
        self._model: Any = None
        self._load_lock = asyncio.Lock()
        self._load_error: str | None = None

    async def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_error is not None:
            return
        async with self._load_lock:
            if self._model is not None or self._load_error is not None:
                return
            loop = asyncio.get_running_loop()
            try:
                self._model = await loop.run_in_executor(None, self._load_model_blocking)
            except Exception as exc:  # pragma: no cover - depends on env/weights
                self._load_error = str(exc)
                log.error("stt.engine.whisper.load_failed", error=str(exc))
                raise

    def _load_model_blocking(self) -> Any:
        # Imported lazily so this module is importable without faster-whisper
        # installed (DummyEngine is usable in a minimal environment).
        from faster_whisper import WhisperModel

        start = time.perf_counter()
        model = WhisperModel(
            self.settings.model,
            device=self.settings.device,
            compute_type=self.settings.compute_type,
            cpu_threads=self.settings.cpu_threads,
        )
        log.info(
            "stt.engine.whisper.loaded",
            model=self.settings.model,
            device=self.settings.device,
            compute_type=self.settings.compute_type,
            elapsed_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        return model

    async def warm_up(self) -> None:
        try:
            await self._ensure_loaded()
        except Exception:
            # Don't crash the process at startup — health endpoint will
            # report degraded, and we retry lazily on first real request.
            pass

    async def transcribe(
        self, pcm_int16: np.ndarray, *, sample_rate: int, is_final: bool, language: str | None = None
    ) -> TranscriptResult:
        await self._ensure_loaded()
        loop = asyncio.get_running_loop()
        if self.settings.gain_normalize:
            pcm_int16 = normalize_gain(pcm_int16)
        waveform = (pcm_int16.astype(np.float32) / 32768.0).copy()
        text, detected_language = await loop.run_in_executor(
            None, self._infer_blocking, waveform, is_final, language
        )
        return TranscriptResult(text=text, is_final=is_final, language=detected_language)

    def _infer_blocking(
        self, waveform: np.ndarray, is_final: bool, language_override: str | None
    ) -> tuple[str, str | None]:
        # Per-session override (e.g. Go's user_settings.Language) wins over
        # the engine's configured STT_LANGUAGE default; "auto" (from either
        # source) means let Whisper guess.
        effective_language = language_override or self.settings.language
        language = None if effective_language == "auto" else effective_language
        # Partials favor latency (greedy decode); finals get full beam
        # search. Whisper's own bundled Silero VAD (vad_filter=True) trims
        # any residual leading/trailing silence inside the utterance —
        # our own SileroVAD gate (stt/silero_vad.py) already screened out
        # utterances with NO speech at all before this is ever called.
        segments, info = self._model.transcribe(
            waveform,
            language=language,
            beam_size=1 if not is_final else self.settings.beam_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False,
            initial_prompt=self.settings.initial_prompt or None,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text, getattr(info, "language", None)

    def health(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "ready": self._model is not None,
            "load_error": self._load_error,
            "model": self.settings.model,
            "device": self.settings.device,
            "compute_type": self.settings.compute_type,
        }


def build_engine(settings: STTSettings) -> STTEngine:
    if settings.engine == "whisper":
        return WhisperEngine(settings)
    return DummyEngine(settings)
