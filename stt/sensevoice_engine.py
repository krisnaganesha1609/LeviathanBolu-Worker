"""
STT inference engines.

`STTEngine` is the abstract interface the worker pipeline talks to.
Two implementations:

  - DummyEngine       No ML deps required. Produces a deterministic,
                      audio-derived placeholder transcript. Lets you run
                      the entire WS/ring-buffer/VAD/protocol pipeline in
                      CI or local dev without downloading any model
                      weights. Selected via STT_ENGINE=dummy (default).

  - SenseVoiceEngine  Wraps FunASR's `AutoModel(model="iic/SenseVoiceSmall",
                      vad_model="fsmn-vad", ...)`. Model load + inference
                      are both blocking/CPU-bound, so they're always run
                      in a thread executor and never block the event loop.
                      Selected via STT_ENGINE=sensevoice.

Swap engines purely via config — the worker (worker.py) never imports a
concrete engine class directly.
"""
from __future__ import annotations

import abc
import asyncio
import time
from typing import Any

import numpy as np

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
        self, pcm_int16: np.ndarray, *, sample_rate: int, is_final: bool
    ) -> TranscriptResult:
        """Transcribe the audio buffered so far. `is_final` hints at effort budget
        (a partial can afford to be cheaper/lower-quality than a final)."""

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
        self, pcm_int16: np.ndarray, *, sample_rate: int, is_final: bool
    ) -> TranscriptResult:
        duration_s = len(pcm_int16) / float(sample_rate) if sample_rate else 0.0
        kind = "final" if is_final else "partial"
        text = f"[dummy-stt {kind} transcript: {duration_s:.2f}s audio, {len(pcm_int16)} samples]"
        return TranscriptResult(text=text, is_final=is_final, language="und", confidence=1.0)

    def health(self) -> dict[str, Any]:
        return {"engine": self.name, "ready": self._ready}


class SenseVoiceEngine(STTEngine):
    """
    Real FunASR/SenseVoice engine.

    pip install -r requirements-stt.txt (torch, torchaudio, funasr) before
    using STT_ENGINE=sensevoice.
    """

    name = "sensevoice"

    def __init__(self, settings: STTSettings) -> None:
        self.settings = settings
        self._model: Any = None
        self._postprocess: Any = None
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
                log.error("stt.engine.sensevoice.load_failed", error=str(exc))
                raise

    def _load_model_blocking(self) -> Any:
        # Imported lazily so this module is importable without the ML stack
        # installed (DummyEngine is usable in a minimal environment).
        from funasr import AutoModel  # type: ignore[import-not-found]
        from funasr.utils.postprocess_utils import (  # type: ignore[import-not-found]
            rich_transcription_postprocess,
        )

        start = time.perf_counter()
        model = AutoModel(
            model=self.settings.model,
            vad_model=self.settings.vad_model,
            device=self.settings.device,
        )
        self._postprocess = rich_transcription_postprocess
        log.info(
            "stt.engine.sensevoice.loaded",
            model=self.settings.model,
            device=self.settings.device,
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
        self, pcm_int16: np.ndarray, *, sample_rate: int, is_final: bool
    ) -> TranscriptResult:
        await self._ensure_loaded()
        loop = asyncio.get_running_loop()
        waveform = (pcm_int16.astype(np.float32) / 32768.0).copy()
        text = await loop.run_in_executor(
            None, self._infer_blocking, waveform, sample_rate
        )
        return TranscriptResult(text=text, is_final=is_final, language=self.settings.language)

    def _infer_blocking(self, waveform: np.ndarray, sample_rate: int) -> str:
        result = self._model.generate(
            input=waveform,
            cache={},
            language=self.settings.language,
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
        )
        if not result:
            return ""
        raw_text = result[0].get("text", "")
        if self._postprocess is not None:
            return str(self._postprocess(raw_text))
        return str(raw_text)

    def health(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "ready": self._model is not None,
            "load_error": self._load_error,
            "model": self.settings.model,
            "device": self.settings.device,
        }


def build_engine(settings: STTSettings) -> STTEngine:
    if settings.engine == "sensevoice":
        return SenseVoiceEngine(settings)
    return DummyEngine(settings)
