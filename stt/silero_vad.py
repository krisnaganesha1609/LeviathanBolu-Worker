"""
Silero VAD, via the ONNX model already bundled inside faster-whisper (no
extra torch/silero-vad package needed — faster-whisper ships
`silero_vad_v6.onnx` and a small onnxruntime-based wrapper around it,
exposed as `faster_whisper.vad`).

This is used for exactly one thing: deciding "is there actually speech in
this buffered utterance at all" right before we'd spend a (relatively
expensive) Whisper inference call on it — see
STTSession._too_quiet_or_short in stt/worker.py. This directly replaces
the RMS-energy-threshold heuristic that turned out to be the root cause
of the CJK-hallucination issue (a naive amplitude threshold can't tell
"wind gust" from "quiet speech"; a trained VAD model can).

The per-frame, rate-limiting logic in stt/ring_buffer.py::SpeechSegmenter
(deciding *when* to attempt a partial) is deliberately left on the cheap
RMS heuristic — that runs on every single 20ms frame, and Silero VAD
wants a wider window (tens of ms minimum) to be meaningful/efficient, so
running it per-frame would be both wasteful and a poor fit. Silero VAD is
reserved for the one-shot "is this buffer worth transcribing" decision.
"""
from __future__ import annotations

import numpy as np

from common.logger import get_logger

log = get_logger(__name__)


class SileroVAD:
    def __init__(self, threshold: float = 0.5, min_speech_duration_ms: int = 100) -> None:
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self._vad_options = None  # lazy: importing faster_whisper.vad loads onnxruntime

    def warm_up(self) -> None:
        """Load the VadOptions + trigger the ONNX session's lazy init with a
        throwaway inference, so the first real request doesn't pay for it.
        Call this from a thread executor at startup (see stt/server.py) —
        it's blocking, same as any other model load (ADR-004)."""
        self._ensure_loaded()
        silence = np.zeros(1600, dtype=np.int16)  # 100ms @ 16kHz, cheap
        self.has_speech(silence, 16000)
        log.info("stt.silero_vad.ready", threshold=self.threshold)

    def _ensure_loaded(self):
        if self._vad_options is not None:
            return
        from faster_whisper.vad import VadOptions

        self._vad_options = VadOptions(
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            speech_pad_ms=0,  # we only need a yes/no + rough timestamps, not padded audio
        )

    def speech_timestamps(self, audio_int16: np.ndarray, sample_rate: int) -> list[dict]:
        """Returns [{"start": sample_idx, "end": sample_idx}, ...] for each
        detected speech chunk. Empty list = no speech found at all."""
        if audio_int16.size == 0:
            return []
        self._ensure_loaded()
        from faster_whisper.vad import get_speech_timestamps

        waveform = audio_int16.astype(np.float32) / 32768.0
        try:
            return get_speech_timestamps(waveform, self._vad_options, sampling_rate=sample_rate)
        except Exception as exc:  # pragma: no cover - defensive, e.g. onnxruntime missing
            log.error("stt.silero_vad.failed", error=str(exc))
            # Fail open: if the VAD itself breaks, don't silently drop every
            # utterance — let it through to the ASR engine instead.
            return [{"start": 0, "end": len(audio_int16)}]

    def has_speech(self, audio_int16: np.ndarray, sample_rate: int) -> bool:
        return len(self.speech_timestamps(audio_int16, sample_rate)) > 0
