"""
VAD-aware wrapper around common.audio.RingBuffer.

We use a cheap energy-based VAD (RMS threshold) rather than pulling in
webrtcvad/silero as a hard dependency — it's good enough to decide "is
there enough new speech + trailing silence to run a partial/final pass",
which is all this layer is responsible for. The actual transcription
quality comes from the ASR engine, not this gate.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from common.audio import RingBuffer, rms_energy


@dataclass
class VADConfig:
    sample_rate: int
    silence_ms: int = 600
    rms_threshold: int = 350
    partial_interval_ms: int = 800


class SpeechSegmenter:
    """
    Feed it PCM int16 frames; it tells you when a "partial" pass is worth
    running (enough new audio + a hint of a pause) and tracks whether the
    tail of the buffer is currently silent, which the worker uses to decide
    when to emit a partial without waiting for the client's "end".
    """

    def __init__(self, config: VADConfig, max_buffer_seconds: float = 30.0) -> None:
        self.config = config
        max_samples = int(config.sample_rate * max_buffer_seconds)
        self.buffer = RingBuffer(max_samples=max_samples)
        self._last_partial_at = 0.0
        self._silence_window_samples = int(config.sample_rate * config.silence_ms / 1000)
        self._has_speech_since_partial = False

    def push(self, samples: np.ndarray) -> None:
        if rms_energy(samples) >= self.config.rms_threshold:
            self._has_speech_since_partial = True
        self.buffer.push(samples)

    def is_tail_silent(self) -> bool:
        tail = self.buffer.tail(self._silence_window_samples)
        if len(tail) < self._silence_window_samples:
            return False
        return rms_energy(tail) < self.config.rms_threshold

    def should_emit_partial(self) -> bool:
        """Rate-limited: only offer a partial every `partial_interval_ms`,
        and only if there's been speech energy since the last one."""
        now = time.monotonic()
        elapsed_ms = (now - self._last_partial_at) * 1000
        if elapsed_ms < self.config.partial_interval_ms:
            return False
        if not self._has_speech_since_partial:
            return False
        return True

    def mark_partial_emitted(self) -> None:
        self._last_partial_at = time.monotonic()
        self._has_speech_since_partial = False

    def snapshot(self) -> np.ndarray:
        """All buffered audio so far, without clearing it."""
        return self.buffer.peek_all()

    def sliding_snapshot(self, window_seconds: float) -> np.ndarray:
        """
        Last `window_seconds` of buffered audio (ADR-001 sliding window),
        used for partial passes so a long utterance doesn't make every
        partial re-transcribe the entire session. Finals still use the
        full buffer via snapshot()/drain().
        """
        n_samples = int(self.config.sample_rate * window_seconds)
        return self.buffer.tail(n_samples)

    def drain(self) -> np.ndarray:
        return self.buffer.drain()

    def __len__(self) -> int:
        return len(self.buffer)
