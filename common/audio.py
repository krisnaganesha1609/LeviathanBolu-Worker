"""
Low-level audio building blocks shared by both workers.

These are deliberately dependency-free (stdlib + numpy only) so they can be
unit tested without any ML stack installed.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


class PCMValidationError(ValueError):
    pass


def validate_pcm16(data: bytes, *, allow_empty: bool = False) -> None:
    """
    Cheap sanity checks on a raw PCM16LE payload before it enters the
    pipeline. Cheap on purpose — this runs on every inbound frame.
    """
    if not data:
        if allow_empty:
            return
        raise PCMValidationError("empty PCM payload")
    if len(data) % 2 != 0:
        raise PCMValidationError(f"PCM payload not 16-bit aligned: {len(data)} bytes")


def pcm16_to_int16(data: bytes) -> np.ndarray:
    """bytes (little-endian int16) -> np.int16 array (copy, safe to mutate)."""
    return np.frombuffer(data, dtype="<i2").astype(np.int16, copy=True)


def int16_to_pcm16(samples: np.ndarray) -> bytes:
    """np.int16 array -> little-endian PCM bytes."""
    return samples.astype("<i2", copy=False).tobytes()


def rms_energy(samples: np.ndarray) -> float:
    """Root-mean-square energy of an int16 sample array; 0.0 for empty input."""
    if samples.size == 0:
        return 0.0
    f = samples.astype(np.float64)
    return float(np.sqrt(np.mean(f * f)))


class RingBuffer:
    """
    A simple growable ring buffer of int16 audio samples.

    Not a fixed-capacity ring in the classic sense — it's a deque-backed
    buffer with an optional `max_samples` cap so long-running sessions
    can't grow unbounded memory. Old samples are dropped from the left
    once the cap is hit (oldest audio is least useful for live STT).
    """

    def __init__(self, max_samples: int | None = None) -> None:
        self._buf: deque[np.int16] = deque()
        self.max_samples = max_samples
        self._dropped = 0

    def push(self, samples: np.ndarray) -> None:
        self._buf.extend(samples.tolist())
        if self.max_samples is not None:
            overflow = len(self._buf) - self.max_samples
            if overflow > 0:
                for _ in range(overflow):
                    self._buf.popleft()
                self._dropped += overflow

    def peek_all(self) -> np.ndarray:
        return np.array(self._buf, dtype=np.int16)

    def drain(self) -> np.ndarray:
        """Return and clear all buffered samples."""
        arr = self.peek_all()
        self._buf.clear()
        return arr

    def tail(self, n_samples: int) -> np.ndarray:
        """Last n_samples without removing them (used for silence detection)."""
        if n_samples <= 0:
            return np.array([], dtype=np.int16)
        n = min(n_samples, len(self._buf))
        return np.array(list(self._buf)[-n:], dtype=np.int16)

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def dropped_samples(self) -> int:
        return self._dropped


class ChunkNormalizer:
    """
    Re-slices arbitrary-sized PCM16 chunks into fixed-size frames.

    This exists specifically because TTS engines (Kokoro included) do NOT
    guarantee their output chunk size lines up with the 320-sample/20ms
    frame the Go orchestrator's Opus encoder needs (see the CATATAN comment
    in assistant/stt_tts.go — `enc.Encode` requires a fixed frame size).
    Feed it PCM bytes of any length via `push()`, and pull exact
    `frame_bytes`-sized frames via `pop_frames()`. Whatever is left over
    that doesn't fill a whole frame stays buffered until more data (or
    `flush()`, which zero-pads the final partial frame).
    """

    def __init__(self, frame_samples: int) -> None:
        if frame_samples <= 0:
            raise ValueError("frame_samples must be > 0")
        self.frame_samples = frame_samples
        self.frame_bytes = frame_samples * 2
        self._buf = bytearray()

    def push(self, pcm_bytes: bytes) -> list[bytes]:
        """Add data, return as many complete frames as are now available."""
        self._buf.extend(pcm_bytes)
        return self._drain_complete_frames()

    def _drain_complete_frames(self) -> list[bytes]:
        frames: list[bytes] = []
        while len(self._buf) >= self.frame_bytes:
            frames.append(bytes(self._buf[: self.frame_bytes]))
            del self._buf[: self.frame_bytes]
        return frames

    def flush(self) -> bytes | None:
        """
        Called at end-of-stream: zero-pad any leftover partial frame and
        return it (so the very last bit of audio isn't silently dropped).
        Returns None if there's nothing left to flush.
        """
        if not self._buf:
            return None
        padded = bytes(self._buf) + b"\x00" * (self.frame_bytes - len(self._buf))
        self._buf.clear()
        return padded

    def __len__(self) -> int:
        """Bytes currently buffered but not yet a complete frame."""
        return len(self._buf)


@dataclass
class AudioMetrics:
    """
    Running counters for one audio session — surfaced via the health/metrics
    endpoint and logged at session end so bottlenecks (e.g. dropped frames,
    corrupt frames) are visible without attaching a debugger.
    """

    frames_in: int = 0
    bytes_in: int = 0
    frames_out: int = 0
    bytes_out: int = 0
    corrupt_frames: int = 0
    dropped_samples: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def record_in(self, n_bytes: int) -> None:
        self.frames_in += 1
        self.bytes_in += n_bytes

    def record_out(self, n_bytes: int) -> None:
        self.frames_out += 1
        self.bytes_out += n_bytes

    def record_corrupt(self) -> None:
        self.corrupt_frames += 1

    @property
    def duration_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def as_dict(self) -> dict[str, object]:
        return {
            "frames_in": self.frames_in,
            "bytes_in": self.bytes_in,
            "frames_out": self.frames_out,
            "bytes_out": self.bytes_out,
            "corrupt_frames": self.corrupt_frames,
            "dropped_samples": self.dropped_samples,
            "duration_seconds": round(self.duration_seconds, 3),
        }


class AudioSession:
    """
    Ties together the per-connection state a worker needs: a unique id,
    wall-clock bookkeeping, and an AudioMetrics instance. Both STT and TTS
    sessions embed one of these for consistent logging/metrics fields.
    """

    _counter = 0

    def __init__(self, kind: str) -> None:
        AudioSession._counter += 1
        self.id = f"{kind}-{int(time.time())}-{AudioSession._counter}"
        self.kind = kind
        self.metrics = AudioMetrics()
        self.created_at = time.monotonic()

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.created_at
