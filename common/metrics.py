"""
Minimal in-process metrics registry — no Prometheus client dependency
required, just enough to answer "where's the bottleneck" per the spec:
STT latency, audio queue depth, active sessions, inference time, dropped
frames, average RTT. Single-process, asyncio-single-threaded, so plain
dicts are safe without locks.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class RollingAverage:
    """Fixed-window rolling average, e.g. for inference time / RTT."""

    window: int = 100
    _values: deque[float] = field(default_factory=deque)

    def add(self, value: float) -> None:
        self._values.append(value)
        if len(self._values) > self.window:
            self._values.popleft()

    @property
    def avg(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    @property
    def count(self) -> int:
        return len(self._values)


class MetricsRegistry:
    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.active_sessions = 0
        self.total_sessions = 0
        self.dropped_frames = 0
        self.audio_queue_depth = 0
        self.inference_time_ms = RollingAverage()
        self.rtt_ms = RollingAverage()

    def session_started(self) -> None:
        self.active_sessions += 1
        self.total_sessions += 1

    def session_ended(self) -> None:
        self.active_sessions = max(0, self.active_sessions - 1)

    def record_dropped_frame(self, n: int = 1) -> None:
        self.dropped_frames += n

    def record_inference_ms(self, ms: float) -> None:
        self.inference_time_ms.add(ms)

    def record_rtt_ms(self, ms: float) -> None:
        self.rtt_ms.add(ms)

    def snapshot(self) -> dict[str, object]:
        return {
            "uptime_seconds": round(time.monotonic() - self.started_at, 1),
            "active_sessions": self.active_sessions,
            "total_sessions": self.total_sessions,
            "dropped_frames": self.dropped_frames,
            "audio_queue_depth": self.audio_queue_depth,
            "inference_time_ms_avg": round(self.inference_time_ms.avg, 2),
            "inference_time_ms_samples": self.inference_time_ms.count,
            "rtt_ms_avg": round(self.rtt_ms.avg, 2),
            "rtt_ms_samples": self.rtt_ms.count,
        }


# Process-wide singletons — one registry per worker process (STT and TTS
# run as separate processes, so there's no cross-contamination).
stt_metrics = MetricsRegistry()
tts_metrics = MetricsRegistry()
