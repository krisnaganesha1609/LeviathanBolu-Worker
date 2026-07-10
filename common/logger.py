"""
Structured logging setup (structlog) shared by both workers, plus a small
`LatencyLogger` helper used to log per-stage timings so bottlenecks are
visible when a session runs slow (this is the "Latency Logger" component
requested in the spec).
"""
from __future__ import annotations

import contextlib
import logging
import sys
import time
from typing import Any, Iterator

import structlog

from common.config import get_general_settings


def configure_logging() -> None:
    """Call once at process startup (from each app.py)."""
    settings = get_general_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.log_json:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


class LatencyLogger:
    """
    Tracks named stage durations for one "unit of work" (one STT session,
    one TTS request, ...) and logs a single summary line at the end so you
    can immediately see which stage is the bottleneck.

    Usage:
        lat = LatencyLogger(log, "stt_session", session_id=sid)
        with lat.stage("decode"):
            ...
        with lat.stage("inference"):
            ...
        lat.emit()
    """

    def __init__(self, logger: Any, unit: str, **context: Any) -> None:
        self._logger = logger
        self._unit = unit
        self._context = context
        self._stages: dict[str, float] = {}
        self._started = time.perf_counter()

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._stages[name] = self._stages.get(name, 0.0) + elapsed_ms

    def mark(self, name: str, elapsed_ms: float) -> None:
        """Record a stage duration measured externally (e.g. inside an executor)."""
        self._stages[name] = self._stages.get(name, 0.0) + elapsed_ms

    def emit(self, **extra: Any) -> None:
        total_ms = (time.perf_counter() - self._started) * 1000
        self._logger.info(
            f"{self._unit}.latency",
            total_ms=round(total_ms, 2),
            stages_ms={k: round(v, 2) for k, v in self._stages.items()},
            **self._context,
            **extra,
        )
