from __future__ import annotations

import asyncio
import functools
import signal
import time
import uuid
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def timed_ms(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Decorator: log-free timing wrapper, returns (result, elapsed_ms) via attribute."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.perf_counter()
        result = await fn(*args, **kwargs)
        wrapper.last_elapsed_ms = (time.perf_counter() - start) * 1000  # type: ignore[attr-defined]
        return result

    wrapper.last_elapsed_ms = 0.0  # type: ignore[attr-defined]
    return wrapper


async def retry_with_backoff(
    fn: Callable[[], Awaitable[T]],
    *,
    retries: int = 3,
    base_delay: float = 0.25,
    max_delay: float = 5.0,
) -> T:
    """Exponential backoff retry, used for model warm-up / transient IO."""
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception:
            attempt += 1
            if attempt > retries:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            await asyncio.sleep(delay)


def install_graceful_shutdown(on_shutdown: Callable[[], Awaitable[None]]) -> None:
    """
    Registers SIGTERM/SIGINT handlers on the running loop that await
    `on_shutdown()` before letting the process exit. Safe to call once per
    process (e.g. from app.py's lifespan startup).
    """
    loop = asyncio.get_event_loop()

    def _handler(sig: signal.Signals) -> None:
        asyncio.ensure_future(_shutdown(sig))

    async def _shutdown(sig: signal.Signals) -> None:
        await on_shutdown()

    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(s, functools.partial(_handler, s))
        except (NotImplementedError, RuntimeError):
            # add_signal_handler unsupported (e.g. Windows) — best effort only.
            pass
