# ADR-004: Model Loading Lifecycle

## Status
Accepted

## Context
faster-whisper (CTranslate2) and Kokoro model load times are on the order of
seconds; loading per-request would add unacceptable latency to every
single utterance ("nanti latency 20 detik" per the project brief).

## Decision
```
Application Start -> engine.warm_up() -> Ready -> (serve requests) -> Shutdown
```
Both `STTEngine` and `TTSEngine` implementations expose an async
`warm_up()` that is awaited exactly once, from each FastAPI app's
`lifespan` context manager (`stt/server.py`, `tts/server.py`), **before**
`uvicorn` starts accepting connections. The concrete model object is
cached on the engine instance (`self._model` / `self._kokoro`), guarded
by an `asyncio.Lock` so concurrent requests during a slow/retried load
can't trigger a duplicate load.

- If `warm_up()` fails (missing weights, no GPU, etc.), the process does
  **not** crash — the error is logged and `/health` reports `"degraded"`
  or `"down"` depending on severity. The engine will lazily retry the
  load on the first real request via `_ensure_loaded()`, so a transient
  failure (e.g. model download not finished yet) can still self-heal
  without a restart.
- Inference calls always run on `loop.run_in_executor(None, ...)` — both
  faster-whisper (CTranslate2) and Kokoro-ONNX are blocking/CPU-bound, and running them
  in-line on the event loop would stall every other concurrent
  session's WebSocket I/O.
- There is deliberately no "unload on idle" path. Model memory is paid
  once at startup and held for the process lifetime; this is a
  small-number-of-long-lived-worker-processes deployment (see
  `docker-compose.yml`), not a scale-to-zero one.

## Consequence
Container startup time now includes model load time (visible in
`/health` → `active_sessions`/`detail.ready` and in the
`*.server.ready` / `warm_up_seconds` log line). `docker-compose.yml`'s
healthcheck accounts for this with a generous `start_period`.
