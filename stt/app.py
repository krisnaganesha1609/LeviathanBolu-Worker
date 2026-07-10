"""
Entrypoint: `python -m stt.app` (or via Docker CMD).

Uvicorn already handles SIGTERM/SIGINT gracefully (finishes in-flight
requests, runs the FastAPI lifespan shutdown block in stt/server.py, then
exits) — that's the graceful-shutdown mechanism for this service.
"""
from __future__ import annotations

import uvicorn

from common.config import get_stt_settings


def main() -> None:
    settings = get_stt_settings()
    uvicorn.run(
        "stt.server:app",
        host=settings.host,
        port=settings.port,
        log_config=None,  # we own logging via common.logger/structlog
        access_log=False,
        timeout_graceful_shutdown=10,
    )


if __name__ == "__main__":
    main()
