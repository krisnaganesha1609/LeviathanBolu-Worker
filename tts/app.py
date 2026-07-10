from __future__ import annotations

import uvicorn

from common.config import get_tts_settings


def main() -> None:
    settings = get_tts_settings()
    uvicorn.run(
        "tts.server:app",
        host=settings.host,
        port=settings.port,
        log_config=None,
        access_log=False,
        timeout_graceful_shutdown=10,
    )


if __name__ == "__main__":
    main()
