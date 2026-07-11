# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-stt.txt ./

# STT_FULL=1 to build with real faster-whisper deps (no torch — much
# smaller/lighter than the old SenseVoice/FunASR stack). Default is
# dummy-engine-only for a lightweight image you can still boot and
# integration-test against.
ARG STT_FULL=0
RUN if [ "$STT_FULL" = "1" ]; then \
        pip install --no-cache-dir -r requirements-stt.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

COPY common ./common
COPY stt ./stt
COPY config ./config

RUN useradd --create-home --uid 1000 leviathan && chown -R leviathan:leviathan /app

# Pre-create the HF cache dir *with correct ownership* before the named
# volume (stt-model-cache in docker-compose.yml) is mounted over it. If
# this path doesn't exist in the image first, Docker initializes the
# volume as root-owned on first mount, and the non-root `leviathan` user
# below can't write to it — that's the "Permission denied .../huggingface"
# error. Docker's local volume driver copies an empty named volume's
# initial ownership from whatever already exists at the mount path in the
# image, so creating+chowning it here fixes that at the source.
ENV HF_HOME=/home/leviathan/.cache/huggingface
RUN mkdir -p "$HF_HOME" && chown -R leviathan:leviathan /home/leviathan/.cache

USER leviathan

EXPOSE 9001

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -fsS http://localhost:9001/health || exit 1

CMD ["python", "-m", "stt.app"]
