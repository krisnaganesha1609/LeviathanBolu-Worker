# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-tts.txt ./

# TTS_FULL=1 to build with real Kokoro deps (kokoro-onnx, soundfile,
# scipy, librosa). Default is dummy-engine-only for a lightweight image.
ARG TTS_FULL=0
RUN if [ "$TTS_FULL" = "1" ]; then \
        pip install --no-cache-dir -r requirements-tts.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

COPY common ./common
COPY tts ./tts
COPY config ./config

RUN useradd --create-home --uid 1000 leviathan && chown -R leviathan:leviathan /app

# Same fix as docker/stt.Dockerfile: pre-create the cache dir with correct
# ownership before the named volume (tts-model-cache) mounts over it, so
# the non-root `leviathan` user can actually write to it.
RUN mkdir -p /home/leviathan/.cache && chown -R leviathan:leviathan /home/leviathan/.cache

USER leviathan

EXPOSE 9002

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -fsS http://localhost:9002/health || exit 1

CMD ["python", "-m", "tts.app"]
