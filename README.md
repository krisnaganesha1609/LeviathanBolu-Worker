# LEVIATHAN — Python Worker Server (STT + TTS)

Asyncio-native STT (faster-whisper + Silero VAD) and TTS (Kokoro) WebSocket
microservices for the LEVIATHAN assistant, built to match the wire
contract of `Orchestrator/internal/assistant/python_workers.go` exactly
("Worker Wire v1.1" — see `common/protocol.py`). See `docs/ADR-*.md` for
the design decisions behind every non-obvious choice in this codebase —
read those before changing protocol-level behavior.

```
python-workers/
  common/          shared config, logging, protocol, audio, metrics, errors
  stt/             faster-whisper + Silero VAD worker (ws://.../stt)
  tts/             Kokoro worker (ws://.../tts)
  docs/            ADR-001..005 (architecture decision records)
  tests/           pytest unit tests + integration_smoke.py (live WS test)
  benchmark/       bench_stt.py / bench_tts.py (latency & throughput)
  docker/          stt.Dockerfile, tts.Dockerfile
  docker-compose.yml
```

## Quickstart (no ML weights required)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env

python -m stt.app &      # ws://localhost:9001/stt   (STT_ENGINE=dummy)
python -m tts.app &      # ws://localhost:9002/tts   (TTS_ENGINE=dummy)

curl localhost:9001/health
curl localhost:9002/health

python tests/integration_smoke.py   # full protocol round-trip against both
```

Dummy engines exercise the *entire* pipeline — WebSocket, ring buffer,
VAD, chunk normalization, state machine, metrics — with zero model
downloads, which is what `integration_smoke.py` proves (it speaks the
exact same wire protocol the Go orchestrator's
`internal/assistant/python_workers.go` does).

## Running the real models

```bash
pip install -r requirements-stt.txt   # faster-whisper + onnxruntime — no torch
pip install -r requirements-tts.txt   # kokoro-onnx, soundfile, scipy, librosa

# .env:
STT_ENGINE=whisper
STT_MODEL=small          # tiny/base/small/medium/large-v3 — see .env.example for RAM guidance
TTS_ENGINE=kokoro
TTS_MODEL_PATH=/path/to/kokoro-v1.0.onnx
TTS_VOICES_PATH=/path/to/voices-v1.0.bin
```

faster-whisper weights download automatically from Hugging Face on first
load (cached under `~/.cache/huggingface`). Unlike SenseVoiceSmall,
Whisper models have an explicit Indonesian (`id`) language code — set
`STT_LANGUAGE=id` or `en` to skip language auto-detection for short
utterances (recommended for wake-word checks; see ADR-004).

Kokoro weights (`kokoro-v1.0.onnx`, `voices-v1.0.bin`) must be downloaded
separately from
[kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) and mounted
at `TTS_MODEL_PATH` / `TTS_VOICES_PATH` (or `./models` in
`docker-compose.yml`).

## Docker

```bash
docker compose up --build                        # dummy engines, fast
STT_FULL=1 TTS_FULL=1 docker compose up --build   # real models (large images)
```

Go should point at `ws://<host>:9001/stt` and `ws://<host>:9002/tts`
(`STT_WORKER_URL` / `TTS_WORKER_URL` in the Orchestrator's `.env`).

## Tests & benchmarks

```bash
pytest                                              # unit tests, no ML deps needed
python tests/integration_smoke.py                    # live WS protocol check
python -m benchmark.bench_stt --n 20 --concurrency 4
python -m benchmark.bench_tts --n 20 --concurrency 4
```

## Configuration

Everything is environment-driven (`common/config.py`, loaded from `.env`
via pydantic-settings) — see `.env.example` for the full list with
inline docs. Highlights:

| Var | Default | Meaning |
|---|---|---|
| `STT_ENGINE` | `dummy` | `dummy` \| `whisper` |
| `STT_MODEL` | `small` | faster-whisper model size (`tiny`/`base`/`small`/`medium`/`large-v3`) |
| `STT_COMPUTE_TYPE` | `int8` | CTranslate2 quantization — `int8` for lowest RAM on CPU |
| `STT_SILERO_VAD_THRESHOLD` | 0.5 | Silero VAD speech-probability threshold (the real "is this speech" gate) |
| `TTS_ENGINE` | `dummy` | `dummy` \| `kokoro` |
| `STT_VAD_RMS_THRESHOLD` | 350 | energy gate for "is this speech" |
| `STT_PARTIAL_INTERVAL_MS` | 500 | min gap between partial transcripts |
| `STT_SLIDING_WINDOW_SECONDS` | 4.0 | bounds cost of each partial pass |
| `STT_MAX_SESSION_SECONDS` | 120 | hard per-connection safety cap |
| `TTS_CHUNK_PACE` | `false` | pace output frames at real-time (20ms) |

There is no personality→voice mapping in this worker: the Go
orchestrator owns that data (`user_settings.domain.go`'s
`WakeWordConfig`) and sends the fully-resolved `voice`/`speed`/`pitch`/
`lang` on every TTS request (`tts/voice_config.py` just validates it).

## Health & metrics

Both workers expose:
- `GET /health` → `{"status", "service", "engine", "active_sessions", "uptime_seconds", "detail"}`
- `GET /metrics` → active/total sessions, dropped frames, rolling-average inference time & RTT

## Architecture Decision Records

- [ADR-001](docs/ADR-001-audio-pipeline.md) — audio contract, ring buffer shape, sliding window, chunk normalization
- [ADR-002](docs/ADR-002-websocket-protocol.md) — wire protocol, why it matches Go exactly, error codes
- [ADR-003](docs/ADR-003-session-lifecycle.md) — STT/TTS state machines, reconnect handling
- [ADR-004](docs/ADR-004-model-loading.md) — load-once-at-startup model lifecycle
- [ADR-005](docs/ADR-005-streaming-policy.md) — partial-transcript emission policy

## Known limitations / next steps

- `KokoroEngine` pitch-shifting uses `librosa.effects.pitch_shift`
  (optional dependency) — reasonable quality, not phase-vocoder-grade.
  Falls back to a no-op (logged warning) if `librosa` isn't installed.
- `SpeechSegmenter`'s **per-frame** gate (deciding *when to attempt* a
  partial, on every 20ms frame — `stt/ring_buffer.py`) is still a cheap
  RMS-energy heuristic; running Silero VAD on every 20ms frame would be
  wasteful and a poor fit for such a short window. The **final "is this
  actually speech" decision** (`STTSession._too_quiet_or_short`, gating
  whether the buffered utterance ever reaches the ASR model at all) uses
  real Silero VAD (`stt/silero_vad.py`, bundled with faster-whisper — no
  extra torch/silero-vad dependency) — this is the gate that matters for
  transcription quality, and it's a trained model, not a threshold.
- Whisper's language auto-detection (`STT_LANGUAGE=auto`) is unreliable
  on short utterances (a few hundred ms) — it can misdetect the language
  entirely. For anything short and latency-sensitive (wake-word checks
  especially), force a language explicitly (`STT_LANGUAGE=en` or `id`)
  instead of `auto`.
- No mTLS/auth on the worker WebSockets — these are intended to run
  on a private network alongside the Go orchestrator, not exposed
  publicly. Add auth at the reverse-proxy / network-policy layer if
  that assumption changes.
