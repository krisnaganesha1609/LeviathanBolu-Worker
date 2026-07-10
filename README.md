# LEVIATHAN — Python Worker Server (STT + TTS)

Asyncio-native STT (SenseVoice/FunASR) and TTS (Kokoro) WebSocket
microservices for the LEVIATHAN assistant, built to match
`assistant/stt_tts.go`'s wire contract exactly. See `docs/ADR-*.md` for
the design decisions behind every non-obvious choice in this codebase —
read those before changing protocol-level behavior.

```
python-workers/
  common/          shared config, logging, protocol, audio, metrics, errors
  stt/             SenseVoice/FunASR worker (ws://.../stt)
  tts/             Kokoro worker (ws://.../tts)
  config/          personalities.yaml (voice mapping, no if/else)
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
VAD, chunk normalization, state machine, metrics, personality mapping —
with zero model downloads, which is what `integration_smoke.py` proves
(it speaks the exact same wire protocol `assistant/stt_tts.go` does).

## Running the real models

```bash
pip install -r requirements-stt.txt   # torch, torchaudio, funasr
pip install -r requirements-tts.txt   # kokoro-onnx, soundfile, scipy, librosa

# .env:
STT_ENGINE=sensevoice
TTS_ENGINE=kokoro
TTS_MODEL_PATH=/path/to/kokoro-v1.0.onnx
TTS_VOICES_PATH=/path/to/voices-v1.0.bin
```

SenseVoice weights (`iic/SenseVoiceSmall`) download automatically via
FunASR/ModelScope on first load. Kokoro weights (`kokoro-v1.0.onnx`,
`voices-v1.0.bin`) must be downloaded separately (see
[kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)) and mounted
at `TTS_MODEL_PATH` / `TTS_VOICES_PATH` (or `./models` in
`docker-compose.yml`).

## Docker

```bash
docker compose up --build                        # dummy engines, fast
STT_FULL=1 TTS_FULL=1 docker compose up --build   # real models (large images)
```

Go should point at `ws://<host>:9001/stt` and `ws://<host>:9002/tts` —
unchanged from what's already in `assistant/stt_tts.go`.

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
| `STT_ENGINE` | `dummy` | `dummy` \| `sensevoice` |
| `TTS_ENGINE` | `dummy` | `dummy` \| `kokoro` |
| `STT_VAD_RMS_THRESHOLD` | 350 | energy gate for "is this speech" |
| `STT_PARTIAL_INTERVAL_MS` | 500 | min gap between partial transcripts |
| `STT_SLIDING_WINDOW_SECONDS` | 4.0 | bounds cost of each partial pass |
| `STT_MAX_SESSION_SECONDS` | 120 | hard per-connection safety cap |
| `TTS_PERSONALITIES_PATH` | `config/personalities.yaml` | voice mapping |
| `TTS_CHUNK_PACE` | `false` | pace output frames at real-time (20ms) |

Personalities are pure data (`config/personalities.yaml`) — add a new
one without touching code:

```yaml
personalities:
  BOLU:
    voice: af_bella
    speed: 1.15
    pitch: 2
    lang: en-us
```

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
- `SpeechSegmenter`'s VAD is energy/RMS-based, not a learned VAD
  (webrtcvad/silero) — simple, dependency-light, and sufficient to gate
  *when to attempt* a partial; transcription quality itself comes from
  the ASR engine, not this gate. Swapping in a learned VAD is a
  contained change inside `stt/ring_buffer.py`.
- No mTLS/auth on the worker WebSockets — these are intended to run
  on a private network alongside the Go orchestrator, not exposed
  publicly. Add auth at the reverse-proxy / network-policy layer if
  that assumption changes.
