# ADR-001: Audio Pipeline

## Status
Accepted

## Context
Audio crosses three language boundaries: Flutter (mic capture) → Go
(orchestrator) → Python (STT/TTS workers). Any ambiguity in sample rate,
channel count, sample format, endianness, or frame size causes silent
misinterpretation at one of those boundaries — the failure mode is
garbled or empty transcripts, not a crash, which makes it expensive to
debug after the fact. This contract is therefore fixed and explicit
rather than inferred.

## Decision

### Audio contract (fixed, must match `assistant/stt_tts.go`)
| Property          | Value              |
|--------------------|--------------------|
| Sample rate         | 16000 Hz           |
| Channels            | 1 (mono)           |
| Sample format        | Signed Int16        |
| Endianness          | Little-endian        |
| Frame duration        | 20 ms              |
| Samples per frame      | 320               |
| Bytes per frame       | 640               |
| Transport codec (Flutter↔Go) | Opus         |
| Inference format (Go↔Python) | Raw PCM Int16 |

`common/config.py::AudioSettings` is the single source of these numbers
on the Python side; `pyAudioSampleRate` / `pyAudioChannels` /
`pyOpusFrameSamples` are the Go-side equivalents. **Changing one without
the other breaks the wire contract** — there is intentionally no
negotiation handshake for this; it's a build-time contract, not a
runtime one.

### Why Opus for transport but PCM for inference
Opus is what actually crosses the network (Flutter↔Go); it's compressed
and designed for that. Once audio reaches a worker it needs to be raw
samples for VAD/energy calculations and for feeding the ASR/TTS models,
neither of which understands Opus. Go owns the Opus codec (decode
inbound, encode outbound) so the Python side never needs an Opus
dependency at all — it only ever sees/produces PCM16LE.

### Ring buffer shape
- **Pre-roll: not clipped by construction.** Rather than a fixed
  circular pre-roll window, `SpeechSegmenter`/`RingBuffer` accumulate
  the *entire* utterance from the moment the client connects until
  `{"action":"end"}` — so there's no VAD onset lag to lose the first
  word to. `STT_PREROLL_MS` exists as a documented target (2s) for a
  future fixed-capacity ring if session memory ever needs bounding more
  aggressively than the existing `max_session_seconds` safety cap.
- **Post-roll (500ms, `STT_POSTROLL_MS`):** trailing consonants are
  preserved the same way — we don't clip on VAD silence detection, we
  only use silence to *decide when to offer a partial*. The client (Go)
  decides when the utterance is actually over by sending `"end"`.
- **Sliding window (4s, `STT_SLIDING_WINDOW_SECONDS`):** partial passes
  only re-transcribe the last N seconds of buffered audio
  (`SpeechSegmenter.sliding_snapshot`), so a long utterance doesn't make
  every partial linearly more expensive. **Finals always use the full
  buffer** (`SpeechSegmenter.drain()`) — correctness over cost for the
  authoritative transcript.

### Chunk size / inference cadence
- **Receive:** 20ms (one frame per inbound WS binary message — matches
  the wire contract exactly, no batching on ingestion).
- **Partial inference:** rate-limited to at most once every
  `STT_PARTIAL_INTERVAL_MS` (default 500ms), and only if there's been
  speech energy since the last partial (`SpeechSegmenter.should_emit_partial`).
- **Final inference:** once, on `{"action":"end"}`, over the full buffer.

### TTS chunk normalization
Kokoro (like most TTS engines) does not guarantee its output chunk size
is a multiple of 320 samples. `common/audio.py::ChunkNormalizer` re-slices
whatever the engine produces into exact 640-byte frames before they ever
reach the WebSocket, specifically so Go's `enc.Encode()` (which requires
a fixed Opus frame size) never receives a malformed chunk. This was the
single highest-risk integration point flagged in the original Go code's
own comments, so it gets a dedicated, unit-tested component
(`tests/test_audio.py::test_chunk_normalizer_across_multiple_pushes`)
rather than being handled ad hoc inside the engine.
