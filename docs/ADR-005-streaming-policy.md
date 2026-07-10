# ADR-005: Streaming / Partial-Transcript Policy

## Status
Accepted

## Context
Left unspecified, "when do we emit a partial transcript" tends to be
answered by the model's own internal cadence, which can spam the
WebSocket (a partial per 20ms frame) or starve it (no partial until the
model feels like it). Both are bad: the former wastes bandwidth/CPU on
mostly-identical partials, the latter makes the UI feel unresponsive.

## Decision
A partial is only offered when **all** of the following hold
(`stt/ring_buffer.py::SpeechSegmenter.should_emit_partial`):

1. At least `STT_PARTIAL_INTERVAL_MS` (default 500ms) has elapsed since
   the last partial was emitted for this session.
2. There has been speech energy (RMS ≥ `STT_VAD_RMS_THRESHOLD`, default
   350) in the buffer since the last partial — a long silent pause does
   not re-trigger partials.
3. No other inference call is currently in flight for this session
   (`STTSession._infer_lock`) — partials never queue up behind each
   other.

Silence thresholds (ADR-001):
- **Speech/silence gate:** RMS ≥ 350 counts as speech (`STT_VAD_RMS_THRESHOLD`).
- **Trailing-silence detection:** `SpeechSegmenter.is_tail_silent()` looks
  at the last `STT_VAD_SILENCE_MS` (default 800ms) of buffered audio —
  exposed for future use (e.g. auto-finalizing without waiting for Go's
  explicit `"end"`), not currently wired to auto-finalize since Go
  already owns that decision today.
- **Stalled-session detection:** `STTSession.stalled` flags a session
  that's received no audio for `STT_MAX_SILENCE_SECONDS` (default 1.5s)
  while mid-utterance — logged as a warning today; a future version
  could use this to proactively finalize instead of waiting for the
  `STT_MAX_SESSION_SECONDS` hard timeout.

This is intentionally **not** "every 500ms no matter what" or "only on
confidence change" — both were considered and rejected: fixed-interval
regardless of content wastes inference cycles during silence, and
confidence-based triggering depends on a confidence score the DummyEngine
(and not all real engines) can reliably produce. The energy-gated,
rate-limited approach works uniformly across engines because it only
depends on the audio itself, not the model's internal state.
