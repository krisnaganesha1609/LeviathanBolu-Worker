# ADR-003: Session Lifecycle

## Status
Accepted

## Context
Without an explicit state machine, reconnect/edge-case handling tends to
be discovered by bug report rather than by design (e.g. "what happens if
`end` arrives with zero audio frames", "what happens if a partial is
still running when `end` arrives"). `common/state_machine.py` makes the
states and legal transitions explicit and raises `InvalidTransition`
immediately if the code tries to do something outside them — this is a
development-time correctness check, not (currently) surfaced to Go.

## Decision

### STT worker
```
DISCONNECTED -> CONNECTED -> RECEIVING_AUDIO <-> PROCESSING -> FINISHED -> IDLE
```
- `CONNECTED`: WebSocket accepted, no audio yet.
- `RECEIVING_AUDIO`: at least one binary frame ingested.
- `PROCESSING`: an inference call (partial or final) is in flight. We
  bounce back to `RECEIVING_AUDIO` after each partial so more frames can
  keep arriving mid-utterance — `PROCESSING` is not a terminal state
  until the *final* pass.
- `FINISHED` → `IDLE`: reached once on `{"action":"end"}`, after the
  final transcript has been computed and sent.

Edge case: `{"action":"end"}` arriving before any audio frame is valid
(empty utterance) — `STTSession.finalize()` explicitly forces a
`CONNECTED`/`PROCESSING` → `RECEIVING_AUDIO` transition first so the
state machine doesn't reject it.

### TTS worker
```
IDLE -> SYNTHESIZING -> STREAMING -> FINISHED -> IDLE
```
One TTS connection = one request/response cycle (matches
`TTSWorkerClient.SynthesizeStream`, which dials fresh per call). There is
currently no reconnect/resume concept for TTS — if the connection drops
mid-stream, Go simply dials again for the next utterance.

### Reconnect handling
Neither worker holds cross-connection state. Every WebSocket connection
gets a brand new `STTSession`/`TTSSession` (and therefore a fresh state
machine starting at `DISCONNECTED`/`IDLE`). This is deliberate: Go's
`websocket.DefaultDialer.DialContext` reconnects per call already
(`Transcribe()` opens and `defer`-closes its own connection each
invocation), so there is no "resume a previous session" scenario to
support in this version of the protocol. `session_id`/`conversation_id`
(ADR-002) exist so that *logging* can still correlate multiple
connections belonging to the same logical conversation, without the
workers themselves needing to track that correlation.
