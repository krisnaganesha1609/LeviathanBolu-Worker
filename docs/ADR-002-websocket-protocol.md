# ADR-002: WebSocket Protocol

## Status
Accepted, amended to **Wire v1.1** (see Addendum below) — **supersedes**
an earlier proposal to wrap audio chunks in a JSON envelope
(`{"event":"audio_chunk","sequence":N}` + binary). See "Rejected
alternative" below.

## Context
The Go worker client (`Orchestrator/internal/assistant/python_workers.go`)
already exists and is treated as a fixed contract for this phase of the
project (per the project owner: "kontrak Go yang kamu buat menurutku
sudah bagus dan tidak perlu diubah"). It implements:

- **STT** (`ws://…/stt`): sends raw binary PCM16LE frames with **no JSON
  envelope**, then a single JSON message `{"action":"end"}`. It reads
  JSON text messages back and branches on `event` values.
- **TTS** (`ws://…/tts`): sends one JSON message with the full resolved
  voice config (`{"text","voice","speed","pitch","lang","personality"}`,
  speed/pitch as JSON numbers), then reads binary frames (assumed PCM)
  until it sees `{"event":"done"}` or the socket closes.

## Decision
The wire framing implemented here matches the Go code **exactly**:
raw binary audio, no per-chunk JSON envelope, `{"action":"end"}` to
terminate an STT utterance, `{"event":"done"}` to terminate a TTS
stream. `common/protocol.py` is the single source of truth for these
shapes and is unit-tested against the literal JSON Go would produce/parse
(`tests/test_protocol.py`).

We *do* add additive, backward-compatible extensions that Go's current
read loops safely ignore (they only match specific `event` values and
fall through / `continue` on anything else):

- `{"event":"partial_transcript","text":...}` — Go's own comment already
  anticipates this; today it's simply not consumed, but nothing breaks
  by sending it.
- `{"event":"error","message":"<CODE>: <message>"}` on both STT and TTS —
  safe today, gives Go an explicit signal to branch on once/if its read
  loop is updated to handle failures instead of blocking until the
  read deadline.
- **Optional** `?session_id=...&conversation_id=...` query parameters on
  the WebSocket URL. Go's current `DialContext` calls pass no query
  string, so these are simply absent and the worker mints a local
  session id — zero Go changes required. If Go is later updated to pass
  these (e.g. from the orchestrator's own conversation tracking), no
  Python change is needed either; `server.py` already reads them via
  `websocket.query_params.get(...)`.

## Rejected alternative
An earlier proposal wrapped every binary audio chunk in a JSON envelope
(`{"event":"audio_chunk","sequence":15}` immediately followed by a binary
frame) to get per-chunk sequence numbers and a uniform "everything is a
tagged event" protocol. **This was rejected** because it does not match
what the existing Go code sends — Go writes raw `websocket.BinaryMessage`
frames directly (see `conn.WriteMessage(websocket.BinaryMessage, ...)` in
`Transcribe()`), never preceded by a JSON tag. Adopting the wrapped
format would require rewriting the Go client, which was explicitly
out of scope. If per-chunk sequence numbers become necessary later
(e.g. for reordering over an unreliable transport), that's a breaking
protocol version bump that should get its own ADR and explicit Go
changes, not something the worker silently assumes.

## Error codes
Structured codes (`common/errors.py`) replace bare exceptions so Go (or
whoever reads the logs) has something greppable instead of free-text
messages:

| Code    | Meaning                                   |
|---------|--------------------------------------------|
| STT_001 | Model not loaded                            |
| STT_002 | Invalid PCM payload                         |
| STT_003 | Session closed                              |
| STT_004 | Timeout                                     |
| STT_005 | Engine failure (inference threw)            |
| TTS_001 | Model not loaded                            |
| TTS_002 | Invalid text (empty/too long)               |
| TTS_003 | Session closed                              |
| TTS_004 | Timeout                                     |
| TTS_005 | Engine failure (inference threw)            |
| TTS_006 | Invalid voice config (missing/blank `voice` — fatal, Go must send a complete voice config) |

## Addendum — Wire v1.1 (runtime-protocol alignment)

The Orchestrator now speaks the runtime protocol (Documentation 05/25)
to Flutter and acts as the translation gateway. To support that, the
worker wire gained the following (all shapes in `common/protocol.py`):

1. **Streaming STT end-to-end.** Go dials the STT worker when the user
   *starts* speaking and forwards PCM frames as they arrive, relaying
   `partial_transcript` upstream as `voice.stt.partial` in real time.
   No worker change was needed — the worker always emitted partials.
2. **`{"action":"cancel"}` on STT** — aborts the session (drops any
   in-flight partial, discards buffered audio, no `final_transcript`).
   Used for voice barge-in / turn interruption. TTS cancellation stays
   implicit: Go closes the socket mid-stream, which the worker now logs
   as a clean client cancel instead of an engine error.
3. **Structured error events** on both workers:
   `{"event":"error","code":"STT_00x","recoverable":bool,"retryable":bool,
   "message":"..."}` — the Go gateway maps these into the runtime error
   contract (03_SYSTEM_CONTRACTS.md) without parsing message strings.
4. **`speed`/`pitch` accept JSON numbers** (what Go actually sends —
   float32/int8), with strings still tolerated for manual testing.

The event names on this hop intentionally stay `partial_transcript` /
`final_transcript` / `done` / `error` (not `voice.stt.*`): the runtime
event vocabulary belongs to the Flutter↔Orchestrator hop, and the Go
gateway owns the mapping. This keeps the worker wire minimal and stable.
