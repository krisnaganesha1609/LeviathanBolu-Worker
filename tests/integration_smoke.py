"""
Simulates exactly what assistant/stt_tts.go does over the wire, against
the real running STT/TTS servers (dummy engines) — this is the closest
thing to an end-to-end integration check without an actual Go binary.

STT: dial -> send N binary PCM16LE frames (320 samples/640 bytes, silence
     then a "speech" burst) -> send {"action":"end"} -> expect exactly one
     final_transcript JSON message (partials may or may not arrive first).

TTS: dial -> send {"text":...,"personality":...} -> expect a stream of
     binary frames, each EXACTLY 640 bytes (320 samples @ int16) -> then
     {"event":"done"}.
"""
import asyncio
import json
import sys

import numpy as np
import websockets

FRAME_SAMPLES = 320
FRAME_BYTES = FRAME_SAMPLES * 2
SAMPLE_RATE = 16000


def _speech_like_pcm(seconds: float, sample_rate: int = SAMPLE_RATE) -> bytes:
    """
    A synthetic signal shaped enough like real speech (pitched harmonics +
    a syllabic on/off amplitude envelope + a little broadband noise) to
    reliably pass Silero VAD — unlike a bare constant-amplitude tone,
    which Silero correctly rejects as non-speech (that discrimination is
    the whole point of switching off the old RMS-threshold gate).
    """
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    f0 = 140
    harmonics = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    envelope = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 4 * t))  # ~4Hz syllable rate
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(len(t)) * 0.05
    signal = harmonics * envelope + noise
    signal = signal / np.max(np.abs(signal)) * 0.6
    return (signal * 32767).astype(np.int16).tobytes()


async def test_stt() -> None:
    uri = "ws://127.0.0.1:9001/stt"
    async with websockets.connect(uri) as ws:
        # ~1s of silence, then ~1.5s of speech-like signal (passes Silero VAD)
        silence = np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes()
        speech_pcm = _speech_like_pcm(1.5)
        speech_frames = [
            speech_pcm[i : i + FRAME_BYTES] for i in range(0, len(speech_pcm), FRAME_BYTES)
        ]

        for _ in range(20):
            await ws.send(silence)
        for frame in speech_frames:
            if len(frame) == FRAME_BYTES:
                await ws.send(frame)
            await asyncio.sleep(0.01)

        await ws.send(json.dumps({"action": "end"}))

        got_final = False
        partial_count = 0
        for _ in range(20):
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            assert isinstance(raw, str), f"expected JSON text, got binary: {raw!r}"
            msg = json.loads(raw)
            assert "event" in msg, f"missing 'event' key: {msg}"
            if msg["event"] == "partial_transcript":
                partial_count += 1
                assert isinstance(msg["text"], str)
            elif msg["event"] == "final_transcript":
                assert isinstance(msg["text"], str) and msg["text"], f"empty final text: {msg}"
                got_final = True
                break
            else:
                raise AssertionError(f"unexpected event: {msg}")

        assert got_final, "never received final_transcript"
        print(f"[STT]  OK — {partial_count} partial(s), then final_transcript received")


async def test_tts(personality: str) -> None:
    uri = "ws://127.0.0.1:9002/tts"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"text": "Halo. Ini tes LEVIATHAN.", "personality": personality}))

        frame_count = 0
        total_bytes = 0
        got_done = False
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            if isinstance(raw, bytes):
                assert len(raw) == FRAME_BYTES, f"frame not {FRAME_BYTES} bytes: got {len(raw)}"
                frame_count += 1
                total_bytes += len(raw)
            else:
                msg = json.loads(raw)
                assert msg == {"event": "done"}, f"unexpected text message: {msg}"
                got_done = True
                break

        assert got_done, "never received done event"
        assert frame_count > 0, "no audio frames received"
        duration_s = total_bytes / 2 / SAMPLE_RATE
        print(
            f"[TTS:{personality}] OK — {frame_count} frames, "
            f"{total_bytes} bytes (~{duration_s:.2f}s audio), all exactly {FRAME_BYTES} bytes"
        )


async def main() -> None:
    await test_stt()
    await test_tts("LEVIATHAN")
    await test_tts("BOLU")
    await test_tts("UNKNOWN_PERSONALITY_TYPO")  # must fall back, not crash
    print("\nALL INTEGRATION CHECKS PASSED")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as e:
        print(f"INTEGRATION CHECK FAILED: {e}", file=sys.stderr)
        sys.exit(1)
