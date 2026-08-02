"""
Benchmark the TTS worker: for N requests (up to --concurrency concurrent),
measures time-to-first-audio-byte and total time-to-done.

Usage:
    python -m benchmark.bench_tts --url ws://localhost:9002/tts --n 20 --concurrency 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import websockets

SAMPLE_TEXT = (
    "Halo, ini adalah tes benchmark untuk LEVIATHAN. "
    "Sistem text to speech ini sedang diukur latensinya."
)


async def run_one(url: str, personality: str, voice: str) -> tuple[float, float]:
    async with websockets.connect(url) as ws:
        start = time.perf_counter()
        # voice is required (Go resolves it from user settings); speed and
        # pitch are JSON numbers, matching the Go wire format.
        await ws.send(json.dumps({
            "text": SAMPLE_TEXT,
            "personality": personality,
            "voice": voice,
            "speed": 1.0,
            "pitch": 0,
            "lang": "id-ID",
        }))
        ttfb_ms: float | None = None
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            if isinstance(raw, bytes):
                if ttfb_ms is None:
                    ttfb_ms = (time.perf_counter() - start) * 1000
            else:
                msg = json.loads(raw)
                if msg.get("event") == "done":
                    total_ms = (time.perf_counter() - start) * 1000
                    return ttfb_ms or total_ms, total_ms


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:9002/tts")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--personality", default="LEVIATHAN")
    ap.add_argument("--voice", default="am_onyx")
    args = ap.parse_args()

    sem = asyncio.Semaphore(args.concurrency)

    async def bounded() -> tuple[float, float]:
        async with sem:
            return await run_one(args.url, args.personality, args.voice)

    start = time.perf_counter()
    results = await asyncio.gather(*(bounded() for _ in range(args.n)))
    wall_s = time.perf_counter() - start

    ttfb = sorted(r[0] for r in results)
    total = sorted(r[1] for r in results)

    def p95(xs: list[float]) -> float:
        return xs[int(len(xs) * 0.95) - 1]

    print(f"TTS benchmark: n={args.n} concurrency={args.concurrency} personality={args.personality}")
    print(f"  wall time:          {wall_s:.2f}s ({args.n / wall_s:.1f} req/s)")
    print(f"  time-to-first-byte: p50={statistics.median(ttfb):.1f}ms  p95={p95(ttfb):.1f}ms")
    print(f"  time-to-done:       p50={statistics.median(total):.1f}ms  p95={p95(total):.1f}ms")


if __name__ == "__main__":
    asyncio.run(main())
