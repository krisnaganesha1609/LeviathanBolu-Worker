"""
Benchmark the STT worker end-to-end: for N simulated utterances (measured
concurrently up to --concurrency), sends a synthetic PCM stream + "end",
and measures time from "end" sent to final_transcript received.

Usage:
    python -m benchmark.bench_stt --url ws://localhost:9001/stt --n 20 --concurrency 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import numpy as np
import websockets

FRAME_SAMPLES = 320


def synth_utterance_frames(seconds: float, sample_rate: int = 16000) -> list[bytes]:
    n_frames = int(seconds * 1000 / 20)
    speech = np.full(FRAME_SAMPLES, 5000, dtype=np.int16).tobytes()
    return [speech] * n_frames


async def run_one(url: str, seconds: float) -> float:
    frames = synth_utterance_frames(seconds)
    async with websockets.connect(url) as ws:
        for f in frames:
            await ws.send(f)
        start = time.perf_counter()
        await ws.send(json.dumps({"action": "end"}))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            if isinstance(raw, str):
                msg = json.loads(raw)
                if msg.get("event") == "final_transcript":
                    return (time.perf_counter() - start) * 1000


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:9001/stt")
    ap.add_argument("--n", type=int, default=20, help="number of utterances")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--utterance-seconds", type=float, default=2.0)
    args = ap.parse_args()

    sem = asyncio.Semaphore(args.concurrency)

    async def bounded() -> float:
        async with sem:
            return await run_one(args.url, args.utterance_seconds)

    start = time.perf_counter()
    latencies = await asyncio.gather(*(bounded() for _ in range(args.n)))
    wall_s = time.perf_counter() - start

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95) - 1]
    print(f"STT benchmark: n={args.n} concurrency={args.concurrency} utterance={args.utterance_seconds}s")
    print(f"  wall time:     {wall_s:.2f}s ({args.n / wall_s:.1f} utterances/s)")
    print(f"  latency p50:   {p50:.1f}ms")
    print(f"  latency p95:   {p95:.1f}ms")
    print(f"  latency min/max: {min(latencies):.1f}ms / {max(latencies):.1f}ms")


if __name__ == "__main__":
    asyncio.run(main())
