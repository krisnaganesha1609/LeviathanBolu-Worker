"""
Sentence -> Personality Mapping -> Kokoro -> PCM -> 20ms Chunk -> WebSocket

This module owns the last two steps. Kokoro (like most TTS engines) does
not guarantee it emits audio in neat 320-sample/20ms pieces — see the
CATATAN comment in assistant/stt_tts.go noting that `enc.Encode()` on the
Go side needs a fixed frame size. `TTSStreamer` uses ChunkNormalizer to
guarantee every binary WS frame we send is exactly `frame_bytes` long,
regardless of what shape the engine handed us.
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from common.audio import ChunkNormalizer
from common.logger import get_logger

log = get_logger(__name__)


class TTSStreamer:
    def __init__(self, frame_samples: int, *, pace: bool = False, frame_ms: int = 20) -> None:
        self.normalizer = ChunkNormalizer(frame_samples)
        self.pace = pace
        self._frame_seconds = frame_ms / 1000.0
        self.frames_sent = 0
        self.bytes_sent = 0

    async def stream(self, pcm_chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        """
        Consumes arbitrarily-sized PCM byte chunks from the engine and
        yields exactly `frame_bytes`-sized frames. Call `flush()` after
        the source is exhausted to emit the zero-padded remainder.
        """
        next_send_at = time.monotonic()
        async for chunk in pcm_chunks:
            for frame in self.normalizer.push(chunk):
                if self.pace:
                    next_send_at += self._frame_seconds
                    delay = next_send_at - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)
                self.frames_sent += 1
                self.bytes_sent += len(frame)
                yield frame

    def flush(self) -> bytes | None:
        return self.normalizer.flush()
