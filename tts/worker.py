"""
Per-request TTS pipeline (ADR-001, ADR-003):

    TTSRequest -> PersonalityRegistry.resolve() -> engine.synthesize()
               -> TTSStreamer (fixed 20ms frames) -> caller yields to WS

State machine: IDLE -> SYNTHESIZING -> STREAMING -> FINISHED -> IDLE
"""
from __future__ import annotations

import time
from typing import AsyncIterator

from common.audio import AudioMetrics
from common.config import AudioSettings, TTSSettings
from common.logger import LatencyLogger, get_logger
from common.metrics import MetricsRegistry
from common.state_machine import TTSState, TTSStateMachine
from common.utils import new_id
from tts.kokoro_engine import TTSEngine
from tts.personalities import PersonalityRegistry
from tts.streamer import TTSStreamer

log = get_logger(__name__)


class TTSSession:
    def __init__(
        self,
        engine: TTSEngine,
        personalities: PersonalityRegistry,
        audio_settings: AudioSettings,
        tts_settings: TTSSettings,
        metrics: MetricsRegistry,
        *,
        session_id: str | None = None,
        conversation_id: str | None = None,
    ) -> None:
        self.engine = engine
        self.personalities = personalities
        self.audio_settings = audio_settings
        self.tts_settings = tts_settings
        self.metrics_registry = metrics
        self.session_id = session_id or new_id("tts")
        self.conversation_id = conversation_id or ""
        self.fsm = TTSStateMachine(self.session_id)
        self.metrics = AudioMetrics()
        self._latency = LatencyLogger(log, "tts_session", session_id=self.session_id)

        self.metrics_registry.session_started()
        log.info("tts.session.started", session_id=self.session_id, conversation_id=self.conversation_id)

    async def stream_reply(self, text: str, personality: str) -> AsyncIterator[bytes]:
        voice_config, matched = self.personalities.resolve(personality)
        if not matched:
            log.warning(
                "tts.session.personality_fallback",
                session_id=self.session_id,
                requested=personality,
                using=self.personalities.default_name,
            )

        self.fsm.transition(TTSState.SYNTHESIZING)
        streamer = TTSStreamer(
            self.audio_settings.frame_samples,
            pace=self.tts_settings.chunk_pace,
            frame_ms=self.audio_settings.frame_ms,
        )

        start = time.perf_counter()
        first_frame_at: float | None = None

        with self._latency.stage("synthesize_and_stream"):
            self.fsm.transition(TTSState.STREAMING)
            async for frame in streamer.stream(self.engine.synthesize(text, voice_config)):
                if first_frame_at is None:
                    first_frame_at = time.perf_counter()
                self.metrics.record_out(len(frame))
                yield frame
            tail = streamer.flush()
            if tail is not None:
                self.metrics.record_out(len(tail))
                yield tail

        total_ms = (time.perf_counter() - start) * 1000
        ttfb_ms = ((first_frame_at - start) * 1000) if first_frame_at else None
        self.metrics_registry.record_inference_ms(total_ms)
        self.fsm.transition(TTSState.FINISHED)
        self.fsm.transition(TTSState.IDLE)
        self._latency.emit(
            personality=personality,
            voice=voice_config.voice,
            frames_sent=streamer.frames_sent,
            bytes_sent=streamer.bytes_sent,
            time_to_first_byte_ms=round(ttfb_ms, 2) if ttfb_ms is not None else None,
        )
        log.info(
            "tts.session.finished",
            session_id=self.session_id,
            conversation_id=self.conversation_id,
            **self.metrics.as_dict(),
        )

    def close(self) -> None:
        self.metrics_registry.session_ended()
