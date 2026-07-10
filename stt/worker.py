"""
Per-connection STT pipeline (ADR-001, ADR-003, ADR-005):

    PCM Frame -> decoder -> RingBuffer/VAD (SpeechSegmenter) -> engine
              -> partial_transcript (sliding 4s window, rate-limited)
              -> [on "end"] -> final_transcript (full buffered utterance)

State machine: DISCONNECTED -> CONNECTED -> RECEIVING_AUDIO <-> PROCESSING
               -> FINISHED -> IDLE

Partial transcription runs as a background task so it never blocks
ingestion of the next incoming frame; only one inference call is in
flight at a time (guarded by a lock) so a slow model doesn't pile up
concurrent work. On "end" we cancel any in-flight partial and run one
last, authoritative pass over the full buffered audio for the final
transcript.

session_id / conversation_id: the Go client today opens a bare connection
with no handshake, so these are OPTIONAL and read from the WS query string
(?session_id=...&conversation_id=...) by server.py — if absent, we mint a
local session_id and leave conversation_id empty. This is additive and
does not require any Go change (see ADR-002).
"""
from __future__ import annotations

import asyncio
import time

from common.audio import AudioMetrics
from common.config import AudioSettings, STTSettings
from common.errors import STTErrorCode, WorkerError
from common.logger import LatencyLogger, get_logger
from common.metrics import MetricsRegistry
from common.state_machine import STTState, STTStateMachine
from common.utils import new_id
from stt.decoder import PCMFrameDecoder
from stt.models import STTSessionState, TranscriptResult
from stt.ring_buffer import SpeechSegmenter, VADConfig
from stt.sensevoice_engine import STTEngine

log = get_logger(__name__)


class STTSession:
    def __init__(
        self,
        engine: STTEngine,
        audio_settings: AudioSettings,
        stt_settings: STTSettings,
        metrics: MetricsRegistry,
        *,
        session_id: str | None = None,
        conversation_id: str | None = None,
    ):
        self.engine = engine
        self.audio_settings = audio_settings
        self.stt_settings = stt_settings
        self.metrics_registry = metrics
        self.state = STTSessionState()
        self.session_id = session_id or new_id("stt")
        self.conversation_id = conversation_id or ""
        self.decoder = PCMFrameDecoder(session_id=self.session_id)
        self.fsm = STTStateMachine(self.session_id)
        self.segmenter = SpeechSegmenter(
            VADConfig(
                sample_rate=audio_settings.sample_rate,
                silence_ms=stt_settings.vad_silence_ms,
                rms_threshold=stt_settings.vad_rms_threshold,
                partial_interval_ms=stt_settings.partial_interval_ms,
            )
        )
        self._infer_lock = asyncio.Lock()
        self._partial_task: asyncio.Task | None = None
        self._latency = LatencyLogger(log, "stt_session", session_id=self.session_id)
        self._deadline = time.monotonic() + stt_settings.max_session_seconds
        self._last_frame_at = time.monotonic()

        self.fsm.transition(STTState.CONNECTED)
        self.metrics_registry.session_started()
        log.info(
            "stt.session.started",
            session_id=self.session_id,
            conversation_id=self.conversation_id,
        )

    @property
    def audio_metrics(self) -> AudioMetrics:
        return self.state.session.metrics

    @property
    def expired(self) -> bool:
        return time.monotonic() > self._deadline

    @property
    def stalled(self) -> bool:
        """No audio for longer than max_silence_seconds while mid-utterance."""
        if self.fsm.state != STTState.RECEIVING_AUDIO:
            return False
        return (time.monotonic() - self._last_frame_at) > self.stt_settings.max_silence_seconds

    def ingest_frame(self, raw: bytes) -> None:
        """Handle one inbound binary WS message. Never raises on bad audio."""
        if self.fsm.state in (STTState.CONNECTED, STTState.PROCESSING):
            self.fsm.transition(STTState.RECEIVING_AUDIO)
        self._last_frame_at = time.monotonic()

        with self._latency.stage("decode"):
            samples = self.decoder.decode(raw)
        if samples is None:
            self.audio_metrics.record_corrupt()
            self.metrics_registry.record_dropped_frame()
            return
        self.audio_metrics.record_in(len(raw))
        self.segmenter.push(samples)

    def maybe_start_partial(self) -> asyncio.Task | None:
        """
        Call after each ingest_frame(). Returns the background task if a new
        partial transcription was kicked off; caller doesn't need to await
        it, server.py polls `pending_partial_result()` for completion.
        """
        if self._partial_task is not None and not self._partial_task.done():
            return None
        if not self.segmenter.should_emit_partial():
            return None
        self.segmenter.mark_partial_emitted()
        self._partial_task = asyncio.create_task(self._run_partial())
        return self._partial_task

    def pending_partial_result(self) -> TranscriptResult | None:
        """Non-blocking check: pop the finished partial task's result, if any."""
        task = self._partial_task
        if task is None or not task.done():
            return None
        self._partial_task = None
        if task.cancelled():
            return None
        exc = task.exception()
        if exc is not None:
            log.error("stt.session.partial_task_error", session_id=self.session_id, error=str(exc))
            return None
        return task.result()

    async def _run_partial(self) -> TranscriptResult | None:
        async with self._infer_lock:
            self.fsm.transition(STTState.PROCESSING)
            window = self.segmenter.sliding_snapshot(self.stt_settings.sliding_window_seconds)
            if window.size == 0:
                self.fsm.transition(STTState.RECEIVING_AUDIO)
                return None
            start = time.perf_counter()
            with self._latency.stage("partial_inference"):
                try:
                    result = await self.engine.transcribe(
                        window, sample_rate=self.audio_settings.sample_rate, is_final=False
                    )
                except Exception as exc:
                    log.error(
                        "stt.session.partial_failed",
                        session_id=self.session_id,
                        error=str(exc),
                        error_code=STTErrorCode.ENGINE_FAILURE.value,
                    )
                    self.fsm.transition(STTState.RECEIVING_AUDIO)
                    return None
            self.metrics_registry.record_inference_ms((time.perf_counter() - start) * 1000)
            self.state.partials_emitted += 1
            self.fsm.transition(STTState.RECEIVING_AUDIO)
            return result

    async def finalize(self) -> TranscriptResult:
        """Called when the client sends {"action":"end"}. Cancels any pending
        partial and runs one authoritative pass over the full buffer."""
        if self._partial_task is not None and not self._partial_task.done():
            self._partial_task.cancel()

        if self.fsm.state not in (STTState.RECEIVING_AUDIO, STTState.PROCESSING):
            # Client sent "end" with zero audio frames — still a valid, if
            # empty, utterance.
            self.fsm.transition(STTState.RECEIVING_AUDIO)

        async with self._infer_lock:
            self.fsm.transition(STTState.PROCESSING)
            full_audio = self.segmenter.drain()
            start = time.perf_counter()
            with self._latency.stage("final_inference"):
                if full_audio.size == 0:
                    result = TranscriptResult(text="", is_final=True)
                else:
                    try:
                        result = await self.engine.transcribe(
                            full_audio, sample_rate=self.audio_settings.sample_rate, is_final=True
                        )
                    except Exception as exc:
                        raise WorkerError(STTErrorCode.ENGINE_FAILURE, str(exc)) from exc
            self.metrics_registry.record_inference_ms((time.perf_counter() - start) * 1000)

        self.state.ended = True
        self.state.final_emitted = True
        self.fsm.transition(STTState.FINISHED)
        self.fsm.transition(STTState.IDLE)
        self._latency.emit(
            frames_in=self.audio_metrics.frames_in,
            corrupt_frames=self.audio_metrics.corrupt_frames,
            partials_emitted=self.state.partials_emitted,
        )
        log.info(
            "stt.session.finished",
            session_id=self.session_id,
            conversation_id=self.conversation_id,
            **self.audio_metrics.as_dict(),
        )
        return result

    def close(self) -> None:
        self.metrics_registry.session_ended()
