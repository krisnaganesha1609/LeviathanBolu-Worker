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
from stt.silero_vad import SileroVAD
from stt.whisper_engine import STTEngine

log = get_logger(__name__)


class STTSession:
    def __init__(
        self,
        engine: STTEngine,
        audio_settings: AudioSettings,
        stt_settings: STTSettings,
        metrics: MetricsRegistry,
        silero_vad: SileroVAD,
        *,
        session_id: str | None = None,
        conversation_id: str | None = None,
    ):
        self.engine = engine
        self.audio_settings = audio_settings
        self.stt_settings = stt_settings
        self.metrics_registry = metrics
        self.silero_vad = silero_vad
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
            self._maybe_dump_wav(full_audio)
            start = time.perf_counter()
            with self._latency.stage("final_inference"):
                if await self._too_quiet_or_short(full_audio):
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

    async def _too_quiet_or_short(self, audio) -> bool:
        """
        Gate final inference on obviously-not-speech audio (ADR-005
        addendum): a misconfigured/over-sensitive client-side VAD sending
        pure noise/silence not only wastes an inference call but,
        for encoder-decoder ASR models like Whisper, tends to produce
        hallucinated-but-plausible-looking text on pure noise input rather
        than an empty string. Better to recognize "this isn't speech" here
        (using a real trained VAD, not an amplitude threshold — an RMS
        cutoff can't tell "wind gust" from "quiet speech") and return ""
        directly than let the model guess.
        """
        duration_ms = (len(audio) / self.audio_settings.sample_rate) * 1000 if len(audio) else 0
        if duration_ms < self.stt_settings.min_utterance_ms:
            log.info(
                "stt.session.skipped_too_short",
                session_id=self.session_id,
                duration_ms=round(duration_ms, 1),
                min_utterance_ms=self.stt_settings.min_utterance_ms,
            )
            return True
        # Silero VAD runs on onnxruntime — blocking/CPU-bound, same as any
        # other model inference (ADR-004): always via run_in_executor, never
        # called directly on the event loop.
        loop = asyncio.get_running_loop()
        has_speech = await loop.run_in_executor(
            None, self.silero_vad.has_speech, audio, self.audio_settings.sample_rate
        )
        if not has_speech:
            log.info(
                "stt.session.skipped_no_speech_detected",
                session_id=self.session_id,
                duration_ms=round(duration_ms, 1),
            )
            return True
        return False

    def _maybe_dump_wav(self, audio) -> None:
        """Debug helper — see STT_DEBUG_DUMP_AUDIO_DIR in .env. Writes the
        exact PCM this session is about to transcribe to a .wav file so you
        can listen to it directly instead of inferring content from the
        (possibly garbage/hallucinated) transcript."""
        dump_dir = self.stt_settings.debug_dump_audio_dir
        if not dump_dir or audio.size == 0:
            return
        import os
        import wave

        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, f"{self.session_id}.wav")
        with wave.open(path, "wb") as wf:
            wf.setnchannels(self.audio_settings.channels)
            wf.setsampwidth(2)  # int16
            wf.setframerate(self.audio_settings.sample_rate)
            wf.writeframes(audio.tobytes())
        log.info("stt.session.debug_wav_dumped", session_id=self.session_id, path=path)

    def close(self) -> None:
        self.metrics_registry.session_ended()
