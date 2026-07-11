"""
Centralized, typed configuration.

Every value the STT/TTS workers need is declared here and loaded from the
process environment (and `.env` if present) via pydantic-settings. Nothing
in the rest of the codebase should call `os.environ` directly — import the
settings singletons from this module instead so behaviour stays consistent
and testable.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AudioSettings(BaseSettings):
    """
    Audio contract shared with the Go orchestrator (assistant/stt_tts.go).

    sample_rate / channels / frame_ms MUST stay in sync with the Go
    constants `pyAudioSampleRate`, `pyAudioChannels`, `pyOpusFrameSamples`.
    Changing any of these without updating Go breaks the wire contract.
    """

    model_config = SettingsConfigDict(env_prefix="AUDIO_", extra="ignore")

    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 20

    @property
    def frame_samples(self) -> int:
        """Samples per frame, e.g. 320 for 16kHz/20ms — matches pyOpusFrameSamples."""
        return self.sample_rate * self.frame_ms // 1000

    @property
    def frame_bytes(self) -> int:
        """Bytes per frame assuming int16 PCM (2 bytes/sample)."""
        return self.frame_samples * 2


class GeneralSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    leviathan_env: Literal["development", "production"] = "development"
    log_level: str = "INFO"
    log_json: bool = False
    health_path: str = "/health"
    metrics_path: str = "/metrics"


class STTSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STT_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 9001
    engine: Literal["dummy", "whisper"] = "dummy"
    # faster-whisper model size/name. "small" is the sweet spot for a 4GB
    # RAM VPS running CPU-only alongside the TTS worker + Go orchestrator +
    # DB — good multilingual (incl. Indonesian, unlike SenseVoiceSmall)
    # accuracy without the ~1GB+ runtime footprint of "medium"/"large-v3".
    # Drop to "base"/"tiny" if memory is still tight; go "medium" only if
    # this VPS is STT-dedicated with headroom to spare.
    model: str = "small"
    device: str = "cpu"  # ctranslate2: "cpu" | "cuda" | "auto"
    # int8 = lowest memory + fastest on CPU, small accuracy tradeoff vs
    # float32. Good default for a RAM-constrained VPS.
    compute_type: str = "int8"
    cpu_threads: int = Field(default=4, ge=1)
    use_gpu: bool = False
    language: str = "auto"  # "auto" | "en" | "id" | any Whisper language code
    # Whisper has no "bias toward these N languages" mode — it picks one
    # language per request. If your users mix Indonesian speech with
    # English wake words, force your primary spoken language here (e.g.
    # "id") rather than "auto" — short utterances are especially prone to
    # language misdetection, and a forced language still transcribes
    # short foreign-sounding proper nouns (like wake words) reasonably.
    # Beam search width. Partials use beam_size=1 (greedy, fast) regardless
    # of this setting — see WhisperEngine.transcribe(); this value is only
    # used for the authoritative final pass.
    beam_size: int = Field(default=5, ge=1)
    # Optional vocabulary/style hint passed to Whisper as `initial_prompt`
    # — a short piece of text whose vocabulary and phrasing biases
    # decoding (e.g. include your wake words / common command phrases).
    # Empty string = no hint.
    initial_prompt: str = ""
    # Peak-normalize quiet audio before it reaches the model (see
    # common.audio.normalize_gain) — ASR accuracy drops noticeably on
    # low-volume input. Only ever boosts, never attenuates.
    gain_normalize: bool = True

    # Silence/segmentation policy (ADR-005): speech considered started once
    # RMS energy crosses vad_rms_threshold; a partial is offered no more
    # often than every partial_interval_ms; max_silence_seconds caps how
    # long we wait for more audio before treating the session as stalled.
    vad_silence_ms: int = Field(default=800, ge=100)
    vad_rms_threshold: int = Field(default=350, ge=0)
    partial_interval_ms: int = Field(default=500, ge=100)
    max_silence_seconds: float = Field(default=1.5, ge=0.1)
    max_session_seconds: int = Field(default=120, ge=1)

    # Ring buffer shape (ADR-001): keep 2s of pre-roll so the first word of
    # an utterance isn't clipped by VAD onset lag, 500ms of post-roll so
    # trailing consonants aren't cut off, and bound partial-pass cost with a
    # 4s sliding window (finals still use the *entire* buffered utterance).
    preroll_ms: int = Field(default=2000, ge=0)
    postroll_ms: int = Field(default=500, ge=0)
    sliding_window_seconds: float = Field(default=4.0, ge=0.5)

    # Debug only: set to a writable path (e.g. /tmp/stt-dumps) to save every
    # finalized utterance as a .wav file, so you can literally listen to
    # what the worker received instead of guessing from transcript output.
    # Empty string (default) = disabled, zero overhead.
    debug_dump_audio_dir: str = ""

    # Defense-in-depth against a noisy/misconfigured client VAD sending
    # near-silent audio: skip the (expensive, and prone to hallucinating
    # garbage on pure noise — see ADR-005) final inference call entirely
    # if the buffered utterance is too short, or Silero VAD (bundled with
    # faster-whisper — see stt/silero_vad.py) finds no speech in it at all.
    # This does NOT replace fixing VAD calibration on the client; it just
    # stops obvious noise from ever reaching the model.
    min_utterance_ms: int = Field(default=300, ge=0)
    silero_vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class TTSSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TTS_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 9002
    engine: Literal["dummy", "kokoro"] = "dummy"
    model_path: str = "/models/kokoro-v1.0.onnx"
    voices_path: str = "/models/voices-v1.0.bin"
    default_personality: str = "LEVIATHAN"
    personalities_path: str = "config/personalities.yaml"
    chunk_pace: bool = False
    use_gpu: bool = False
    max_session_seconds: int = Field(default=60, ge=1)


@lru_cache(maxsize=1)
def get_general_settings() -> GeneralSettings:
    return GeneralSettings()


@lru_cache(maxsize=1)
def get_audio_settings() -> AudioSettings:
    return AudioSettings()


@lru_cache(maxsize=1)
def get_stt_settings() -> STTSettings:
    return STTSettings()


@lru_cache(maxsize=1)
def get_tts_settings() -> TTSSettings:
    return TTSSettings()