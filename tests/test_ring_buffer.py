import numpy as np

from stt.ring_buffer import SpeechSegmenter, VADConfig


def _silence(n):
    return np.zeros(n, dtype=np.int16)


def _speech(n, amplitude=5000):
    return np.full(n, amplitude, dtype=np.int16)


def test_should_emit_partial_requires_speech_energy():
    cfg = VADConfig(sample_rate=16000, silence_ms=200, rms_threshold=350, partial_interval_ms=0)
    seg = SpeechSegmenter(cfg)
    seg.push(_silence(1600))
    assert seg.should_emit_partial() is False


def test_should_emit_partial_true_after_speech():
    cfg = VADConfig(sample_rate=16000, silence_ms=200, rms_threshold=350, partial_interval_ms=0)
    seg = SpeechSegmenter(cfg)
    seg.push(_speech(1600))
    assert seg.should_emit_partial() is True


def test_partial_interval_rate_limits():
    cfg = VADConfig(sample_rate=16000, silence_ms=200, rms_threshold=350, partial_interval_ms=10_000)
    seg = SpeechSegmenter(cfg)
    seg.push(_speech(1600))
    assert seg.should_emit_partial() is True
    seg.mark_partial_emitted()
    # No new speech energy and inside the rate-limit window -> no partial
    assert seg.should_emit_partial() is False


def test_is_tail_silent_detects_trailing_silence():
    cfg = VADConfig(sample_rate=16000, silence_ms=100, rms_threshold=350)  # 1600 samples
    seg = SpeechSegmenter(cfg)
    seg.push(_speech(3200))
    assert seg.is_tail_silent() is False
    seg.push(_silence(1600))
    assert seg.is_tail_silent() is True


def test_sliding_snapshot_bounds_window():
    cfg = VADConfig(sample_rate=100, silence_ms=100, rms_threshold=350)
    seg = SpeechSegmenter(cfg, max_buffer_seconds=10)
    seg.push(_speech(1000))  # 10 seconds @ 100Hz
    window = seg.sliding_snapshot(2.0)  # last 2 seconds = 200 samples
    assert len(window) == 200


def test_drain_clears_buffer():
    cfg = VADConfig(sample_rate=16000, silence_ms=100, rms_threshold=350)
    seg = SpeechSegmenter(cfg)
    seg.push(_speech(500))
    assert len(seg) == 500
    drained = seg.drain()
    assert len(drained) == 500
    assert len(seg) == 0
