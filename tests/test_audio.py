import numpy as np
import pytest

from common.audio import (
    ChunkNormalizer,
    PCMValidationError,
    RingBuffer,
    int16_to_pcm16,
    pcm16_to_int16,
    rms_energy,
    validate_pcm16,
)


def test_validate_pcm16_rejects_odd_length():
    with pytest.raises(PCMValidationError):
        validate_pcm16(b"\x01\x02\x03")


def test_validate_pcm16_rejects_empty_by_default():
    with pytest.raises(PCMValidationError):
        validate_pcm16(b"")


def test_pcm16_roundtrip():
    samples = np.array([0, 1, -1, 32767, -32768], dtype=np.int16)
    data = int16_to_pcm16(samples)
    assert len(data) == len(samples) * 2
    back = pcm16_to_int16(data)
    np.testing.assert_array_equal(back, samples)


def test_rms_energy_silence_is_zero():
    assert rms_energy(np.zeros(100, dtype=np.int16)) == 0.0


def test_rms_energy_of_constant_signal():
    samples = np.full(100, 1000, dtype=np.int16)
    assert rms_energy(samples) == pytest.approx(1000.0)


def test_ring_buffer_push_and_drain():
    rb = RingBuffer()
    rb.push(np.array([1, 2, 3], dtype=np.int16))
    rb.push(np.array([4, 5], dtype=np.int16))
    assert len(rb) == 5
    drained = rb.drain()
    np.testing.assert_array_equal(drained, [1, 2, 3, 4, 5])
    assert len(rb) == 0


def test_ring_buffer_respects_max_samples():
    rb = RingBuffer(max_samples=3)
    rb.push(np.array([1, 2, 3, 4, 5], dtype=np.int16))
    assert len(rb) == 3
    np.testing.assert_array_equal(rb.peek_all(), [3, 4, 5])
    assert rb.dropped_samples == 2


def test_ring_buffer_tail():
    rb = RingBuffer()
    rb.push(np.array([1, 2, 3, 4, 5], dtype=np.int16))
    np.testing.assert_array_equal(rb.tail(2), [4, 5])
    np.testing.assert_array_equal(rb.tail(100), [1, 2, 3, 4, 5])


def test_chunk_normalizer_exact_multiple():
    norm = ChunkNormalizer(frame_samples=4)  # 8 bytes/frame
    frames = norm.push(b"\x00" * 16)
    assert len(frames) == 2
    assert all(len(f) == 8 for f in frames)
    assert norm.flush() is None


def test_chunk_normalizer_leftover_and_flush():
    norm = ChunkNormalizer(frame_samples=4)  # 8 bytes/frame
    frames = norm.push(b"\x01" * 10)  # 1 full frame + 2 leftover bytes
    assert len(frames) == 1
    assert len(norm) == 2
    tail = norm.flush()
    assert tail is not None
    assert len(tail) == 8
    assert tail[:2] == b"\x01\x01"
    assert tail[2:] == b"\x00" * 6


def test_chunk_normalizer_across_multiple_pushes():
    """Simulates Kokoro emitting irregularly-sized chunks — output frames
    must still always be exactly frame_bytes long."""
    norm = ChunkNormalizer(frame_samples=4)
    all_frames = []
    for size in (3, 5, 1, 7, 2):
        all_frames.extend(norm.push(b"\xff" * size))
    tail = norm.flush()
    if tail:
        all_frames.append(tail)
    assert all(len(f) == 8 for f in all_frames)
    total_bytes = sum(len(f) for f in all_frames)
    assert total_bytes >= 3 + 5 + 1 + 7 + 2
