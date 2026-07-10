"""
Turns raw inbound WebSocket binary messages (already-decoded PCM16LE from
Go — see assistant/stt_tts.go's opus.Decoder usage) into validated
np.int16 arrays ready for the ring buffer / VAD / engine.

Named `decoder.py` for parity with the requested project layout; the
*Opus* decoding itself happens on the Go side, this module only validates
and converts the raw PCM bytes it receives.
"""
from __future__ import annotations

import numpy as np

from common.audio import PCMValidationError, pcm16_to_int16, validate_pcm16
from common.logger import get_logger

log = get_logger(__name__)


class PCMFrameDecoder:
    """Stateless-ish helper; one instance per session is fine but not required."""

    def __init__(self, *, session_id: str) -> None:
        self.session_id = session_id
        self.corrupt_count = 0

    def decode(self, raw: bytes) -> np.ndarray | None:
        """
        Returns an int16 sample array, or None if the frame was corrupt
        (mirrors the Go side's "1 frame korup jangan gagalkan seluruh
        transkrip" policy — we skip bad frames rather than killing the
        session).
        """
        try:
            validate_pcm16(raw)
        except PCMValidationError as exc:
            self.corrupt_count += 1
            log.warning("stt.decoder.corrupt_frame", session_id=self.session_id, error=str(exc))
            return None
        return pcm16_to_int16(raw)
