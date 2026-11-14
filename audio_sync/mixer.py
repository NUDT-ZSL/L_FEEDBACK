"""Multi-stream frame-aligned mixer.

Streams are mixed either linearly (uniform ``1/N`` gain) or with equal-power
weights (``1/sqrt(N)``), where *N* counts currently active streams. Inactive
(near-silent) streams are gated so they do not consume headroom. The output is
soft-limited to int16 range and per-frame gains / activity are exposed for the
diagnostic log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .constants import ACTIVITY_RMS_THRESHOLD, FRAME_SAMPLES
from .signals import rms_level


@dataclass
class MixerFrame:
    """Mixed output plus the per-stream diagnostics used by the JSONL log."""

    mixed: np.ndarray
    gains: tuple
    active: tuple
    peak: int


class Mixer:
    """Mix 2-4 aligned float32 frames into one mono float32 frame."""

    def __init__(self, mode: str = "equal_power") -> None:
        if mode not in {"equal_power", "linear"}:
            raise ValueError(f"unknown mixer mode {mode!r}")
        self.mode = mode

    def mix(self, frames: Sequence[np.ndarray]) -> MixerFrame:
        if not frames:
            raise ValueError("at least one input frame is required")

        active_flags = tuple(
            rms_level(frame) >= ACTIVITY_RMS_THRESHOLD for frame in frames
        )
        active_count = max(1, sum(active_flags))
        per_stream_gain = (
            1.0 / np.sqrt(active_count)
            if self.mode == "equal_power"
            else 1.0 / len(frames)
        )

        gains = tuple(
            per_stream_gain if active else 0.0 for active in active_flags
        )
        mixed = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        for frame, gain in zip(frames, gains):
            if gain > 0.0:
                mixed += frame.astype(np.float32) * np.float32(gain)

        # Soft clip guard: hard clamp only protects int16 conversion.
        mixed = np.clip(mixed, -1.0, 1.0)
        peak = int(np.max(np.abs(mixed)) * 32767.0) if mixed.size else 0
        return MixerFrame(
            mixed=mixed.astype(np.float32),
            gains=gains,
            active=active_flags,
            peak=peak,
        )
