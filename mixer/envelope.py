"""Sample-accurate fades and equal-power crossfade envelopes."""

from __future__ import annotations

import numpy as np

from .config import ShotSpec
from .timeline import Overlap


def fade_in(length: int) -> np.ndarray:
    if length <= 0:
        return np.ones(0, dtype=np.float64)
    points = (np.arange(length, dtype=np.float64) + 0.5) / float(length)
    return np.sin(points * (np.pi / 2.0))


def fade_out(length: int) -> np.ndarray:
    if length <= 0:
        return np.ones(0, dtype=np.float64)
    points = (np.arange(length, dtype=np.float64) + 0.5) / float(length)
    return np.cos(points * (np.pi / 2.0))


def track_envelope(
    duration_samples: int, fade_in_samples: int, fade_out_samples: int
) -> np.ndarray:
    envelope = np.ones(duration_samples, dtype=np.float64)
    if fade_in_samples:
        envelope[:fade_in_samples] *= fade_in(fade_in_samples)
    if fade_out_samples:
        envelope[-fade_out_samples:] *= fade_out(fade_out_samples)
    return envelope


def crossfade_envelopes(
    shots: list[ShotSpec], overlaps: list[Overlap]
) -> dict[str, np.ndarray]:
    """Build per-shot equal-power gains for overlapping intervals.

    For every overlap pair, the later shot receives ``sin(a)`` and the earlier
    shot ``cos(a)``. Because the pair is evaluated at identical sample
    positions, ``sin(a)^2 + cos(a)^2 == 1`` for the whole crossfade.
    """

    envelopes = {
        shot.shot_id: np.ones(shot.duration_samples, dtype=np.float64)
        for shot in shots
    }
    for overlap in overlaps:
        length = overlap.duration_samples
        if length <= 0:
            continue
        progress = (np.arange(length, dtype=np.float64) + 0.5) / float(length)
        angle = progress * (np.pi / 2.0)

        later_start = overlap.start_sample - overlap.later.start_sample
        earlier_start = overlap.start_sample - overlap.earlier.start_sample
        envelopes[overlap.later.shot_id][
            later_start : later_start + length
        ] *= np.sin(angle)
        envelopes[overlap.earlier.shot_id][
            earlier_start : earlier_start + length
        ] *= np.cos(angle)
    return envelopes
