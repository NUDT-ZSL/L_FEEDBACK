"""Millisecond <-> sample conversion helpers.

Conversion rule (fixed for the whole pipeline, as required by the spec):
``samples = floor(round(ms * sr / 1000))`` where ``round`` is Python's
banker's-rounding builtin, which is deterministic across platforms.
"""

from __future__ import annotations

import numpy as np


def ms_to_samples(ms: float, sample_rate: int) -> int:
    """Convert milliseconds to a sample count with the fixed rounding rule."""
    return int(np.floor(round(float(ms) * sample_rate / 1000.0)))


def samples_to_seconds(samples: int, sample_rate: int) -> float:
    """Convert a sample count to seconds (used by the loudness meter)."""
    return int(samples) / float(sample_rate)
