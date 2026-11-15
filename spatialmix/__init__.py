"""Deterministic offline spatial audio mixer."""

from .audio import AUDIO_SAMPLE_RATE
from .spatial import (
    DEFAULT_MIN_GAIN,
    SpatialGains,
    compute_distance_gain,
    compute_spatial_gains,
    equal_power_pan,
)

__all__ = [
    "AUDIO_SAMPLE_RATE",
    "DEFAULT_MIN_GAIN",
    "SpatialGains",
    "compute_distance_gain",
    "compute_spatial_gains",
    "equal_power_pan",
]

__version__ = "1.0.0"
