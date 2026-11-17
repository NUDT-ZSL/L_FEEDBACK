"""BPM timeline alignment: snap skill triggers to a musical grid."""

from __future__ import annotations

GRID_DIVISIONS = {"1/4": 1, "1/8": 2, "1/16": 4}  # steps per beat


def quantize_time(
    time_s: float, first_beat_offset_s: float, bpm: float, grid: str = "1/16"
) -> tuple[float, float]:
    """Snap ``time_s`` to the nearest grid point.

    Grid points are ``first_beat_offset_s + k * (60/bpm) / division``.
    Returns (quantized_time_s, signed_error_ms).
    """
    if grid not in GRID_DIVISIONS:
        raise ValueError(f"unknown grid {grid!r}; expected one of {sorted(GRID_DIVISIONS)}")
    step = 60.0 / bpm / GRID_DIVISIONS[grid]
    k = round((time_s - first_beat_offset_s) / step)
    quantized = first_beat_offset_s + k * step
    return quantized, (quantized - time_s) * 1000.0
