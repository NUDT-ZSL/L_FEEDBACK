"""Global soft peak limiter with a -1 dBFS ceiling."""

from __future__ import annotations

import math

import numpy as np

from . import PEAK_CEILING_DBFS


def ceiling_linear(ceiling_dbfs: float = PEAK_CEILING_DBFS) -> float:
    return 10.0 ** (ceiling_dbfs / 20.0)


def soft_limit(signal: np.ndarray,
               ceiling_dbfs: float = PEAK_CEILING_DBFS) -> tuple[np.ndarray, dict]:
    """Limit a mono/float mix.

    Below the ceiling the signal passes unchanged; above it a cubic soft-knee
    curve eases the signal toward a hard ceiling, so output never exceeds
    -1 dBFS while the knee stays derivative-continuous at the threshold.
    """
    ceiling = ceiling_linear(ceiling_dbfs)
    signal = np.asarray(signal, dtype=np.float64)
    output = signal.copy()

    over = np.abs(signal) > ceiling
    trigger_sample_count = int(np.count_nonzero(over))
    trigger_event_count = _count_events(over)

    # Soft knee starts at 75% of the ceiling (~-3.5 dBFS). The quadratic curve
    # joins the unity line with matching derivative 1 at the knee and flattens
    # to derivative 0 at the ceiling, so the output is transparent below the
    # knee and can never cross -1 dBFS.
    knee = 0.75 * ceiling
    knee_width = 2.0 * (ceiling - knee)
    signs = np.sign(signal)
    magnitude = np.abs(signal)
    beyond_knee = magnitude > knee
    depth = magnitude - knee
    curved = knee + depth - depth**2 / (2.0 * knee_width)
    softened = np.where(depth < knee_width, curved, ceiling)
    output[beyond_knee] = signs[beyond_knee] * softened[beyond_knee]

    stats = {
        "ceiling_dbfs": ceiling_dbfs,
        "trigger_sample_count": trigger_sample_count,
        "trigger_event_count": int(trigger_event_count),
        "limiter_engaged": bool(trigger_sample_count > 0),
        "pre_limit_peak_dbfs": _peak_dbfs(signal),
        "post_limit_peak_dbfs": _peak_dbfs(output),
    }
    return output, stats


def _count_events(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    return int(np.count_nonzero((~mask[:-1]) & (mask[1:])) + int(mask[0]))


def _peak_dbfs(signal: np.ndarray) -> float:
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(peak)
