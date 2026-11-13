"""Per-segment adaptive peak gain."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

MAX_GAIN_DB = 20.0


@dataclass(frozen=True)
class GainResult:
    gain_linear: float
    gain_db: float
    input_peak_linear: float
    input_peak_dbfs: float | None
    output_peak_linear: float
    output_peak_dbfs: float | None
    input_peak_above_target: bool


def dbfs_to_linear(dbfs: float) -> float:
    return 10.0 ** (float(dbfs) / 20.0)


def linear_to_dbfs(value: float) -> float | None:
    if value <= 0.0:
        return None
    return 20.0 * math.log10(value)


def _peak(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.max(np.abs(samples)))


def adaptive_gain(
    samples: np.ndarray,
    target_peak_dbfs: float = -1.0,
    max_gain_db: float = 20.0,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, GainResult]:
    target_peak = dbfs_to_linear(target_peak_dbfs)
    max_gain_linear = dbfs_to_linear(min(float(max_gain_db), MAX_GAIN_DB))
    peak = _peak(samples)

    if peak <= 0.0:
        gain = 1.0
    elif peak > target_peak:
        gain = 1.0
    else:
        gain = min(max(1.0, target_peak / peak), max_gain_linear)

    gained = np.asarray(samples, dtype=np.float64) * gain
    output_peak = _peak(gained)
    if gain > 1.0 and output_peak > target_peak:
        correction = max(1.0, min(gain, target_peak / output_peak))
        gain = correction
        gained = np.asarray(samples, dtype=np.float64) * gain
        output_peak = _peak(gained)

    if transform is not None:
        gained = transform(gained)

    return gained, GainResult(
        gain_linear=float(gain),
        gain_db=float(20.0 * math.log10(gain)) if gain > 0 else 0.0,
        input_peak_linear=float(peak),
        input_peak_dbfs=linear_to_dbfs(peak),
        output_peak_linear=float(output_peak),
        output_peak_dbfs=linear_to_dbfs(output_peak),
        input_peak_above_target=peak > target_peak,
    )
