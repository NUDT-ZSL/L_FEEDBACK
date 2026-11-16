"""Grid alignment, tempo conversion, tiling, and bus limiting."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .analysis import CONFIDENCE_THRESHOLD, round_half_up

LIMIT_CEILING = 0.98
ORIGINAL_GAIN = 0.90
PARTICLE_GAIN = 0.85
PERCUSSION_GAIN = 0.72
DB_FLOOR = -120.0


@dataclass(frozen=True)
class LimiterResult:
    audio: np.ndarray
    applied: bool
    pre_peak: float
    post_peak: float
    gain_adjustment_db: float


def tile_to_length(source: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(source, dtype=np.float64)
    if length <= 0:
        return np.zeros((0, values.shape[1]), dtype=np.float64)
    if values.shape[0] == 0:
        return np.zeros((length, values.shape[1]), dtype=np.float64)
    indices = np.arange(length, dtype=np.int64) % values.shape[0]
    return values[indices].copy()


def resample_linear(source: np.ndarray, ratio: float) -> np.ndarray:
    """Linear sample-rate conversion by a positive output/input length ratio."""

    values = np.asarray(source, dtype=np.float64)
    if values.shape[0] == 0:
        return values.copy()
    if ratio <= 0.0 or not math.isfinite(ratio):
        raise ValueError("resample ratio must be positive and finite")
    output_length = max(1, round_half_up(values.shape[0] * ratio))
    if output_length == values.shape[0] and abs(ratio - 1.0) < 1e-12:
        return values.copy()

    positions = np.arange(output_length, dtype=np.float64) / ratio
    positions = np.minimum(positions, values.shape[0] - 1)
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, values.shape[0] - 1)
    fraction = (positions - left)[:, None]
    return values[left] * (1.0 - fraction) + values[right] * fraction


def prepare_percussion_loop(
    source: np.ndarray,
    *,
    target_beat_samples: int,
    source_beat_samples: int,
    confidence: float,
) -> tuple[np.ndarray, int, float]:
    """Tempo-map percussion to the master beat grid and pad to whole bars."""

    loop_samples = target_beat_samples * 4
    if confidence >= CONFIDENCE_THRESHOLD:
        tempo_ratio = target_beat_samples / float(source_beat_samples)
    else:
        tempo_ratio = 1.0
    mapped = resample_linear(source, tempo_ratio)

    bars = max(1, int(math.ceil(mapped.shape[0] / float(loop_samples))))
    aligned_length = bars * loop_samples
    aligned = np.zeros((aligned_length, mapped.shape[1]), dtype=np.float64)
    copy_length = min(aligned_length, mapped.shape[0])
    aligned[:copy_length] = mapped[:copy_length]
    return aligned, aligned_length, tempo_ratio


def db_from_linear(value: float) -> float:
    if value <= 0.0:
        return DB_FLOOR
    return max(DB_FLOOR, 20.0 * math.log10(float(value)))


def apply_soft_limiter(mix: np.ndarray) -> LimiterResult:
    """Apply a tanh soft limiter only when the bus exceeds 0 dBFS."""

    values = np.asarray(mix, dtype=np.float64)
    pre_peak = float(np.max(np.abs(values))) if values.size else 0.0
    if pre_peak <= 1.0:
        return LimiterResult(
            audio=values,
            applied=False,
            pre_peak=pre_peak,
            post_peak=pre_peak,
            gain_adjustment_db=0.0,
        )

    limited = LIMIT_CEILING * np.tanh(values / LIMIT_CEILING)
    post_peak = float(np.max(np.abs(limited))) if limited.size else 0.0
    gain_db = db_from_linear(post_peak) - db_from_linear(pre_peak)
    return LimiterResult(
        audio=limited,
        applied=True,
        pre_peak=pre_peak,
        post_peak=post_peak,
        gain_adjustment_db=round(float(gain_db), 6),
    )

