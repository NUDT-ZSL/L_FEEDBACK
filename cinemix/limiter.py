from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import LimiterSpec
from .dsp import true_peak


@dataclass(frozen=True)
class LimiterResult:
    samples: np.ndarray
    sample_ceiling_dbtp: float
    max_reduction_db: float
    passes: int


def _db_to_linear(value_db: float) -> float:
    return 10.0 ** (value_db / 20.0)


def _sliding_future_min(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    if window <= 1:
        return values.copy()
    result = values.copy()
    shift = 1
    while shift < window:
        result[:-shift] = np.minimum(result[:-shift], result[shift:])
        shift *= 2
    return result


def _limit_pass(
    samples: np.ndarray,
    sample_rate: int,
    sample_ceiling_dbtp: float,
    spec: LimiterSpec,
) -> tuple[np.ndarray, float]:
    working = np.asarray(samples, dtype=np.float64)
    if working.ndim == 1:
        working = working[:, None]
        squeeze = True
    else:
        squeeze = False
    frames = working.shape[0]
    if frames == 0:
        return (working[:, 0] if squeeze else working.copy()), 0.0

    ceiling = _db_to_linear(sample_ceiling_dbtp)
    compression_threshold_db = spec.compression_threshold_dbtp
    if compression_threshold_db is None:
        compression_threshold_db = sample_ceiling_dbtp - 0.5
    compression_threshold_db = min(compression_threshold_db, sample_ceiling_dbtp)
    compression_threshold = _db_to_linear(compression_threshold_db)
    exponent = 1.0 - 1.0 / spec.ratio
    detector = np.max(np.abs(working), axis=1)
    detector_safe = np.maximum(detector, np.finfo(np.float64).tiny)

    target = np.ones_like(detector)
    compressed = detector > compression_threshold
    target[compressed] = (
        compression_threshold / detector_safe[compressed]
    ) ** exponent
    target = np.minimum(target, ceiling / detector_safe)
    target = np.clip(target, 0.0, 1.0)

    lookahead_samples = max(
        1, round(spec.lookahead_ms * 0.001 * sample_rate)
    )
    future_minimum = _sliding_future_min(target, lookahead_samples)
    attack = math.exp(-1.0 / max(1.0, spec.attack_ms * 0.001 * sample_rate))
    release = math.exp(-1.0 / max(1.0, spec.release_ms * 0.001 * sample_rate))
    hard_gain = np.minimum(1.0, ceiling / detector_safe)

    gain = 1.0
    max_reduction = 0.0
    output = np.empty_like(working)
    for index in range(frames):
        desired = float(future_minimum[index])
        coefficient = attack if desired < gain else release
        gain = coefficient * gain + (1.0 - coefficient) * desired
        current_hard = float(hard_gain[index])
        gain = min(gain, current_hard)
        gain = max(gain, 0.0)
        max_reduction = max(max_reduction, -20.0 * math.log10(max(gain, 1e-12)))
        output[index, :] = working[index, :] * gain

    if squeeze:
        return output[:, 0], max_reduction
    return output, max_reduction


def apply_limiter(
    samples: np.ndarray,
    sample_rate: int,
    true_peak_limit_dbtp: float,
    spec: LimiterSpec,
    *,
    max_true_peak_passes: int = 5,
) -> LimiterResult:
    """Apply a feed-forward lookahead compressor/limiter.

    The sidechain is the unlimitted mix bus. A sample-domain hard ceiling
    prevents local overshoot. Because intersample peaks can sit above sample
    peaks, the complete limiter is re-run with a lower internal ceiling until
    a 4x true-peak measurement satisfies the configured limit.
    """
    target_limit = true_peak_limit_dbtp
    sample_ceiling = target_limit - spec.safety_margin_db
    passes = 0
    result: np.ndarray | None = None
    reduction = 0.0
    for _ in range(max_true_peak_passes):
        passes += 1
        result, reduction = _limit_pass(
            samples, sample_rate, sample_ceiling, spec
        )
        measured = true_peak(result, sample_rate)
        if measured.dbtp <= target_limit - (spec.safety_margin_db * 0.25):
            break
        excess = measured.dbtp - (target_limit - spec.safety_margin_db * 0.5)
        sample_ceiling -= max(0.05, excess + 0.05)
        if sample_ceiling < -40.0:
            break
    assert result is not None
    return LimiterResult(
        samples=result,
        sample_ceiling_dbtp=sample_ceiling,
        max_reduction_db=reduction,
        passes=passes,
    )
