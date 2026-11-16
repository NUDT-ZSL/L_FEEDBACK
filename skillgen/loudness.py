from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal


def _k_weighting_coefficients(sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fs = float(sample_rate)
    f0 = 1681.974450955533
    gain_db = 3.999843853973347
    q = 0.7071752369554196
    k = math.tan(math.pi * f0 / fs)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh**0.4996667741545416
    a0 = 1.0 + k / q + k * k
    shelf_b = np.array(
        [
            (vh + vb * k / q + k * k) / a0,
            2.0 * (k * k - vh) / a0,
            (vh - vb * k / q + k * k) / a0,
        ],
        dtype=np.float64,
    )
    shelf_a = np.array(
        [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
        dtype=np.float64,
    )

    f0 = 38.13547087602444
    q = 0.5003270373238773
    k = math.tan(math.pi * f0 / fs)
    a0 = 1.0 + k / q + k * k
    highpass_b = np.array([1.0, -2.0, 1.0], dtype=np.float64)
    highpass_a = np.array(
        [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
        dtype=np.float64,
    )
    return shelf_b, shelf_a, highpass_b, highpass_a


def k_weight(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    shelf_b, shelf_a, highpass_b, highpass_a = _k_weighting_coefficients(sample_rate)
    filtered = signal.lfilter(shelf_b, shelf_a, samples)
    return signal.lfilter(highpass_b, highpass_a, filtered)


def integrated_loudness_mlfs(samples: np.ndarray, sample_rate: int) -> float | None:
    """BS.1770-style gated integrated loudness for a mono signal."""
    if samples.size == 0 or sample_rate <= 0:
        return None
    weighted = k_weight(np.asarray(samples, dtype=np.float64), sample_rate)
    block = max(1, int(round(0.400 * sample_rate)))
    hop = max(1, int(round(0.100 * sample_rate)))
    if weighted.size < block:
        padded = np.zeros(block, dtype=np.float64)
        padded[: weighted.size] = weighted
        weighted = padded
    starts = list(range(0, weighted.size - block + 1, hop))
    if not starts:
        starts = [0]
    energies = np.array(
        [np.mean(weighted[start : start + block] ** 2, dtype=np.float64) for start in starts],
        dtype=np.float64,
    )
    if energies.size == 0:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        block_loudness = -0.691 + 10.0 * np.log10(energies + 1e-30)
    absolute = block_loudness > -70.0
    if not np.any(absolute):
        return None
    absolute_energy = float(np.mean(energies[absolute], dtype=np.float64))
    relative_loudness = -0.691 + 10.0 * math.log10(absolute_energy + 1e-30) - 10.0
    gated = absolute & (block_loudness > relative_loudness)
    if not np.any(gated):
        return None
    final_energy = float(np.mean(energies[gated], dtype=np.float64))
    if final_energy <= 0.0:
        return None
    return -0.691 + 10.0 * math.log10(final_energy)


def soft_limit(samples: np.ndarray, knee: float = 0.90, ceiling: float = 0.999) -> np.ndarray:
    """Smooth peak limiter with an asymptotic ceiling below 1.0."""
    array = np.asarray(samples, dtype=np.float64)
    magnitude = np.abs(array)
    limited = magnitude.copy()
    mask = magnitude > knee
    if np.any(mask):
        span = ceiling - knee
        limited[mask] = knee + span * np.tanh((magnitude[mask] - knee) / span)
    return np.sign(array) * limited


@dataclass
class LoudnessCalibration:
    samples: np.ndarray
    target_mlfs: float
    before_mlfs: float | None
    after_mlfs: float | None
    gain_db: float
    peak: float
    pre_limit_peak: float
    limited: bool
    error_code: str | None = None
    error_message: str | None = None


def calibrate_loudness(
    samples: np.ndarray,
    sample_rate: int,
    target_mlfs: float,
    tolerance_lu: float = 0.5,
) -> LoudnessCalibration:
    original = np.asarray(samples, dtype=np.float64)
    before = integrated_loudness_mlfs(original, sample_rate)
    if before is None:
        return LoudnessCalibration(
            samples=original,
            target_mlfs=target_mlfs,
            before_mlfs=None,
            after_mlfs=None,
            gain_db=0.0,
            peak=float(np.max(np.abs(original))) if original.size else 0.0,
            pre_limit_peak=float(np.max(np.abs(original))) if original.size else 0.0,
            limited=False,
            error_code="E_LOUDNESS",
            error_message="input is too quiet for gated integrated loudness measurement",
        )

    desired_gain_db = target_mlfs - before
    total_gain_db = desired_gain_db
    current = original * (10.0 ** (desired_gain_db / 20.0))
    pre_limit_peak = float(np.max(np.abs(current))) if current.size else 0.0
    limited = pre_limit_peak > 1.0
    if limited:
        current = soft_limit(current)

    after = integrated_loudness_mlfs(current, sample_rate)
    for _ in range(10):
        if after is None:
            break
        delta = target_mlfs - after
        if abs(delta) <= 0.05:
            break
        total_gain_db += delta
        current = current * (10.0 ** (delta / 20.0))
        peak_now = float(np.max(np.abs(current))) if current.size else 0.0
        if peak_now > 1.0:
            limited = True
            current = soft_limit(current)
        after = integrated_loudness_mlfs(current, sample_rate)

    if after is None:
        return LoudnessCalibration(
            samples=current,
            target_mlfs=target_mlfs,
            before_mlfs=before,
            after_mlfs=None,
            gain_db=total_gain_db,
            peak=float(np.max(np.abs(current))) if current.size else 0.0,
            pre_limit_peak=pre_limit_peak,
            limited=limited,
            error_code="E_LOUDNESS",
            error_message="unable to measure loudness after gain calibration",
        )

    error_code = None
    error_message = None
    if abs(after - target_mlfs) > tolerance_lu:
        error_code = "E_LOUDNESS"
        error_message = (
            f"loudness outside target tolerance after calibration: "
            f"{after:.4f} MLFS vs {target_mlfs:.4f}"
        )

    return LoudnessCalibration(
        samples=current,
        target_mlfs=target_mlfs,
        before_mlfs=before,
        after_mlfs=after,
        gain_db=total_gain_db,
        peak=float(np.max(np.abs(current))) if current.size else 0.0,
        pre_limit_peak=pre_limit_peak,
        limited=limited,
        error_code=error_code,
        error_message=error_message,
    )
