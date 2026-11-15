"""Host-triggered, one-pole sidechain ducking for the music bed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .manifest import DuckingConfig


@dataclass(frozen=True)
class DuckingResult:
    trigger_count: int
    ducked_samples: int
    max_reduction_db: float
    energy_window_samples: int
    attack_samples: int
    release_samples: int


def _milliseconds_to_samples(milliseconds: float, sample_rate: int) -> int:
    if milliseconds <= 0:
        return 0
    return max(1, int(round(milliseconds * sample_rate / 1000.0)))


def _one_pole_alpha(time_samples: int) -> float:
    if time_samples <= 0:
        return 1.0
    return float(1.0 - np.exp(-1.0 / time_samples))


def short_term_energy_db(signal: np.ndarray, window_samples: int) -> np.ndarray:
    """Causal mean-square energy in dBFS, averaged over channels."""

    if signal.ndim != 2:
        raise ValueError("signal must be shaped (frames, channels)")
    frames = signal.shape[0]
    window = max(1, min(window_samples, frames))
    power = np.mean(np.square(signal), axis=1)
    cumulative = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(power)))
    indices = np.arange(frames, dtype=np.int64)
    starts = np.maximum(0, indices + 1 - window)
    lengths = indices + 1 - starts
    mean_power = (cumulative[indices + 1] - cumulative[starts]) / lengths
    return 10.0 * np.log10(np.maximum(mean_power, 1.0e-30))


def apply_ducking(
    tracks: dict[str, np.ndarray],
    config: DuckingConfig,
    sample_rate: int,
) -> DuckingResult:
    """Apply a smoothed gain reduction to the configured music track in-place."""

    window_samples = _milliseconds_to_samples(config.energy_window_ms, sample_rate)
    attack_samples = _milliseconds_to_samples(config.attack_ms, sample_rate)
    release_samples = _milliseconds_to_samples(config.release_ms, sample_rate)
    if not config.enabled:
        return DuckingResult(0, 0, 0.0, window_samples, attack_samples, release_samples)

    trigger_signal = tracks[config.trigger_track]
    energy_db = short_term_energy_db(trigger_signal, window_samples)
    active = energy_db > config.threshold_dbfs
    previous = np.concatenate((np.array([False]), active[:-1]))
    rising = active & ~previous
    trigger_count = int(np.count_nonzero(rising))

    attack_alpha = _one_pole_alpha(attack_samples)
    release_alpha = _one_pole_alpha(release_samples)
    reduction_db = np.zeros(energy_db.shape[0], dtype=np.float64)
    current = 0.0
    for index, is_active in enumerate(active):
        target = config.depth_db if is_active else 0.0
        alpha = attack_alpha if target > current else release_alpha
        current += (target - current) * alpha
        reduction_db[index] = current

    gain = np.power(10.0, -reduction_db / 20.0)
    tracks[config.target_track] *= gain[:, None]
    return DuckingResult(
        trigger_count=trigger_count,
        ducked_samples=int(np.count_nonzero(reduction_db > 1.0e-6)),
        max_reduction_db=float(np.max(reduction_db)) if reduction_db.size else 0.0,
        energy_window_samples=window_samples,
        attack_samples=attack_samples,
        release_samples=release_samples,
    )

