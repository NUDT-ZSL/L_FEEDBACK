"""Side-chain dynamic ducking of the music bus.

The control signal is the amplitude envelope of the aligned dialogue bus.
Music gain falls at 3 dB per octave of dialogue level above an onset
threshold, is capped at -12 dB, attacks in ~10 ms and recovers to at least
99% (within ~0.09 dB of unity) no later than 200 ms after dialogue ends.
"""

from __future__ import annotations

import math

import numpy as np

from . import TARGET_SR

ONSET_THRESHOLD_DBFS = -45.0
MAX_REDUCTION_DB = 12.0
SLOPE_DB_PER_OCTAVE = 3.0
ATTACK_TAU_SECONDS = 0.010
RECOVERY_WINDOW_SECONDS = 0.200
RECOVERY_TARGET_RATIO = 0.99


def _one_pole_coefficient(tau_seconds: float, sample_rate: int) -> float:
    return math.exp(-1.0 / (tau_seconds * sample_rate))


def target_music_gain_db(control: np.ndarray) -> np.ndarray:
    """Instantaneous target music gain (dB) from the dialogue amplitude bus."""
    level_db = 20.0 * np.log10(np.maximum(np.abs(control), 1e-6))
    octaves_above = np.maximum(0.0, level_db - ONSET_THRESHOLD_DBFS) / 3.0102999566
    reduction = SLOPE_DB_PER_OCTAVE * octaves_above
    return np.maximum(-MAX_REDUCTION_DB, -reduction)


def smooth_gain(target_gain_db: np.ndarray, sample_rate: int = TARGET_SR) -> np.ndarray:
    """Attack/release one-pole smoothing of the target gain in linear domain."""
    target_linear = 10.0 ** (target_gain_db / 20.0)
    attack_alpha = _one_pole_coefficient(ATTACK_TAU_SECONDS, sample_rate)
    # Choose release tau so recovery reaches 99% within the 200 ms window.
    release_tau = RECOVERY_WINDOW_SECONDS / -math.log(1.0 - RECOVERY_TARGET_RATIO)
    release_alpha = _one_pole_coefficient(release_tau, sample_rate)

    smoothed = np.empty_like(target_linear)
    current = 1.0
    for index, target in enumerate(target_linear):
        alpha = attack_alpha if target < current else release_alpha
        current = alpha * current + (1.0 - alpha) * target
        smoothed[index] = current
    return smoothed


def build_duck_gain(dialogue_bus: np.ndarray, sample_rate: int = TARGET_SR) -> np.ndarray:
    """Full music gain envelope (linear, length of dialogue bus)."""
    target_db = target_music_gain_db(dialogue_bus)
    return smooth_gain(target_db, sample_rate)


def apply_ducking(music: np.ndarray, dialogue_bus: np.ndarray,
                  sample_rate: int = TARGET_SR) -> tuple[np.ndarray, np.ndarray]:
    """Return (ducked_music, linear_gain_envelope)."""
    gain = build_duck_gain(dialogue_bus, sample_rate)
    length = min(music.size, gain.size)
    ducked = music.copy()
    ducked[:length] *= gain[:length]
    return ducked, gain


def measure_recovery_time(gain: np.ndarray, active_mask: np.ndarray,
                          sample_rate: int = TARGET_SR) -> float | None:
    """Time from last active dialogue sample until gain >= 99% of unity."""
    active_indices = np.flatnonzero(active_mask)
    if active_indices.size == 0:
        return None
    last_active = int(active_indices[-1])
    recovered = np.flatnonzero(gain[last_active:] >= RECOVERY_TARGET_RATIO)
    if recovered.size == 0:
        return None
    return float(recovered[0]) / sample_rate
