"""Small NumPy DSP helpers used by the offline renderer."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .scene import AttenuationConfig


def db_to_gain(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def distance_vectors(source: Tuple[float, float, float],
                     listener: Tuple[float, float, float]) -> Tuple[float, np.ndarray]:
    delta = np.asarray(source, dtype=np.float64) - np.asarray(listener, dtype=np.float64)
    return float(np.linalg.norm(delta)), delta


def distance_gain(distance: float, config: AttenuationConfig) -> float:
    if config.model == "none":
        return float(config.max_gain)
    if config.max_distance is not None and distance > config.max_distance:
        return 0.0
    if config.model == "linear":
        if config.max_distance is None or config.max_distance <= config.min_distance:
            return float(config.max_gain)
        span = config.max_distance - config.min_distance
        normalized = (config.max_distance - distance) / span
        return float(config.max_gain * min(1.0, max(0.0, normalized)))
    if config.model == "inverse_distance":
        clamped = max(distance, config.min_distance)
        reference = config.reference_distance
        if clamped <= reference:
            return float(config.max_gain)
        gain = (reference / clamped) ** config.rolloff
        return float(min(config.max_gain, gain))
    raise AssertionError("unsupported attenuation model: %s" % config.model)


def equal_power_pan(pan: float) -> Tuple[float, float]:
    clamped = min(1.0, max(-1.0, float(pan)))
    angle = (clamped + 1.0) * math.pi / 4.0
    return math.cos(angle), math.sin(angle)


def automatic_pan(delta_source_listener: np.ndarray,
                  listener_forward: Tuple[float, float, float],
                  listener_up: Tuple[float, float, float]) -> float:
    forward = np.asarray(listener_forward, dtype=np.float64)
    up = np.asarray(listener_up, dtype=np.float64)
    forward_norm = float(np.linalg.norm(forward))
    up_norm = float(np.linalg.norm(up))
    if forward_norm <= 1e-12 or up_norm <= 1e-12:
        return 0.0
    forward = forward / forward_norm
    up = up / up_norm
    right = np.cross(forward, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-12:
        return 0.0
    right = right / right_norm
    delta_norm = float(np.linalg.norm(delta_source_listener))
    if delta_norm <= 1e-12:
        return 0.0
    direction = delta_source_listener / delta_norm
    return float(min(1.0, max(-1.0, float(np.dot(direction, right)))))


def occlusion_cutoff(amount: float, open_cutoff_hz: float,
                     closed_cutoff_hz: float, sample_rate: int) -> float:
    amount = min(1.0, max(0.0, float(amount)))
    cutoff = open_cutoff_hz + amount * (closed_cutoff_hz - open_cutoff_hz)
    nyquist_limit = sample_rate * 0.45
    return float(min(max(20.0, cutoff), nyquist_limit))


def one_pole_lowpass(signal: np.ndarray, cutoff_hz: float,
                     sample_rate: int, initial: float = 0.0) -> np.ndarray:
    """Apply a causal first-order low-pass filter.

    A short explicit loop keeps the implementation dependency-free while the
    filtering itself is deterministic and independent of chunk boundaries.
    """

    source = np.asarray(signal, dtype=np.float64)
    if source.size == 0:
        return source.copy()
    if cutoff_hz >= sample_rate * 0.499:
        return source.copy()
    cutoff_hz = min(max(1.0, float(cutoff_hz)), sample_rate * 0.499)
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz / float(sample_rate))
    output = np.empty_like(source)
    previous = float(initial)
    for index in range(source.size):
        previous += alpha * (float(source[index]) - previous)
        output[index] = previous
    return output


def maybe_lowpass(signal: np.ndarray, cutoff_hz: Optional[float],
                  sample_rate: int) -> np.ndarray:
    if cutoff_hz is None:
        return np.asarray(signal, dtype=np.float64).copy()
    if cutoff_hz >= sample_rate * 0.45:
        return np.asarray(signal, dtype=np.float64).copy()
    return one_pole_lowpass(signal, cutoff_hz, sample_rate)


def fft_convolve(signal: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    """Full linear convolution using a deterministic power-of-two FFT."""

    source = np.asarray(signal, dtype=np.float64)
    kernel = np.asarray(impulse, dtype=np.float64)
    if source.size == 0 or kernel.size == 0:
        return np.zeros(0, dtype=np.float64)
    output_length = source.size + kernel.size - 1
    fft_length = 1 << (output_length - 1).bit_length()
    spectrum = np.fft.rfft(source, fft_length) * np.fft.rfft(kernel, fft_length)
    return np.fft.irfft(spectrum, fft_length)[:output_length]

