"""Deterministic stereo reverb impulse generation and shared-bus convolution."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .dsp import fft_convolve
from .scene import ReverbConfig


@dataclass(frozen=True)
class ReverbImpulse:
    stereo: np.ndarray
    pre_delay_samples: int
    length_samples: int
    sha256: str


def _channel_seed(base_seed: int, channel: int) -> int:
    sequence = np.random.SeedSequence([int(base_seed), int(channel)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _apply_spectral_damping(impulse: np.ndarray,
                            damping_hz: Optional[float],
                            sample_rate: int) -> np.ndarray:
    if damping_hz is None or damping_hz >= sample_rate * 0.45:
        return impulse
    fft_length = 1 << (impulse.size - 1).bit_length() if impulse.size > 1 else 1
    spectrum = np.fft.rfft(impulse, fft_length)
    frequencies = np.fft.rfftfreq(fft_length, d=1.0 / sample_rate)
    magnitude = 1.0 / np.sqrt(1.0 + (frequencies / float(damping_hz)) ** 2)
    return np.fft.irfft(spectrum * magnitude, fft_length)[:impulse.size]


def _normalized_channel(noise: np.ndarray, envelope: np.ndarray,
                        sample_rate: int) -> np.ndarray:
    impulse = noise * envelope
    fade_samples = min(impulse.size, max(1, int(round(0.01 * sample_rate))))
    if impulse.size > fade_samples:
        fade = 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, fade_samples)))
        impulse[-fade_samples:] *= fade
    energy = float(np.sum(impulse.astype(np.float64) ** 2))
    if energy > 0.0:
        impulse = impulse / math.sqrt(energy)
    return impulse


def build_reverb_impulse(config: ReverbConfig, sample_rate: int) -> ReverbImpulse:
    pre_delay_samples = int(round(config.pre_delay_seconds * sample_rate))
    decay_samples = max(1, int(round(config.rt60_seconds * sample_rate)))
    length = pre_delay_samples + decay_samples
    time_after_predelay = np.arange(decay_samples, dtype=np.float64) / float(sample_rate)
    envelope = 10.0 ** (-3.0 * time_after_predelay / config.rt60_seconds)
    padding = np.zeros(pre_delay_samples, dtype=np.float64)

    channels = []
    for channel in range(2):
        rng = np.random.Generator(np.random.PCG64(_channel_seed(config.seed, channel)))
        noise = rng.standard_normal(decay_samples)
        channel_signal = _normalized_channel(noise, envelope, sample_rate)
        channel_signal = _apply_spectral_damping(channel_signal, config.damping_hz, sample_rate)
        channel_signal = np.concatenate((padding, channel_signal))
        energy = float(np.sum(channel_signal.astype(np.float64) ** 2))
        if energy > 0.0:
            channel_signal = channel_signal / math.sqrt(energy)
        channels.append(channel_signal)

    stereo = np.stack(channels, axis=0).astype(np.float64, copy=False)
    digest = hashlib.sha256(np.ascontiguousarray(stereo).tobytes()).hexdigest()
    return ReverbImpulse(stereo, pre_delay_samples, length, digest)


def convolve_shared_bus(wet_input: np.ndarray, impulse: ReverbImpulse,
                        total_samples: int,
                        wet_gain: float) -> Tuple[np.ndarray, np.ndarray]:
    """Convolve one mono send into the shared stereo wet bus."""

    if impulse.stereo.shape[0] != 2:
        raise AssertionError("reverb impulse must have two channels")
    left = fft_convolve(wet_input, impulse.stereo[0])[:total_samples]
    right = fft_convolve(wet_input, impulse.stereo[1])[:total_samples]
    if left.size < total_samples:
        left = np.pad(left, (0, total_samples - left.size))
    if right.size < total_samples:
        right = np.pad(right, (0, total_samples - right.size))
    return left * wet_gain, right * wet_gain

