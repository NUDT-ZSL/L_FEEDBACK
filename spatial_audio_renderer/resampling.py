"""Band-limited, continuously phased time-varying sinc resampling."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

from .errors import TrackLoadError


def _sinc_kernel(
    fractions: np.ndarray,
    rates: np.ndarray,
    taps: int,
) -> np.ndarray:
    if taps < 8 or taps % 2 != 0:
        raise ValueError("taps must be an even integer >= 8")
    half = taps // 2
    offsets = np.arange(-half + 1, half + 1, dtype=np.float64)
    distances = fractions[:, None] - offsets[None, :]
    cutoff = np.minimum(1.0, 1.0 / np.maximum(rates, 1.0e-6))
    kernel = (
        cutoff[:, None]
        * np.sinc(cutoff[:, None] * distances)
        * np.kaiser(taps, 8.0)[None, :]
    )
    weight_sum = np.sum(kernel, axis=1, keepdims=True)
    kernel /= np.where(np.abs(weight_sum) > 1.0e-12, weight_sum, 1.0)
    return kernel


def sinc_interpolate(
    samples: np.ndarray,
    positions: np.ndarray,
    rates: np.ndarray,
    loop: bool,
    taps: int = 32,
) -> np.ndarray:
    """Evaluate a mono clip at fractional sample positions.

    The low-pass cutoff follows ``min(1, 1/rate)``, which suppresses input
    frequencies that would otherwise fold back when playback is sped up.
    """
    samples = np.asarray(samples, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    rates = np.asarray(rates, dtype=np.float64)
    if len(samples) == 0:
        return np.zeros(len(positions), dtype=np.float64)
    if positions.shape != rates.shape:
        raise ValueError("positions and rates must have the same shape")
    if len(positions) == 0:
        return np.zeros(0, dtype=np.float64)
    half = taps // 2
    offsets = np.arange(-half + 1, half + 1, dtype=np.int64)
    base = np.floor(positions).astype(np.int64)
    fractions = positions - base
    kernel = _sinc_kernel(fractions, rates, taps)
    indices = base[:, None] + offsets[None, :]
    if loop:
        values = samples[np.mod(indices, len(samples))]
        weighted = values * kernel
    else:
        valid = (indices >= 0) & (indices < len(samples))
        safe_indices = np.clip(indices, 0, len(samples) - 1)
        kernel = kernel * valid
        values = samples[safe_indices]
        weighted = values * kernel
    denominator = np.sum(kernel, axis=1)
    numerator = np.sum(weighted, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=np.abs(denominator) > 1.0e-12,
    )


def resample_audio(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int,
    taps: int = 32,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if source_rate == target_rate:
        return samples.copy()
    output_length = max(
        1, int(round(len(samples) * float(target_rate) / float(source_rate)))
    )
    positions = (
        np.arange(output_length, dtype=np.float64)
        * float(source_rate)
        / float(target_rate)
    )
    rates = np.full(output_length, float(source_rate) / float(target_rate))
    return sinc_interpolate(samples, positions, rates, loop=False, taps=taps)


class DopplerClock:
    """A continuous fractional clip-position clock shared across frames."""

    def __init__(self, start_position: float = 0.0) -> None:
        self.cursor = float(start_position)

    def positions(self, rates: np.ndarray) -> np.ndarray:
        rates = np.asarray(rates, dtype=np.float64)
        if len(rates) == 0:
            return np.zeros(0, dtype=np.float64)
        if np.any(~np.isfinite(rates)) or np.any(rates <= 0.0):
            raise ValueError("Doppler rates must be finite and positive")
        positions = np.empty(len(rates), dtype=np.float64)
        positions[0] = self.cursor
        if len(rates) > 1:
            positions[1:] = self.cursor + np.cumsum(rates[:-1])
        self.cursor += float(np.sum(rates))
        return positions


class SincTrackRenderer:
    """Stateful mono clip reader with looping and Doppler phase continuity."""

    def __init__(
        self,
        samples: np.ndarray,
        loop: bool,
        taps: int = 32,
    ) -> None:
        self.samples = np.asarray(samples, dtype=np.float64)
        self.loop = bool(loop)
        self.taps = int(taps)
        self.clock = DopplerClock()

    def render(self, rates: np.ndarray) -> np.ndarray:
        positions = self.clock.positions(rates)
        return sinc_interpolate(
            self.samples,
            positions,
            np.asarray(rates, dtype=np.float64),
            loop=self.loop,
            taps=self.taps,
        )


def load_audio_track(
    path: Path,
    target_sample_rate: int,
    taps: int = 32,
) -> np.ndarray:
    """Decode a WAV, downmix channels, and resample to the scene rate."""
    try:
        samples, sample_rate = sf.read(
            str(path), dtype="float64", always_2d=True
        )
    except Exception as exc:
        raise TrackLoadError(
            "cannot decode WAV %s: %s" % (path, exc)
        ) from exc
    if samples.size == 0 or samples.shape[1] == 0:
        raise TrackLoadError("WAV has no audio samples: %s" % path)
    if sample_rate <= 0:
        raise TrackLoadError("WAV has an invalid sample rate: %s" % path)
    if not np.all(np.isfinite(samples)):
        raise TrackLoadError("WAV contains non-finite samples: %s" % path)
    mono = np.mean(samples, axis=1)
    if sample_rate != target_sample_rate:
        mono = resample_audio(
            mono, int(sample_rate), int(target_sample_rate), taps=taps
        )
    if len(mono) == 0:
        raise TrackLoadError("WAV produced no usable samples: %s" % path)
    return np.asarray(mono, dtype=np.float64)
