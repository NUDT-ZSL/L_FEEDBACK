"""Deterministic WAV loading, resampling, and event fitting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

from .errors import ToolError

AUDIO_SAMPLE_RATE = 48000


@dataclass(frozen=True)
class MonoClip:
    """A mono float64 clip and its source metadata."""

    samples: np.ndarray
    source_sample_rate: int
    source_frames: int
    source_channels: int


def read_wav_mono(path: Path, field: str) -> MonoClip:
    """Read a WAV file as a deterministic mono float64 signal."""

    try:
        data, sample_rate = sf.read(
            str(path),
            dtype="float64",
            always_2d=True,
        )
    except Exception as exc:
        raise ToolError(
            "{}: cannot read WAV file '{}': {}".format(field, path.name, exc)
        ) from exc

    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ToolError("{}: WAV file '{}' contains no audio frames".format(field, path.name))
    if sample_rate <= 0:
        raise ToolError(
            "{}: WAV file '{}' has an invalid sample rate {}".format(
                field, path.name, sample_rate
            )
        )

    source_channels = int(data.shape[1])
    mono = np.mean(data, axis=1, dtype=np.float64)
    if not np.all(np.isfinite(mono)):
        raise ToolError(
            "{}: WAV file '{}' contains non-finite samples".format(field, path.name)
        )

    return MonoClip(
        samples=np.ascontiguousarray(mono, dtype=np.float64),
        source_sample_rate=int(sample_rate),
        source_frames=int(data.shape[0]),
        source_channels=source_channels,
    )


def _interpolate_at(signal: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Linearly interpolate a one-dimensional signal at sample positions."""

    source_length = int(signal.shape[0])
    if source_length == 0:
        raise ValueError("cannot interpolate an empty signal")

    positions = np.asarray(positions, dtype=np.float64)
    positions = np.clip(positions, 0.0, float(source_length - 1))
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, source_length - 1)
    fraction = positions - lower.astype(np.float64)
    result = (
        signal[lower] * (1.0 - fraction)
        + signal[upper] * fraction
    )
    return np.ascontiguousarray(result, dtype=np.float64)


def _scaled_length(source_length: int, source_rate: int, target_rate: int) -> int:
    scaled = int(source_length) * int(target_rate) / int(source_rate)
    return max(1, int(math.floor(scaled + 0.5)))


def resample_to_rate(
    signal: np.ndarray,
    source_rate: int,
    target_rate: int = AUDIO_SAMPLE_RATE,
) -> np.ndarray:
    """Resample with deterministic linear interpolation."""

    if int(source_rate) <= 0 or int(target_rate) <= 0:
        raise ValueError("sample rates must be positive")
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1 or signal.shape[0] == 0:
        raise ValueError("signal must be a non-empty one-dimensional array")
    if int(source_rate) == int(target_rate):
        return np.ascontiguousarray(signal.copy(), dtype=np.float64)

    target_length = _scaled_length(signal.shape[0], source_rate, target_rate)
    positions = (
        np.arange(target_length, dtype=np.float64)
        * float(source_rate)
        / float(target_rate)
    )
    return _interpolate_at(signal, positions)


def stretch_to_length(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Time-stretch a signal to an exact length using linear interpolation."""

    signal = np.asarray(signal, dtype=np.float64)
    target_length = int(target_length)
    if signal.ndim != 1 or signal.shape[0] == 0:
        raise ValueError("signal must be a non-empty one-dimensional array")
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if signal.shape[0] == target_length:
        return np.ascontiguousarray(signal.copy(), dtype=np.float64)

    source_length = int(signal.shape[0])
    positions = (
        np.arange(target_length, dtype=np.float64)
        * float(source_length)
        / float(target_length)
    )
    return _interpolate_at(signal, positions)


def fit_to_duration(
    signal: np.ndarray,
    target_length: int,
    strategy: str,
) -> Tuple[np.ndarray, str]:
    """Fit a clip to an event duration and return (samples, fit_kind)."""

    target_length = int(target_length)
    if target_length <= 0:
        raise ValueError("target_length must be positive")

    source_length = int(signal.shape[0])
    if source_length == target_length:
        return (
            np.ascontiguousarray(signal.copy(), dtype=np.float64),
            "none",
        )

    if strategy == "stretch":
        return stretch_to_length(signal, target_length), "stretched"

    if strategy == "crop":
        if source_length > target_length:
            return (
                np.ascontiguousarray(signal[:target_length].copy(), dtype=np.float64),
                "cropped",
            )
        padding = np.zeros(target_length - source_length, dtype=np.float64)
        return (
            np.ascontiguousarray(np.concatenate((signal, padding)), dtype=np.float64),
            "padded",
        )

    raise ValueError("unknown fit strategy: {}".format(strategy))
