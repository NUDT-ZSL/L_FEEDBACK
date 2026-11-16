"""Sample-domain audio helpers."""

from __future__ import annotations

import math

import numpy as np


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def linear_to_db(value: float) -> float:
    if value <= 0.0 or not math.isfinite(value):
        return -math.inf
    return 20.0 * math.log10(value)


def trim_leading_trailing_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    absolute_threshold_dbfs: float = -60.0,
    relative_threshold_db: float = -50.0,
    frame_seconds: float = 0.010,
    padding_seconds: float = 0.002,
) -> np.ndarray:
    """Trim silent head and tail using overlapping-independent 10 ms frames."""

    working = np.asarray(audio, dtype=np.float64)
    if working.ndim != 2 or working.shape[0] == 0:
        return working[:0].copy()

    mono_peak = np.max(np.abs(working), axis=1)
    global_peak = float(np.max(mono_peak))
    if global_peak <= 0.0:
        return working[:0].copy()

    frame_size = max(1, int(round(frame_seconds * sample_rate)))
    frame_count = int(math.ceil(working.shape[0] / frame_size))
    padded = np.zeros((frame_count * frame_size, working.shape[1]))
    padded[: working.shape[0]] = working
    frames = padded.reshape(frame_count, frame_size, working.shape[1])
    frame_rms = np.sqrt(np.mean(frames * frames, axis=(1, 2)))

    threshold = max(
        db_to_linear(absolute_threshold_dbfs),
        global_peak * db_to_linear(relative_threshold_db),
    )
    active = np.flatnonzero(frame_rms >= threshold)
    if active.size == 0:
        return working[:0].copy()

    padding = max(0, int(round(padding_seconds * sample_rate)))
    start = max(0, int(active[0]) * frame_size - padding)
    end = min(
        working.shape[0],
        (int(active[-1]) + 1) * frame_size + padding,
    )
    return working[start:end].copy()


def equal_power_fades(length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return fade-out/fade-in gains whose squared sum is exactly one."""

    if length <= 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    if length == 1:
        midpoint = math.pi / 4.0
        return (
            np.array([math.cos(midpoint)], dtype=np.float64),
            np.array([math.sin(midpoint)], dtype=np.float64),
        )
    phase = np.linspace(
        0.0, math.pi / 2.0, length, endpoint=True, dtype=np.float64
    )
    return np.cos(phase), np.sin(phase)


def quantize_pcm16(audio: np.ndarray) -> np.ndarray:
    """Convert float samples to the exact values represented by PCM16."""

    clipped = np.clip(audio, -1.0, 1.0)
    integers = np.clip(
        np.round(clipped * 32768.0), -32768.0, 32767.0
    ).astype(np.int16)
    return integers.astype(np.float64) / 32768.0

