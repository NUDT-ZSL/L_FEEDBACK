from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from .config import AudioSettings


INT16_SCALE = 32768.0
LOUDNESS_FLOOR = -120.0


def generate_source_pcm(config: AudioSettings, seed: int) -> np.ndarray:
    """Generate deterministic, speech-like test audio as int16 samples."""
    rng = np.random.default_rng((seed ^ 0x5EED5EED) & 0xFFFFFFFF)
    total = config.total_samples
    sample_index = np.arange(total, dtype=np.float64)
    seconds = sample_index / config.sample_rate

    vibrato = 0.09 * np.sin(2.0 * np.pi * 0.37 * seconds)
    f0 = config.base_frequency_hz * (1.0 + vibrato)
    phase = 2.0 * np.pi * np.cumsum(f0) / config.sample_rate
    slow_amplitude = 0.88 + 0.12 * np.sin(2.0 * np.pi * 0.21 * seconds + 0.4)
    waveform = (
        np.sin(phase)
        + 0.34 * np.sin(2.0 * phase + 0.2)
        + 0.16 * np.sin(3.0 * phase + 0.7)
        + 0.07 * np.sin(5.0 * phase + 0.1)
    )
    waveform /= 1.45
    noise = rng.normal(0.0, 0.004, total)
    signal = config.amplitude * slow_amplitude * waveform + noise

    fade_samples = min(config.sample_rate // 100, total // 2)
    if fade_samples > 1:
        fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float64)
        signal[:fade_samples] *= fade
        signal[-fade_samples:] *= fade[::-1]

    pcm = np.clip(np.rint(signal * INT16_SCALE), -32768, 32767).astype("<i2")
    return pcm.reshape(config.total_packets, config.frame_samples)


def pcm_bytes(frame: np.ndarray) -> bytes:
    return np.asarray(frame, dtype="<i2").tobytes(order="C")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pcm_square_sum(samples: np.ndarray) -> int:
    values = np.asarray(samples, dtype=np.int64)
    return int(np.sum(values * values, dtype=np.int64))


def loudness_lufs_approx(square_sum: int, sample_count: int) -> float:
    if sample_count <= 0 or square_sum <= 0:
        return LOUDNESS_FLOOR
    mean_square = square_sum / (float(sample_count) * INT16_SCALE * INT16_SCALE)
    if mean_square <= 0.0:
        return LOUDNESS_FLOOR
    value = -0.691 + 10.0 * math.log10(mean_square)
    return round(max(value, LOUDNESS_FLOOR), 3)


def rms_int16(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.asarray(samples, dtype=np.float64) ** 2)))
