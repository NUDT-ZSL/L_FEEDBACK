from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int


def _to_float(samples: np.ndarray) -> np.ndarray:
    if samples.dtype == np.int16:
        return samples.astype(np.float64) / 32768.0
    if samples.dtype == np.int32:
        return samples.astype(np.float64) / 2147483648.0
    if samples.dtype == np.uint8:
        return (samples.astype(np.float64) - 128.0) / 128.0
    if np.issubdtype(samples.dtype, np.floating):
        return samples.astype(np.float64)
    raise ValueError(f"unsupported WAV sample dtype: {samples.dtype}")


def load_wav(path: str | Path) -> AudioData:
    sample_rate, raw = wavfile.read(str(path))
    if sample_rate <= 0:
        raise ValueError(f"invalid sample rate: {sample_rate}")
    samples = _to_float(np.asarray(raw))
    if samples.ndim == 1:
        mono = samples
    elif samples.ndim == 2:
        if samples.shape[1] < 1:
            raise ValueError("WAV contains no channels")
        mono = samples.mean(axis=1, dtype=np.float64)
    else:
        raise ValueError(f"unsupported WAV shape: {samples.shape}")
    mono = np.ascontiguousarray(mono, dtype=np.float64)
    if mono.size == 0:
        raise ValueError("WAV contains no audio frames")
    if not np.all(np.isfinite(mono)):
        raise ValueError("WAV contains NaN or infinite samples")
    return AudioData(samples=mono, sample_rate=int(sample_rate))


def write_float_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    array = np.ascontiguousarray(samples, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError("float WAV writer expects mono samples")
    if not np.all(np.isfinite(array)):
        raise ValueError("refusing to write non-finite samples")
    wavfile.write(str(path), int(sample_rate), array)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
