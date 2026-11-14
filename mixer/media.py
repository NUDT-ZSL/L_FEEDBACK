"""Audio loading, downmixing, and deterministic linear resampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import SUPPORTED_SAMPLE_RATES
from .errors import ErrorCode, MixerError


@dataclass(frozen=True)
class MonoAudio:
    samples: np.ndarray
    sample_rate: int


class MediaLoader:
    def __init__(self, target_sample_rate: int) -> None:
        self.target_sample_rate = target_sample_rate
        self._cache: dict[Path, MonoAudio] = {}

    def load(self, path: Path) -> MonoAudio:
        if path not in self._cache:
            self._cache[path] = self._load_and_resample(path)
        return self._cache[path]

    def _load_and_resample(self, path: Path) -> MonoAudio:
        try:
            info = sf.info(str(path))
            if info.frames <= 0 or info.channels <= 0:
                raise ValueError("audio file contains no samples")
            if info.samplerate not in SUPPORTED_SAMPLE_RATES:
                raise ValueError(
                    f"unsupported input sample rate {info.samplerate}; "
                    "expected 44100 or 48000"
                )
            data, source_rate = sf.read(
                str(path),
                dtype="float64",
                always_2d=True,
            )
        except Exception as exc:
            raise MixerError(
                ErrorCode.E_MEDIA_LOAD,
                f"cannot load media {path}: {exc}",
            ) from exc

        if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
            raise MixerError(
                ErrorCode.E_MEDIA_LOAD, f"media {path} has no usable audio samples"
            )
        mono = np.mean(data, axis=1, dtype=np.float64)
        if not np.all(np.isfinite(mono)):
            raise MixerError(
                ErrorCode.E_MEDIA_LOAD, f"media {path} contains non-finite samples"
            )

        if source_rate != self.target_sample_rate:
            mono = linear_resample(mono, source_rate, self.target_sample_rate)
        return MonoAudio(samples=mono, sample_rate=self.target_sample_rate)


def linear_resample(
    samples: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    """Resample a mono signal with linear interpolation.

    The frame count uses deterministic half-up rounding. Time positions are
    computed directly from source and target rates to avoid accumulating
    rational-rate drift.
    """

    signal = np.asarray(samples, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("linear_resample expects a mono signal")
    if signal.size == 0:
        return signal.copy()
    if source_rate == target_rate:
        return signal.copy()
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")

    numerator = signal.size * target_rate
    output_size = (numerator + source_rate // 2) // source_rate
    output_size = max(1, output_size)
    source_positions = (
        np.arange(output_size, dtype=np.float64) * float(source_rate) / float(target_rate)
    )
    return np.interp(
        source_positions,
        np.arange(signal.size, dtype=np.float64),
        signal,
    )
