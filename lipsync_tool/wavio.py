"""PCM WAV IO, resampling and channel handling.

Only the Python standard library (``wave``) and NumPy are required.  SciPy
is used for high quality polyphase resampling when present, otherwise a
linear-interpolation resampler keeps the tool fully functional.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

try:  # scipy is an optional dependency
    from scipy.signal import resample_poly as _scipy_resample_poly
except Exception:  # pragma: no cover - exercised only without scipy
    _scipy_resample_poly = None


@dataclass
class WavData:
    """In-memory audio container.

    ``samples`` is float64 in the range [-1.0, 1.0], shape (frames, channels).
    """

    samples: np.ndarray
    sample_rate: int
    sample_width: int

    @property
    def channels(self) -> int:
        return 1 if self.samples.ndim == 1 else self.samples.shape[1]

    @property
    def duration(self) -> float:
        frames = self.samples.shape[0]
        return frames / float(self.sample_rate)

    def mono(self) -> np.ndarray:
        """Return a mono (1-D) float array, averaging channels if needed."""
        if self.samples.ndim == 1:
            return self.samples.astype(np.float64, copy=True)
        return self.samples.mean(axis=1).astype(np.float64)


def _decode_pcm(raw: bytes, width: int, channels: int) -> np.ndarray:
    if width == 1:  # 8-bit PCM is unsigned
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        data = (data - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    elif width == 3:
        # 24-bit little-endian PCM, unpack into signed 32-bit integers
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        as32 = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        as32 = np.where(as32 & 0x800000, as32 - 0x1000000, as32)
        data = as32.astype(np.float64) / 8388608.0
    else:
        raise ValueError(f"unsupported PCM sample width: {width} bytes")
    return data.reshape(-1, channels) if channels > 1 else data


def _encode_pcm(samples: np.ndarray, width: int) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    if width == 1:
        quantized = np.round(clipped * 128.0 + 128.0).astype(np.uint8)
        return quantized.tobytes()
    if width == 2:
        quantized = np.round(clipped * 32767.0).astype("<i2")
        return quantized.tobytes()
    if width == 4:
        quantized = np.round(clipped * 2147483647.0).astype("<i4")
        return quantized.tobytes()
    if width == 3:
        quantized = np.round(clipped * 8388607.0).astype(np.int32)
        packed = np.empty((quantized.shape[0], 3), dtype=np.uint8)
        packed[:, 0] = quantized & 0xFF
        packed[:, 1] = (quantized >> 8) & 0xFF
        packed[:, 2] = (quantized >> 16) & 0xFF
        return packed.reshape(-1).tobytes()
    raise ValueError(f"unsupported PCM sample width: {width} bytes")


def read_wav(path: str | Path) -> WavData:
    """Read a PCM WAV file into a :class:`WavData` instance."""
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)
        if wav.getcomptype() != "NONE":
            raise ValueError(f"only uncompressed PCM WAV is supported ({path})")
    samples = _decode_pcm(raw, sample_width, channels)
    return WavData(samples=samples, sample_rate=sample_rate, sample_width=sample_width)


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int, sample_width: int = 2) -> None:
    """Write a mono (1-D) or multi-channel (frames x channels) float PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(samples, dtype=np.float64)
    channels = 1 if array.ndim == 1 else array.shape[1]
    payload = _encode_pcm(array.reshape(-1), sample_width)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(payload)


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Bandwidth-unaware linear interpolation resampler (fallback)."""
    if source_rate == target_rate or samples.size == 0:
        return samples.astype(np.float64, copy=True)
    duration = samples.shape[0] / float(source_rate)
    out_frames = max(1, int(round(duration * target_rate)))
    source_positions = np.arange(out_frames, dtype=np.float64) * (source_rate / float(target_rate))
    index_lo = np.floor(source_positions).astype(np.int64)
    index_lo = np.minimum(index_lo, samples.shape[0] - 1)
    index_hi = np.minimum(index_lo + 1, samples.shape[0] - 1)
    fraction = source_positions - index_lo
    return samples[index_lo] * (1.0 - fraction) + samples[index_hi] * fraction


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample to ``target_rate``.

    Uses SciPy polyphase filtering when available, otherwise linear
    interpolation.  Output duration matches the input duration.
    """
    if source_rate == target_rate or samples.size == 0:
        return samples.astype(np.float64, copy=True)
    if _scipy_resample_poly is not None:
        gcd = math.gcd(int(source_rate), int(target_rate))
        up = int(target_rate) // gcd
        down = int(source_rate) // gcd
        return _scipy_resample_poly(samples, up, down).astype(np.float64)
    return resample_linear(samples, source_rate, target_rate)


def normalize_to_mono(
    wav: WavData, target_rate: int
) -> Tuple[np.ndarray, int, list[str]]:
    """Resample to the declared target rate and downmix to mono.

    Returns the mono float signal, its sample rate and human-readable notes
    describing every normalization step performed.
    """
    notes: list[str] = []
    signal = wav.samples
    if wav.channels > 1:
        notes.append(f"downmixed {wav.channels} channels to mono")
        signal = signal.mean(axis=1)
    else:
        signal = signal.reshape(-1)
    if wav.sample_rate != target_rate:
        notes.append(
            f"resampled {wav.sample_rate} Hz -> {target_rate} Hz "
            f"(backend: {'scipy' if _scipy_resample_poly is not None else 'linear'})"
        )
        signal = resample(signal, wav.sample_rate, target_rate)
    return signal.astype(np.float64), target_rate, notes
