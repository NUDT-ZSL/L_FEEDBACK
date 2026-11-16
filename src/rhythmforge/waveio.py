"""Deterministic PCM WAV input/output.

Only uncompressed PCM WAV files are supported (sample widths 1, 2, 3 or 4
bytes).  The stdlib :mod:`wave` module handles the container while the
sample conversion is implemented here so that 24-bit PCM - a format wave
cannot read natively - is fully supported.

Arrays are always shaped ``(frames, channels)`` and use ``float32`` in the
internal pipeline.
"""

from __future__ import annotations

import os
import wave

import numpy as np

from .errors import BuildError, ExitCode

SUPPORTED_SAMPLE_WIDTHS = (2, 3, 4)


class WavInfo:
    """Metadata describing a PCM WAV file."""

    __slots__ = ("sample_rate", "channels", "sample_width", "frames", "path")

    def __init__(self, path: str, sample_rate: int, channels: int,
                 sample_width: int, frames: int):
        self.path = path
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.frames = frames

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width_bytes": self.sample_width,
            "frames": self.frames,
            "duration_seconds": self.frames / self.sample_rate,
        }


def probe_wav(path: str) -> WavInfo:
    """Return container metadata without decoding sample data."""
    if not os.path.isfile(path):
        raise BuildError(ExitCode.MISSING_SOURCE, f"source wav not found: {path}",
                         {"path": path})
    try:
        with wave.open(path, "rb") as handle:
            params = handle.getparams()
    except (wave.Error, EOFError, OSError) as exc:
        raise BuildError(ExitCode.UNSUPPORTED_FORMAT,
                         f"not a readable PCM wav container: {path}",
                         {"path": path, "reason": str(exc)}) from exc
    if params.sampwidth not in SUPPORTED_SAMPLE_WIDTHS:
        raise BuildError(
            ExitCode.UNSUPPORTED_FORMAT,
            f"unsupported sample width {params.sampwidth} bytes in {path}; "
            f"expected 16/24/32-bit PCM",
            {"path": path, "sample_width_bytes": params.sampwidth},
        )
    if params.comptype != "NONE":
        raise BuildError(ExitCode.UNSUPPORTED_FORMAT,
                         f"compressed wav is not supported: {path}",
                         {"path": path, "comptype": params.comptype})
    return WavInfo(path, params.framerate, params.nchannels,
                   params.sampwidth, params.nframes)


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.int32)
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.int64)
    # 24-bit little endian: expand each triple into signed 32-bit integers.
    data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    value = data[:, 0] | (data[:, 1] << 8) | (data[:, 2] << 16)
    sign_mask = value >= (1 << 23)
    value = np.where(sign_mask, value - (1 << 24), value)
    return value


def _to_float(samples: np.ndarray, sample_width: int) -> np.ndarray:
    max_abs = float(1 << (8 * sample_width - 1))
    return (samples.astype(np.float64) / max_abs).astype(np.float32)


def read_wav(path: str) -> tuple[np.ndarray, WavInfo]:
    """Decode a PCM WAV into a ``(frames, channels)`` float32 array."""
    info = probe_wav(path)
    with wave.open(path, "rb") as handle:
        raw = handle.readframes(info.frames)
    flat = _to_float(_decode_pcm(raw, info.sample_width), info.sample_width)
    data = flat.reshape(info.frames, info.channels)
    return np.ascontiguousarray(data), info


def _encode_pcm(data: np.ndarray, sample_width: int) -> bytes:
    """Clip float samples to full scale and encode as little-endian PCM."""
    clipped = np.clip(data, -1.0, 1.0).reshape(-1)
    if sample_width == 2:
        scale = float(1 << 15) - 1
        quantized = np.rint(clipped * scale).astype("<i2")
        return quantized.tobytes()
    if sample_width == 4:
        scale = float(1 << 31) - 1
        quantized = np.rint(clipped * scale).astype("<i4")
        return quantized.tobytes()
    # 24-bit: quantize then emit three bytes per sample, little endian.
    scale = float(1 << 23) - 1
    quantized = np.rint(clipped * scale).astype(np.int32)
    packed = np.empty((quantized.size, 3), dtype=np.uint8)
    unsigned = quantized.astype(np.uint32)
    packed[:, 0] = unsigned & 0xFF
    packed[:, 1] = (unsigned >> 8) & 0xFF
    packed[:, 2] = (unsigned >> 16) & 0xFF
    return packed.tobytes()


def write_wav(path: str, data: np.ndarray, sample_rate: int,
              sample_width: int = 3) -> None:
    """Write a ``(frames, channels)`` float array as deterministic PCM WAV."""
    if sample_width not in SUPPORTED_SAMPLE_WIDTHS:
        raise BuildError(ExitCode.UNSUPPORTED_FORMAT,
                         f"cannot write sample width {sample_width} bytes",
                         {"sample_width_bytes": sample_width})
    if data.ndim != 2 or data.shape[1] not in (1, 2):
        raise BuildError(ExitCode.UNSUPPORTED_FORMAT,
                         "wav export expects a (frames, 1|2) array",
                         {"shape": list(data.shape)})
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    payload = _encode_pcm(np.asarray(data, dtype=np.float32), sample_width)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(data.shape[1])
        handle.setsampwidth(sample_width)
        handle.setframerate(int(sample_rate))
        handle.writeframes(payload)
