"""Deterministic PCM WAV reader/writer using only the standard library + NumPy.

Samples are exchanged as float64 arrays in [-1.0, 1.0], shaped
``(frames, channels)``. Supported on-disk formats: 16/24/32-bit signed PCM,
mono or stereo. Anything else raises E_UNSUPPORTED_FORMAT.
"""

from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path

import numpy as np

from .errors import ErrorCode, MixerError


def _pcm_max(sample_width: int) -> int:
    return 1 << (8 * sample_width - 1)


def read_wav(path: str | Path) -> tuple[np.ndarray, int, int]:
    """Return ``(samples float64 [frames, channels], sample_rate, sample_width)``."""
    wav_path = Path(path)
    if not wav_path.is_file():
        raise MixerError(
            ErrorCode.MISSING_ASSET,
            f"audio asset not found: {wav_path}",
            {"path": str(wav_path)},
        )
    try:
        with wave.open(str(wav_path), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
            raw = wf.readframes(frames)
    except wave.Error as exc:
        raise MixerError(
            ErrorCode.UNSUPPORTED_FORMAT,
            f"unsupported or corrupt WAV asset {wav_path}: {exc}",
            {"path": str(wav_path)},
        ) from exc

    if channels not in (1, 2):
        raise MixerError(
            ErrorCode.UNSUPPORTED_FORMAT,
            f"asset {wav_path.name}: {channels} channels are not supported (mono/stereo only)",
            {"path": str(wav_path), "channels": channels},
        )
    if sample_width not in (2, 3, 4):
        raise MixerError(
            ErrorCode.UNSUPPORTED_FORMAT,
            f"asset {wav_path.name}: {sample_width * 8}-bit PCM is not supported",
            {"path": str(wav_path), "sample_width_bytes": sample_width},
        )

    dtype = {2: "<i2", 3: None, 4: "<i4"}[sample_width]
    if sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        ints = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
    else:
        ints = np.frombuffer(raw, dtype=np.dtype(dtype)).astype(np.int32)

    samples = ints.astype(np.float64) / _pcm_max(sample_width)
    samples = samples.reshape(-1, channels)
    return samples, sample_rate, sample_width


def quantize_pcm(samples: np.ndarray, sample_width: int) -> np.ndarray:
    """Clip to [-1, 1) and quantise float samples to signed integer PCM."""
    peak_max = _pcm_max(sample_width)
    clipped = np.clip(samples, -1.0, 1.0 - 1.0 / peak_max)
    scaled = np.round(clipped * peak_max)
    return scaled.astype(np.int64)


def _encode_pcm(ints: np.ndarray, sample_width: int) -> bytes:
    flat = ints.reshape(-1)
    if sample_width == 2:
        return flat.astype("<i2").tobytes()
    if sample_width == 4:
        return flat.astype("<i4").tobytes()
    masked = (flat & 0xFFFFFF).astype(np.uint32)
    packed = np.empty((masked.size, 3), dtype=np.uint8)
    packed[:, 0] = masked & 0xFF
    packed[:, 1] = (masked >> 8) & 0xFF
    packed[:, 2] = (masked >> 16) & 0xFF
    return packed.tobytes()


def write_wav_atomic(
    path: str | Path, samples: np.ndarray, sample_rate: int, sample_width: int
) -> None:
    """Write a WAV via a temporary file + rename so failures leave no partial WAV."""
    if sample_width not in (2, 3, 4):
        raise MixerError(
            ErrorCode.UNSUPPORTED_FORMAT,
            f"cannot write {sample_width * 8}-bit PCM",
            {"sample_width_bytes": sample_width},
        )
    if samples.ndim == 1:
        samples = samples[:, None]
    channels = samples.shape[1]
    if channels not in (1, 2):
        raise MixerError(
            ErrorCode.UNSUPPORTED_FORMAT,
            f"cannot write {channels}-channel WAV (mono/stereo only)",
            {"channels": channels},
        )

    wav_path = Path(path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    ints = quantize_pcm(samples, sample_width)
    payload = _encode_pcm(ints, sample_width)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{wav_path.name}.", suffix=".tmp", dir=str(wav_path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            with wave.open(handle, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(payload)
        os.replace(tmp_name, wav_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def remove_if_exists(path: str | Path) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
