"""Small deterministic WAV helpers built on the standard-library wave module."""

from __future__ import annotations

import hashlib
import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class WavReadError(ValueError):
    pass


@dataclass(frozen=True)
class WavInfo:
    sample_rate_hz: int
    channels: int
    frames: int
    sample_width_bytes: int
    compression: str


def inspect_wav(path: str | Path) -> WavInfo:
    try:
        with wave.open(str(path), "rb") as wav:
            return WavInfo(
                sample_rate_hz=wav.getframerate(),
                channels=wav.getnchannels(),
                frames=wav.getnframes(),
                sample_width_bytes=wav.getsampwidth(),
                compression=wav.getcomptype(),
            )
    except (wave.Error, EOFError, OSError) as exc:
        raise WavReadError(f"cannot inspect WAV: {exc}") from exc


def read_wav_mono_float64(path: str | Path) -> tuple[np.ndarray, WavInfo]:
    info = inspect_wav(path)
    if info.compression != "NONE":
        raise WavReadError(f"unsupported WAV compression: {info.compression}")
    if info.channels != 1:
        raise WavReadError("only mono WAV files are supported")
    if info.frames < 0:
        raise WavReadError("negative WAV frame count")
    try:
        with wave.open(str(path), "rb") as wav:
            raw = wav.readframes(info.frames)
    except (wave.Error, EOFError, OSError) as exc:
        raise WavReadError(f"cannot read WAV: {exc}") from exc
    expected_bytes = info.frames * info.channels * info.sample_width_bytes
    if len(raw) != expected_bytes:
        raise WavReadError(
            f"WAV payload is truncated: {len(raw)} bytes, expected {expected_bytes}"
        )

    width = info.sample_width_bytes
    if width == 1:
        values = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        samples = (values - 128.0) / 128.0
    elif width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float64)
        samples = values / 32768.0
    elif width == 3:
        values = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        signed = values[:, 0] | (values[:, 1] << 8) | (values[:, 2] << 16)
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        samples = signed.astype(np.float64) / 8388608.0
    elif width == 4:
        values = np.frombuffer(raw, dtype="<i4").astype(np.float64)
        samples = values / 2147483648.0
    else:
        raise WavReadError(f"unsupported PCM sample width: {width} bytes")
    return samples, info


def floats_to_pcm16(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    scaled = np.rint(clipped * 32767.0)
    return np.clip(scaled, -32768.0, 32767.0).astype("<i2")


def write_wav_pcm16(
    path: str | Path,
    samples: np.ndarray,
    sample_rate_hz: int = 48_000,
    channels: int = 1,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if channels != 1:
        raise ValueError("replay output must be mono")
    pcm = floats_to_pcm16(samples)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with wave.open(str(temp_path), "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate_hz)
            wav.writeframes(pcm.tobytes(order="C"))
        os.replace(temp_path, output)
    except BaseException:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise
    return output


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
