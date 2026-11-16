"""Strict PCM WAV I/O used by the command-line pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import wave

import numpy as np

from .errors import BadFormatError

SAMPLE_RATE = 44_100
BITS_PER_SAMPLE = 16


@dataclass(frozen=True)
class WavData:
    samples: np.ndarray
    sample_rate: int
    channels: int
    source_path: Path

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])


def read_pcm_wav(path: str | os.PathLike[str]) -> WavData:
    """Read a 44.1 kHz, 16-bit, mono/stereo integer PCM WAV file."""

    source = Path(path)
    try:
        with wave.open(str(source), "rb") as handle:
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            bits_per_sample = handle.getsampwidth() * 8
            compression = handle.getcomptype()
            frame_count = handle.getnframes()
            raw = handle.readframes(frame_count)
    except (OSError, EOFError, ValueError, wave.Error) as exc:
        raise BadFormatError(f"cannot read WAV header/data from {source}: {exc}") from exc

    if compression != "NONE":
        raise BadFormatError(f"{source}: compressed WAV is not supported")
    if sample_rate != SAMPLE_RATE:
        raise BadFormatError(
            f"{source}: sample rate must be {SAMPLE_RATE} Hz, got {sample_rate} Hz"
        )
    if bits_per_sample != BITS_PER_SAMPLE:
        raise BadFormatError(
            f"{source}: bit depth must be {BITS_PER_SAMPLE}-bit, "
            f"got {bits_per_sample}-bit"
        )
    if channels not in (1, 2):
        raise BadFormatError(
            f"{source}: channel count must be mono or stereo, got {channels}"
        )
    if frame_count <= 0:
        raise BadFormatError(f"{source}: WAV contains no audio frames")

    expected_bytes = frame_count * channels * (BITS_PER_SAMPLE // 8)
    if len(raw) != expected_bytes:
        raise BadFormatError(
            f"{source}: truncated PCM data "
            f"(expected {expected_bytes} bytes, got {len(raw)})"
        )

    pcm = np.frombuffer(raw, dtype="<i2")
    if pcm.size != frame_count * channels:
        raise BadFormatError(f"{source}: PCM data is not frame-aligned")

    samples = pcm.reshape(-1, channels).astype(np.float64) / 32768.0
    if not np.all(np.isfinite(samples)):
        raise BadFormatError(f"{source}: PCM data contains non-finite values")

    return WavData(
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
        source_path=source,
    )


def samples_to_pcm16(samples: np.ndarray) -> bytes:
    """Quantize normalized floating-point samples to little-endian PCM16."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] not in (1, 2):
        raise ValueError("samples must have shape (frames, 1) or (frames, 2)")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples contain NaN or infinity")

    quantized = np.rint(np.clip(values, -1.0, 1.0) * 32767.0)
    return quantized.astype("<i2", copy=False).tobytes(order="C")


def write_pcm16_wav(
    path: str | os.PathLike[str],
    samples: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    channels: int | None = None,
) -> None:
    """Write PCM16 WAV data to a destination path.

    This function writes the named file directly. Pipeline callers stage it on
    a temporary path and atomically replace the final destination.
    """

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("samples must be two-dimensional")
    actual_channels = int(values.shape[1])
    if channels is not None and actual_channels != channels:
        raise ValueError("channel count does not match sample shape")
    if actual_channels not in (1, 2):
        raise ValueError("only mono or stereo output is supported")

    destination = Path(path)
    with destination.open("wb") as file_handle:
        with wave.open(file_handle, "wb") as wav_handle:
            wav_handle.setnchannels(actual_channels)
            wav_handle.setsampwidth(BITS_PER_SAMPLE // 8)
            wav_handle.setframerate(sample_rate)
            wav_handle.writeframes(samples_to_pcm16(values))
        file_handle.flush()
        os.fsync(file_handle.fileno())

