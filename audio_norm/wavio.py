"""Minimal PCM WAV reader/writer supporting 16-bit and 24-bit samples."""
from __future__ import annotations

import wave
from dataclasses import dataclass

import numpy as np

SUPPORTED_SAMPLE_RATES = (44100, 48000)
SUPPORTED_CHANNEL_COUNTS = (1, 2, 6)
SUPPORTED_SAMPLE_WIDTHS = (2, 3)  # 16-bit, 24-bit inputs/outputs
INTERNAL_SAMPLE_WIDTH = 4  # 32-bit float intermediates produced by ffmpeg
TARGET_SAMPLE_RATE = 48000
TARGET_SAMPLE_WIDTH = 3


class WavFormatError(Exception):
    """Raised when a WAV file is corrupt or uses an unsupported format."""


@dataclass
class WavData:
    samples: np.ndarray  # shape (n_samples, n_channels), float64 in [-1, 1]
    sample_rate: int
    sample_width: int
    n_channels: int

    @property
    def duration_seconds(self) -> float:
        return self.samples.shape[0] / float(self.sample_rate)


def _decode_int(raw: np.ndarray, width: int) -> np.ndarray:
    if width == 2:
        value = raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8)
        value = np.where(value & 0x8000, value - 0x10000, value)
        return value.astype(np.float64) / 32768.0
    if width == 3:
        # Little-endian 24-bit signed integers packed as uint8 triples.
        byte0 = raw[:, 0].astype(np.int32)
        byte1 = raw[:, 1].astype(np.int32)
        byte2 = raw[:, 2].astype(np.int32)
        value = byte0 | (byte1 << 8) | (byte2 << 16)
        value = np.where(value & 0x800000, value - 0x1000000, value)
        return value.astype(np.float64) / 8388608.0
    raise WavFormatError(f"unsupported sample width: {width * 8}-bit")


def _decode_float32(raw_bytes: bytes, n_frames: int, n_channels: int) -> np.ndarray:
    decoded = np.frombuffer(raw_bytes, dtype="<f4").astype(np.float64)
    return decoded.reshape(n_frames, n_channels)


def _encode_int24(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    value = np.round(clipped * 8388607.0).astype(np.int32)
    value = np.where(value < 0, value + 0x1000000, value)
    packed = np.empty((value.size, 3), dtype=np.uint8)
    packed[:, 0] = value & 0xFF
    packed[:, 1] = (value >> 8) & 0xFF
    packed[:, 2] = (value >> 16) & 0xFF
    return packed.tobytes()


def read_wav(path: str) -> WavData:
    try:
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            comp = wf.getcomptype()
            if comp != "NONE":
                raise WavFormatError(f"compressed WAV not supported: {comp}")
            raw_bytes = wf.readframes(n_frames)
    except (wave.Error, EOFError, OSError) as exc:
        raise WavFormatError(f"cannot read WAV {path}: {exc}") from exc

    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise WavFormatError(f"unsupported sample rate: {sample_rate} Hz")
    if n_channels not in SUPPORTED_CHANNEL_COUNTS:
        raise WavFormatError(f"unsupported channel count: {n_channels}")
    if sample_width not in SUPPORTED_SAMPLE_WIDTHS + (INTERNAL_SAMPLE_WIDTH,):
        raise WavFormatError(f"unsupported bit depth: {sample_width * 8}-bit")

    expected = n_frames * sample_width * n_channels
    if len(raw_bytes) != expected:
        raise WavFormatError(
            f"truncated WAV data: expected {expected} bytes, got {len(raw_bytes)}"
        )
    if sample_width == INTERNAL_SAMPLE_WIDTH:
        samples = _decode_float32(raw_bytes, n_frames, n_channels)
    else:
        raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, sample_width)
        decoded = _decode_int(raw, sample_width)
        samples = decoded.reshape(n_frames, n_channels)
    return WavData(samples, sample_rate, sample_width, n_channels)


def write_wav_24bit(path: str, samples: np.ndarray, sample_rate: int) -> None:
    """Write 24-bit PCM WAV. `samples` has shape (n, channels) in [-1, 1]."""
    n_channels = samples.shape[1]
    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(TARGET_SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(_encode_int24(samples.reshape(-1)))
