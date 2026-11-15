"""Minimal WAV I/O: 16-bit read, 16/24-bit PCM write (little-endian)."""

from __future__ import annotations

import wave

import numpy as np


def write_wav16(path, data: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(data, -1.0, 1.0 - 2 ** -15)
    ints = np.round(pcm * (2 ** 15)).astype("<i2")
    channels = 1 if ints.ndim == 1 else ints.shape[1]
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(ints.reshape(-1).tobytes())


def write_wav24(path, data: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(data, -1.0, 1.0 - 2 ** -23)
    ints = np.round(pcm * (2 ** 23)).astype("<i4")
    channels = 1 if ints.ndim == 1 else ints.shape[1]
    raw = np.frombuffer(ints.reshape(-1).tobytes(), dtype=np.uint8).reshape(-1, 4)[:, :3]
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(3)
        wf.setframerate(sample_rate)
        wf.writeframes(raw.tobytes())


def read_wav(path):
    """Return (float64 data in [-1,1], sample_rate, channels, sampwidth)."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if width == 2:
        ints = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 2 ** 15
    elif width == 3:
        b = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        padded = np.zeros((b.shape[0], 4), dtype=np.uint8)
        padded[:, :3] = b
        padded[b[:, 2] >= 0x80, 3] = 0xFF  # sign extend
        ints = padded.view("<i4").reshape(-1).astype(np.float64) / 2 ** 23
    else:
        raise ValueError(f"unsupported sample width: {width}")
    if channels > 1:
        ints = ints.reshape(-1, channels)
    return ints, rate, channels, width
