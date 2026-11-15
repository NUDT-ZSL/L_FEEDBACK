"""Deterministic 24-bit PCM WAV output."""

from __future__ import annotations

import wave

import numpy as np

from .config import CHANNELS, SAMPLE_RATE

_MAX24 = (1 << 23) - 1  # 8_388_607


def _to_int24(audio: np.ndarray) -> bytes:
    """Clip, round-to-nearest-even and pack float samples into signed 24-bit."""
    clipped = np.clip(audio, -1.0, 1.0)
    scaled = clipped * _MAX24
    ints = np.rint(scaled).astype(np.int32)  # banker's rounding, no dither
    unsigned = ints.reshape(-1).view(np.uint32) & 0x00FFFFFF
    # Little-endian: byte 0 is the least significant byte.  On a
    # little-endian host the uint32 byte view is [b0, b1, b2, 0]; keep b0..b2.
    bytes_view = unsigned[:, None].view(np.uint8).reshape(-1, 4)[:, :3]
    return np.ascontiguousarray(bytes_view).tobytes()


def write_wav24(path: str, audio: np.ndarray) -> None:
    """Write a ``(frames, 2)`` float array as a 48 kHz / 24-bit stereo WAV."""
    if audio.ndim != 2 or audio.shape[1] != CHANNELS:
        raise ValueError(f"期望 ({audio.shape[0]}, {CHANNELS}) 立体声音频")
    payload = _to_int24(np.asarray(audio, dtype=np.float32))
    with wave.open(path, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(3)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(payload)
