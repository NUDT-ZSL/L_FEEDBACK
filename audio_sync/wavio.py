"""Minimal 48 kHz mono 16-bit PCM WAV writer (standard library only)."""

from __future__ import annotations

import wave

from .constants import SAMPLE_RATE


def write_wav(path: str, pcm_bytes: bytearray | bytes) -> None:
    """Write mono 16-bit PCM WAV at 48 kHz."""
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(bytes(pcm_bytes))


def read_wav_duration(path: str) -> float:
    """Return WAV duration in seconds (used by the verifier)."""
    with wave.open(path, "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())
