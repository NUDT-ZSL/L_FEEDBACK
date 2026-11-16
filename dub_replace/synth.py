"""Deterministic PCM audio synthesis for samples and tests (48 kHz / 16-bit)."""

from __future__ import annotations

import numpy as np
import soundfile as sf

SAMPLE_RATE = 48000


def tone(freq_hz: float, duration_s: float, sr: int = SAMPLE_RATE, amp: float = 0.4,
         phase: float = 0.0) -> np.ndarray:
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / sr
    return amp * np.sin(2.0 * np.pi * freq_hz * t + phase)


def speech_like(duration_s: float, base_freq: float, sr: int = SAMPLE_RATE,
                amp: float = 0.4, seed: int = 0) -> np.ndarray:
    """Deterministic voiced-signal surrogate: two harmonics x syllable envelope.

    The envelope modulation (~4 Hz) mimics syllable rhythm while the carrier
    stays perfectly periodic, so pitch preservation is easy to assert.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / sr
    carrier = np.sin(2 * np.pi * base_freq * t) + 0.35 * np.sin(2 * np.pi * base_freq * 2 * t)
    syllable = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, np.pi))
    edge = np.minimum(np.arange(n) / max(1, int(0.01 * sr)), 1.0)
    edge *= np.minimum(np.arange(n)[::-1] / max(1, int(0.01 * sr)), 1.0)
    return amp * carrier * syllable * edge


def write_pcm16(path, samples, sr: int = SAMPLE_RATE) -> None:
    sf.write(str(path), np.clip(np.asarray(samples, dtype=np.float64), -1, 1), sr, subtype="PCM_16")


def write_corrupt_wav(path) -> None:
    """Write a file with a .wav extension whose contents are not valid RIFF."""
    path = str(path)
    with open(path, "wb") as fh:
        fh.write(b"NOT-A-VALID-WAV-FILE" * 128)
