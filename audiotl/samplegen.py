"""Deterministic local WAV sample generation (no external assets).

Samples are pure analytic tones (fundamental + second harmonic), so they
are byte-identical across runs and machines. Used by tests and by the
tools/make_samples.py helper. FFmpeg is not required.
"""

import math

import numpy as np

from .wavio import write_wav16


def tone_data(rate, channels, seconds, base_freq, amplitude=0.6):
    frames = int(round(rate * seconds))
    t = np.arange(frames, dtype=np.float64) / float(rate)
    cols = []
    for ch in range(channels):
        freq = base_freq * (1.0 + 0.5 * ch)
        wave = amplitude * np.sin(2.0 * math.pi * freq * t)
        wave = wave + 0.3 * amplitude * np.sin(2.0 * math.pi * 2.0 * freq * t)
        cols.append(wave)
    data = np.stack(cols, axis=1)
    return data.astype(np.float32)


def write_tone(path, rate, channels, seconds, base_freq, amplitude=0.6):
    write_wav16(path, tone_data(rate, channels, seconds, base_freq, amplitude), rate)
    return path
