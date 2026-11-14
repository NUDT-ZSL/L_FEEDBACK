"""Deterministic mono 48 kHz / 24-bit sample generation.

No external audio assets are needed: every track is synthesised from a
fixed-seed NumPy RNG so repeated runs are bit-identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from .wavio import write_wav_24bit

SAMPLE_RATE = 48000


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_strings(seconds: float, seed: int = 101) -> np.ndarray:
    rng = _rng(seed)
    n = int(round(seconds * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    base = 220.0 * (1.0 + 0.01 * np.sin(2.0 * np.pi * 0.13 * t))
    signal = (0.55 * np.sin(2.0 * np.pi * base * t)
              + 0.25 * np.sin(2.0 * np.pi * base * 2.0 * t + 0.7)
              + 0.12 * np.sin(2.0 * np.pi * base * 3.01 * t))
    vibrato = 0.9 + 0.1 * np.sin(2.0 * np.pi * 0.21 * t)
    return signal * vibrato * 0.4


def generate_percussion(seconds: float, seed: int = 202) -> np.ndarray:
    rng = _rng(seed)
    n = int(round(seconds * SAMPLE_RATE))
    signal = np.zeros(n, dtype=np.float64)
    period = int(0.5 * SAMPLE_RATE)  # 120 BPM eighth-note grid
    for onset in range(0, n, period):
        length = min(int(0.18 * SAMPLE_RATE), n - onset)
        if length <= 0:
            break
        tt = np.arange(length) / SAMPLE_RATE
        thump = np.sin(2.0 * np.pi * (90.0 * np.exp(-18.0 * tt)) * tt)
        noise = rng.standard_normal(length) * np.exp(-28.0 * tt)
        hit = (0.8 * thump + 0.5 * noise) * np.exp(-14.0 * tt)
        signal[onset: onset + length] += hit
    return signal * 0.55


def generate_ambience(seconds: float, seed: int = 303) -> np.ndarray:
    rng = _rng(seed)
    n = int(round(seconds * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    white = rng.standard_normal(n)
    # One-pole low-pass for a soft atmospheric bed, applied via a truncated
    # impulse response so long tracks stay on the vectorised NumPy path.
    coeff = 0.985
    tail = int(np.ceil(np.log(1e-6) / np.log(coeff)))
    ir = (1.0 - coeff) * coeff ** np.arange(tail)
    filtered = np.convolve(white, ir, mode="full")[:n]
    pad = (0.35 * np.sin(2.0 * np.pi * 110.0 * t)
           + 0.25 * np.sin(2.0 * np.pi * 164.8 * t + 1.3))
    drift = 0.6 + 0.4 * np.sin(2.0 * np.pi * 0.07 * t)
    return (filtered * 0.5 + pad * 0.5) * drift * 0.35


TRACK_GENERATORS = {
    "strings": generate_strings,
    "percussion": generate_percussion,
    "ambience": generate_ambience,
}


def generate_tracks(directory, track_names: List[str],
                    total_seconds: float) -> Dict[str, str]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    for index, name in enumerate(track_names):
        generator = TRACK_GENERATORS.get(name, generate_ambience)
        samples = generator(total_seconds, seed=1001 + index * 77)
        out_path = directory / f"{name}.wav"
        write_wav_24bit(out_path, samples, SAMPLE_RATE)
        paths[name] = str(out_path)
    return paths
