"""Deterministic WAV synthesis driven by event/source identifiers.

Same (event_id, source, duration) triple always yields bit-identical
44.1 kHz / 16-bit / mono PCM.
"""

from __future__ import annotations

import hashlib

import numpy as np

SYNTH_SAMPLE_RATE = 44100


def synth_event(event_id: str, source: str, duration: float,
                sample_rate: int = SYNTH_SAMPLE_RATE) -> np.ndarray:
    """Generate a deterministic mono signal for one timeline event."""
    digest = hashlib.sha256(f"{event_id}|{source}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = np.random.default_rng(seed)

    n = int(round(duration * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    base_freq = 160.0 + (seed % 640)
    amplitude = 0.22 + 0.2 * (digest[8] / 255.0)

    signal = np.zeros(n, dtype=np.float64)
    n_harmonics = 2 + digest[9] % 3
    for h in range(1, n_harmonics + 1):
        harm_amp = amplitude / h
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        vibrato = 1.0 + 0.002 * np.sin(2.0 * np.pi * (3.0 + h) * t)
        signal += harm_amp * np.sin(2.0 * np.pi * base_freq * h * t * vibrato + phase)

    # Deterministic attack/release envelope so onsets are measurable.
    attack = max(1, int(0.010 * sample_rate))
    release = max(1, int(0.050 * sample_rate))
    env = np.ones(n, dtype=np.float64)
    env[:attack] = np.linspace(0.0, 1.0, attack)
    env[-release:] = np.linspace(1.0, 0.0, release)
    return signal * env
