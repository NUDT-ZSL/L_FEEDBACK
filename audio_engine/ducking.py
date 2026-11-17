"""BGM ducking envelope driven by skill SFX playback intervals."""

from __future__ import annotations

import numpy as np


def build_duck_envelope(
    n_samples: int,
    sample_rate: int,
    intervals: list[tuple[float, float]],
    duck_gain_db: float = -6.0,
    attack_s: float = 0.020,
    release_s: float = 0.100,
) -> np.ndarray:
    """Build a per-sample BGM gain envelope.

    For each SFX interval [start, end]: the gain ramps linearly from 1.0
    down to the duck gain over ``attack_s``, holds during the interval,
    and recovers linearly over ``release_s``. Overlapping intervals take
    the minimum gain. Ramps start/end exactly on existing curve values,
    so the envelope has no single-sample discontinuities.
    """
    env = np.ones(n_samples, dtype=np.float64)
    duck = 10.0 ** (duck_gain_db / 20.0)
    attack_n = max(1, int(round(attack_s * sample_rate)))
    release_n = max(1, int(round(release_s * sample_rate)))

    for start_s, end_s in intervals:
        s0 = int(round(start_s * sample_rate))
        s1 = int(round(end_s * sample_rate))
        if s1 <= s0:
            s1 = s0 + 1
        lo = max(0, s0)
        hi = min(n_samples, s1 + release_n)
        if hi <= lo:
            continue
        curve = np.ones(hi - lo, dtype=np.float64)

        a_lo = max(s0, lo) - lo
        a_hi = min(s0 + attack_n, hi) - lo
        if a_hi > a_lo:
            count = a_hi - a_lo
            ramp = np.linspace(1.0, duck, count + 1)
            curve[a_lo:a_hi] = ramp[:count]

        h_lo = max(s0 + attack_n, lo) - lo
        h_hi = min(s1, hi) - lo
        if h_hi > h_lo:
            curve[h_lo:h_hi] = duck

        r_lo = max(s1, lo) - lo
        r_hi = min(s1 + release_n, hi) - lo
        if r_hi > r_lo:
            count = r_hi - r_lo
            ramp = np.linspace(duck, 1.0, count + 1)
            curve[r_lo:r_hi] = np.minimum(curve[r_lo:r_hi], ramp[:count])

        env[lo:hi] = np.minimum(env[lo:hi], curve)
    return env


def envelope_max_slope_db_per_s(env: np.ndarray, sample_rate: int) -> float:
    """Maximum absolute slope of the envelope measured in dB per second."""
    floor = 1e-6
    db = 20.0 * np.log10(np.maximum(env, floor))
    if db.size < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(db))) * sample_rate)
