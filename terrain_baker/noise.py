"""Self-contained, bit-reproducible 2D value noise and fBm.

No global PRNG state is used: every lattice random value is derived from
the integer lattice coordinates and the seed through a splitmix64-style
integer hash. Repeated runs with identical parameters therefore produce
identical floating point values on the same platform/NumPy build.
"""

from __future__ import annotations

import numpy as np

_U64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_SPLIT_A = np.uint64(0x9E3779B97F4A7C15)
_SPLIT_B = np.uint64(0xC2B2AE3D27D4EB4F)
_SPLIT_C = np.uint64(0xBF58476D1CE4E5B9)
_SPLIT_D = np.uint64(0x94D049BB133111EB)


def _hash64(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Return a deterministic uint64 hash for integer lattice points."""
    xi = x.astype(np.uint64, copy=False)
    yi = y.astype(np.uint64, copy=False)
    z = (np.uint64(seed) + xi * _SPLIT_A + yi * _SPLIT_B) & _U64_MASK
    z = ((z ^ (z >> np.uint64(30))) * _SPLIT_C) & _U64_MASK
    z = ((z ^ (z >> np.uint64(27))) * _SPLIT_D) & _U64_MASK
    return z ^ (z >> np.uint64(31))


def _lattice_random(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Uniform random value in [-1, 1] for integer lattice points."""
    h = _hash64(x, y, seed)
    unit = (h >> np.uint64(32)).astype(np.float64) / np.float64(4294967296.0)
    return unit * np.float64(2.0) * unit - np.float64(1.0)


def _quintic(t: np.ndarray) -> np.ndarray:
    """Quintic smoothstep 6t^5 - 15t^4 + 10t^3 with zero edge derivatives."""
    return t * t * t * (t * (t * np.float64(6.0) - np.float64(15.0)) + np.float64(10.0))


def value_noise_2d(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Smoothly interpolated value noise at float coordinates x/y."""
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    xf = x - xi.astype(np.float64)
    yf = y - yi.astype(np.float64)
    u = _quintic(xf)
    v = _quintic(yf)

    h00 = _lattice_random(xi, yi, seed)
    h10 = _lattice_random(xi + np.int64(1), yi, seed)
    h01 = _lattice_random(xi, yi + np.int64(1), seed)
    h11 = _lattice_random(xi + np.int64(1), yi + np.int64(1), seed)

    top = h00 + (h10 - h00) * u
    bottom = h01 + (h11 - h01) * u
    return top + (bottom - top) * v


def fbm_2d(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    octaves: int = 5,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> np.ndarray:
    """Fractal Brownian motion built from octaves of value noise.

    Output is normalized to roughly the [-1, 1] range.
    """
    if octaves < 1:
        raise ValueError("octaves must be >= 1")
    total = np.zeros_like(x, dtype=np.float64)
    amplitude = np.float64(1.0)
    frequency = np.float64(1.0)
    norm = np.float64(0.0)
    for octave in range(octaves):
        total += amplitude * value_noise_2d(
            x * frequency, y * frequency, seed + octave * 1013
        )
        norm += amplitude
        amplitude *= np.float64(gain)
        frequency *= np.float64(lacunarity)
    return total / norm
