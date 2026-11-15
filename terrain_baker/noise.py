"""Deterministic procedural noise.

All noise is a pure function of (seed, integer pixel x, integer pixel y),
so chunked baking with halos produces bit-identical results to a single
full-frame bake. No global RNG state is used anywhere.
"""

from __future__ import annotations

import numpy as np

_U64 = np.uint64


def _k(value: int) -> np.uint64:
    return _U64(value & 0xFFFFFFFFFFFFFFFF)


def lattice_hash(seed: int, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """Hash integer lattice coordinates to floats in [0, 1).

    Uses 64-bit wrapping integer mixing (splitmix-style). The result is
    platform independent because uint64 overflow is well defined.
    """
    x = np.asarray(ix, dtype=np.int64).astype(np.uint64)
    y = np.asarray(iy, dtype=np.int64).astype(np.uint64)
    h = (x * _k(0x9E3779B97F4A7C15)) ^ _k(seed)
    h = h ^ (y * _k(0xBF58476D1CE4E5B9))
    h = (h ^ (h >> _U64(29))) * _k(0x94D049BB133111EB)
    h = h ^ (h >> _U64(31))
    h = (h ^ (h >> _U64(32))) * _k(0xD6E8FEB86659FD93)
    h = h ^ (h >> _U64(32))
    return (h >> _U64(11)).astype(np.float64) / float(1 << 53)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def value_noise(
    seed: int,
    u: np.ndarray,
    v: np.ndarray,
    period: int,
) -> np.ndarray:
    """Tiling value noise sampled at continuous coordinates (u, v).

    ``u``/``v`` are in lattice units; the lattice wraps with the given
    integer ``period`` so the pattern tiles seamlessly.
    """
    period = max(1, int(period))
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    fx = _smoothstep(u - x0)
    fy = _smoothstep(v - y0)
    x1 = x0 + 1
    y1 = y0 + 1
    x0m = x0 % period
    x1m = x1 % period
    y0m = y0 % period
    y1m = y1 % period
    n00 = lattice_hash(seed, x0m, y0m)
    n10 = lattice_hash(seed, x1m, y0m)
    n01 = lattice_hash(seed, x0m, y1m)
    n11 = lattice_hash(seed, x1m, y1m)
    nx0 = n00 + (n10 - n00) * fx
    nx1 = n01 + (n11 - n01) * fx
    return nx0 + (nx1 - nx0) * fy


def fbm(seed: int, u: np.ndarray, v: np.ndarray, period: int,
        octaves: int = 4) -> np.ndarray:
    """Fractal value noise, normalized to roughly [0, 1]."""
    total = np.zeros_like(u, dtype=np.float64)
    amplitude = 1.0
    norm = 0.0
    for octave in range(octaves):
        octave_seed = seed + 0x9E37 * (octave + 1)
        total += amplitude * value_noise(
            octave_seed, u * (1 << octave), v * (1 << octave),
            period * (1 << octave),
        )
        norm += amplitude
        amplitude *= 0.5
    return total / norm


def rotated_uv(
    xs: np.ndarray,
    ys: np.ndarray,
    width: int,
    height: int,
    rotation_deg: float,
) -> tuple:
    """Rotate normalized pixel coordinates around the image center."""
    u = xs.astype(np.float64) / float(width) - 0.5
    v = ys.astype(np.float64) / float(height) - 0.5
    angle = np.deg2rad(rotation_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return u * cos_a - v * sin_a, u * sin_a + v * cos_a
