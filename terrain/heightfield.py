"""Deterministic procedural heightfield built on a self-contained splitmix64.

All hashing is performed with explicit 64-bit unsigned integer arithmetic
(``numpy.uint64``), so the generated bits are identical on every platform and
Python build.  Sampling and tile grids are evaluated in float64.
"""

from __future__ import annotations

import numpy as np

UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
GOLDEN_GAMMA = np.uint64(0x9E3779B97F4A7C15)
MIX_A = np.uint64(0xBF58476D1CE4E5B9)
MIX_B = np.uint64(0x94D049BB133111EB)
TWO_POW_64 = 18446744073709551616.0


def splitmix64(state: np.uint64 | int) -> np.uint64:
    """One splitmix64 state transition; wrapping 64-bit arithmetic throughout."""
    with np.errstate(over="ignore"):
        x = np.uint64(state) + GOLDEN_GAMMA
        z = x
        z = (z ^ (z >> np.uint64(30))) * MIX_A
        z = (z ^ (z >> np.uint64(27))) * MIX_B
        return z ^ (z >> np.uint64(31))


def _u64(value: int) -> np.uint64:
    return np.uint64(int(value) & int(UINT64_MASK))


def _u64_array(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.int64) & np.int64(UINT64_MASK)).astype(np.uint64)


def _unit_float(value: np.uint64) -> float:
    """Map a uint64 hash to [0.0, 1.0) deterministically."""
    return float(value) / TWO_POW_64


def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


class Heightfield:
    """Fractal value-noise heightfield sampled at arbitrary world (x, z)."""

    def __init__(
        self,
        seed: int,
        height_scale: float,
        octaves: int,
        lattice_spacing: float = 2.0,
    ):
        self.seed = int(seed)
        self.height_scale = float(height_scale)
        self.octaves = int(octaves)
        self.lattice_spacing = float(lattice_spacing)

    def lattice_hash(self, ix: int, iz: int, octave: int) -> np.uint64:
        """Raw 64-bit hash of lattice point (ix, iz) for one octave."""
        with np.errstate(over="ignore"):
            state = _u64(self.seed)
            state = splitmix64(state + _u64(ix))
            state = splitmix64(state + _u64(iz))
            state = splitmix64(state + _u64(0x9E37 + octave * 0x85EB))
            return state

    def lattice_value(self, ix: int, iz: int, octave: int) -> float:
        """Hashed lattice value in [0, 1)."""
        return _unit_float(self.lattice_hash(ix, iz, octave))

    def lattice_values(
        self, ix: np.ndarray, iz: np.ndarray, octave: int
    ) -> np.ndarray:
        """Vectorised lattice hashes -> values in [0, 1) (uint64 arithmetic)."""
        ix64 = _u64_array(ix)
        iz64 = _u64_array(iz)
        with np.errstate(over="ignore"):
            state = _u64(self.seed)
            state = splitmix64(state + ix64)
            state = splitmix64(state + iz64)
            state = splitmix64(state + _u64(0x9E37 + octave * 0x85EB))
        return state.astype(np.float64) / TWO_POW_64

    def _octave_noise(self, x: float, z: float, octave: int) -> float:
        frequency = float(1 << octave) / self.lattice_spacing
        fx = x * frequency
        fz = z * frequency
        ix = int(np.floor(fx))
        iz = int(np.floor(fz))
        tx = _smoothstep(fx - ix)
        tz = _smoothstep(fz - iz)
        v00 = self.lattice_value(ix, iz, octave)
        v10 = self.lattice_value(ix + 1, iz, octave)
        v01 = self.lattice_value(ix, iz + 1, octave)
        v11 = self.lattice_value(ix + 1, iz + 1, octave)
        top = v00 + (v10 - v00) * tx
        bottom = v01 + (v11 - v01) * tx
        return top + (bottom - top) * tz

    def sample(self, x: float, z: float) -> float:
        """Height at arbitrary world coordinate (float64 bilinear noise)."""
        total = 0.0
        weight_sum = 0.0
        amplitude = 1.0
        for octave in range(self.octaves):
            total += amplitude * self._octave_noise(x, z, octave)
            weight_sum += amplitude
            amplitude *= 0.5
        normalized = total / weight_sum
        return normalized * self.height_scale

    def sample_array(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Vectorised arbitrary-point sampler; same math as :meth:`sample`."""
        x = np.asarray(x, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        total = np.zeros_like(x, dtype=np.float64)
        weight_sum = 0.0
        amplitude = 1.0
        for octave in range(self.octaves):
            frequency = float(1 << octave) / self.lattice_spacing
            fx = x * frequency
            fz = z * frequency
            ix = np.floor(fx).astype(np.int64)
            iz = np.floor(fz).astype(np.int64)
            tx = _smoothstep(fx - ix)
            tz = _smoothstep(fz - iz)
            v00 = self.lattice_values(ix, iz, octave)
            v10 = self.lattice_values(ix + 1, iz, octave)
            v01 = self.lattice_values(ix, iz + 1, octave)
            v11 = self.lattice_values(ix + 1, iz + 1, octave)
            top = v00 + (v10 - v00) * tx
            bottom = v01 + (v11 - v01) * tx
            total += amplitude * (top + (bottom - top) * tz)
            weight_sum += amplitude
            amplitude *= 0.5
        return (total / weight_sum) * self.height_scale

    def tile_grid(
        self, x0: float, z0: float, x1: float, z1: float, resolution: int
    ) -> np.ndarray:
        """Float64 ``resolution x resolution`` height grid for one tile."""
        grid = np.zeros((resolution, resolution), dtype=np.float64)
        if resolution == 1:
            grid[0, 0] = self.sample((x0 + x1) * 0.5, (z0 + z1) * 0.5)
            return grid
        xs = x0 + (x1 - x0) * np.arange(resolution, dtype=np.float64) / (
            resolution - 1
        )
        zs = z0 + (z1 - z0) * np.arange(resolution, dtype=np.float64) / (
            resolution - 1
        )
        xx, zz = np.meshgrid(xs, zs)
        grid[:, :] = self.sample_array(xx, zz)
        return grid
