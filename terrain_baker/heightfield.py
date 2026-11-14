"""Deterministic seeded heightfield generator with chunked queries.

The noise is a gradient (Perlin-style) fBm built on a splitmix64 integer
hash.  Only uint64 arithmetic is used, so results are bit-exact across
platforms and numpy versions (no reliance on numpy.random stream stability).
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .errors import BakeError, ErrorCode

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _splitmix64(z: np.ndarray) -> np.ndarray:
    z = (z + np.uint64(0x9E3779B97F4A7C15)) & _MASK64
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _MASK64
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _MASK64
    return z ^ (z >> np.uint64(31))


def _lattice_hash(seed: int, octave: int, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    z = (np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
         ^ (ix.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))
         ^ (iy.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F))
         ^ (np.uint64(octave) * np.uint64(0x165667B19E3779F9)))
    return _splitmix64(z)


def _gradient_noise(seed: int, octave: int, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Gradient noise sampled at fractional lattice coordinates xs/ys."""
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = xs - x0
    fy = ys - y0
    # quintic fade
    u = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0)
    v = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0)

    def corner(ix: np.ndarray, iy: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        h = _lattice_hash(seed, octave, ix, iy)
        angle = (h >> np.uint64(11)).astype(np.float64) * (2.0 * np.pi / 2.0**53)
        return np.cos(angle) * dx + np.sin(angle) * dy

    n00 = corner(x0, y0, fx, fy)
    n10 = corner(x1, y0, fx - 1.0, fy)
    n01 = corner(x0, y1, fx, fy - 1.0)
    n11 = corner(x1, y1, fx - 1.0, fy - 1.0)
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    return nx0 + v * (nx1 - nx0)


def generate_heightfield(cfg: Config) -> np.ndarray:
    """Generate the full (resolution, resolution) float64 heightfield, row = y."""
    hf = cfg.heightfield
    res = hf.resolution
    noise = hf.noise
    try:
        ys, xs = np.meshgrid(
            np.arange(res, dtype=np.float64),
            np.arange(res, dtype=np.float64),
            indexing="ij",
        )
        total = np.zeros((res, res), dtype=np.float64)
        amplitude_total = 0.0
        freq = noise.base_frequency
        amp = 1.0
        for octave in range(noise.octaves):
            total += amp * _gradient_noise(
                cfg.seed, octave, xs * freq / (res - 1), ys * freq / (res - 1))
            amplitude_total += amp
            freq *= noise.lacunarity
            amp *= noise.gain
        total /= amplitude_total
        # total is roughly in [-1, 1]; map linearly to [min_height, max_height]
        t = np.clip(total * 0.5 + 0.5, 0.0, 1.0)
        return hf.min_height + t * (hf.max_height - hf.min_height)
    except Exception as exc:  # pragma: no cover - defensive
        raise BakeError(ErrorCode.E_HEIGHTFIELD, f"heightfield generation failed: {exc}")


class HeightField:
    """Heightfield with clamped sampling shared by mesh and normal bakers."""

    def __init__(self, cfg: Config, heights: np.ndarray | None = None):
        self.cfg = cfg
        self.heights = heights if heights is not None else generate_heightfield(cfg)
        self.resolution = cfg.heightfield.resolution
        self.spacing = cfg.spacing

    # -- clamped sampling rules (shared by every consumer) -----------------
    def clamp_index(self, i: int) -> int:
        return min(max(i, 0), self.resolution - 1)

    def sample_grid(self, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        """Nearest-grid sample with index clamping."""
        cx = np.clip(ix, 0, self.resolution - 1)
        cy = np.clip(iy, 0, self.resolution - 1)
        return self.heights[cy, cx]

    def sample_world(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Bilinear sample at world coordinates, clamped to the field bounds."""
        gx = np.clip(np.asarray(xs, dtype=np.float64) / self.spacing, 0.0, self.resolution - 1.0)
        gy = np.clip(np.asarray(ys, dtype=np.float64) / self.spacing, 0.0, self.resolution - 1.0)
        x0 = np.floor(gx).astype(np.int64)
        y0 = np.floor(gy).astype(np.int64)
        x1 = np.minimum(x0 + 1, self.resolution - 1)
        y1 = np.minimum(y0 + 1, self.resolution - 1)
        fx = gx - x0
        fy = gy - y0
        h00 = self.heights[y0, x0]
        h10 = self.heights[y0, x1]
        h01 = self.heights[y1, x0]
        h11 = self.heights[y1, x1]
        return ((h00 * (1 - fx) + h10 * fx) * (1 - fy)
                + (h01 * (1 - fx) + h11 * fx) * fy)

    # -- chunked access -----------------------------------------------------
    def chunk_vertex_indices(self, cx: int, cy: int, step: int = 1):
        """Grid vertex indices (ix, iy arrays) covered by chunk (cx, cy)."""
        hf = self.cfg.heightfield
        cells = hf.chunk_size - 1
        if cells % step != 0:
            raise BakeError(ErrorCode.E_HEIGHTFIELD,
                            f"chunk cells {cells} not divisible by step {step}")
        n = cells // step + 1
        chunks = self.cfg.chunks_per_side
        if not (0 <= cx < chunks and 0 <= cy < chunks):
            raise BakeError(ErrorCode.E_HEIGHTFIELD,
                            f"chunk ({cx}, {cy}) out of range {chunks}")
        local = np.arange(n, dtype=np.int64) * step
        ix = cx * cells + local
        iy = cy * cells + local
        return ix, iy

    def chunk_grid(self, cx: int, cy: int, step: int = 1):
        """Return (xs, ys, heights) world-space grid for a chunk.

        xs/ys are 1-D world coordinates of length n; heights is (n, n) with
        row = y.  Border samples use the shared clamping rule.
        """
        ix, iy = self.chunk_vertex_indices(cx, cy, step)
        xs = ix.astype(np.float64) * self.spacing
        ys = iy.astype(np.float64) * self.spacing
        heights = self.heights[np.ix_(iy, ix)]
        return xs, ys, heights
