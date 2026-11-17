"""Procedural tangent-space normal maps (RGB8, +Y up) and sampling."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .errors import AssetError


def _tileable_value_noise(rng: np.random.Generator, size: int, period: int) -> np.ndarray:
    """Tileable value noise on a ``size`` x ``size`` grid with lattice ``period``."""
    lattice = rng.random((period, period))
    ys = np.linspace(0.0, period, size, endpoint=False)
    xs = np.linspace(0.0, period, size, endpoint=False)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    fy = fy * fy * (3.0 - 2.0 * fy)
    fx = fx * fx * (3.0 - 2.0 * fx)
    y1 = (y0 + 1) % period
    x1 = (x0 + 1) % period
    n00 = lattice[np.ix_(y0, x0)]
    n01 = lattice[np.ix_(y0, x1)]
    n10 = lattice[np.ix_(y1, x0)]
    n11 = lattice[np.ix_(y1, x1)]
    return (n00 * (1 - fx) + n01 * fx) * (1 - fy) + (n10 * (1 - fx) + n11 * fx) * fy


def generate_heightfield(seed: int, size: int, kind: str) -> np.ndarray:
    """Deterministic tileable heightfield in [0, 1]."""
    rng = np.random.default_rng(seed)
    h = np.zeros((size, size), dtype=np.float64)
    period = 4
    amp = 1.0
    total = 0.0
    for _ in range(4):
        h += amp * _tileable_value_noise(rng, size, period)
        total += amp
        amp *= 0.5
        period *= 2
    h /= total
    if kind == "panels":
        # Hard-surface panel grooves; the 32px period tiles exactly.
        ys, xs = np.mgrid[0:size, 0:size]
        groove_x = (xs % 32) < 2
        groove_y = (ys % 32) < 2
        h = h * 0.55 + 0.45
        h = h - 0.35 * (groove_x | groove_y)
        rivet = ((xs % 32) == 16) & ((ys % 32) == 16)
        h = h + 0.25 * rivet
        h = np.clip(h, 0.0, 1.0)
    return h


def heightfield_to_rgb8(h: np.ndarray, strength: float = 2.6) -> np.ndarray:
    """Convert heightfield to tangent-space RGB8 normal map (+Y up)."""
    dhdx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5
    dhdy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5
    nx = -dhdx * strength
    ny = dhdy * strength  # image row grows downward; +Y is up
    nz = np.ones_like(h)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    rgb = np.stack([nx * inv, ny * inv, nz * inv], axis=-1) * 0.5 + 0.5
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def write_normal_map(path: str, seed: int, size: int, kind: str) -> None:
    h = generate_heightfield(seed, size, kind)
    Image.fromarray(heightfield_to_rgb8(h), mode="RGB").save(path)


class NormalMap:
    """Bilinear sampler with repeat addressing; tangent space, +Y up."""

    def __init__(self, rgb01: np.ndarray):
        # rgb01: (H, W, 3) in [0, 1]
        self.map = rgb01.astype(np.float64) * 2.0 - 1.0
        self.height, self.width = self.map.shape[:2]

    @classmethod
    def from_file(cls, path: str) -> "NormalMap":
        try:
            with Image.open(path) as img:
                arr = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
        except OSError as exc:
            raise AssetError(f"failed to read normal map {path}: {exc}")
        return cls(arr)

    def sample(self, uv: np.ndarray) -> np.ndarray:
        """uv: (N, 2), any range (repeat). Returns (N, 3) normalized tangents."""
        u = uv[:, 0] - np.floor(uv[:, 0])
        v = uv[:, 1] - np.floor(uv[:, 1])
        # OBJ convention: v = 1 is the top row of the image.
        x = u * self.width - 0.5
        y = (1.0 - v) * self.height - 0.5
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        fx = (x - x0)[:, None]
        fy = (y - y0)[:, None]
        x1 = (x0 + 1) % self.width
        y1 = (y0 + 1) % self.height
        x0 %= self.width
        y0 %= self.height
        c00 = self.map[y0, x0]
        c01 = self.map[y0, x1]
        c10 = self.map[y1, x0]
        c11 = self.map[y1, x1]
        n = (c00 * (1 - fx) + c01 * fx) * (1 - fy) + (c10 * (1 - fx) + c11 * fx) * fy
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-12)
        return n / norm
