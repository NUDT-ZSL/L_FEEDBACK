"""Regenerate the deterministic built-in .npy textures."""

from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
TEXTURE_DIR = ROOT / "assets" / "textures"


def _grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:size, 0:size]
    return x.astype(np.float32), y.astype(np.float32)


def metal_texture(size: int = 32) -> np.ndarray:
    x, y = _grid(size)
    checker = (((x // 4) + (y // 4)) % 2).astype(np.float32)
    stripe = (0.80 + 0.20 * np.sin((x + 2.0 * y) * np.pi / 8.0)).astype(np.float32)
    red = np.clip(0.55 + 0.40 * stripe, 0.0, 1.0)
    green = np.clip(0.42 + 0.36 * checker + 0.12 * stripe, 0.0, 1.0)
    blue = np.clip(0.20 + 0.55 * checker, 0.0, 1.0)
    alpha = np.ones_like(red)
    return np.stack((red, green, blue, alpha), axis=-1).astype(np.float32)


def plastic_texture(size: int = 32) -> np.ndarray:
    x, y = _grid(size)
    dots = (
        (((x + 1.5) % 8.0) - 3.5) ** 2 + (((y + 1.5) % 8.0) - 3.5) ** 2 < 2.0
    ).astype(np.float32)
    red = np.clip(0.22 + 0.20 * dots, 0.0, 1.0)
    green = np.clip(0.42 + 0.28 * dots, 0.0, 1.0)
    blue = np.clip(0.75 + 0.22 * dots, 0.0, 1.0)
    alpha = np.ones_like(red)
    return np.stack((red, green, blue, alpha), axis=-1).astype(np.float32)


def emissive_texture(size: int = 24) -> np.ndarray:
    x, y = _grid(size)
    cx = (x - (size - 1) * 0.5) / max(size * 0.5, 1.0)
    cy = (y - (size - 1) * 0.5) / max(size * 0.5, 1.0)
    radius = np.sqrt(cx * cx + cy * cy)
    glow = np.clip(1.0 - radius, 0.0, 1.0) ** 2
    pulse = (0.75 + 0.25 * np.cos(radius * np.pi * 3.0)).astype(np.float32)
    red = np.clip(0.85 * pulse + 0.40 * glow, 0.0, 2.0)
    green = np.clip(0.18 * pulse + 0.25 * glow, 0.0, 2.0)
    blue = np.clip(0.04 * pulse + 0.08 * glow, 0.0, 2.0)
    alpha = np.ones_like(red)
    return np.stack((red, green, blue, alpha), axis=-1).astype(np.float32)


def main() -> None:
    TEXTURE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(TEXTURE_DIR / "pbr_metal_albedo.npy", metal_texture().astype(np.uint8))
    np.save(TEXTURE_DIR / "rough_plastic_albedo.npy", plastic_texture().astype(np.uint8))
    np.save(TEXTURE_DIR / "emissive_albedo.npy", emissive_texture().astype(np.float32))
    for path in sorted(TEXTURE_DIR.glob("*.npy")):
        array = np.load(path, allow_pickle=False)
        print(f"{path.relative_to(ROOT)}: shape={array.shape} dtype={array.dtype}")


if __name__ == "__main__":
    main()

