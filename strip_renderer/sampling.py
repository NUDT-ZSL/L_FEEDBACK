from __future__ import annotations

import numpy as np


def sample_nearest(
    texture: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Sample with integer texel centers and edge clamping."""
    height, width = texture.shape[:2]
    u_clamped = np.clip(u, 0.0, 1.0)
    v_clamped = np.clip(v, 0.0, 1.0)
    ix = np.floor(u_clamped * width).astype(np.int64)
    iy = np.floor(v_clamped * height).astype(np.int64)
    ix = np.clip(ix, 0, width - 1)
    iy = np.clip(iy, 0, height - 1)
    return texture[iy, ix]


def sample_bilinear(
    texture: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Sample between texel centers, clamping indices outside the texture."""
    height, width = texture.shape[:2]
    u_clamped = np.clip(u, 0.0, 1.0)
    v_clamped = np.clip(v, 0.0, 1.0)

    x = u_clamped * width - 0.5
    y = v_clamped * height - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    wx = x - x0
    wy = y - y0
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)

    if texture.ndim == 3:
        wx = wx[..., None]
        wy = wy[..., None]
    top = texture[y0, x0] * (1.0 - wx) + texture[y0, x1] * wx
    bottom = texture[y1, x0] * (1.0 - wx) + texture[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def sample_texture(
    texture: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "nearest":
        return sample_nearest(texture, u, v)
    if mode == "bilinear":
        return sample_bilinear(texture, u, v)
    raise ValueError(f"unsupported texture filter: {mode}")
