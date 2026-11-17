"""HDR (float32 RGB) -> ACES tone mapping -> sRGB -> 8-bit."""

from __future__ import annotations

import numpy as np


def aces_tonemap(x: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """ACES fitted curve (Narkowicz), input/output in [0, 1]."""
    x = np.maximum(x.astype(np.float64) * exposure, 0.0)
    return np.clip(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def srgb_encode(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    lo = x * 12.92
    hi = 1.055 * np.power(x, 1.0 / 2.4) - 0.055
    return np.where(x <= 0.0031308, lo, hi)


def hdr_to_rgb8(hdr: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """Float32 HDR (H, W, 3) -> uint8 (H, W, 3), all channels 8-bit."""
    return np.clip(np.rint(srgb_encode(aces_tonemap(hdr, exposure)) * 255.0), 0, 255).astype(np.uint8)
