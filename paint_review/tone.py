from __future__ import annotations

import numpy as np


def aces_tonemap(linear_rgb: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    x = np.maximum(linear_rgb, 0.0) * float(exposure)
    mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    return np.clip(mapped, 0.0, 1.0)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    x = np.clip(image, 0.0, 1.0)
    return np.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * np.power(x, 1.0 / 2.4) - 0.055,
    )

