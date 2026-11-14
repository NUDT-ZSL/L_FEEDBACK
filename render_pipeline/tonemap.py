from __future__ import annotations

import numpy as np

from .scene import LightSpec


def tonemap_linear(linear_rgb: np.ndarray, preset: LightSpec) -> np.ndarray:
    """Apply exposure in linear space, tone-map, and keep result finite."""

    exposed = np.maximum(np.asarray(linear_rgb, dtype=np.float64), 0.0) * preset.exposure
    if preset.tone_operator == "reinhard":
        white_squared = preset.white_point * preset.white_point
        mapped = exposed * (1.0 + exposed / white_squared) / (1.0 + exposed)
    elif preset.tone_operator == "aces":
        numerator = exposed * (2.51 * exposed + 0.03)
        denominator = exposed * (2.43 * exposed + 0.59) + 0.14
        mapped = numerator / denominator
    else:
        raise ValueError(f"unsupported tone-mapping operator {preset.tone_operator!r}")
    return np.clip(mapped, 0.0, 1.0)


def linear_to_srgb8(linear_rgb: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(linear_rgb, dtype=np.float64), 0.0, 1.0)
    encoded = np.where(
        value <= 0.0031308,
        12.92 * value,
        1.055 * np.power(value, 1.0 / 2.4) - 0.055,
    )
    return np.floor(np.clip(encoded, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
