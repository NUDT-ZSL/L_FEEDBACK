"""Color-space conversion and display transforms."""

from __future__ import annotations

import numpy as np


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return np.where(
        value <= 0.04045,
        value / np.float32(12.92),
        np.power((value + np.float32(0.055)) / np.float32(1.055), np.float32(2.4)),
    ).astype(np.float32)


def linear_to_srgb(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    return np.where(
        value <= 0.0031308,
        value * np.float32(12.92),
        np.float32(1.055) * np.power(value, np.float32(1.0 / 2.4))
        - np.float32(0.055),
    ).astype(np.float32)


def tone_map_aces(value: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    x = np.maximum(np.asarray(value, dtype=np.float32) * np.float32(exposure), 0.0)
    mapped = (x * (np.float32(2.51) * x + np.float32(0.03))) / (
        x * (np.float32(2.43) * x + np.float32(0.59)) + np.float32(0.14)
    )
    return np.clip(mapped, 0.0, 1.0).astype(np.float32)


def display_transform(value: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    return linear_to_srgb(tone_map_aces(value, exposure))
