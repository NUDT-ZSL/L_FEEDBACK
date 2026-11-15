from __future__ import annotations

import numpy as np


def as_vec3(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 numbers")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numbers")
    return array


def normalize(value: np.ndarray, name: str) -> np.ndarray:
    length = float(np.linalg.norm(value))
    if not np.isfinite(length) or length < 1e-12:
        raise ValueError(f"{name} must be a non-zero finite vector")
    return value / length


def normalize_rows(value: np.ndarray, fallback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.linalg.norm(value, axis=-1)
    bad = (~np.isfinite(lengths)) | (lengths < 1e-12)
    result = np.array(value, dtype=np.float64, copy=True)
    good = ~bad
    result[good] /= lengths[good, None]
    if np.any(bad):
        result[bad] = fallback
    return result, bad


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.power((values + 0.055) / 1.055, 2.4),
    )


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )
