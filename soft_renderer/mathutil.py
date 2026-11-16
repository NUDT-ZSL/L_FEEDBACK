from __future__ import annotations

from typing import Iterable

import numpy as np


EPSILON = 1.0e-12


def normalize(vector: np.ndarray, axis: int = -1) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    length = np.sqrt(np.sum(array * array, axis=axis, keepdims=True))
    return array / np.maximum(length, EPSILON)


def normalize_single(vector: Iterable[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    length = float(np.sqrt(np.dot(array, array)))
    if not np.isfinite(length) or length <= EPSILON:
        raise ValueError("vector must have non-zero length")
    return array / length


def clamp(value: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    return np.minimum(np.maximum(value, low), high)


def smoothstep(edge0: np.ndarray, edge1: np.ndarray, value: np.ndarray) -> np.ndarray:
    denominator = np.maximum(edge1 - edge0, EPSILON)
    t = clamp((value - edge0) / denominator)
    return t * t * (3.0 - 2.0 * t)


def srgb_to_linear(color: np.ndarray) -> np.ndarray:
    value = np.asarray(color, dtype=np.float64)
    return np.where(
        value <= 0.04045,
        value / 12.92,
        np.power((value + 0.055) / 1.055, 2.4),
    )


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        dtype=np.float64,
    )
