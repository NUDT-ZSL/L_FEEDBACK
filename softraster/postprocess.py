"""Independent tone mapping and sRGB transfer post-processing steps."""

from __future__ import annotations

import numpy as np


def apply_tonemap(image: np.ndarray, method: str) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if method == "aces":
        return _aces_fitted(image)
    if method == "reinhard":
        return image / (1.0 + image)
    raise ValueError(f"unsupported tone mapping method: {method}")


def apply_srgb_gamma(image: np.ndarray) -> np.ndarray:
    """Apply only the sRGB transfer function, without tone mapping."""
    image = np.asarray(image, dtype=np.float64)
    nonnegative = np.maximum(image, 0.0)
    return np.where(
        nonnegative <= 0.0031308,
        12.92 * nonnegative,
        1.055 * np.power(nonnegative, 1.0 / 2.4) - 0.055,
    )


def postprocess(
    image: np.ndarray,
    *,
    tonemap: str | None = None,
    gamma: bool = False,
) -> np.ndarray:
    result = np.asarray(image, dtype=np.float64)
    if tonemap is not None:
        result = apply_tonemap(result, tonemap)
    if gamma:
        result = apply_srgb_gamma(result)
    return result.astype(np.float32)


def _aces_fitted(image: np.ndarray) -> np.ndarray:
    # Narkowicz's commonly used ACES fit. It is applied independently of gamma.
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return np.clip((image * (a * image + b)) / (image * (c * image + d) + e), 0.0, 1.0)
