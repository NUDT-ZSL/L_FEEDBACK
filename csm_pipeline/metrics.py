"""Quantitative shadow quality metrics and difference-image helpers."""

from __future__ import annotations

import hashlib

import numpy as np


# Explicit, documented PSNR value for two bit-identical zero-error images.
PSNR_IDENTICAL_DB = 99.0
_MAX_PIXEL_VALUE = 1.0


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape:
        raise ValueError("PSNR inputs must share a shape")
    mse = float(np.mean((ref - cand) ** 2))
    if mse == 0.0:
        return PSNR_IDENTICAL_DB
    return float(10.0 * np.log10((_MAX_PIXEL_VALUE ** 2) / mse))


def max_absolute_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    return float(np.max(np.abs(ref - cand)))


def normalized_cross_correlation(
    reference: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> float:
    """NCC over a boolean mask; returns 1.0 for constant/undefined regions."""
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    sel_mask = np.asarray(mask, dtype=bool)
    if int(sel_mask.sum()) < 2:
        return 1.0
    a = ref[sel_mask]
    b = cand[sel_mask]
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denominator = float(
        np.sqrt(np.sum(a_centered ** 2) * np.sum(b_centered ** 2))
    )
    if denominator == 0.0:
        return 1.0
    return float(np.sum(a_centered * b_centered) / denominator)


def shadow_region_mask(factor_a: np.ndarray, factor_b: np.ndarray) -> np.ndarray:
    """Pixels that either image considers at least partially shadowed."""
    return (np.asarray(factor_a) < 0.999) | (np.asarray(factor_b) < 0.999)


def per_cascade_stats(
    coverage: np.ndarray,
    hit: np.ndarray,
    depth_bias_ndc: np.ndarray,
) -> list:
    stats: list = []
    total_hit = max(int(np.asarray(hit).sum()), 1)
    for index in range(coverage.shape[0]):
        own = coverage[index]
        pixel_count = int(own.sum())
        if pixel_count:
            average_bias = float(
                np.mean(np.asarray(depth_bias_ndc, dtype=np.float64)[own])
            )
        else:
            average_bias = 0.0
        stats.append(
            {
                "cascade_index": index,
                "pixel_count": pixel_count,
                "shadow_coverage": pixel_count / total_hit,
                "average_depth_bias_ndc": average_bias,
            }
        )
    return stats


def array_checksum(arr: np.ndarray) -> str:
    """Deterministic little-endian SHA-256 used by the JSON contract."""
    encoded = np.asarray(arr).astype("<f4", copy=False).tobytes()
    return hashlib.sha256(encoded).hexdigest()


def write_pgm(path: str, image: np.ndarray) -> None:
    """Write an 8-bit binary PGM (P5), viewable without any image library."""
    arr = np.asarray(image, dtype=np.float64)
    scaled = np.clip(arr, 0.0, 1.0) * 255.0
    pixels = np.rint(scaled).astype(np.uint8)
    height, width = pixels.shape
    with open(path, "wb") as handle:
        header = f"P5\n{width} {height}\n255\n".encode("ascii")
        handle.write(header)
        handle.write(pixels.tobytes())

