"""Deterministic procedural heightfield and material-mask generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .coordinates import HEIGHT_SCALE, MATERIAL_NAMES, heightfield_grid
from .errors import E_INVALID_HEIGHTFIELD, E_WEIGHT_MISMATCH, BakeError


@dataclass(frozen=True)
class TerrainSamples:
    heightfield: np.ndarray
    weights: np.ndarray
    seed: int
    heightfield_sha256: str
    weights_sha256: str


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    if edge0 == edge1:
        return (value >= edge1).astype(np.float64)
    t = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _stable_seed(seed: int, salt: int) -> int:
    value = (int(seed) & 0xFFFFFFFFFFFFFFFF) ^ (int(salt) * 0x9E3779B97F4A7C15)
    value &= 0xFFFFFFFFFFFFFFFF
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 31
    return value


def _value_noise(
    rows: int,
    cols: int,
    seed: int,
    octave: int,
    cells: int,
) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(_stable_seed(seed, 1009 + octave)))
    lattice = rng.random((cells + 1, cells + 1), dtype=np.float64)

    x = np.linspace(0.0, 1.0, cols, dtype=np.float64) * cells
    z = np.linspace(0.0, 1.0, rows, dtype=np.float64) * cells
    gx = np.minimum(np.floor(x).astype(np.int64), cells - 1)
    gz = np.minimum(np.floor(z).astype(np.int64), cells - 1)
    tx = x - gx
    tz = z - gz
    tx = tx * tx * (3.0 - 2.0 * tx)
    tz = tz * tz * (3.0 - 2.0 * tz)

    gxx, gzz = np.meshgrid(gx, gz)
    txx, tzz = np.meshgrid(tx, tz)
    n00 = lattice[gzz, gxx]
    n10 = lattice[gzz, gxx + 1]
    n01 = lattice[gzz + 1, gxx]
    n11 = lattice[gzz + 1, gxx + 1]
    nx0 = n00 * (1.0 - txx) + n10 * txx
    nx1 = n01 * (1.0 - txx) + n11 * txx
    return nx0 * (1.0 - tzz) + nx1 * tzz


def _fbm(rows: int, cols: int, seed: int, salt: int, octave_count: int = 6) -> np.ndarray:
    total = np.zeros((rows, cols), dtype=np.float64)
    weight_sum = 0.0
    for octave in range(octave_count):
        amplitude = 0.5 ** octave
        cells = 3 * (2 ** octave)
        total += amplitude * _value_noise(rows, cols, seed + salt, octave, cells)
        weight_sum += amplitude
    return total / weight_sum


def generate_heightfield(rows: int, cols: int, seed: int) -> np.ndarray:
    """Generate a deterministic normalized [0, 1] heightfield."""

    if rows < 2 or cols < 2:
        raise BakeError(
            E_INVALID_HEIGHTFIELD,
            "heightfield dimensions must both be at least 2",
            {"shape": [int(rows), int(cols)]},
        )
    if seed < 0 or seed > 0xFFFFFFFFFFFFFFFF:
        raise BakeError(
            E_INVALID_HEIGHTFIELD,
            "seed must fit in an unsigned 64-bit integer",
            {"seed": seed},
        )

    base = _fbm(rows, cols, seed, 0)
    detail = _fbm(rows, cols, seed, 7919, octave_count=4)
    ridge_source = _fbm(rows, cols, seed, 15485863, octave_count=5)
    ridge = 1.0 - np.abs(2.0 * ridge_source - 1.0)

    height = 0.10 + 0.70 * base + 0.12 * detail + 0.14 * ridge
    value_range = float(height.max() - height.min())
    if value_range <= 1e-15:
        height = np.zeros_like(height)
    else:
        height = (height - height.min()) / value_range
    return np.ascontiguousarray(height, dtype=np.float64)


def validate_heightfield(heightfield: np.ndarray) -> np.ndarray:
    array = np.asarray(heightfield)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 2:
        raise BakeError(
            E_INVALID_HEIGHTFIELD,
            "heightfield must be a 2-D array with both dimensions >= 2",
            {"shape": list(array.shape)},
        )
    try:
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise BakeError(
            E_INVALID_HEIGHTFIELD,
            "heightfield must contain numeric samples",
        ) from exc
    if not np.all(np.isfinite(array)):
        bad = int(array.size - np.count_nonzero(np.isfinite(array)))
        raise BakeError(
            E_INVALID_HEIGHTFIELD,
            "heightfield contains NaN or Inf",
            {"non_finite_samples": bad},
        )
    return np.ascontiguousarray(array, dtype=np.float64)


def validate_weights(weights: np.ndarray, shape: tuple[int, int], tolerance: float) -> np.ndarray:
    array = np.asarray(weights)
    expected = (int(shape[0]), int(shape[1]), len(MATERIAL_NAMES))
    if array.shape != expected:
        raise BakeError(
            E_WEIGHT_MISMATCH,
            f"material weights must have shape {expected}",
            {"shape": list(array.shape), "expected": list(expected)},
        )
    try:
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise BakeError(E_WEIGHT_MISMATCH, "material weights must be numeric") from exc
    if not np.all(np.isfinite(array)):
        raise BakeError(E_WEIGHT_MISMATCH, "material weights contain NaN or Inf")
    if np.min(array) < -tolerance:
        raise BakeError(
            E_WEIGHT_MISMATCH,
            "material weights contain negative values",
            {"minimum": float(np.min(array))},
        )
    residual = np.abs(np.sum(array, axis=2) - 1.0)
    if np.max(residual) > tolerance:
        raise BakeError(
            E_WEIGHT_MISMATCH,
            "material weight rows must sum to 1 within tolerance",
            {
                "max_abs_residual": float(np.max(residual)),
                "tolerance": float(tolerance),
            },
        )
    array = np.clip(array, 0.0, None)
    sums = np.sum(array, axis=2, keepdims=True)
    if np.any(sums <= 0.0):
        raise BakeError(E_WEIGHT_MISMATCH, "material weight row sums to zero")
    return np.ascontiguousarray(array / sums, dtype=np.float64)


def _world_slope(heightfield: np.ndarray) -> np.ndarray:
    rows, cols = heightfield.shape
    dz = 1.0 / max(rows - 1, 1)
    dx = 1.0 / max(cols - 1, 1)
    grad_z, grad_x = np.gradient(heightfield * HEIGHT_SCALE, dz, dx)
    return np.sqrt(grad_x * grad_x + grad_z * grad_z)


def generate_material_weights(
    heightfield: np.ndarray,
    seed: int,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Generate per-vertex grass/rock/snow weights from the shared height grid."""

    height = validate_heightfield(heightfield)
    rows, cols = height.shape
    altitude_range = float(height.max() - height.min())
    if altitude_range <= 1e-15:
        altitude = np.zeros_like(height)
    else:
        altitude = (height - height.min()) / altitude_range
    slope = _world_slope(height)
    detail = _fbm(rows, cols, seed, 32452843, octave_count=4)

    steep = _smoothstep(0.18, 0.85, slope)
    high = _smoothstep(0.58, 0.88, altitude)
    snow_line = _smoothstep(0.62, 0.92, altitude) * (1.0 - 0.72 * steep)
    detail_centered = detail - 0.5

    grass = 0.08 + 0.82 * (1.0 - high) * (1.0 - 0.88 * steep)
    rock = 0.08 + 1.15 * steep + 0.22 * detail_centered
    snow = 0.015 + 1.65 * snow_line + 0.08 * detail_centered
    grass *= 0.92 + 0.16 * detail
    rock *= 0.88 + 0.24 * detail

    weights = np.stack([grass, rock, snow], axis=2)
    weights = np.clip(weights, 0.0, None)
    sums = np.sum(weights, axis=2, keepdims=True)
    if not np.all(np.isfinite(weights)) or np.any(sums <= 0.0):
        raise BakeError(
            E_INVALID_HEIGHTFIELD,
            "heightfield produced non-finite material weights",
        )
    weights /= sums
    return validate_weights(weights, height.shape, tolerance)


def _sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def generate_samples(
    rows: int,
    cols: int,
    seed: int,
    weight_tolerance: float,
) -> TerrainSamples:
    heightfield = generate_heightfield(rows, cols, seed)
    weights = generate_material_weights(heightfield, seed, tolerance=weight_tolerance)
    return TerrainSamples(
        heightfield=heightfield,
        weights=weights,
        seed=int(seed),
        heightfield_sha256=_sha256_array(heightfield),
        weights_sha256=_sha256_array(weights),
    )


def samples_from_external(
    heightfield: np.ndarray,
    weights: np.ndarray | None,
    seed: int,
    tolerance: float,
) -> TerrainSamples:
    height = validate_heightfield(heightfield)
    if weights is None:
        weights_array = generate_material_weights(height, seed, tolerance=tolerance)
    else:
        weights_array = validate_weights(weights, height.shape, tolerance)
    return TerrainSamples(
        heightfield=height,
        weights=weights_array,
        seed=int(seed),
        heightfield_sha256=_sha256_array(height),
        weights_sha256=_sha256_array(weights_array),
    )


def samples_summary(samples: TerrainSamples) -> dict[str, Any]:
    rows, cols = samples.heightfield.shape
    _, x_grid = heightfield_grid(rows, cols)
    return {
        "shape": [int(rows), int(cols)],
        "seed": int(samples.seed),
        "height_min": float(samples.heightfield.min()),
        "height_max": float(samples.heightfield.max()),
        "weight_sum_max_error": float(
            np.max(np.abs(np.sum(samples.weights, axis=2) - 1.0))
        ),
        "heightfield_sha256": samples.heightfield_sha256,
        "weights_sha256": samples.weights_sha256,
        "x_resolution": int(x_grid.shape[1]),
    }
