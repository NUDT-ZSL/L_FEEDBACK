"""Real orthonormal spherical-harmonic basis through degree three."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

Y00 = 0.28209479177387814
Y1 = 0.4886025119029199
Y2_2 = 1.0925484305920792
Y20 = 0.31539156525252005
Y22 = 0.5462742152960396
SH_COEFFICIENT_COUNT = 9
IRRADIANCE_FROM_C00 = math.sqrt(4.0 * math.pi)


def basis(directions: np.ndarray) -> np.ndarray:
    """Evaluate the nine real SH basis functions for unit directions."""

    d = np.asarray(directions, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != 3:
        raise ValueError("directions must have shape (N, 3)")
    x = d[:, 0]
    y = d[:, 1]
    z = d[:, 2]
    result = np.empty((d.shape[0], SH_COEFFICIENT_COUNT), dtype=np.float64)
    result[:, 0] = Y00
    result[:, 1] = Y1 * y
    result[:, 2] = Y1 * z
    result[:, 3] = Y1 * x
    result[:, 4] = Y2_2 * x * y
    result[:, 5] = Y2_2 * y * z
    result[:, 6] = Y20 * (3.0 * z * z - 1.0)
    result[:, 7] = Y2_2 * x * z
    result[:, 8] = Y22 * (x * x - y * y)
    return result


def project(
    directions: np.ndarray,
    values: np.ndarray,
    solid_angle_per_sample: float,
) -> np.ndarray:
    """Project RGB directional samples onto SH coefficients.

    The returned array has shape ``(9, 3)``. The projection uses a fixed,
    sample-order accumulation in float64 so threaded and serial execution are
    bit-identical before the final float32 cache conversion.
    """

    dirs = np.asarray(directions, dtype=np.float64)
    sample_values = np.asarray(values, dtype=np.float64)
    if sample_values.ndim == 1:
        sample_values = sample_values[:, None]
    if dirs.shape[0] == 0 or dirs.shape[0] != sample_values.shape[0]:
        raise ValueError("directions and values must have matching non-zero length")
    if not math.isfinite(solid_angle_per_sample):
        raise ValueError("solid angle weight must be finite")
    matrix = basis(dirs)
    return (matrix.T @ sample_values) * float(solid_angle_per_sample)


def reconstruct(directions: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    dirs = np.asarray(directions, dtype=np.float64)
    coeffs = np.asarray(coefficients, dtype=np.float64)
    if coeffs.shape[0] != SH_COEFFICIENT_COUNT:
        raise ValueError("coefficients must have shape (9, C)")
    return basis(dirs) @ coeffs


def relative_residual(
    directions: np.ndarray,
    values: np.ndarray,
    coefficients: np.ndarray,
) -> float:
    """Scalar relative RMS residual over all RGB samples."""

    sample_values = np.asarray(values, dtype=np.float64)
    if sample_values.ndim == 1:
        sample_values = sample_values[:, None]
    fitted = reconstruct(directions, coefficients)
    error_rms = math.sqrt(float(np.mean((fitted - sample_values) ** 2)))
    signal_rms = math.sqrt(float(np.mean(sample_values ** 2)))
    if signal_rms <= 1.0e-15:
        return 0.0
    return error_rms / signal_rms


def irradiance_from_coefficients(coefficients: np.ndarray) -> np.ndarray:
    """Return integrated directional irradiance for each RGB channel."""

    coeffs = np.asarray(coefficients, dtype=np.float64)
    return coeffs[0] * IRRADIANCE_FROM_C00


def gpu_style_convolved_coefficients(coefficients: np.ndarray) -> np.ndarray:
    """Apply the standard Lambertian SH convolution to radiance SH bands."""

    coeffs = np.asarray(coefficients, dtype=np.float64)
    if coeffs.shape[0] != SH_COEFFICIENT_COUNT:
        raise ValueError("coefficients must have shape (9, C)")
    result = coeffs.copy()
    # A0 = pi, A1 = 2*pi/3, A2 = pi/4 (Ramamoorthi-Hanrahan).
    result[0] /= math.pi
    result[1:4] *= (2.0 / 3.0)
    result[4:9] *= 0.25
    return result


def sh_orthonormality_error(sample_count: int = 4096) -> Tuple[float, float]:
    """Numerically check basis orthonormality on a Fibonacci sphere."""

    count = int(sample_count)
    indices = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    phi = indices * golden_angle
    dirs = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))
    matrix = basis(dirs)
    gram = (matrix.T @ matrix) * (4.0 * math.pi / count)
    identity = np.eye(SH_COEFFICIENT_COUNT)
    diagonal_error = float(np.max(np.abs(np.diag(gram) - 1.0)))
    off_diagonal_error = float(np.max(np.abs(gram - identity)))
    return diagonal_error, off_diagonal_error
