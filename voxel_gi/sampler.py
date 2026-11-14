"""Deterministic low-discrepancy hemisphere and disk sampling."""

from __future__ import annotations

import math

import numpy as np

GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
_PROBE_SALT = np.uint64(0x9E3779B97F4A7C15)


def normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(value))
    if not math.isfinite(length) or length <= 1.0e-15:
        raise ValueError("vector must be finite and non-zero")
    return value / length


def orthonormal_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic right-handed basis whose third axis is ``up``."""

    axis = normalize(up)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(axis, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    tangent = normalize(np.cross(helper, axis))
    bitangent = normalize(np.cross(axis, tangent))
    return tangent, bitangent, axis


def probe_phase(seed: int, probe_index: int) -> float:
    mask = (1 << 64) - 1
    mixed = (int(seed) ^ (((int(probe_index) + 1) * int(_PROBE_SALT)) & mask)) & mask
    rng = np.random.Generator(np.random.PCG64(mixed))
    return float(rng.random() * 2.0 * math.pi)


def hemisphere_directions(
    count: int,
    seed: int,
    probe_index: int,
    up: np.ndarray,
) -> np.ndarray:
    """Uniform solid-angle Fibonacci hemisphere with a seed-derived rotation."""

    n = int(count)
    if n <= 0:
        raise ValueError("sample count must be positive")
    tangent, bitangent, axis = orthonormal_basis(up)
    index = np.arange(n, dtype=np.float64)
    local_z = (index + 0.5) / float(n)
    radius = np.sqrt(np.maximum(0.0, 1.0 - local_z * local_z))
    phase = probe_phase(seed, probe_index)
    phi = index * GOLDEN_ANGLE + phase
    local_x = radius * np.cos(phi)
    local_y = radius * np.sin(phi)
    dirs = (
        local_x[:, None] * tangent[None, :]
        + local_y[:, None] * bitangent[None, :]
        + local_z[:, None] * axis[None, :]
    )
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def disk_samples(
    radius: float,
    count: int,
    normal: np.ndarray,
    seed: int,
    light_index: int,
) -> np.ndarray:
    """Return deterministic planar-disk offsets in the light tangent plane."""

    n = int(count)
    if n <= 0:
        raise ValueError("sample count must be positive")
    if radius <= 0.0:
        return np.zeros((1, 3), dtype=np.float64)
    tangent, bitangent, _ = orthonormal_basis(normal)
    index = np.arange(n, dtype=np.float64)
    area_fraction = (index + 0.5) / float(n)
    r = radius * np.sqrt(area_fraction)
    phase = probe_phase(seed, 100000 + int(light_index))
    phi = index * GOLDEN_ANGLE + phase
    local_x = r * np.cos(phi)
    local_y = r * np.sin(phi)
    offsets = local_x[:, None] * tangent[None, :] + local_y[:, None] * bitangent[None, :]
    return offsets
