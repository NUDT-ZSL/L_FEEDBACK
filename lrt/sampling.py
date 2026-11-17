"""Counter-based deterministic sequences and random-number helpers."""

from __future__ import annotations

import math
from typing import Tuple

_MASK64 = (1 << 64) - 1
_INV_2_53 = 1.0 / float(1 << 53)


def _mix64(value: int) -> int:
    x = value & _MASK64
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def hash_u64(*values: int) -> int:
    """Stable, platform-independent integer hash (counter based)."""

    state = 0x243F6A8885A308D3
    for value in values:
        state = _mix64(state ^ (int(value) & _MASK64))
        state = _mix64(state + 0x9E3779B97F4A7C15)
    return state


def hash01(*values: int) -> float:
    """Map a tuple of integer stream coordinates into [0, 1)."""

    bits = hash_u64(*values) >> 11
    return float(bits) * _INV_2_53


def radical_inverse(index: int, base: int) -> float:
    """Van der Corput radical inverse for an integer index >= 1."""

    if index <= 0:
        raise ValueError("radical_inverse index must be positive")
    result = 0.0
    factor = 1.0 / float(base)
    current = float(index)
    while current > 0.0:
        digit = int(current % base)
        result += float(digit) * factor
        current = float(int(current / base))
        factor /= float(base)
    return result


def low_discrepancy_2d(index: int, shift_u: float, shift_v: float) -> Tuple[float, float]:
    """A pixel-local, Cranley-Patterson rotated 2D Halton sequence."""

    u = (radical_inverse(index, 2) + shift_u) % 1.0
    v = (radical_inverse(index, 3) + shift_v) % 1.0
    return u, v


def tangent_basis(normal: Tuple[float, float, float]) -> Tuple[
    Tuple[float, float, float], Tuple[float, float, float]
]:
    nx, ny, nz = normal
    if abs(nx) > abs(nz):
        inv_len = 1.0 / math.sqrt(nx * nx + ny * ny)
        tangent = (-ny * inv_len, nx * inv_len, 0.0)
    else:
        inv_len = 1.0 / math.sqrt(ny * ny + nz * nz)
        tangent = (0.0, -nz * inv_len, ny * inv_len)
    bitangent = (
        normal[1] * tangent[2] - normal[2] * tangent[1],
        normal[2] * tangent[0] - normal[0] * tangent[2],
        normal[0] * tangent[1] - normal[1] * tangent[0],
    )
    return tangent, bitangent


def cosine_hemisphere(
    normal: Tuple[float, float, float], u1: float, u2: float
) -> Tuple[float, float, float]:
    """Sample a cosine-weighted direction in the hemisphere around normal."""

    radius = math.sqrt(max(0.0, u1))
    phi = 2.0 * math.pi * u2
    x = radius * math.cos(phi)
    y = radius * math.sin(phi)
    z = math.sqrt(max(0.0, 1.0 - u1))
    tangent, bitangent = tangent_basis(normal)
    return (
        tangent[0] * x + bitangent[0] * y + normal[0] * z,
        tangent[1] * x + bitangent[1] * y + normal[1] * z,
        tangent[2] * x + bitangent[2] * y + normal[2] * z,
    )


def perturb_around_axis(
    axis: Tuple[float, float, float],
    normal: Tuple[float, float, float],
    roughness: float,
    u1: float,
    u2: float,
) -> Tuple[float, float, float]:
    """Sample a narrow cone around a reflected direction, kept above a surface."""

    if roughness <= 1.0e-8:
        return axis
    max_angle = max(1.0e-5, min(0.95, roughness) * (0.5 * math.pi))
    radius = math.tan(max_angle) * math.sqrt(max(0.0, u1))
    phi = 2.0 * math.pi * u2
    tangent, bitangent = tangent_basis(axis)
    direction = (
        axis[0] + tangent[0] * radius * math.cos(phi) + bitangent[0] * radius * math.sin(phi),
        axis[1] + tangent[1] * radius * math.cos(phi) + bitangent[1] * radius * math.sin(phi),
        axis[2] + tangent[2] * radius * math.cos(phi) + bitangent[2] * radius * math.sin(phi),
    )
    length = math.sqrt(
        direction[0] * direction[0]
        + direction[1] * direction[1]
        + direction[2] * direction[2]
    )
    if length <= 1.0e-12:
        return axis
    direction = (direction[0] / length, direction[1] / length, direction[2] / length)
    if (
        direction[0] * normal[0]
        + direction[1] * normal[1]
        + direction[2] * normal[2]
    ) <= 1.0e-6:
        return axis
    return direction

