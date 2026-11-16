"""Tile- and scheduling-independent deterministic random numbers.

The renderer uses a counter-based SplitMix64 stream.  Every random value is a
pure function of (seed, pixel/photon id, sample id, stream id, event id, draw
id), so changing tile boundaries or resuming a checkpoint cannot change a
pixel's Monte Carlo samples.
"""

from __future__ import annotations

import numpy as np

_MASK = (1 << 64) - 1
_C1 = np.uint64(0xD6E8FEB86659FD93)
_C2 = np.uint64(0xA24BAED4963EE407)
_C3 = np.uint64(0x9FB21C651E98DF25)
_C4 = np.uint64(0xC2B2AE3D27D4EB4F)
_C5 = np.uint64(0x165667B19E3779F9)
_ADD = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def _as_u64(value: np.ndarray | int) -> np.ndarray:
    return np.asarray(value, dtype=np.uint64)


def _mix64(value: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = value + _ADD
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        z = z ^ (z >> np.uint64(31))
    return z


def uniform(
    seed: int,
    a: np.ndarray | int,
    b: np.ndarray | int = 0,
    c: np.ndarray | int = 0,
    d: np.ndarray | int = 0,
    e: np.ndarray | int = 0,
) -> np.ndarray:
    """Return floats in [0, 1), keyed by five stable integer coordinates."""

    seed_u = np.uint64(seed & _MASK)
    with np.errstate(over="ignore"):
        key = (
            seed_u
            ^ (_as_u64(a) * _C1)
            ^ (_as_u64(b) * _C2)
            ^ (_as_u64(c) * _C3)
            ^ (_as_u64(d) * _C4)
            ^ (_as_u64(e) * _C5)
        )
        bits = _mix64(key) >> np.uint64(11)
    return bits.astype(np.float64) * (1.0 / 9007199254740992.0)


_R2_A1 = 0.7548776662466927
_R2_A2 = 0.5698402909980532


def r2_2d(
    sample_ids: np.ndarray,
    rotation_1: np.ndarray | float,
    rotation_2: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified 2D sequence with a deterministic Cranley-Patterson rotation."""

    sample = np.asarray(sample_ids, dtype=np.float64)
    u = np.mod(0.5 + sample * _R2_A1 + np.asarray(rotation_1), 1.0)
    v = np.mod(0.5 + sample * _R2_A2 + np.asarray(rotation_2), 1.0)
    return u, v


def cosine_hemisphere(
    normal: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
) -> np.ndarray:
    """Sample the cosine-weighted hemisphere around unit vectors in `normal`."""

    radius = np.sqrt(u1)
    phi = 2.0 * np.pi * u2

    nx = normal[:, 0]
    ny = normal[:, 1]
    nz = normal[:, 2]
    use_x_axis = np.abs(nx) > np.abs(nz)
    tangent = np.empty_like(normal)
    tangent[use_x_axis] = np.column_stack(
        (
            -ny[use_x_axis],
            nx[use_x_axis],
            np.zeros(np.count_nonzero(use_x_axis)),
        )
    )
    tangent[~use_x_axis] = np.column_stack(
        (
            np.zeros(np.count_nonzero(~use_x_axis)),
            -nz[~use_x_axis],
            ny[~use_x_axis],
        )
    )
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1.0e-30)
    bitangent = np.cross(normal, tangent)

    local = np.column_stack(
        (
            radius * np.cos(phi),
            radius * np.sin(phi),
            np.sqrt(np.maximum(0.0, 1.0 - u1)),
        )
    )
    direction = tangent * local[:, 0:1] + bitangent * local[:, 1:2] + normal * local[:, 2:3]
    return direction / np.maximum(
        np.linalg.norm(direction, axis=1, keepdims=True), 1.0e-30
    )

