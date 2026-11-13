"""Small deterministic scalar vector and oriented-box intersection utilities."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable, NamedTuple, Sequence

Vec3 = tuple[float, float, float]

EPSILON = 1.0e-12


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def mul_components(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] * b[0], a[1] * b[1], a[2] * b[2])


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def length_sq(a: Vec3) -> float:
    return dot(a, a)


def length(a: Vec3) -> float:
    return math.sqrt(length_sq(a))


def normalize(a: Vec3, label: str = "vector") -> Vec3:
    n = length(a)
    if n <= EPSILON:
        raise ValueError(f"{label} must have non-zero length")
    return (a[0] / n, a[1] / n, a[2] / n)


def mat_mul_vec(m: Sequence[float], v: Vec3) -> Vec3:
    return (
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    )


def mat_transpose_mul_vec(m: Sequence[float], v: Vec3) -> Vec3:
    return (
        m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
        m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
        m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
    )


def mat_mul(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return (
        a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
        a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
        a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
        a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
        a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
        a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
        a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
        a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
        a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
    )


@lru_cache(maxsize=8192)
def rotation_matrix_deg(rotation_deg: Vec3) -> tuple[float, ...]:
    rx, ry, rz = (math.radians(component) for component in rotation_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    # R = Rz * Ry * Rx.  Every box in the scene therefore has one
    # unambiguous local-to-world transform derivable from its canonical data.
    return (
        cz * cy,
        cz * sy * sx - sz * cx,
        cz * sy * cx + sz * sx,
        sz * cy,
        sz * sy * sx + cz * cx,
        sz * sy * cx - cz * sx,
        -sy,
        cy * sx,
        cy * cx,
    )


class Hit(NamedTuple):
    object_id: int
    t: float
    point: Vec3
    normal: Vec3


def ray_obb_interval(
    origin: Vec3,
    direction: Vec3,
    center: Vec3,
    half_size: Vec3,
    rotation_deg: Vec3,
) -> tuple[float, float, Vec3, Vec3] | None:
    """Return (t_enter, t_exit, local_entry_normal, local_exit_normal)."""

    matrix = rotation_matrix_deg(rotation_deg)
    local_origin = mat_transpose_mul_vec(matrix, sub(origin, center))
    local_direction = mat_transpose_mul_vec(matrix, direction)

    t_enter = -math.inf
    t_exit = math.inf
    entry_normal: Vec3 = (0.0, 0.0, 0.0)
    exit_normal: Vec3 = (0.0, 0.0, 0.0)

    for axis in range(3):
        origin_axis = local_origin[axis]
        direction_axis = local_direction[axis]
        half_axis = half_size[axis]
        if abs(direction_axis) <= EPSILON:
            if origin_axis < -half_axis or origin_axis > half_axis:
                return None
            continue

        t1 = (-half_axis - origin_axis) / direction_axis
        t2 = (half_axis - origin_axis) / direction_axis
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_enter:
            t_enter = t1
            entry_normal = [0.0, 0.0, 0.0]
            entry_normal[axis] = -1.0 if direction_axis > 0.0 else 1.0
            entry_normal = (entry_normal[0], entry_normal[1], entry_normal[2])
        if t2 < t_exit:
            t_exit = t2
            exit_normal = [0.0, 0.0, 0.0]
            exit_normal[axis] = 1.0 if direction_axis > 0.0 else -1.0
            exit_normal = (exit_normal[0], exit_normal[1], exit_normal[2])
        if t_enter > t_exit:
            return None

    return t_enter, t_exit, entry_normal, exit_normal


def trace_boxes(
    origin: Vec3,
    direction: Vec3,
    boxes: Iterable[tuple[int, Vec3, Vec3, Vec3]],
    t_min: float,
    t_max: float = math.inf,
) -> Hit | None:
    """Nearest oriented-box hit.

    `boxes` must be iterated in ascending object-id order.  The strict
    comparison below makes an equidistant tie resolve to the lower id.
    """

    best_t = t_max
    best: Hit | None = None
    for object_id, center, half_size, rotation_deg in boxes:
        interval = ray_obb_interval(origin, direction, center, half_size, rotation_deg)
        if interval is None:
            continue
        t_enter, t_exit, entry_local_normal, exit_local_normal = interval
        if t_exit <= t_min:
            continue
        if t_enter > t_min:
            t = t_enter
            local_normal = entry_local_normal
        else:
            # Origin started inside this box (normally prevented by probe
            # culling).  Use the exit surface so shadow queries stay robust.
            t = t_exit
            local_normal = exit_local_normal
        if t >= best_t:
            continue
        best_t = t
        matrix = rotation_matrix_deg(rotation_deg)
        world_normal = mat_mul_vec(matrix, local_normal)
        point = add(origin, mul(direction, t))
        best = Hit(object_id, t, point, world_normal)
    return best


def box_signed_distance(
    point: Vec3,
    center: Vec3,
    half_size: Vec3,
    rotation_deg: Vec3,
) -> float:
    matrix = rotation_matrix_deg(rotation_deg)
    local = mat_transpose_mul_vec(matrix, sub(point, center))

    outside_sq = 0.0
    inside_depth = -math.inf
    for axis in range(3):
        distance_to_plane = abs(local[axis]) - half_size[axis]
        inside_depth = max(inside_depth, distance_to_plane)
        if distance_to_plane > 0.0:
            outside_sq += distance_to_plane * distance_to_plane
    if outside_sq > 0.0:
        return math.sqrt(outside_sq)
    return inside_depth


def orthonormal_hemisphere_basis(normal: Vec3) -> tuple[Vec3, Vec3, Vec3]:
    normal = normalize(normal, "probe normal")
    reference: Vec3 = (0.0, 0.0, 1.0) if abs(normal[2]) < 0.999 else (0.0, 1.0, 0.0)
    tangent = normalize(cross(reference, normal), "hemisphere tangent")
    bitangent = cross(normal, tangent)
    return tangent, bitangent, normal

