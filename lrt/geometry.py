"""Ray/primitive intersection code using scalar Python arithmetic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a: Vec3, scalar: float) -> Vec3:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def hadamard(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] * b[0], a[1] * b[1], a[2] * b[2])


def length_squared(a: Vec3) -> float:
    return dot(a, a)


def length(a: Vec3) -> float:
    return math.sqrt(max(0.0, dot(a, a)))


def normalize(a: Vec3) -> Vec3:
    magnitude = length(a)
    if magnitude <= 1.0e-15:
        raise ValueError("zero-length vector")
    inverse = 1.0 / magnitude
    return (a[0] * inverse, a[1] * inverse, a[2] * inverse)


def reflect(direction: Vec3, normal: Vec3) -> Vec3:
    scale = 2.0 * dot(direction, normal)
    return (
        direction[0] - scale * normal[0],
        direction[1] - scale * normal[1],
        direction[2] - scale * normal[2],
    )


def offset_point(point: Vec3, normal: Vec3, hit_distance: float, epsilon: float) -> Vec3:
    """Adaptive self-intersection offset proportional to the hit distance."""

    scale = epsilon * max(1.0, abs(hit_distance))
    return (
        point[0] + normal[0] * scale,
        point[1] + normal[1] * scale,
        point[2] + normal[2] * scale,
    )


class Hit(NamedTuple):
    t: float
    point: Vec3
    normal: Vec3
    front: bool
    obj: Any


@dataclass(frozen=True)
class Sphere:
    center: Vec3
    radius: float
    material: Any

    def intersect(
        self, origin: Vec3, direction: Vec3, t_min: float, t_max: float
    ) -> Optional[Hit]:
        oc = sub(origin, self.center)
        b = dot(oc, direction)
        c = dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - c
        if discriminant < 0.0:
            return None
        root = math.sqrt(discriminant)
        t0 = -b - root
        t1 = -b + root
        if t0 >= t_min and t0 <= t_max:
            t = t0
        elif t1 >= t_min and t1 <= t_max:
            t = t1
        else:
            return None
        point = (
            origin[0] + direction[0] * t,
            origin[1] + direction[1] * t,
            origin[2] + direction[2] * t,
        )
        normal = normalize(sub(point, self.center))
        return Hit(t, point, normal, dot(direction, normal) < 0.0, self)


@dataclass(frozen=True)
class Plane:
    point: Vec3
    normal: Vec3
    material: Any

    def intersect(
        self, origin: Vec3, direction: Vec3, t_min: float, t_max: float
    ) -> Optional[Hit]:
        denominator = dot(direction, self.normal)
        if abs(denominator) <= 1.0e-14:
            return None
        t = dot(sub(self.point, origin), self.normal) / denominator
        if t < t_min or t > t_max:
            return None
        point = (
            origin[0] + direction[0] * t,
            origin[1] + direction[1] * t,
            origin[2] + direction[2] * t,
        )
        return Hit(t, point, self.normal, denominator < 0.0, self)


@dataclass(frozen=True)
class AABB:
    minimum: Vec3
    maximum: Vec3
    material: Any

    def intersect(
        self, origin: Vec3, direction: Vec3, t_min: float, t_max: float
    ) -> Optional[Hit]:
        near = -float("inf")
        far = float("inf")
        near_axis = -1
        far_axis = -1
        for axis in range(3):
            axis_min = self.minimum[axis]
            axis_max = self.maximum[axis]
            component = direction[axis]
            if abs(component) <= 1.0e-15:
                if origin[axis] < axis_min or origin[axis] > axis_max:
                    return None
                continue
            inverse = 1.0 / component
            t0 = (axis_min - origin[axis]) * inverse
            t1 = (axis_max - origin[axis]) * inverse
            if t0 > t1:
                t0, t1 = t1, t0
            if t0 > near:
                near = t0
                near_axis = axis
            if t1 < far:
                far = t1
                far_axis = axis
            if near > far:
                return None
        if far < t_min or near > t_max:
            return None
        if near >= t_min:
            t = near
            hit_axis = near_axis
            inside = False
        else:
            t = far
            hit_axis = far_axis
            inside = True
        if hit_axis < 0:
            return None
        point = (
            origin[0] + direction[0] * t,
            origin[1] + direction[1] * t,
            origin[2] + direction[2] * t,
        )
        normal = [0.0, 0.0, 0.0]
        if inside:
            normal[hit_axis] = 1.0 if direction[hit_axis] > 0.0 else -1.0
        else:
            normal[hit_axis] = -1.0 if direction[hit_axis] > 0.0 else 1.0
        normal_tuple = (normal[0], normal[1], normal[2])
        return Hit(t, point, normal_tuple, dot(direction, normal_tuple) < 0.0, self)


@dataclass(frozen=True)
class Triangle:
    v0: Vec3
    v1: Vec3
    v2: Vec3
    material: Any

    def intersect(
        self, origin: Vec3, direction: Vec3, t_min: float, t_max: float
    ) -> Optional[Hit]:
        edge1 = sub(self.v1, self.v0)
        edge2 = sub(self.v2, self.v0)
        pvec = cross(direction, edge2)
        determinant = dot(edge1, pvec)
        if abs(determinant) <= 1.0e-14:
            return None
        inverse = 1.0 / determinant
        tvec = sub(origin, self.v0)
        u = dot(tvec, pvec) * inverse
        if u < 0.0 or u > 1.0:
            return None
        qvec = cross(tvec, edge1)
        v = dot(direction, qvec) * inverse
        if v < 0.0 or u + v > 1.0:
            return None
        t = dot(edge2, qvec) * inverse
        if t < t_min or t > t_max:
            return None
        point = (
            origin[0] + direction[0] * t,
            origin[1] + direction[1] * t,
            origin[2] + direction[2] * t,
        )
        normal = normalize(cross(edge1, edge2))
        return Hit(t, point, normal, dot(direction, normal) < 0.0, self)


@dataclass(frozen=True)
class RectLight:
    center: Vec3
    right: Vec3
    up: Vec3
    width: float
    height: float
    intensity: Vec3
    name: str = "rect_light"

    def __post_init__(self) -> None:
        normal = normalize(cross(self.right, self.up))
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "area", self.width * self.height)

    def sample_point(self, u: float, v: float) -> Vec3:
        return (
            self.center[0] + (u - 0.5) * self.width * self.right[0]
            + (v - 0.5) * self.height * self.up[0],
            self.center[1] + (u - 0.5) * self.width * self.right[1]
            + (v - 0.5) * self.height * self.up[1],
            self.center[2] + (u - 0.5) * self.width * self.right[2]
            + (v - 0.5) * self.height * self.up[2],
        )

    def intersect(
        self, origin: Vec3, direction: Vec3, t_min: float, t_max: float
    ) -> Optional[Hit]:
        denominator = dot(direction, self.normal)
        if abs(denominator) <= 1.0e-14:
            return None
        t = dot(sub(self.center, origin), self.normal) / denominator
        if t < t_min or t > t_max:
            return None
        point = (
            origin[0] + direction[0] * t,
            origin[1] + direction[1] * t,
            origin[2] + direction[2] * t,
        )
        local = sub(point, self.center)
        half_width = 0.5 * self.width
        half_height = 0.5 * self.height
        if abs(dot(local, self.right)) > half_width + 1.0e-12:
            return None
        if abs(dot(local, self.up)) > half_height + 1.0e-12:
            return None
        normal = self.normal if dot(direction, self.normal) < 0.0 else mul(self.normal, -1.0)
        return Hit(t, point, normal, dot(direction, self.normal) < 0.0, self)


@dataclass(frozen=True)
class SphereLight:
    center: Vec3
    radius: float
    intensity: Vec3
    name: str = "sphere_light"

    def intersect(
        self, origin: Vec3, direction: Vec3, t_min: float, t_max: float
    ) -> Optional[Hit]:
        oc = sub(origin, self.center)
        b = dot(oc, direction)
        c = dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - c
        if discriminant < 0.0:
            return None
        root = math.sqrt(discriminant)
        t0 = -b - root
        t1 = -b + root
        if t0 >= t_min and t0 <= t_max:
            t = t0
        elif t1 >= t_min and t1 <= t_max:
            t = t1
        else:
            return None
        point = (
            origin[0] + direction[0] * t,
            origin[1] + direction[1] * t,
            origin[2] + direction[2] * t,
        )
        normal = normalize(sub(point, self.center))
        return Hit(t, point, normal, dot(direction, normal) < 0.0, self)

    def sample_visible_point(self, receiver: Vec3, u: float, v: float) -> Tuple[Vec3, float]:
        """Sample the sphere's receiver-visible cap uniformly by area.

        Returns the point and the visible cap area. Uniform area sampling makes
        the direct-lighting estimator's area pdf equal 1 / cap_area.
        """

        toward_receiver = sub(receiver, self.center)
        distance = length(toward_receiver)
        if distance <= self.radius + 1.0e-9:
            # Degenerate receiver-inside-light case: use the full sphere area.
            cos_alpha = -1.0
            visible_area = 4.0 * math.pi * self.radius * self.radius
        else:
            axis = mul(toward_receiver, 1.0 / distance)
            cos_alpha = self.radius / distance
            cap_height = self.radius * (1.0 - cos_alpha)
            visible_area = 2.0 * math.pi * self.radius * cap_height
            z = cos_alpha + u * (1.0 - cos_alpha)
            phi = 2.0 * math.pi * v
            radial = math.sqrt(max(0.0, 1.0 - z * z))
            tangent = (-axis[1], axis[0], 0.0)
            tangent_len = length(tangent)
            if tangent_len <= 1.0e-12:
                tangent = (1.0, 0.0, 0.0)
            else:
                tangent = mul(tangent, 1.0 / tangent_len)
            bitangent = cross(axis, tangent)
            normal = (
                axis[0] * z + tangent[0] * radial * math.cos(phi)
                + bitangent[0] * radial * math.sin(phi),
                axis[1] * z + tangent[1] * radial * math.cos(phi)
                + bitangent[1] * radial * math.sin(phi),
                axis[2] * z + tangent[2] * radial * math.cos(phi)
                + bitangent[2] * radial * math.sin(phi),
            )
            point = add(self.center, mul(normal, self.radius))
            return point, visible_area

        phi = 2.0 * math.pi * v
        z = -1.0 + 2.0 * u
        radial = math.sqrt(max(0.0, 1.0 - z * z))
        normal = (radial * math.cos(phi), z, radial * math.sin(phi))
        return add(self.center, mul(normal, self.radius)), visible_area


Emitter = (RectLight, SphereLight)


def nearest_geometry(
    objects: Sequence[Any], origin: Vec3, direction: Vec3, t_min: float, t_max: float
) -> Optional[Hit]:
    nearest: Optional[Hit] = None
    closest = t_max
    for obj in objects:
        hit = obj.intersect(origin, direction, t_min, closest)
        if hit is not None and hit.t < closest:
            closest = hit.t
            nearest = hit
    return nearest


def any_occlusion(
    geometry: Sequence[Any], origin: Vec3, direction: Vec3, max_distance: float
) -> bool:
    limit = max_distance - 1.0e-9
    for obj in geometry:
        hit = obj.intersect(origin, direction, 1.0e-9, limit)
        if hit is not None:
            return True
    return False

