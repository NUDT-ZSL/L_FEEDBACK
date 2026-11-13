"""Small deterministic vector and projection helpers used by the renderer."""

from __future__ import annotations

from dataclasses import dataclass
import math


Vec3 = tuple[float, float, float]


def add(left: Vec3, right: Vec3) -> Vec3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def sub(left: Vec3, right: Vec3) -> Vec3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def scale(vector: Vec3, factor: float) -> Vec3:
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def dot(left: Vec3, right: Vec3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def cross(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def length(vector: Vec3) -> float:
    return math.sqrt(dot(vector, vector))


def normalize(vector: Vec3, fallback: Vec3 = (0.0, 0.0, 1.0)) -> Vec3:
    magnitude = length(vector)
    if magnitude <= 1.0e-15:
        return fallback
    inverse = 1.0 / magnitude
    return (vector[0] * inverse, vector[1] * inverse, vector[2] * inverse)


def lerp(left: Vec3, right: Vec3, amount: float) -> Vec3:
    inverse = 1.0 - amount
    return (
        left[0] * inverse + right[0] * amount,
        left[1] * inverse + right[1] * amount,
        left[2] * inverse + right[2] * amount,
    )


def subtract_product(origin: Vec3, direction: Vec3, amount: float) -> Vec3:
    return (
        origin[0] - direction[0] * amount,
        origin[1] - direction[1] * amount,
        origin[2] - direction[2] * amount,
    )


@dataclass(frozen=True)
class Basis:
    eye: Vec3
    right: Vec3
    up: Vec3
    forward: Vec3

    def to_camera(self, point: Vec3) -> Vec3:
        relative = sub(point, self.eye)
        return (dot(relative, self.right), dot(relative, self.up), dot(relative, self.forward))

    def from_camera(self, point: Vec3) -> Vec3:
        return (
            self.eye[0] + self.right[0] * point[0] + self.up[0] * point[1] + self.forward[0] * point[2],
            self.eye[1] + self.right[1] * point[0] + self.up[1] * point[1] + self.forward[1] * point[2],
            self.eye[2] + self.right[2] * point[0] + self.up[2] * point[1] + self.forward[2] * point[2],
        )


def make_basis(eye: Vec3, target: Vec3, up_hint: Vec3) -> Basis:
    forward = normalize(sub(target, eye))
    if abs(dot(forward, up_hint)) > 0.999:
        up_hint = (0.0, 0.0, 1.0) if abs(forward[2]) < 0.9 else (0.0, 1.0, 0.0)
    right = normalize(cross(forward, up_hint))
    up = normalize(cross(right, forward))
    return Basis(eye=eye, right=right, up=up, forward=forward)


def make_basis_from_forward(eye: Vec3, forward: Vec3, up_hint: Vec3) -> Basis:
    return make_basis(eye=eye, target=add(eye, forward), up_hint=up_hint)


@dataclass(frozen=True)
class Projection:
    width: int
    height: int
    basis: Basis
    projection_kind: str
    near: float
    far: float
    tan_half_fov_y: float = 1.0
    aspect: float = 1.0
    ortho_half_width: float = 1.0
    ortho_half_height: float = 1.0

    def camera_to_screen_depth(self, camera_point: Vec3) -> tuple[float, float, float]:
        cam_x, cam_y, cam_z = camera_point
        if self.projection_kind == "orthographic":
            ndc_x = cam_x / self.ortho_half_width
            ndc_y = cam_y / self.ortho_half_height
        else:
            denominator = cam_z * self.tan_half_fov_y
            ndc_x = cam_x / (denominator * self.aspect)
            ndc_y = cam_y / denominator
        screen_x = (ndc_x * 0.5 + 0.5) * float(self.width)
        screen_y = (0.5 - ndc_y * 0.5) * float(self.height)
        return screen_x, screen_y, cam_z

    def camera_to_world(self, camera_point: Vec3) -> Vec3:
        return self.basis.from_camera(camera_point)

    def screen_to_camera(self, screen_x: float, screen_y: float, depth: float) -> Vec3:
        ndc_x = screen_x / float(self.width) * 2.0 - 1.0
        ndc_y = 1.0 - screen_y / float(self.height) * 2.0
        if self.projection_kind == "orthographic":
            return (
                ndc_x * self.ortho_half_width,
                ndc_y * self.ortho_half_height,
                depth,
            )
        return (
            ndc_x * depth * self.tan_half_fov_y * self.aspect,
            ndc_y * depth * self.tan_half_fov_y,
            depth,
        )

    def screen_to_world(self, screen_x: float, screen_y: float, depth: float) -> Vec3:
        return self.camera_to_world(self.screen_to_camera(screen_x, screen_y, depth))


def aabb_corners(bounds_min: Vec3, bounds_max: Vec3) -> tuple[Vec3, ...]:
    return (
        (bounds_min[0], bounds_min[1], bounds_min[2]),
        (bounds_max[0], bounds_min[1], bounds_min[2]),
        (bounds_min[0], bounds_max[1], bounds_min[2]),
        (bounds_max[0], bounds_max[1], bounds_min[2]),
        (bounds_min[0], bounds_min[1], bounds_max[2]),
        (bounds_max[0], bounds_min[1], bounds_max[2]),
        (bounds_min[0], bounds_max[1], bounds_max[2]),
        (bounds_max[0], bounds_max[1], bounds_max[2]),
    )
