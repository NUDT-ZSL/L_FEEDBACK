from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from .constants import EPSILON, SHADOW_EPSILON
from .sampling import sample_offsets
from .spec import FrameState, RenderSpec


class RenderError(RuntimeError):
    pass


def _normalize(vector: np.ndarray, label: str) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length <= EPSILON:
        raise RenderError(f"{label} is degenerate")
    return vector / length


class RayScene:
    def __init__(self, spec: RenderSpec) -> None:
        scene = spec.scene
        self.width = spec.width
        self.height = spec.height
        self.fov_y_deg = scene.camera.fov_y_deg
        self.centers = np.asarray(
            [sphere.center for sphere in scene.spheres], dtype=np.float64
        )
        self.radii = np.asarray(
            [sphere.radius for sphere in scene.spheres], dtype=np.float64
        )
        self.albedo = np.asarray(
            [sphere.albedo for sphere in scene.spheres], dtype=np.float64
        )
        self.specular = np.asarray(
            [sphere.specular for sphere in scene.spheres], dtype=np.float64
        )
        self.shininess = np.asarray(
            [sphere.shininess for sphere in scene.spheres], dtype=np.float64
        )
        self.light_position = np.asarray(scene.light.position, dtype=np.float64)
        self.light_color = np.asarray(scene.light.color, dtype=np.float64)
        self.light_intensity = float(scene.light.intensity)
        self.ambient = np.asarray(scene.ambient, dtype=np.float64)
        self.background = np.asarray(scene.background, dtype=np.float64)
        self.sampling = spec.sampling
        self.samples_per_axis = spec.sampling.samples_per_axis
        self.jitter = spec.sampling.jitter

    def render_frame(self, frame: FrameState) -> np.ndarray:
        position = np.asarray(frame.position, dtype=np.float64)
        target = np.asarray(frame.target, dtype=np.float64)
        up = np.asarray(frame.up, dtype=np.float64)
        forward = _normalize(target - position, f"frame {frame.frame} camera target")
        right = _normalize(
            np.cross(forward, up), f"frame {frame.frame} camera up"
        )
        true_up = np.cross(right, forward)

        count = self.samples_per_axis
        offsets = sample_offsets(
            self.sampling, self.width, self.height, frame.seed
        )
        pixel_x = np.arange(self.width, dtype=np.float64)[None, :, None, None]
        pixel_y = np.arange(self.height, dtype=np.float64)[:, None, None, None]
        ndc_x = ((pixel_x + offsets[..., 0]) / self.width) * 2.0 - 1.0
        ndc_y = 1.0 - ((pixel_y + offsets[..., 1]) / self.height) * 2.0

        aspect = self.width / self.height
        half_height = math.tan(math.radians(self.fov_y_deg) * 0.5)
        half_width = half_height * aspect
        directions = (
            forward[None, None, None, None, :]
            + ndc_x[..., None] * half_width * right[None, None, None, None, :]
            + ndc_y[..., None] * half_height * true_up[None, None, None, None, :]
        )
        directions = directions.reshape(-1, 3)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        ray_count = directions.shape[0]
        best_t = np.full(ray_count, np.inf, dtype=np.float64)
        hit_index = np.full(ray_count, -1, dtype=np.int32)

        for index in range(self.centers.shape[0]):
            center = self.centers[index]
            radius = self.radii[index]
            offset = position - center
            half_b = directions @ offset
            c_term = float(offset @ offset) - radius * radius
            discriminant = half_b * half_b - c_term
            valid = discriminant >= 0.0
            root = np.sqrt(np.maximum(discriminant, 0.0))
            near_t = -half_b - root
            far_t = -half_b + root
            candidate = np.full(ray_count, np.inf, dtype=np.float64)
            use_near = near_t > EPSILON
            candidate[use_near] = near_t[use_near]
            use_far = (~use_near) & (far_t > EPSILON)
            candidate[use_far] = far_t[use_far]
            better = valid & (candidate < best_t)
            best_t[better] = candidate[better]
            hit_index[better] = index

        hit = hit_index >= 0
        hit_points = np.zeros((ray_count, 3), dtype=np.float64)
        hit_points[hit] = position + best_t[hit, None] * directions[hit]

        normals = np.zeros((ray_count, 3), dtype=np.float64)
        albedo = np.zeros((ray_count, 3), dtype=np.float64)
        specular = np.zeros(ray_count, dtype=np.float64)
        shininess = np.ones(ray_count, dtype=np.float64)
        for index in range(self.centers.shape[0]):
            selected = hit_index == index
            if not np.any(selected):
                continue
            normals[selected] = (
                hit_points[selected] - self.centers[index]
            ) / self.radii[index]
            albedo[selected] = self.albedo[index]
            specular[selected] = self.specular[index]
            shininess[selected] = self.shininess[index]

        to_light = self.light_position[None, :] - hit_points
        light_distance = np.linalg.norm(to_light, axis=1)
        light_direction = np.zeros_like(to_light)
        nonzero = light_distance > EPSILON
        light_direction[nonzero] = to_light[nonzero] / light_distance[nonzero, None]

        visible = hit.copy()
        shadow_origins = hit_points + normals * SHADOW_EPSILON
        for index in range(self.centers.shape[0]):
            offset = shadow_origins - self.centers[index]
            half_b = np.einsum("ij,ij->i", light_direction, offset)
            c_term = np.einsum("ij,ij->i", offset, offset) - self.radii[index] ** 2
            discriminant = half_b * half_b - c_term
            valid = discriminant >= 0.0
            root = np.sqrt(np.maximum(discriminant, 0.0))
            near_t = -half_b - root
            far_t = -half_b + root
            shadow_t = np.full(ray_count, np.inf, dtype=np.float64)
            use_near = near_t > EPSILON
            shadow_t[use_near] = near_t[use_near]
            use_far = (~use_near) & (far_t > EPSILON)
            shadow_t[use_far] = far_t[use_far]
            blocked = (
                hit
                & valid
                & (shadow_t > EPSILON)
                & (shadow_t < light_distance - SHADOW_EPSILON)
            )
            visible[blocked] = False

        n_dot_l = np.einsum("ij,ij->i", normals, light_direction)
        n_dot_l = np.clip(n_dot_l, 0.0, 1.0)
        view = -directions
        half_vector = light_direction + view
        half_length = np.linalg.norm(half_vector, axis=1)
        valid_half = half_length > EPSILON
        half_vector[valid_half] /= half_length[valid_half, None]
        n_dot_h = np.clip(
            np.einsum("ij,ij->i", normals, half_vector), 0.0, 1.0
        )

        ambient = albedo * self.ambient[None, :]
        lighting = n_dot_l[:, None] * self.light_intensity
        diffuse = albedo * self.light_color[None, :] * lighting
        highlight = (
            np.power(n_dot_h, shininess)
            * specular
            * self.light_intensity
        )[:, None] * self.light_color[None, :]
        shaded = ambient + (diffuse + highlight) * visible[:, None]

        samples = np.broadcast_to(self.background, (ray_count, 3)).copy()
        samples[hit] = shaded[hit]
        linear = samples.reshape(
            self.height, self.width, count, count, 3
        ).mean(axis=(2, 3))
        return linear_to_srgb8(linear)


def linear_to_srgb8(linear: np.ndarray) -> np.ndarray:
    clamped = np.clip(linear, 0.0, 1.0)
    encoded = np.where(
        clamped <= 0.0031308,
        clamped * 12.92,
        1.055 * np.power(clamped, 1.0 / 2.4) - 0.055,
    )
    encoded = np.clip(encoded, 0.0, 1.0)
    return np.floor(encoded * 255.0 + 0.5).astype(np.uint8)


def save_png(image: np.ndarray, path: str | Path) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise RenderError("PNG image must be an HxWx3 uint8 array")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(
        target, format="PNG", compress_level=6, optimize=False
    )

