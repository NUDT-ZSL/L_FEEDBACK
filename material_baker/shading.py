"""Cook-Torrance direct lighting and roughness-aware indirect injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import MaterialError
from .scene import Scene

PI = np.float64(np.pi)
EPSILON = np.float64(1e-7)


@dataclass
class ShadingComponents:
    diffuse: np.ndarray
    specular: np.ndarray
    indirect: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return (
            self.diffuse.astype(np.float64)
            + self.specular.astype(np.float64)
            + self.indirect.astype(np.float64)
        ).astype(np.float32)

    def packed(self) -> np.ndarray:
        return np.concatenate(
            [self.diffuse, self.specular, self.indirect], axis=2
        ).astype(np.float32)


def _validate_inputs(
    base_color: np.ndarray,
    roughness: np.ndarray,
    metallic: np.ndarray,
    object_normals: np.ndarray,
) -> None:
    if base_color.ndim != 3 or base_color.shape[2] != 3:
        raise MaterialError("E003", "base color chunk must have three channels")
    shape = base_color.shape[:2]
    if roughness.shape[:2] != shape or roughness.ndim != 3 or roughness.shape[2] != 1:
        raise MaterialError("E003", "roughness chunk must be single-channel")
    if metallic.shape[:2] != shape or metallic.ndim != 3 or metallic.shape[2] != 1:
        raise MaterialError("E003", "metallic chunk must be single-channel")
    if object_normals.shape != base_color.shape:
        raise MaterialError("E003", "object normal chunk dimensions are inconsistent")


def _direct_light(
    base: np.ndarray,
    roughness: np.ndarray,
    metallic: np.ndarray,
    normals: np.ndarray,
    view: np.ndarray,
    light: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(light["direction"], dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    radiance = np.asarray(light["color"], dtype=np.float64) * float(
        light["intensity"]
    )

    n = normals.astype(np.float64)
    v = view.astype(np.float64)
    l = np.broadcast_to(direction, n.shape)
    h = l + v
    h /= np.maximum(np.linalg.norm(h, axis=2, keepdims=True), EPSILON)

    n_dot_l = np.sum(n * l, axis=2, keepdims=True)
    n_dot_v = np.sum(n * v, axis=2, keepdims=True)
    n_dot_h = np.sum(n * h, axis=2, keepdims=True)
    h_dot_v = np.sum(h * v, axis=2, keepdims=True)
    n_dot_l = np.clip(n_dot_l, 0.0, 1.0)
    n_dot_v = np.clip(n_dot_v, EPSILON, 1.0)
    n_dot_h = np.clip(n_dot_h, 0.0, 1.0)

    roughness_value = np.clip(roughness.astype(np.float64), 0.015, 1.0)
    metallic_value = np.clip(metallic.astype(np.float64), 0.0, 1.0)
    alpha = roughness_value * roughness_value
    alpha2 = alpha * alpha

    denominator = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0
    distribution = alpha2 / (PI * np.maximum(denominator * denominator, EPSILON))

    k = (roughness_value + 1.0) ** 2 / 8.0
    geometry_l = n_dot_l / (n_dot_l * (1.0 - k) + k + EPSILON)
    geometry_v = n_dot_v / (n_dot_v * (1.0 - k) + k + EPSILON)
    geometry = geometry_l * geometry_v

    f0 = 0.04 * (1.0 - metallic_value) + base * metallic_value
    fresnel = f0 + (1.0 - f0) * np.power(1.0 - h_dot_v, 5)
    specular_brdf = distribution * geometry * fresnel / (
        4.0 * n_dot_l * n_dot_v + EPSILON
    )
    diffuse_weight = (1.0 - fresnel) * (1.0 - metallic_value)
    diffuse_brdf = diffuse_weight * base / PI

    contribution = n_dot_l * radiance[None, None, :]
    return (
        (diffuse_brdf * contribution).astype(np.float32),
        (specular_brdf * contribution).astype(np.float32),
    )


def shade_chunk(
    scene: Scene,
    base_color: np.ndarray,
    roughness: np.ndarray,
    metallic: np.ndarray,
    object_normals: np.ndarray,
    *,
    lights: list[dict[str, Any]] | None = None,
    include_environment: bool = True,
) -> ShadingComponents:
    _validate_inputs(base_color, roughness, metallic, object_normals)
    base = base_color.astype(np.float64)
    view = np.asarray(scene.baking["view_direction"], dtype=np.float64)
    view /= max(float(np.linalg.norm(view)), 1e-12)
    view = np.broadcast_to(view, object_normals.shape)

    selected_lights = scene.lights if lights is None else lights
    diffuse = np.zeros_like(base, dtype=np.float32)
    specular = np.zeros_like(base, dtype=np.float32)
    for light in selected_lights:
        light_diffuse, light_specular = _direct_light(
            base,
            roughness,
            metallic,
            object_normals,
            view,
            light,
        )
        # canonical_lights() fixed this order before shading.
        diffuse += light_diffuse
        specular += light_specular

    if include_environment:
        environment = np.asarray(
            scene.environment["color"], dtype=np.float64
        ) * float(scene.environment["intensity"])
        roughness_value = np.clip(roughness.astype(np.float64), 0.0, 1.0)
        metallic_value = np.clip(metallic.astype(np.float64), 0.0, 1.0)
        f0 = 0.04 * (1.0 - metallic_value) + base * metallic_value
        diffuse_weight = 0.35 + 0.65 * (1.0 - roughness_value)
        specular_weight = 0.15 + 0.85 * (1.0 - roughness_value) ** 2
        indirect = (
            base
            * (1.0 - metallic_value)
            * environment[None, None, :]
            * diffuse_weight
            + f0 * environment[None, None, :] * specular_weight
        ).astype(np.float32)
    else:
        indirect = np.zeros_like(base, dtype=np.float32)
    return ShadingComponents(diffuse=diffuse, specular=specular, indirect=indirect)
