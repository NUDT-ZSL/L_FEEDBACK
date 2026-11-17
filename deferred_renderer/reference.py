from __future__ import annotations

from typing import Any

import numpy as np

from .math3d import reconstruct_world
from .scene import Scene

PI = np.float64(np.pi)


def fullscreen_shading_reference(
    scene: Scene,
    albedo_roughness: np.ndarray,
    normal_side: np.ndarray,
    linear_depth: np.ndarray,
    material_id: np.ndarray,
) -> np.ndarray:
    """CPU per-pixel deferred shading reference using all lights."""

    depth = np.asarray(linear_depth, dtype=np.float64)
    height, width = depth.shape
    world = reconstruct_world(depth, scene.inv_projection, scene.camera.inv_view)
    albedo = np.asarray(albedo_roughness[..., :3], dtype=np.float64)
    roughness = np.asarray(albedo_roughness[..., 3], dtype=np.float64)
    normal = np.asarray(normal_side[..., :3], dtype=np.float64)
    normal_length = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.maximum(normal_length, 1e-8)
    side = np.asarray(normal_side[..., 3], dtype=np.float64)
    ids = np.asarray(material_id, dtype=np.int64)

    camera_position = np.asarray(scene.camera.position, dtype=np.float64)
    view_vector = camera_position[None, None, :] - world
    view_direction = view_vector / np.maximum(
        np.linalg.norm(view_vector, axis=-1, keepdims=True),
        1e-8,
    )
    flip = np.sum(normal * view_direction, axis=-1) < 0.0
    normal[flip] *= -1.0

    roughness = np.clip(roughness, 0.04, 1.0)
    back_scale = np.ones(33, dtype=np.float64)
    back_bias = np.zeros(33, dtype=np.float64)
    for material in scene.materials:
        back_scale[material.id] = material.back_roughness_scale
        back_bias[material.id] = material.back_roughness_bias
    safe_ids = np.clip(ids, 0, 32)
    back_mask = side < 0.0
    roughness = np.where(
        back_mask,
        np.clip(
            roughness * back_scale[safe_ids] + back_bias[safe_ids],
            0.04,
            1.0,
        ),
        roughness,
    )

    background = np.asarray(scene.render.background, dtype=np.float64)
    color = albedo * background[None, None, :] * scene.render.ambient_strength
    n_dot_v = np.maximum(np.sum(normal * view_direction, axis=-1), 1e-4)
    f0 = np.full(3, 0.04, dtype=np.float64)
    alpha = np.maximum(roughness * roughness, 0.0025)
    alpha_sq = alpha * alpha

    for light in scene.lights:
        light_position = np.asarray(light.position, dtype=np.float64)
        to_light = light_position[None, None, :] - world
        distance_sq = np.sum(to_light * to_light, axis=-1)
        distance = np.sqrt(np.maximum(distance_sq, 1e-8))
        light_direction = to_light / distance[..., None]
        n_dot_l = np.maximum(np.sum(normal * light_direction, axis=-1), 0.0)
        halfway = view_direction + light_direction
        halfway = halfway / np.maximum(
            np.linalg.norm(halfway, axis=-1, keepdims=True),
            1e-8,
        )
        n_dot_h = np.maximum(np.sum(normal * halfway, axis=-1), 0.0)
        v_dot_h = np.maximum(np.sum(view_direction * halfway, axis=-1), 0.0)
        fresnel_scalar = np.power(1.0 - v_dot_h, 5.0)
        fresnel = f0[None, None, :] + (
            1.0 - f0[None, None, :]
        ) * fresnel_scalar[..., None]

        denominator = n_dot_h * n_dot_h * (alpha_sq - 1.0) + 1.0
        distribution = alpha_sq / np.maximum(PI * denominator * denominator, 1e-7)
        k = alpha * 0.5
        geometry = (
            n_dot_l / np.maximum(n_dot_l * (1.0 - k) + k, 1e-7)
        ) * (
            n_dot_v / np.maximum(n_dot_v * (1.0 - k) + k, 1e-7)
        )
        specular = (
            distribution[..., None]
            * geometry[..., None]
            * fresnel
            / np.maximum(4.0 * n_dot_l * n_dot_v, 1e-5)[..., None]
        )
        diffuse = (1.0 - fresnel) * albedo / PI

        attenuation_fraction = np.clip(
            1.0 - distance / light.radius,
            0.0,
            1.0,
        )
        attenuation = attenuation_fraction * attenuation_fraction
        radiance = (
            np.asarray(light.color, dtype=np.float64)[None, None, :]
            * light.intensity
            * attenuation[..., None]
            / np.maximum(distance_sq, 0.02)[..., None]
        )
        contribution = (diffuse + specular) * radiance * n_dot_l[..., None]
        color += np.where(
            (distance_sq <= light.radius * light.radius)[..., None],
            contribution,
            0.0,
        )

    valid = depth > 0.0
    color[~valid] = background
    return np.asarray(color, dtype=np.float32)


def compare_reference(
    rendered: np.ndarray,
    reference: np.ndarray,
    linear_depth: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    rendered_rgb = np.asarray(rendered[..., :3], dtype=np.float64)
    reference_rgb = np.asarray(reference[..., :3], dtype=np.float64)
    absolute = np.abs(rendered_rgb - reference_rgb)
    valid = np.asarray(linear_depth) > 0.0
    valid_absolute = absolute[valid] if valid.any() else absolute
    max_abs = float(valid_absolute.max()) if valid_absolute.size else 0.0
    mean_abs = float(valid_absolute.mean()) if valid_absolute.size else 0.0
    return {
        "threshold_abs": threshold,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "compared_pixels": int(valid.sum()) if valid.any() else int(valid.size),
        "passed": bool(max_abs <= threshold),
        "algorithm": "CPU full-screen all-light BRDF against tiled GPU HDR",
    }

