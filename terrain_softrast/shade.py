from __future__ import annotations

import math

import numpy as np

from .config import Camera, Material, PointLight, RenderSettings
from .math3d import normalize


def _perturb_normal(
    normal: np.ndarray,
    world: np.ndarray,
    material: Material,
) -> np.ndarray:
    x_axis = np.broadcast_to(np.asarray((1.0, 0.0, 0.0)), normal.shape)
    tangent = x_axis - normal * np.sum(x_axis * normal, axis=1, keepdims=True)
    tangent_length = np.linalg.norm(tangent, axis=1)
    fallback = np.cross(normal, np.asarray((0.0, 0.0, 1.0)))
    fallback = fallback / np.maximum(np.linalg.norm(fallback, axis=1, keepdims=True), 1e-15)
    tangent = np.where(tangent_length[:, None] > 1e-7, tangent / np.maximum(tangent_length[:, None], 1e-15), fallback)
    bitangent = np.cross(normal, tangent)
    scale = material.detail_scale
    phase = material.detail_phase
    bump_u = material.normal_strength * 0.32 * np.cos(
        scale * (world[:, 0] * 2.1 + world[:, 2] * 0.7) + phase
    )
    bump_v = material.normal_strength * 0.32 * np.sin(
        scale * (world[:, 2] * 2.3 - world[:, 0] * 0.45) + phase * 1.31
    )
    perturbed = normal + tangent * bump_u[:, None] + bitangent * bump_v[:, None]
    length = np.linalg.norm(perturbed, axis=1, keepdims=True)
    output = perturbed / np.maximum(length, 1e-15)
    return np.where(length > 1e-12, output, normal)


def _stable_half_vector(light: np.ndarray, view: np.ndarray, fallback_normal: np.ndarray) -> np.ndarray:
    total = light + view
    length = np.linalg.norm(total, axis=1, keepdims=True)
    good = length[:, 0] > 1e-7
    half = total / np.maximum(length, 1e-15)
    return np.where(good[:, None], half, fallback_normal)


def _attenuation(distance: np.ndarray, light_range: float) -> np.ndarray:
    normalized = np.clip(distance / light_range, 0.0, 1.0)
    window = np.maximum(0.0, 1.0 - normalized ** 4) ** 2
    return window / (1.0 + (distance / light_range) ** 2)


def material_layer_colors(
    world: np.ndarray,
    normals: np.ndarray,
    visibility: np.ndarray,
    materials: tuple[Material, ...],
    camera: Camera,
    directional_light,
    point_lights: tuple[PointLight, ...],
    settings: RenderSettings,
    material_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-layer linear colors and blended layer normals.

    Lighting is intentionally evaluated per layer and then combined by the
    normalized material weights. This makes the full material blend exactly
    equal to the weighted sum of one-hot layer renders.
    """
    count = world.shape[0]
    if count == 0:
        return np.zeros((0, len(materials), 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    view = np.asarray(camera.eye, dtype=np.float64)[None, :] - world
    view_distance = np.linalg.norm(view, axis=1, keepdims=True)
    view = view / np.maximum(view_distance, 1e-15)
    light_direction = normalize(np.asarray(directional_light.direction, dtype=np.float64))
    light_to_surface = -light_direction
    light_color = np.asarray(directional_light.color, dtype=np.float64) * float(directional_light.intensity)
    ambient = np.asarray(settings.ambient, dtype=np.float64)
    layer_colors = np.empty((count, len(materials), 3), dtype=np.float64)
    if material_weights is None:
        blend_weights = np.full((count, len(materials)), 1.0 / len(materials), dtype=np.float64)
    else:
        blend_weights = np.asarray(material_weights, dtype=np.float64)
        blend_weights = blend_weights / np.maximum(
            np.sum(blend_weights, axis=1, keepdims=True),
            1e-15,
        )
    blended_normal = np.zeros((count, 3), dtype=np.float64)
    for index, material in enumerate(materials):
        perturbed = _perturb_normal(normals, world, material)
        blended_normal += blend_weights[:, index : index + 1] * perturbed
        albedo = np.asarray(material.albedo, dtype=np.float64)
        ndotl = np.maximum(np.sum(perturbed * light_to_surface[None, :], axis=1), 0.0)
        half = _stable_half_vector(
            np.broadcast_to(light_to_surface, perturbed.shape),
            view,
            perturbed,
        )
        ndoth = np.maximum(np.sum(perturbed * half, axis=1), 0.0)
        shininess = min(max(2.0 / max(material.roughness ** 4, 1e-6) - 2.0, 2.0), 512.0)
        specular_strength = 0.035 + (1.0 - material.roughness) * 0.16
        color = np.broadcast_to(albedo * ambient, (count, 3)).copy()
        color += (albedo[None, :] * ndotl[:, None] * light_color[None, :]) * visibility[:, None]
        color += (light_color[None, :] * specular_strength * (ndoth ** shininess)[:, None]) * visibility[:, None]
        for light in point_lights:
            delta = np.asarray(light.position, dtype=np.float64)[None, :] - world
            distance = np.linalg.norm(delta, axis=1)
            direction = delta / np.maximum(distance[:, None], 1e-15)
            attenuation = _attenuation(distance, light.range)
            point_ndotl = np.maximum(np.sum(perturbed * direction, axis=1), 0.0)
            point_half = _stable_half_vector(direction, view, perturbed)
            point_ndoth = np.maximum(np.sum(perturbed * point_half, axis=1), 0.0)
            point_color = np.asarray(light.color, dtype=np.float64) * float(light.intensity)
            energy = attenuation * point_ndotl
            color += albedo[None, :] * energy[:, None] * point_color[None, :]
            color += (
                point_color[None, :]
                * (attenuation * specular_strength * point_ndoth ** shininess)[:, None]
            )
        layer_colors[:, index, :] = color
    blended_normal = blended_normal / np.maximum(np.linalg.norm(blended_normal, axis=1, keepdims=True), 1e-15)
    return layer_colors, blended_normal


def shade_fragments(
    world: np.ndarray,
    normals: np.ndarray,
    weights: np.ndarray,
    visibility: np.ndarray,
    materials: tuple[Material, ...],
    camera: Camera,
    directional_light,
    point_lights: tuple[PointLight, ...],
    settings: RenderSettings,
) -> np.ndarray:
    if weights.shape[1] != len(materials):
        raise ValueError("材质图层数与权重通道数不一致")
    layer_colors, _ = material_layer_colors(
        world,
        normals,
        visibility,
        materials,
        camera,
        directional_light,
        point_lights,
        settings,
        weights,
    )
    normalized_weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-15)
    color = np.sum(layer_colors * normalized_weights[:, :, None], axis=1)
    return np.nan_to_num(color, nan=0.0, posinf=0.0, neginf=0.0)


def tone_map(color_linear: np.ndarray, exposure: float, gamma: float) -> np.ndarray:
    color = np.maximum(np.asarray(color_linear, dtype=np.float64), 0.0) * exposure
    mapped = 1.0 - np.exp(-color)
    mapped = np.power(np.clip(mapped, 0.0, 1.0), 1.0 / gamma)
    return mapped


def quantize_u8(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return np.floor(value * 255.0 + 0.5).astype(np.uint8)


def quantize_u16(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return np.floor(value * 65535.0 + 0.5).astype(np.uint16)
