from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .clustering import VirtualPointLight
from .config import Material, KNOWN_MATERIAL_TYPES
from .mathutil import normalize


@dataclass(frozen=True)
class MaterialSample:
    albedo: np.ndarray
    diffuse: np.ndarray
    specular: float
    shininess: float
    roughness: float
    type_index: int


def material_sample(material: Material) -> MaterialSample:
    type_index = KNOWN_MATERIAL_TYPES.get(material.type, 0)
    return MaterialSample(
        albedo=np.asarray(material.albedo, dtype=np.float64),
        diffuse=np.asarray(material.albedo, dtype=np.float64) / math.pi,
        specular=material.specular,
        shininess=material.shininess,
        roughness=material.roughness,
        type_index=type_index,
    )


def oren_nayar_factor(
    normal: np.ndarray,
    view_dir: np.ndarray,
    light_dir: np.ndarray,
    roughness: float,
) -> float:
    n_dot_l = max(0.0, float(np.dot(normal, light_dir)))
    n_dot_v = max(0.0, float(np.dot(normal, view_dir)))
    if n_dot_l <= 0.0 or n_dot_v <= 0.0:
        return 0.0
    sigma = max(1.0e-4, roughness) * (math.pi * 0.5)
    sigma_sq = sigma * sigma
    a = 1.0 - 0.5 * sigma_sq / (sigma_sq + 0.33)
    b = 0.45 * sigma_sq / (sigma_sq + 0.09)
    light_tangent = normalize(light_dir - normal * n_dot_l, fallback=(1.0, 0.0, 0.0))
    view_tangent = normalize(view_dir - normal * n_dot_v, fallback=(1.0, 0.0, 0.0))
    cosine_delta = max(0.0, float(np.dot(light_tangent, view_tangent)))
    sine_alpha = math.sqrt(max(0.0, 1.0 - n_dot_l * n_dot_l))
    sine_beta = math.sqrt(max(0.0, 1.0 - n_dot_v * n_dot_v))
    tangent_term = max(sine_alpha, sine_beta)
    return n_dot_l * (a + b * cosine_delta * tangent_term)


def evaluate_brdf(
    material: Material,
    normal_vector: np.ndarray,
    view_vector: np.ndarray,
    light_vector: np.ndarray,
) -> np.ndarray:
    normal = normalize(normal_vector)
    view_dir = normalize(view_vector)
    light_dir = normalize(light_vector)
    n_dot_l = max(0.0, float(np.dot(normal, light_dir)))
    if n_dot_l <= 0.0:
        return np.zeros(3, dtype=np.float64)
    sample = material_sample(material)
    if sample.type_index == 1:
        factor = oren_nayar_factor(normal, view_dir, light_dir, sample.roughness)
        return sample.albedo * factor / math.pi
    diffuse = sample.diffuse * n_dot_l
    if sample.type_index == 2:
        half_vector = normalize(view_dir + light_dir)
        n_dot_h = max(0.0, float(np.dot(normal, half_vector)))
        n_dot_v = max(1.0e-4, float(np.dot(normal, view_dir)))
        specular_norm = (sample.shininess + 8.0) / (8.0 * math.pi)
        specular = sample.specular * specular_norm * (n_dot_h**sample.shininess)
        return diffuse + specular * n_dot_l / max(1.0e-4, n_dot_v)
    return diffuse


def smooth_range_window(distance: float, light_range: float) -> float:
    if distance >= light_range:
        return 0.0
    normalized = distance / max(1.0e-5, light_range)
    return (1.0 - normalized**4) ** 2


def vpl_contribution(
    vpl: VirtualPointLight,
    material: Material,
    position: np.ndarray,
    normal: np.ndarray,
    view_dir: np.ndarray,
    *,
    visibility: float = 1.0,
) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    delta = np.asarray(vpl.position, dtype=np.float64) - position
    distance_sq = float(np.dot(delta, delta))
    distance = math.sqrt(distance_sq)
    if distance <= 1.0e-6 or distance >= vpl.range:
        return np.zeros(3, dtype=np.float64)
    light_dir = delta / distance
    if vpl.is_spot:
        cone = float(
            np.dot(
                normalize(-delta),
                normalize(vpl.direction),
            )
        )
        outer = vpl.cone_cos_outer
        inner = min(0.9999, outer + 0.08)
        if cone <= outer:
            return np.zeros(3, dtype=np.float64)
        spot = (cone - outer) / max(1.0e-5, inner - outer)
    else:
        spot = 1.0
    brdf = evaluate_brdf(material, normal, view_dir, light_dir)
    radiance = np.asarray(vpl.color, dtype=np.float64) * vpl.intensity
    attenuation = smooth_range_window(distance, vpl.range) / (distance_sq + 0.03)
    return brdf * radiance * attenuation * spot * visibility


def accumulate_reference(
    vpls: list[VirtualPointLight],
    material: Material,
    position: np.ndarray,
    normal: np.ndarray,
    view_dir: np.ndarray,
    visibilities: list[float] | None = None,
) -> np.ndarray:
    visibilities = visibilities or [1.0] * len(vpls)
    result = np.zeros(3, dtype=np.float64)
    for vpl, visibility in zip(vpls, visibilities):
        result += vpl_contribution(
            vpl,
            material,
            position,
            normal,
            view_dir,
            visibility=visibility,
        )
    return result

