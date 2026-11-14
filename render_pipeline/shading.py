from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import GBuffer
from .material import MaterialMaps, sample_bilinear
from .scene import LightSpec
from .tonemap import linear_to_srgb8, tonemap_linear


@dataclass(frozen=True)
class ShadeResult:
    linear_irradiance: np.ndarray
    linear_color: np.ndarray
    mapped_linear: np.ndarray
    srgb8: np.ndarray
    albedo: np.ndarray
    roughness: np.ndarray
    metallic: np.ndarray
    reflectance_peak: float


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.maximum(norm, 1e-15)


def integrate_lighting(
    normals: np.ndarray,
    view_direction: tuple[float, float, float],
    albedo_linear: np.ndarray,
    roughness: np.ndarray,
    metallic: np.ndarray,
    preset: LightSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return diffuse/specular reflectance, incident irradiance and linear color.

    The direct-light response uses a Lambert lobe and a Smith-GGX/Schlick
    microfacet lobe. The raw specular response is bounded with ``x/(1+x)`` and
    lobe weights are partitioned so ``diffuse_weight + specular_weight <= 1``.
    A final reflectance normalization is a hard conservation guard for the
    approximate model.
    """

    normals = _normalize(np.asarray(normals, dtype=np.float64))
    view = np.asarray(view_direction, dtype=np.float64)
    view = view / max(float(np.linalg.norm(view)), 1e-15)
    light = np.asarray(preset.key_direction, dtype=np.float64)
    light = light / max(float(np.linalg.norm(light)), 1e-15)

    n_dot_l = np.clip(normals @ light, 0.0, 1.0)
    n_dot_v = np.clip(normals @ view, 1e-4, 1.0)
    half_vector = _normalize((light + view)[None, :])[0]
    n_dot_h = np.clip(normals @ half_vector, 0.0, 1.0)
    v_dot_h = np.clip(view @ half_vector, 0.0, 1.0)

    perceptual_roughness = np.clip(roughness, 0.02, 1.0)
    alpha = perceptual_roughness * perceptual_roughness
    alpha_squared = alpha * alpha
    denominator = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0
    distribution = alpha_squared / (np.pi * np.maximum(denominator * denominator, 1e-15))

    k = alpha * 0.5
    geometry_v = n_dot_v / np.maximum(n_dot_v * (1.0 - k) + k, 1e-15)
    geometry_l = n_dot_l / np.maximum(n_dot_l * (1.0 - k) + k, 1e-15)
    geometry = geometry_v * geometry_l

    metal = np.clip(metallic, 0.0, 1.0)[..., None]
    dielectric_f0 = np.full_like(albedo_linear, 0.04)
    f0 = dielectric_f0 * (1.0 - metal) + albedo_linear * metal
    fresnel = f0 + (1.0 - f0) * np.power(1.0 - v_dot_h[..., None], 5.0)

    specular_raw = distribution[..., None] * geometry[..., None] * fresnel
    specular_raw /= np.maximum(4.0 * n_dot_l[..., None] * n_dot_v[..., None], 1e-15)
    specular_bounded = specular_raw / (1.0 + specular_raw)

    f_avg = dielectric_f0 * (1.0 - metal) + np.mean(albedo_linear, axis=-1, keepdims=True) * metal
    diffuse_weight = (1.0 - metal) * (1.0 - f_avg)
    specular_weight = 1.0 - diffuse_weight

    diffuse_reflectance = diffuse_weight * albedo_linear
    specular_reflectance = specular_weight * specular_bounded
    reflectance = diffuse_reflectance + specular_reflectance
    peak = np.max(np.concatenate((reflectance.reshape(-1, 3), diffuse_weight + specular_weight), axis=0), axis=0)
    reflectance_peak = float(np.max(peak))
    if reflectance_peak > 1.0:
        reflectance /= reflectance_peak

    key_irradiance = np.asarray(preset.key_irradiance, dtype=np.float64)
    ambient_irradiance = np.asarray(preset.ambient_irradiance, dtype=np.float64)
    irradiance = ambient_irradiance[None, :] + key_irradiance[None, :] * n_dot_l[..., None]
    linear_color = ambient_irradiance[None, :] * reflectance + key_irradiance[None, :] * n_dot_l[..., None] * reflectance
    return reflectance, irradiance, linear_color, diffuse_weight + specular_weight, specular_bounded


def shade_gbuffer(gbuffer: GBuffer, material: MaterialMaps, preset: LightSpec) -> ShadeResult:
    height, width = gbuffer.coverage.shape
    linear_irradiance = np.zeros((height, width, 3), dtype=np.float64)
    linear_color = np.zeros((height, width, 3), dtype=np.float64)
    mapped_linear = np.zeros((height, width, 3), dtype=np.float64)
    srgb8 = np.zeros((height, width, 3), dtype=np.uint8)
    albedo = np.zeros((height, width, 3), dtype=np.float64)
    roughness = np.zeros((height, width), dtype=np.float64)
    metallic = np.zeros((height, width), dtype=np.float64)

    covered = gbuffer.coverage
    if not np.any(covered):
        return ShadeResult(
            linear_irradiance=linear_irradiance,
            linear_color=linear_color,
            mapped_linear=mapped_linear,
            srgb8=srgb8,
            albedo=albedo,
            roughness=roughness,
            metallic=metallic,
            reflectance_peak=0.0,
        )
    uv = gbuffer.uv[covered]
    albedo[covered] = sample_bilinear(material.albedo_linear, uv)
    roughness[covered] = sample_bilinear(material.roughness, uv)
    metallic[covered] = sample_bilinear(material.metallic, uv)

    normals = gbuffer.normal[covered]
    reflectance, irradiance, color, weights, _ = integrate_lighting(
        normals,
        gbuffer.camera_view_direction,
        albedo[covered],
        roughness[covered],
        metallic[covered],
        preset,
    )
    mapped = tonemap_linear(color, preset)
    encoded = linear_to_srgb8(mapped)
    linear_irradiance[covered] = irradiance
    linear_color[covered] = color
    mapped_linear[covered] = mapped
    srgb8[covered] = encoded
    peak = float(max(np.max(reflectance), np.max(weights)))
    return ShadeResult(
        linear_irradiance=linear_irradiance,
        linear_color=linear_color,
        mapped_linear=mapped_linear,
        srgb8=srgb8,
        albedo=albedo,
        roughness=roughness,
        metallic=metallic,
        reflectance_peak=peak,
    )
