"""Cook-Torrance direct lighting and hemisphere-environment approximation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DirectionalLight, HemisphereLight


PI = np.pi
DIELECTRIC_F0 = 0.04
DENOMINATOR_EPSILON = 1.0e-12


@dataclass
class LightingResult:
    irradiance: np.ndarray
    specular: np.ndarray
    final_color: np.ndarray
    direct_diffuse: np.ndarray
    direct_specular: np.ndarray
    environment_diffuse: np.ndarray
    environment_specular: np.ndarray


def _as_rgb(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float64)


def _safe_normalize(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    safe = np.maximum(lengths, DENOMINATOR_EPSILON)
    normalized = vectors / safe
    zero = lengths[..., 0] <= 1.0e-2
    if np.any(zero):
        normalized[zero] = (0.0, 0.0, 1.0)
    return normalized


def ggx_distribution(ndot_h: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_sq = alpha * alpha
    denominator_base = ndot_h * ndot_h * (alpha_sq - 1.0) + 1.0
    denominator = PI * (denominator_base * denominator_base + DENOMINATOR_EPSILON)
    return alpha_sq / denominator


def smith_g1_ggx(ndot_x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_sq = alpha * alpha
    root = np.sqrt(
        alpha_sq + (1.0 - alpha_sq) * ndot_x * ndot_x
    )
    return 2.0 * ndot_x / np.maximum(ndot_x + root, DENOMINATOR_EPSILON)


def fresnel_schlick(
    f0: np.ndarray, cos_theta: np.ndarray
) -> np.ndarray:
    factor = np.power(
        np.clip(1.0 - cos_theta, 0.0, 1.0), 5.0
    )
    return f0 + (1.0 - f0) * factor


def hemisphere_irradiance(
    normal: np.ndarray, sky_color: np.ndarray, ground_color: np.ndarray
) -> np.ndarray:
    """Analytic normal-facing hemisphere irradiance approximation."""

    up_weight = np.clip(0.5 * normal[..., 2] + 0.5, 0.0, 1.0)[..., None]
    return ground_color + (sky_color - ground_color) * up_weight


def _environment_radiance(
    direction: np.ndarray, sky_color: np.ndarray, ground_color: np.ndarray
) -> np.ndarray:
    up_weight = np.clip(0.5 * direction[..., 2] + 0.5, 0.0, 1.0)[..., None]
    return ground_color + (sky_color - ground_color) * up_weight


def prefiltered_environment(
    reflection: np.ndarray,
    roughness: np.ndarray,
    sky_color: np.ndarray,
    ground_color: np.ndarray,
) -> np.ndarray:
    """Roughness-dependent diffuse-like prefilter of the hemisphere gradient."""

    sharp = _environment_radiance(reflection, sky_color, ground_color)
    broad = 0.5 * (sky_color + ground_color)
    roughness_values = np.asarray(roughness, dtype=np.float64)
    if roughness_values.ndim == 2:
        roughness_values = roughness_values[:, :, None]
    blend = np.clip(roughness_values * roughness_values, 0.0, 1.0)
    return sharp * (1.0 - blend) + broad * blend


def evaluate_material(
    albedo: np.ndarray,
    roughness: np.ndarray,
    metallic: np.ndarray,
    normal: np.ndarray,
    view_direction: tuple[float, float, float],
    directional: DirectionalLight,
    hemisphere: HemisphereLight,
    *,
    alpha_epsilon: float = 1.0e-3,
) -> LightingResult:
    """Evaluate direct and environment lighting for every texel."""

    albedo = _as_rgb(albedo)
    roughness = np.clip(_as_rgb(roughness), 0.0, 1.0)
    metallic = np.clip(_as_rgb(metallic), 0.0, 1.0)
    if roughness.ndim == 2:
        roughness = roughness[:, :, None]
    if metallic.ndim == 2:
        metallic = metallic[:, :, None]
    if albedo.ndim != 3 or albedo.shape[2] != 3:
        raise ValueError("albedo must be HxWx3")
    if normal.shape != albedo.shape:
        raise ValueError("normal must match albedo dimensions")

    normal = _safe_normalize(normal)
    light_direction = np.asarray(directional.direction, dtype=np.float64)
    view = np.asarray(view_direction, dtype=np.float64)
    view = view / max(float(np.linalg.norm(view)), DENOMINATOR_EPSILON)
    half_vector = _safe_normalize(light_direction[None, None, :] + view[None, None, :])

    ndot_l = np.clip(np.sum(normal * light_direction, axis=-1), 0.0, 1.0)[..., None]
    ndot_v = np.clip(np.sum(normal * view, axis=-1), 0.0, 1.0)[..., None]
    ndot_h = np.clip(np.sum(normal * half_vector, axis=-1), 0.0, 1.0)[..., None]
    vdot_h = np.clip(np.sum(view * half_vector, axis=-1), 0.0, 1.0)[..., None]

    perceptual_roughness = np.clip(roughness, 0.0, 1.0)
    alpha = np.maximum(perceptual_roughness * perceptual_roughness, alpha_epsilon)
    distribution = ggx_distribution(ndot_h, alpha)
    geometry = smith_g1_ggx(ndot_v, alpha) * smith_g1_ggx(ndot_l, alpha)
    f0 = DIELECTRIC_F0 * (1.0 - metallic) + albedo * metallic
    fresnel = fresnel_schlick(f0, vdot_h)
    specular_brdf = (
        distribution * geometry * fresnel
        / np.maximum(4.0 * ndot_v * ndot_l, DENOMINATOR_EPSILON)
    )
    diffuse_brdf = (1.0 - fresnel) * (1.0 - metallic) * albedo / PI

    light_color = np.asarray(directional.color, dtype=np.float64)
    incident = light_color * directional.intensity * ndot_l
    direct_diffuse = diffuse_brdf * incident
    direct_specular = specular_brdf * incident

    combined_direct = direct_diffuse + direct_specular
    safety_scale = np.minimum(
        1.0,
        incident / np.maximum(combined_direct, DENOMINATOR_EPSILON),
    )
    direct_diffuse = direct_diffuse * safety_scale
    direct_specular = direct_specular * safety_scale

    sky = np.asarray(hemisphere.sky_color, dtype=np.float64)
    ground = np.asarray(hemisphere.ground_color, dtype=np.float64)
    ambient_irradiance = (
        hemisphere_irradiance(normal, sky, ground) * hemisphere.intensity
    )
    fresnel_ambient = fresnel_schlick(f0, ndot_v)
    environment_diffuse = (
        (1.0 - fresnel_ambient) * (1.0 - metallic) * albedo * ambient_irradiance
    )

    reflection = _safe_normalize(2.0 * ndot_v * normal - view[None, None, :])
    reflected_radiance = (
        prefiltered_environment(
            reflection, perceptual_roughness, sky, ground
        )
        * hemisphere.intensity
    )
    environment_specular = fresnel_ambient * reflected_radiance

    irradiance_map = environment_diffuse
    specular_map = direct_specular + environment_specular
    final_color = (
        direct_diffuse
        + direct_specular
        + environment_diffuse
        + environment_specular
    )

    return LightingResult(
        irradiance=irradiance_map,
        specular=specular_map,
        final_color=final_color,
        direct_diffuse=direct_diffuse,
        direct_specular=direct_specular,
        environment_diffuse=environment_diffuse,
        environment_specular=environment_specular,
    )

