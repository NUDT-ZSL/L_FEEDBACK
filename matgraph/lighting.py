from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evaluate import MaterialBuffers
from .errors import PipelineError
from .raster import GBuffer
from .scene import DirectionalLight, Scene
from .mathutil import linear_to_srgb


@dataclass(frozen=True)
class LightingResult:
    radiance: np.ndarray
    ldr_linear: np.ndarray
    ldr_srgb: np.ndarray
    ldr_u8: np.ndarray


def tone_map_aces(value: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    mapped = (x * (a * x + b)) / (x * (c * x + d) + e)
    return np.clip(mapped, 0.0, 1.0)


def _light_contribution(
    normal: np.ndarray,
    view: np.ndarray,
    diffuse_color: np.ndarray,
    f0: np.ndarray,
    roughness: np.ndarray,
    metallic: np.ndarray,
    light: DirectionalLight,
) -> np.ndarray:
    light_direction = np.broadcast_to(light.direction[None, :], normal.shape)
    half_direction = light_direction + view
    half_lengths = np.linalg.norm(half_direction, axis=1)
    half_direction /= np.maximum(half_lengths, 1e-12)[:, None]

    n_dot_l = np.clip(np.sum(normal * light_direction, axis=1), 0.0, 1.0)
    n_dot_v = np.clip(np.sum(normal * view, axis=1), 1e-6, 1.0)
    n_dot_h = np.clip(np.sum(normal * half_direction, axis=1), 0.0, 1.0)
    v_dot_h = np.clip(np.sum(view * half_direction, axis=1), 0.0, 1.0)

    alpha = np.square(roughness)
    alpha2 = np.square(alpha)
    denominator = np.square(n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0)
    distribution = alpha2 / (np.pi * denominator + 1e-12)

    k = np.square(roughness + 1.0) / 8.0
    geometry_l = n_dot_l / (n_dot_l * (1.0 - k) + k + 1e-12)
    geometry_v = n_dot_v / (n_dot_v * (1.0 - k) + k + 1e-12)
    geometry = geometry_l * geometry_v

    fresnel = f0 + (1.0 - f0) * np.power(1.0 - v_dot_h, 5.0)[:, None]
    specular = (
        distribution[:, None]
        * geometry[:, None]
        * fresnel
        / (4.0 * n_dot_v[:, None] * n_dot_l[:, None] + 1e-12)
    )
    diffuse = diffuse_color * (1.0 - fresnel) / np.pi
    return (diffuse + specular) * light.color[None, :] * n_dot_l[:, None]


def shade_scene(
    scene: Scene,
    gbuffer: GBuffer,
    material: MaterialBuffers,
) -> LightingResult:
    height = scene.height
    width = scene.width
    radiance = np.zeros((height, width, 3), dtype=np.float64)
    radiance[~gbuffer.valid] = scene.background

    rows, columns = np.nonzero(gbuffer.valid)
    if rows.size:
        normal = material.normal[rows, columns]
        base_color = material.base_color[rows, columns]
        roughness = np.clip(material.roughness[rows, columns], 0.045, 1.0)
        metallic = np.clip(material.metallic[rows, columns], 0.0, 1.0)
        emissive = material.emissive[rows, columns]
        world = gbuffer.world[rows, columns]
        view = scene.camera.position[None, :] - world
        view /= np.linalg.norm(view, axis=-1, keepdims=True)

        f0 = 0.04 * (1.0 - metallic[:, None]) + base_color * metallic[:, None]
        diffuse_color = base_color * (1.0 - metallic[:, None])
        direct = np.zeros_like(base_color)
        for light in scene.lights:
            direct += _light_contribution(
                normal,
                view,
                diffuse_color,
                f0,
                roughness,
                metallic,
                light,
            )

        hemisphere_mix = np.clip(0.5 * (normal[:, 1] + 1.0), 0.0, 1.0)[:, None]
        ambient_irradiance = (
            scene.ambient.ground[None, :] * (1.0 - hemisphere_mix)
            + scene.ambient.sky[None, :] * hemisphere_mix
        )
        ambient_diffuse = diffuse_color * ambient_irradiance
        ambient_specular = (
            f0
            * scene.ambient.sky[None, :]
            * np.square(1.0 - roughness)[:, None]
        )
        surface = emissive + direct + ambient_diffuse + ambient_specular
        surface = np.maximum(surface, 0.0)
        if not np.all(np.isfinite(surface)):
            raise PipelineError(
                code="E_NUMERIC",
                message="lighting produced a non-finite radiance value",
                stage="render",
            )
        radiance[rows, columns] = surface

    exposed = np.maximum(radiance * scene.exposure, 0.0)
    ldr_linear = np.clip(tone_map_aces(exposed), 0.0, 1.0)
    ldr_srgb = np.clip(linear_to_srgb(ldr_linear), 0.0, 1.0)
    ldr_u8 = np.floor(ldr_srgb * 255.0 + 0.5).astype(np.uint8)
    return LightingResult(
        radiance=np.ascontiguousarray(radiance),
        ldr_linear=np.ascontiguousarray(ldr_linear),
        ldr_srgb=np.ascontiguousarray(ldr_srgb),
        ldr_u8=np.ascontiguousarray(ldr_u8),
    )
