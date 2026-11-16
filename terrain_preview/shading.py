"""Combine layered material, direct light, PCF shadow, and hemisphere ambient."""

from __future__ import annotations

import numpy as np

from .config import AppConfig
from .light import LightingResources, pcf_visibility
from .material import layer_linear_colors, material_weights, srgb_to_linear


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(lengths, 1e-15)


def shade_fragments(
    positions: np.ndarray,
    normals: np.ndarray,
    config: AppConfig,
    lighting: LightingResources,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sRGB uint8 color, normal debug RGB, and shadow mask."""

    normals = _normalize_rows(normals.astype(np.float64))
    altitude = np.clip(
        positions[:, 1].astype(np.float64) / config.terrain.height_scale, 0.0, 1.0
    )
    weights = material_weights(altitude, normals[:, 1], config.materials)
    colors = layer_linear_colors(config.materials)
    albedo = np.sum(weights[..., None] * colors[None, :, :], axis=1)

    visibility = pcf_visibility(positions, normals, config, lighting)
    light_direction = np.array(
        (
            np.cos(np.radians(config.light.elevation_deg))
            * np.cos(np.radians(config.light.azimuth_deg)),
            np.sin(np.radians(config.light.elevation_deg)),
            np.cos(np.radians(config.light.elevation_deg))
            * np.sin(np.radians(config.light.azimuth_deg)),
        ),
        dtype=np.float64,
    )
    ndotl = np.maximum(normals @ light_direction, 0.0)
    direct_color = srgb_to_linear(config.light.color_srgb)
    direct = (
        direct_color[None, :]
        * (config.light.intensity * ndotl * visibility)[:, None]
    )

    hemi_t = 0.5 * (normals[:, 1] + 1.0)
    sky = srgb_to_linear(config.light.ambient.sky_color_srgb)
    ground = srgb_to_linear(config.light.ambient.ground_color_srgb)
    ambient = (
        ground[None, :] + (sky - ground)[None, :] * hemi_t[:, None]
    ) * config.light.ambient.intensity
    linear = albedo * (direct + ambient)

    mapped = 1.0 - np.exp(-linear * config.render.exposure)
    encoded = np.power(np.clip(mapped, 0.0, 1.0), 1.0 / config.render.gamma)
    rgb = np.clip(np.rint(encoded * 255.0), 0, 255).astype(np.uint8)

    normal_debug = np.clip(np.rint((normals * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
    shadow_debug = np.clip(np.rint(visibility * 255.0), 0, 255).astype(np.uint8)
    return rgb, normal_debug, shadow_debug

