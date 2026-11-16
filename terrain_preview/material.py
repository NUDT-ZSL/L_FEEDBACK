"""Altitude/slope layer blending and color conversion."""

from __future__ import annotations

import numpy as np

from .config import MaterialConfig


def smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return (value >= edge1).astype(np.float64)
    t = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def srgb_to_linear(color: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    value = np.asarray(color, dtype=np.float64)
    return np.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.where(
        value <= 0.0031308,
        value * 12.92,
        1.055 * np.power(np.maximum(value, 0.0031308), 1.0 / 2.4) - 0.055,
    )


def layer_linear_colors(config: MaterialConfig) -> np.ndarray:
    return np.stack([srgb_to_linear(layer.color_srgb) for layer in config.layers])


def material_weights(
    altitude: np.ndarray, normal_y: np.ndarray, config: MaterialConfig
) -> np.ndarray:
    """Return normalized sand/grass/rock/snow weights for height and normal Y."""

    altitude = np.asarray(altitude, dtype=np.float64)
    normal_y = np.asarray(normal_y, dtype=np.float64)
    half_width = config.elevation_transition * 0.5
    cumulative = []
    for threshold in config.elevation_thresholds:
        cumulative.append(
            smoothstep(threshold - half_width, threshold + half_width, altitude)
        )
    s0, s1, s2 = cumulative
    weights = np.stack(
        (
            1.0 - s0,
            np.maximum(s0 - s1, 0.0),
            np.maximum(s1 - s2, 0.0),
            s2,
        ),
        axis=-1,
    )

    layer_weights = np.array([layer.weight for layer in config.layers], dtype=np.float64)
    if np.any(layer_weights > 0.0):
        weights *= layer_weights[None, :]
    totals = np.sum(weights, axis=-1, keepdims=True)
    weights = weights / np.maximum(totals, 1e-15)

    slope = np.clip(1.0 - normal_y, 0.0, 1.0)
    steep = smoothstep(config.slope_start, config.slope_end, slope)
    steep *= config.slope_rock_strength
    if np.any(steep > 0.0):
        mixed = weights * (1.0 - steep)[..., None]
        mixed[..., 2] += steep
        weights = mixed
        totals = np.sum(weights, axis=-1, keepdims=True)
        weights = weights / np.maximum(totals, 1e-15)
    return weights


def material_albedo(
    altitude: np.ndarray,
    normal_y: np.ndarray,
    config: MaterialConfig,
    *,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if weights is None:
        weights = material_weights(altitude, normal_y, config)
    colors = layer_linear_colors(config)
    albedo = np.sum(weights[..., None] * colors[None, :, :], axis=-2)
    return weights, albedo

