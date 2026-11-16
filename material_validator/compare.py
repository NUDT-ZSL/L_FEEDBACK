from __future__ import annotations

import math
from typing import Any

import numpy as np

from .manifest import ComparisonSettings, RenderSettings


CHANNEL_INDEX = {"R": 0, "G": 1, "B": 2, "A": 3}


def _location(index: int, width: int) -> dict[str, int]:
    return {"x": int(index % width), "y": int(index // width)}


def _max_channel(diff: np.ndarray, selected: list[int]) -> tuple[float, str | None, dict[str, int] | None]:
    if diff.size == 0:
        return 0.0, None, None
    channel_names = [name for name, index in CHANNEL_INDEX.items() if index in selected]
    best_value = -1.0
    best_channel: str | None = None
    best_flat = 0
    width = diff.shape[1]
    for channel_name in channel_names:
        channel_index = CHANNEL_INDEX[channel_name]
        values = diff[:, :, channel_index]
        finite = np.isfinite(values)
        if np.any(~finite):
            flat_index = int(np.argmin(finite))
            return math.inf, channel_name, _location(flat_index, width)
        flat_index = int(np.argmax(values))
        value = float(values.reshape(-1)[flat_index])
        if value > best_value:
            best_value = value
            best_channel = channel_name
            best_flat = flat_index
    if best_channel is None:
        return 0.0, None, None
    return best_value, best_channel, _location(best_flat, width)


def _category_stats(
    mask: np.ndarray,
    diff: np.ndarray,
    selected: list[int],
) -> dict[str, Any]:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {"pixels": 0, "max_abs_error": 0.0, "channel": None, "location": None}
    masked_diff = np.where(mask[:, :, None], diff, 0.0)
    maximum, channel, location = _max_channel(masked_diff, selected)
    return {
        "pixels": count,
        "max_abs_error": maximum,
        "channel": channel,
        "location": location,
    }


def analyze_gbuffer(
    position: np.ndarray,
    normal: np.ndarray,
    albedo: np.ndarray,
    material_params: np.ndarray,
    render_settings: RenderSettings,
) -> dict[str, Any]:
    coverage = albedo[:, :, 3] > 0.5
    normal_xyz = normal[:, :, :3]
    normal_length = np.sqrt(np.sum(normal_xyz * normal_xyz, axis=2))
    normal_flagged = normal[:, :, 3] < 0.5
    fallback = coverage & normal_flagged
    non_unit = coverage & (~normal_flagged) & (np.abs(normal_length - 1.0) > render_settings.normal_tolerance)
    fallback_vector = np.asarray(render_settings.normal_fallback, dtype=np.float32)
    fallback_error = np.max(np.abs(normal_xyz - fallback_vector[None, None, :]), axis=2)
    fallback_mismatch = fallback & (fallback_error > render_settings.normal_tolerance)
    metallic = material_params[:, :, 0]
    roughness = material_params[:, :, 1]
    metallic_flag = material_params[:, :, 2] > 0.5
    roughness_flag = material_params[:, :, 3] > 0.5
    metallic_invalid = coverage & (
        metallic_flag
        | (metallic < -render_settings.normal_tolerance)
        | (metallic > 1.0 + render_settings.normal_tolerance)
    )
    roughness_invalid = coverage & (
        roughness_flag
        | (roughness < -render_settings.normal_tolerance)
        | (roughness > 1.0 + render_settings.normal_tolerance)
    )
    nonfinite = coverage & (
        ~np.isfinite(position).all(axis=2)
        | ~np.isfinite(normal).all(axis=2)
        | ~np.isfinite(albedo).all(axis=2)
        | ~np.isfinite(material_params).all(axis=2)
    )
    return {
        "coverage_pixels": int(np.count_nonzero(coverage)),
        "normal_fallback_pixels": int(np.count_nonzero(fallback)),
        "normal_fallback_mismatch_pixels": int(np.count_nonzero(fallback_mismatch)),
        "normal_non_unit_pixels": int(np.count_nonzero(non_unit)),
        "metallic_out_of_range_pixels": int(np.count_nonzero(metallic_invalid)),
        "roughness_out_of_range_pixels": int(np.count_nonzero(roughness_invalid)),
        "non_finite_pixels": int(np.count_nonzero(nonfinite)),
        "metallic_range": [float(np.min(metallic[coverage])) if np.any(coverage) else 0.0, float(np.max(metallic[coverage])) if np.any(coverage) else 0.0],
        "roughness_range": [float(np.min(roughness[coverage])) if np.any(coverage) else 0.0, float(np.max(roughness[coverage])) if np.any(coverage) else 0.0],
    }


def compare_images(
    deferred: np.ndarray,
    forward: np.ndarray,
    gbuffer_normal: np.ndarray,
    material_params: np.ndarray,
    blend_mask: np.ndarray,
    comparison: ComparisonSettings,
    render_settings: RenderSettings,
) -> dict[str, Any]:
    if deferred.shape != forward.shape or deferred.ndim != 3 or deferred.shape[2] != 4:
        raise ValueError(f"Expected matching RGBA images, got {deferred.shape} and {forward.shape}")
    selected = sorted({CHANNEL_INDEX[name] for name in comparison.channels})
    diff = np.abs(deferred.astype(np.float64) - forward.astype(np.float64))
    magnitude = np.maximum(np.abs(deferred.astype(np.float64)), np.abs(forward.astype(np.float64)))
    allowed = comparison.absolute + comparison.relative * magnitude
    with np.errstate(invalid="ignore"):
        exceeded_channels = diff > allowed
    nonfinite = ~np.isfinite(diff)
    pixel_over = np.zeros(deferred.shape[:2], dtype=bool)
    for channel_index in selected:
        pixel_over |= exceeded_channels[:, :, channel_index] | nonfinite[:, :, channel_index]
    material_cause = (
        (material_params[:, :, 2] > 0.5)
        | (material_params[:, :, 3] > 0.5)
        | (material_params[:, :, 0] < -render_settings.normal_tolerance)
        | (material_params[:, :, 0] > 1.0 + render_settings.normal_tolerance)
        | (material_params[:, :, 1] < -render_settings.normal_tolerance)
        | (material_params[:, :, 1] > 1.0 + render_settings.normal_tolerance)
    )
    normal_length = np.sqrt(np.sum(gbuffer_normal[:, :, :3] * gbuffer_normal[:, :, :3], axis=2))
    normal_cause = (gbuffer_normal[:, :, 3] < 0.5) | (
        np.abs(normal_length - 1.0) > render_settings.normal_tolerance
    )
    blend_cause = blend_mask[:, :, 0] > 0.5
    material_mask = pixel_over & material_cause
    normal_mask = pixel_over & (~material_mask) & normal_cause
    blend_order_mask = pixel_over & (~material_mask) & (~normal_mask) & blend_cause
    unclassified_mask = pixel_over & (~material_mask) & (~normal_mask) & (~blend_order_mask)
    maximum, maximum_channel, maximum_location = _max_channel(diff, selected)
    per_channel: dict[str, Any] = {}
    for channel_name in comparison.channels:
        channel_index = CHANNEL_INDEX[channel_name]
        values = diff[:, :, channel_index]
        finite_values = np.where(np.isfinite(values), values, np.nan)
        channel_max = float(np.nanmax(finite_values)) if np.any(np.isfinite(finite_values)) else math.inf
        channel_count = int(np.count_nonzero(exceeded_channels[:, :, channel_index] | nonfinite[:, :, channel_index]))
        per_channel[channel_name] = {
            "pixels_over_threshold": channel_count,
            "max_abs_error": channel_max,
        }
    categories = {
        "normal_encoding_precision": _category_stats(normal_mask, diff, selected),
        "floating_blend_order": _category_stats(blend_order_mask, diff, selected),
        "material_parameter_range": _category_stats(material_mask, diff, selected),
        "unclassified": _category_stats(unclassified_mask, diff, selected),
    }
    return {
        "pass": not bool(np.any(pixel_over)),
        "pixels_total": int(deferred.shape[0] * deferred.shape[1]),
        "pixels_over_threshold": int(np.count_nonzero(pixel_over)),
        "max_abs_error": maximum,
        "max_channel": maximum_channel,
        "max_location": maximum_location,
        "non_finite_pixels": int(np.count_nonzero(nonfinite[:, :, selected].any(axis=2))),
        "per_channel": per_channel,
        "categories": categories,
        "thresholds": comparison.to_dict(),
    }
