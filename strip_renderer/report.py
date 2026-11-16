from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .render import RenderResult
from .scene import Scene


CHANNEL_FILES = {
    "base_color": "base_color.png",
    "direct": "direct_light.png",
    "ambient": "ambient_light.png",
    "composite": "composite.png",
}
REPORT_FILE = "self_check.json"


def quantize_channel(channel: np.ndarray) -> np.ndarray:
    """Quantize with round-to-nearest, ties-to-even."""
    clipped = np.clip(channel, 0.0, 1.0)
    return np.rint(clipped * 255.0).astype(np.uint8)


def quantize_rgb(channel: np.ndarray) -> np.ndarray:
    return quantize_channel(channel)


def _array_finite_counts(array: np.ndarray) -> tuple[int, int]:
    finite = np.isfinite(array)
    return int(np.count_nonzero(np.isnan(array))), int(
        np.count_nonzero(np.isinf(array))
    )


def _safe_mean(array: np.ndarray, mask: np.ndarray | None = None) -> float:
    values = array if mask is None else array[mask]
    finite = np.isfinite(values)
    if not np.any(finite):
        return 0.0
    return float(np.mean(values[finite], dtype=np.float64))


def _rgb_mean(array: np.ndarray, mask: np.ndarray | None = None) -> float:
    return _safe_mean(array, mask)


def build_report(scene: Scene, result: RenderResult) -> dict[str, Any]:
    channels = {
        "base_color": result.base_color,
        "direct": result.direct_light,
        "ambient": result.ambient_light,
        "composite": result.composite,
    }
    nan_counts: dict[str, int] = {}
    inf_counts: dict[str, int] = {}
    for name, channel in channels.items():
        nan_counts[name], inf_counts[name] = _array_finite_counts(channel)

    weights = scene.output.weights
    expected_composite = (
        result.base_color * weights.base_color
        + result.direct_light * weights.direct
        + result.ambient_light * weights.ambient
    )
    errors = np.abs(result.composite - expected_composite)
    finite_errors = errors[np.isfinite(errors)]
    max_error = float(np.max(finite_errors)) if finite_errors.size else 0.0

    hit_mask = result.buffers.hit_mask
    hit_pixels = int(np.count_nonzero(hit_mask))
    quantized_composite = quantize_rgb(result.composite)
    quantized_hashes = {
        name: hashlib.sha256(quantize_rgb(channel).tobytes(order="C")).hexdigest()
        for name, channel in channels.items()
    }

    return {
        "status": "ok",
        "schema_version": scene.schema_version,
        "scene_file": str(scene.scene_path),
        "output": {
            "width": scene.output.width,
            "height": scene.output.height,
            "background_color": list(scene.output.background_color),
            "texture_filter": scene.output.texture_filter,
            "block_size": scene.output.block_size,
            "block_axis": scene.output.block_axis,
            "block_order": scene.output.block_order,
            "weights": {
                "base_color": weights.base_color,
                "direct": weights.direct,
                "ambient": weights.ambient,
            },
        },
        "render": {
            "pixel_count": scene.output.width * scene.output.height,
            "hit_pixels": hit_pixels,
            "miss_pixels": scene.output.width * scene.output.height - hit_pixels,
            "normal_fallback_count": result.normal_fallback_count,
            "normal_fallback_breakdown": result.normal_fallback_breakdown,
            "nan_count": int(sum(nan_counts.values())),
            "inf_count": int(sum(inf_counts.values())),
            "nan_counts": nan_counts,
            "inf_counts": inf_counts,
            "channel_means": {
                name: _rgb_mean(channel) for name, channel in channels.items()
            },
            "channel_means_hit": {
                name: _rgb_mean(channel, hit_mask)
                for name, channel in channels.items()
            },
            "composite_check": {
                "max_abs_error": max_error,
                "tolerance": 1.0e-6,
                "passed": bool(max_error <= 1.0e-6 and np.all(np.isfinite(errors))),
            },
        },
        "textures": {
            "base_color_size": list(result.textures.base_color_size),
            "roughness_size": list(result.textures.roughness_size),
            "normal_size": (
                None
                if result.textures.normal_size is None
                else list(result.textures.normal_size)
            ),
            "normal_texture_missing": result.textures.normal_texture_missing,
        },
        "outputs": {
            "channels": CHANNEL_FILES,
            "report": REPORT_FILE,
            "channel_quantized_sha256": quantized_hashes,
            "composite_quantized_sha256": hashlib.sha256(
                quantized_composite.tobytes(order="C")
            ).hexdigest(),
        },
    }
