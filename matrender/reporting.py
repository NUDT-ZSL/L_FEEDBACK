"""Statistics derived directly from the merged linear HDR image."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .color import luminance, rgb_to_hue_degrees
from .scene import Sphere


def _json_float(value: float) -> float:
    return float(value)


def summarize_linear(
    linear_hdr: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, object]:
    """Summarize RGB and luminance values, optionally inside a boolean mask."""

    rgb = np.asarray(linear_hdr, dtype=np.float64)
    if mask is None:
        values = rgb.reshape(-1, 3)
    else:
        values = rgb[np.asarray(mask, dtype=bool)]

    if values.size == 0:
        return {
            "pixels": 0,
            "linear_hdr_min": None,
            "linear_hdr_max": None,
            "linear_scalar_min": None,
            "linear_scalar_max": None,
            "luminance_min": None,
            "luminance_max": None,
            "mean_linear_rgb": None,
        }

    lum = luminance(values)
    return {
        "pixels": int(values.shape[0]),
        "linear_hdr_min": [
            _json_float(v) for v in np.min(values, axis=0).tolist()
        ],
        "linear_hdr_max": [
            _json_float(v) for v in np.max(values, axis=0).tolist()
        ],
        "linear_scalar_min": _json_float(np.min(values)),
        "linear_scalar_max": _json_float(np.max(values)),
        "luminance_min": _json_float(np.min(lum)),
        "luminance_max": _json_float(np.max(lum)),
        "mean_linear_rgb": [
            _json_float(v) for v in np.mean(values, axis=0).tolist()
        ],
    }


def build_material_regions(
    linear_hdr: np.ndarray,
    masks: Sequence[np.ndarray],
    spheres: Sequence[Sphere],
) -> list[dict[str, object]]:
    """Build one report entry per scene sphere, preserving scene order."""

    entries: list[dict[str, object]] = []
    for sphere, mask in zip(spheres, masks, strict=True):
        stats = summarize_linear(linear_hdr, mask)
        mean_rgb_value = stats["mean_linear_rgb"]
        hue = (
            rgb_to_hue_degrees(tuple(mean_rgb_value))
            if mean_rgb_value is not None
            else None
        )
        entries.append(
            {
                "index": sphere.index,
                "name": sphere.name,
                "mask_pixels": int(np.count_nonzero(mask)),
                "average_luminance": stats["luminance_min"]
                if stats["luminance_min"] is None
                else _json_float(
                    float(np.mean(luminance(linear_hdr[np.asarray(mask, bool)])))
                ),
                "max_luminance": stats["luminance_max"],
                "dominant_hue_deg": hue,
                "mean_linear_rgb": mean_rgb_value,
            }
        )
    return entries

