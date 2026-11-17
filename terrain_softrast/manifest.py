from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def array_digest(array: np.ndarray, dtype: str = "<f8") -> str:
    array = np.asarray(array, dtype=dtype)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(dtype.encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _plain_number(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_plain_number(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return [_plain_number(item) for item in value]
    if isinstance(value, list):
        return [_plain_number(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain_number(item) for key, item in value.items()}
    return value


def material_dict(materials) -> list[dict]:
    result = []
    for material in materials:
        result.append(
            {
                "name": material.name,
                "albedo": list(material.albedo),
                "roughness": material.roughness,
                "normal_strength": material.normal_strength,
                "detail_scale": material.detail_scale,
                "detail_phase": material.detail_phase,
            }
        )
    return result


def structural_config(camera, directional_light, point_lights, materials, settings) -> dict:
    return _plain_number(
        {
            "camera": camera.to_dict(),
            "directional_light": {
                "direction": list(directional_light.direction),
                "color": list(directional_light.color),
                "intensity": directional_light.intensity,
            },
            "point_lights": [
                {
                    "position": list(light.position),
                    "color": list(light.color),
                    "intensity": light.intensity,
                    "range": light.range,
                }
                for light in point_lights
            ],
            "materials": material_dict(materials),
            "settings": {
                "width": settings.width,
                "height": settings.height,
                "tile_size": settings.tile_size,
                "cascades": {
                    "count": settings.cascades.count,
                    "shadow_size": settings.cascades.shadow_size,
                    "split_lambda": settings.cascades.split_lambda,
                    "overlap_fraction": settings.cascades.overlap_fraction,
                },
                "ambient": list(settings.ambient),
                "tone": {"exposure": settings.tone.exposure, "gamma": settings.tone.gamma},
                "backface_culling": settings.backface_culling,
                "shadow_constant_bias": settings.shadow_constant_bias,
                "shadow_slope_bias": settings.shadow_slope_bias,
                "shadow_normal_offset": settings.shadow_normal_offset,
                "shadow_pcf_radius": settings.shadow_pcf_radius,
            },
        }
    )


def build_manifest(
    *,
    settings,
    camera,
    directional_light,
    point_lights,
    materials,
    height: np.ndarray,
    source_weights: np.ndarray,
    height_hash: str,
    weights_hash: str,
    structural_hash: str,
    cascades,
    warnings,
    timings: dict,
    incremental: dict,
    output_files: dict,
    input_metadata: dict | None = None,
) -> dict:
    normalized_summary = {
        "heightfield": {
            "shape": list(height.shape),
            "data_sha256": height_hash,
            "unit_scale": incremental["unit_scale"],
            "height_scale": incremental["height_scale"],
        },
        "material_weights": {
            "shape": list(source_weights.shape),
            "data_sha256": weights_hash,
            "normalize_before_interpolation": True,
        },
        "source_metadata": _plain_number(input_metadata or {}),
        "structural": structural_config(camera, directional_light, point_lights, materials, settings),
    }
    summary_hash = digest_json(normalized_summary)
    warning_codes = sorted(set(warnings["codes"]))
    return {
        "schema": "terrain-softrast-manifest",
        "version": 1,
        "renderer": "terrain_softrast",
        "summary_sha256": summary_hash,
        "normalized_inputs": normalized_summary,
        "resolution": [settings.width, settings.height],
        "outputs": {
            "color": output_files["color"],
            "depth": output_files["depth"],
            "shadow_visibility": output_files["shadow_visibility"],
            "format": {
                "color": "RGB8 PNG, deterministic exponential tone map and gamma encode",
                "depth": "Gray16 PNG, linear camera depth normalized from near to far; 0=near, 65535=far",
                "shadow_visibility": "Gray8 PNG, 0=fully occluded, 255=fully lit",
            },
        },
        "cascades": {
            "count": settings.cascades.count,
            "split_lambda": settings.cascades.split_lambda,
            "overlap_fraction": settings.cascades.overlap_fraction,
            "splits": [cascade.split_far for cascade in cascades],
            "levels": [cascade.to_dict() for cascade in cascades],
        },
        "warning_codes": warning_codes,
        "warnings": warnings["details"],
        "timings_ms": timings,
        "incremental": incremental,
        "structural_hash": structural_hash,
        "determinism": {
            "png_encoder": "internal-png-zlib-9-no-timestamps",
            "quantization": "floor(clamp(x,0,1)*max+0.5)",
            "summary_excludes": ["timings_ms"],
        },
    }
