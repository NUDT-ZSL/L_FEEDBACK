from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import ErrorCode, diagnostic
from .patterns import evaluate_pattern_rgb, evaluate_pattern_scalar, pattern_error
from .scene import UVLayout
from .util import clamp01


SUPPORTED_BLENDS = {"normal", "multiply", "screen", "overlay", "add"}


@dataclass(frozen=True)
class MaterialMaps:
    width: int
    height: int
    albedo_linear: np.ndarray
    roughness: np.ndarray
    metallic: np.ndarray
    layer_order: tuple[str, ...]


@dataclass(frozen=True)
class MaterialResult:
    maps: MaterialMaps | None
    errors: tuple[dict[str, Any], ...]


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float64)
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def _blend(destination: np.ndarray, source: np.ndarray, mode: str) -> np.ndarray:
    if mode == "normal":
        return source
    if mode == "multiply":
        return destination * source
    if mode == "screen":
        return 1.0 - (1.0 - destination) * (1.0 - source)
    if mode == "overlay":
        return np.where(
            destination <= 0.5,
            2.0 * destination * source,
            1.0 - 2.0 * (1.0 - destination) * (1.0 - source),
        )
    if mode == "add":
        return destination + source
    raise ValueError(f"unsupported blend mode {mode!r}")


def _layer_tiling(layer: dict[str, Any]) -> tuple[float, float]:
    tiling = layer.get("tiling", [1.0, 1.0])
    if not isinstance(tiling, (list, tuple)) or len(tiling) != 2:
        raise ValueError("tiling must contain two positive numbers")
    tiling_tuple = (float(tiling[0]), float(tiling[1]))
    if tiling_tuple[0] <= 0.0 or tiling_tuple[1] <= 0.0:
        raise ValueError("tiling values must be positive")
    return tiling_tuple


def _layer_rotation(layer: dict[str, Any]) -> float:
    value = layer.get("rotation_deg", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        raise ValueError("rotation_deg must be finite")
    return float(value)


def compile_material(layout: UVLayout, material: dict[str, Any]) -> MaterialResult:
    errors: list[dict[str, Any]] = []
    width, height = layout.width, layout.height
    if "base_color" not in material:
        errors.append(
            diagnostic(
                ErrorCode.MATERIAL_FIELD_MISSING,
                "material.base_color is required",
                "material",
                "material.base_color",
            )
        )
        base = np.zeros(3, dtype=np.float64)
    else:
        base_raw = material["base_color"]
        if not isinstance(base_raw, (list, tuple)) or len(base_raw) != 3:
            errors.append(
                diagnostic(
                    ErrorCode.MATERIAL_FIELD_MISSING,
                    "material.base_color must be an RGB array",
                    "material",
                    "material.base_color",
                )
            )
            base = np.zeros(3, dtype=np.float64)
        else:
            try:
                base_srgb = np.asarray([float(item) for item in base_raw], dtype=np.float64)
                if not np.all(np.isfinite(base_srgb)) or np.any(base_srgb < 0.0) or np.any(base_srgb > 1.0):
                    raise ValueError
                base = srgb_to_linear(base_srgb)
            except (TypeError, ValueError):
                errors.append(
                    diagnostic(
                        ErrorCode.MATERIAL_FIELD_MISSING,
                        "material.base_color values must be finite numbers in [0, 1]",
                        "material",
                        "material.base_color",
                    )
                )
                base = np.zeros(3, dtype=np.float64)

    try:
        base_roughness = float(material.get("base_roughness", 0.5))
        base_metallic = float(material.get("base_metallic", 0.0))
        if not np.isfinite(base_roughness) or not 0.0 <= base_roughness <= 1.0:
            raise ValueError("base_roughness must be in [0, 1]")
        if not np.isfinite(base_metallic) or not 0.0 <= base_metallic <= 1.0:
            raise ValueError("base_metallic must be in [0, 1]")
    except (TypeError, ValueError) as exc:
        errors.append(
            diagnostic(
                ErrorCode.MATERIAL_FIELD_MISSING,
                str(exc),
                "material",
                "material.base_roughness/base_metallic",
            )
        )
        base_roughness = 0.5
        base_metallic = 0.0

    layers_raw = material.get("layers", [])
    if not isinstance(layers_raw, list):
        errors.append(
            diagnostic(
                ErrorCode.MATERIAL_LAYER_INVALID,
                "material.layers must be an array",
                "material",
                "material.layers",
            )
        )
        layers_raw = []

    albedo = np.broadcast_to(base, (height, width, 3)).copy()
    roughness = np.full((height, width), base_roughness, dtype=np.float64)
    metallic = np.full((height, width), base_metallic, dtype=np.float64)
    layer_ids: list[str] = []
    v, u = np.meshgrid(
        (np.arange(height, dtype=np.float64) + 0.5) / height,
        (np.arange(width, dtype=np.float64) + 0.5) / width,
        indexing="ij",
    )
    uv = np.stack((u, v), axis=-1)

    for index, layer_raw in enumerate(layers_raw):
        path = f"material.layers[{index}]"
        context: dict[str, Any] = {"layer_index": index}
        if not isinstance(layer_raw, dict):
            errors.append(
                diagnostic(
                    ErrorCode.MATERIAL_LAYER_INVALID,
                    f"{path} must be an object",
                    "material_layer",
                    path,
                    **context,
                )
            )
            continue
        layer_id = layer_raw.get("id", f"layer_{index}")
        if not isinstance(layer_id, str) or not layer_id:
            errors.append(
                diagnostic(
                    ErrorCode.MATERIAL_LAYER_INVALID,
                    f"{path}.id must be a non-empty string",
                    "material_layer",
                    f"{path}.id",
                    **context,
                )
            )
            continue
        context["layer_id"] = layer_id
        layer_ids.append(layer_id)
        channel = layer_raw.get("channel")
        if channel not in {"albedo", "roughness", "metallic"}:
            errors.append(
                diagnostic(
                    ErrorCode.MATERIAL_LAYER_INVALID,
                    f"{path}.channel must be albedo, roughness or metallic",
                    "material_layer",
                    f"{path}.channel",
                    **context,
                )
            )
            continue

        if channel in {"roughness", "metallic"}:
            if "mask_size" not in layer_raw:
                errors.append(
                    diagnostic(
                        ErrorCode.MAT_MASK_SIZE_MISSING,
                        f"{path}.mask_size is required for {channel} layers",
                        "material_layer",
                        f"{path}.mask_size",
                        expected=[width, height],
                        **context,
                    )
                )
                continue
            mask_size = layer_raw["mask_size"]
            if not isinstance(mask_size, (list, tuple)) or len(mask_size) != 2:
                errors.append(
                    diagnostic(
                        ErrorCode.MAT_MASK_SIZE_MISMATCH,
                        f"{path}.mask_size must be [width, height]",
                        "material_layer",
                        f"{path}.mask_size",
                        expected=[width, height],
                        actual=mask_size,
                        **context,
                    )
                )
                continue
            valid_ints = all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in mask_size
            )
            actual_size = [mask_size[0], mask_size[1]]
            if not valid_ints or actual_size != [width, height]:
                errors.append(
                    diagnostic(
                        ErrorCode.MAT_MASK_SIZE_MISMATCH,
                        f"{path}.mask_size {actual_size} does not match UV layout {[width, height]}",
                        "material_layer",
                        f"{path}.mask_size",
                        expected=[width, height],
                        actual=actual_size,
                        **context,
                    )
                )
                continue

        blend = layer_raw.get("blend", "normal")
        if blend not in SUPPORTED_BLENDS:
            errors.append(
                diagnostic(
                    ErrorCode.MAT_BLEND_UNSUPPORTED,
                    f"unsupported blend mode {blend!r}",
                    "material_layer",
                    f"{path}.blend",
                    supported=sorted(SUPPORTED_BLENDS),
                    **context,
                )
            )
            continue
        opacity_raw = layer_raw.get("opacity", 1.0)
        try:
            opacity = float(opacity_raw)
            if not np.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
                raise ValueError("opacity must be in [0, 1]")
        except (TypeError, ValueError) as exc:
            errors.append(
                diagnostic(
                    ErrorCode.MATERIAL_LAYER_INVALID,
                    str(exc),
                    "material_layer",
                    f"{path}.opacity",
                    **context,
                )
            )
            continue
        pattern = layer_raw.get("pattern")
        if not isinstance(pattern, dict):
            errors.append(
                diagnostic(
                    ErrorCode.MAT_PATTERN_FIELD_MISSING,
                    f"{path}.pattern must be an object",
                    "material_layer",
                    f"{path}.pattern",
                    **context,
                )
            )
            continue
        try:
            tiling = _layer_tiling(layer_raw)
            rotation = _layer_rotation(layer_raw)
            if channel == "albedo":
                source = clamp01(evaluate_pattern_rgb(uv, pattern, tiling, rotation))
                source = srgb_to_linear(source)
                destination = albedo
                blended = clamp01(_blend(destination, source, blend))
                albedo = destination * (1.0 - opacity) + blended * opacity
            elif channel == "roughness":
                source = clamp01(evaluate_pattern_scalar(uv, pattern, tiling, rotation))
                destination = roughness
                blended = clamp01(_blend(destination, source, blend))
                roughness = destination * (1.0 - opacity) + blended * opacity
            else:
                source = clamp01(evaluate_pattern_scalar(uv, pattern, tiling, rotation))
                destination = metallic
                blended = clamp01(_blend(destination, source, blend))
                metallic = destination * (1.0 - opacity) + blended * opacity
        except (KeyError, TypeError, ValueError) as exc:
            code, message = pattern_error(exc, path=f"{path}.pattern")
            errors.append(
                diagnostic(
                    code,
                    message,
                    "material_layer",
                    f"{path}.pattern",
                    **context,
                )
            )
            continue

    if errors:
        return MaterialResult(maps=None, errors=tuple(errors))
    return MaterialResult(
        maps=MaterialMaps(
            width=width,
            height=height,
            albedo_linear=clamp01(albedo),
            roughness=clamp01(roughness),
            metallic=clamp01(metallic),
            layer_order=tuple(layer_ids),
        ),
        errors=(),
    )


def sample_bilinear(map_array: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample an atlas at UV pixel centers with clamped edge behavior."""

    height, width = map_array.shape[:2]
    coords = np.asarray(uv, dtype=np.float64)
    x = coords[..., 0] * width - 0.5
    y = coords[..., 1] * height - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[..., None] if map_array.ndim == 3 else x - x0
    fy = (y - y0)[..., None] if map_array.ndim == 3 else y - y0
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x0 = np.clip(x0, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    c00 = map_array[y0, x0]
    c10 = map_array[y0, x1]
    c01 = map_array[y1, x0]
    c11 = map_array[y1, x1]
    top = c00 * (1.0 - fx) + c10 * fx
    bottom = c01 * (1.0 - fx) + c11 * fx
    return top * (1.0 - fy) + bottom * fy
