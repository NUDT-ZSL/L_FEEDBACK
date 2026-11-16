"""PNG channel export, pixel statistics, and consistency-report generation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import __version__
from .gbuffer import GBUFFER_ATTACHMENTS, layout_manifest
from .mathutil import sha256_bytes, tone_map
from .renderer import RenderResult
from .scene import Scene


REPORT_SCHEMA = "deferred-validation-report"
REPORT_SCHEMA_VERSION = 1


def _as_rgb(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    return np.where(value <= 0.0031308, value * 12.92, 1.055 * np.power(value, 1.0 / 2.4) - 0.055)


def _u8(rgb: np.ndarray) -> np.ndarray:
    return np.rint(_as_rgb(rgb) * 255.0).astype(np.uint8)


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_u8(rgb), mode="RGB").save(path, format="PNG", compress_level=9, optimize=False)


def _save_gray(path: Path, values: np.ndarray, coverage: np.ndarray | None = None) -> None:
    gray = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    if coverage is not None:
        rgb *= np.clip(coverage[..., None], 0.0, 1.0)
    _save_rgb(path, rgb)


def _stats(array: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 2:
        values = values[..., None]
    if mask is not None:
        flat_mask = np.asarray(mask, dtype=bool).reshape(-1)
        values = values.reshape(-1, values.shape[-1])[flat_mask]
    else:
        values = values.reshape(-1, values.shape[-1])
    count = int(values.shape[0])
    channel_names = ("r", "g", "b", "a")
    channels: dict[str, Any] = {}
    for index in range(values.shape[1]):
        channel = values[:, index]
        if count == 0:
            channels[channel_names[index]] = {
                "mean": 0.0,
                "variance": 0.0,
                "stddev": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
            continue
        channels[channel_names[index]] = {
            "mean": float(np.mean(channel)),
            "variance": float(np.var(channel)),
            "stddev": float(np.std(channel)),
            "min": float(np.min(channel)),
            "max": float(np.max(channel)),
        }
    luminance = {}
    if values.shape[1] >= 3 and count:
        lum = values[:, 0] * 0.2126 + values[:, 1] * 0.7152 + values[:, 2] * 0.0722
        luminance = {
            "mean": float(np.mean(lum)),
            "variance": float(np.var(lum)),
            "stddev": float(np.std(lum)),
            "min": float(np.min(lum)),
            "max": float(np.max(lum)),
        }
    return {
        "pixel_count": count,
        "channels": channels,
        "luminance": luminance,
    }


def _difference_stats(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(f"difference shape mismatch: {first.shape} vs {second.shape}")
    if first.ndim == 2:
        first = first[..., None]
        second = second[..., None]
    first = first[..., :3]
    second = second[..., :3]
    if mask is not None:
        valid = np.asarray(mask, dtype=bool)
        first = first[valid]
        second = second[valid]
    signed = second - first
    absolute = np.abs(signed)
    if absolute.size == 0:
        return {
            "pixel_count": 0,
            "mean_signed": 0.0,
            "mean_abs": 0.0,
            "rms": 0.0,
            "max_abs": 0.0,
            "p95_abs": 0.0,
            "p99_abs": 0.0,
            "fraction_over_1e-5": 0.0,
            "fraction_over_1_over_255": 0.0,
            "fraction_over_1_percent": 0.0,
        }
    per_pixel = np.max(absolute, axis=1)
    return {
        "pixel_count": int(absolute.shape[0]),
        "mean_signed": float(np.mean(signed)),
        "mean_abs": float(np.mean(absolute)),
        "rms": float(np.sqrt(np.mean(signed * signed))),
        "max_abs": float(np.max(absolute)),
        "p95_abs": float(np.percentile(absolute, 95.0)),
        "p99_abs": float(np.percentile(absolute, 99.0)),
        "fraction_over_1e-5": float(np.mean(per_pixel > 1e-5)),
        "fraction_over_1_over_255": float(np.mean(per_pixel > (1.0 / 255.0))),
        "fraction_over_1_percent": float(np.mean(per_pixel > 0.01)),
    }


def _artifact(path: Path, report_dir: Path, encoding: str) -> dict[str, Any]:
    data = path.read_bytes()
    with Image.open(path) as image:
        width, height = image.size
    return {
        "file": str(path.relative_to(report_dir)),
        "sha256": sha256_bytes(data),
        "width": width,
        "height": height,
        "encoding": encoding,
    }


def build_channel_images(scene: Scene, result: RenderResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    coverage = result.coverage
    bounds_min = scene.bounds_min.astype(np.float64)
    bounds_extent = np.maximum(scene.bounds_max.astype(np.float64) - bounds_min, 1e-8)

    position = result.gbuffer_target["position"]
    position_vis = np.clip((position[..., :3].astype(np.float64) - bounds_min) / bounds_extent, 0.0, 1.0)
    position_vis *= coverage[..., None]
    paths["g_position"] = output_dir / "g_position.png"
    _save_rgb(paths["g_position"], position_vis)

    normal = result.gbuffer_target["normal"]
    normal_vis = normal[..., :3].astype(np.float64) * 0.5 + 0.5
    normal_vis *= coverage[..., None]
    paths["g_normal"] = output_dir / "g_normal.png"
    _save_rgb(paths["g_normal"], normal_vis)

    albedo = result.gbuffer_target["albedo"]
    albedo_vis = linear_to_srgb(albedo[..., :3]) * coverage[..., None]
    paths["g_albedo"] = output_dir / "g_albedo.png"
    _save_rgb(paths["g_albedo"], albedo_vis)

    material = result.gbuffer_target["material"]
    paths["g_metallic"] = output_dir / "g_metallic.png"
    _save_gray(paths["g_metallic"], material[..., 0], coverage)
    paths["g_roughness"] = output_dir / "g_roughness.png"
    _save_gray(paths["g_roughness"], material[..., 1], coverage)
    depth_normalized = np.clip(
        (position[..., 3] - scene.camera.near) / max(scene.camera.far - scene.camera.near, 1e-8),
        0.0,
        1.0,
    )
    paths["g_depth"] = output_dir / "g_depth.png"
    _save_gray(paths["g_depth"], depth_normalized, coverage)

    direct_preview = tone_map(result.direct, scene.rendering.exposure, scene.rendering.tone_mapper)
    indirect_preview = tone_map(result.indirect, scene.rendering.exposure, scene.rendering.tone_mapper)
    paths["direct"] = output_dir / "direct.png"
    _save_rgb(paths["direct"], direct_preview)
    paths["indirect"] = output_dir / "indirect.png"
    _save_rgb(paths["indirect"], indirect_preview)
    paths["final"] = output_dir / "final.png"
    _save_rgb(paths["final"], result.final_rgb.astype(np.float64) / 255.0)
    return paths


def build_report(scene: Scene, result: RenderResult, output_dir: Path, paths: dict[str, Path]) -> dict[str, Any]:
    coverage_mask = result.coverage > 0.5
    direct_plus_indirect = result.direct + result.indirect
    reconstructed_final = tone_map(
        result.composite, scene.rendering.exposure, scene.rendering.tone_mapper
    )
    final_normalized = result.final_rgb.astype(np.float64) / 255.0
    nonlinear_preview_sum = (
        tone_map(result.direct, scene.rendering.exposure, scene.rendering.tone_mapper)
        + tone_map(result.indirect, scene.rendering.exposure, scene.rendering.tone_mapper)
    )

    material = result.gbuffer_target["material"]
    position = result.gbuffer_target["position"]
    statistics = {
        "gbuffer_position": _stats(position, coverage_mask),
        "gbuffer_normal": _stats(result.gbuffer_target["normal"], coverage_mask),
        "gbuffer_albedo": _stats(result.gbuffer_target["albedo"], coverage_mask),
        "gbuffer_metallic": _stats(material[..., 0], coverage_mask),
        "gbuffer_roughness": _stats(material[..., 1], coverage_mask),
        "gbuffer_depth": _stats(position[..., 3], coverage_mask),
        "direct_linear": _stats(result.direct),
        "indirect_linear": _stats(result.indirect),
        "direct_plus_indirect_linear": _stats(direct_plus_indirect),
        "composite_linear": _stats(result.composite),
        "final_srgb_u8": _stats(result.final_rgb.astype(np.float64) / 255.0),
    }
    consistency = {
        "direct_plus_indirect_vs_composite": {
            "a": "direct_linear + indirect_linear",
            "b": "composite_linear",
            **_difference_stats(direct_plus_indirect, result.composite),
        },
        "reconstructed_tone_map_vs_final": {
            "a": "tone_map(composite_linear)",
            "b": "final.png / 255",
            **_difference_stats(reconstructed_final, final_normalized),
        },
        "direct_vs_indirect": {
            "a": "direct_linear",
            "b": "indirect_linear",
            **_difference_stats(result.direct, result.indirect),
        },
        "nonlinear_preview_sum_vs_final": {
            "a": "tone_map(direct_linear) + tone_map(indirect_linear)",
            "b": "final.png / 255",
            "note": "Diagnostic only; tonal compression makes this sum non-additive.",
            **_difference_stats(nonlinear_preview_sum, final_normalized),
        },
    }
    coverage_fraction = float(np.mean(coverage_mask))
    checks = [
        {
            "name": "direct_plus_indirect_matches_composite",
            "ok": consistency["direct_plus_indirect_vs_composite"]["max_abs"] <= 0.02,
            "value": consistency["direct_plus_indirect_vs_composite"]["max_abs"],
            "limit": 0.02,
        },
        {
            "name": "tone_map_reconstruction_matches_final_png",
            "ok": consistency["reconstructed_tone_map_vs_final"]["max_abs"] <= (2.5 / 255.0),
            "value": consistency["reconstructed_tone_map_vs_final"]["max_abs"],
            "limit": 2.5 / 255.0,
        },
        {
            "name": "scene_visible_coverage",
            "ok": coverage_fraction >= 0.01,
            "value": coverage_fraction,
            "limit": 0.01,
        },
    ]
    artifact_meta = {
        name: _artifact(path, output_dir, _artifact_encoding(name))
        for name, path in sorted(paths.items())
    }
    report = {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "generator_version": __version__,
        "scene": {
            "source_name": scene.source_path.name,
            "checksums": scene.checksums,
            "bounds_min": scene.bounds_min.astype(float).tolist(),
            "bounds_max": scene.bounds_max.astype(float).tolist(),
            "node_count": len(scene.nodes),
            "mesh_count": len(scene.meshes),
            "material_count": len(scene.materials),
            "light_count": len(scene.lights),
            "lights": [
                {
                    "name": light.name,
                    "type": light.type,
                    "casts_shadow": light.casts_shadow,
                }
                for light in scene.lights
            ],
        },
        "gl": result.gl_info,
        "rendering": {
            "width": scene.rendering.width,
            "height": scene.rendering.height,
            "ssaa": scene.rendering.ssaa,
            "internal_width": result.internal_width,
            "internal_height": result.internal_height,
            "exposure": scene.rendering.exposure,
            "tone_mapper": scene.rendering.tone_mapper,
            "shadows": scene.rendering.shadows.__dict__,
            "shadow_resources": result.shadow_resources,
            "indirect": {
                "enabled": scene.indirect.enabled,
                "probe_type": scene.indirect.probe_type,
                "sky_color": scene.indirect.sky_color.astype(float).tolist(),
                "ground_color": scene.indirect.ground_color.astype(float).tolist(),
                "intensity": scene.indirect.intensity,
            },
        },
        "gbuffer_layout": layout_manifest(),
        "artifacts": artifact_meta,
        "statistics": statistics,
        "consistency": consistency,
        "coverage": {
            "pixel_count": int(np.count_nonzero(coverage_mask)),
            "pixel_fraction": coverage_fraction,
            "partial_coverage_fraction": float(np.mean((result.coverage > 0.0) & (result.coverage < 0.999))),
        },
        "analysis_policy": {
            "pixel_tolerance_default": 0.01,
            "float_noise_tolerance": 1.0 / 255.0,
            "significant_pixel_fraction": 0.0005,
            "comparison": "max per-pixel RGB absolute difference in normalized sRGB space",
        },
        "validation": {
            "status": "pass" if all(check["ok"] for check in checks) else "fail",
            "checks": checks,
        },
    }
    return report


def _artifact_encoding(name: str) -> str:
    for attachment in GBUFFER_ATTACHMENTS:
        if name == f"g_{attachment.name}":
            return attachment.encoding
    return {
        "direct": "tonemapped_linear_rgb",
        "indirect": "tonemapped_linear_rgb",
        "final": "tonemapped_srgb",
        "g_metallic": "linear_scalar",
        "g_roughness": "linear_scalar",
        "g_depth": "near_far_normalized",
    }.get(name, "unknown")


def write_outputs(scene: Scene, result: RenderResult, output_dir: str | Path) -> dict[str, Any]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = build_channel_images(scene, result, directory)
    report = build_report(scene, result, directory, paths)
    report_path = directory / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report
