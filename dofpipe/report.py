from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig

REPORT_SCHEMA_VERSION = 1


def global_report_parameters(
    config: PipelineConfig, width: int, height: int
) -> dict[str, Any]:
    return {
        "frame_rate": float(config.frame_rate),
        "frame_start": int(config.frame_start),
        "frame_count": int(config.frame_count),
        "image_width": int(width),
        "image_height": int(height),
        "near_plane_m": float(config.lens.near_plane_m),
        "far_plane_m": float(config.lens.far_plane_m),
        "focal_length_mm": float(config.lens.focal_length_mm),
        "f_number": float(config.lens.f_number),
        "sensor_width_mm": float(config.lens.sensor_width_mm),
        "max_blur_radius_px": float(config.lens.max_blur_radius_px),
        "invalid_depth_fallback_m": float(config.lens.invalid_depth_fallback_m),
        "depth_tolerance_m": float(config.lens.depth_tolerance_m),
        "aperture_blades": int(config.aperture.blades),
        "aperture_rotation_deg": float(config.aperture.rotation_deg),
        "aperture_supersample": int(config.aperture.supersample),
        "depth_slices": int(config.render.depth_slices),
        "gather_rounds": int(config.render.gather_rounds),
        "seed": int(config.seed),
        "coc_units": "pixel_radius",
        "kernel_radius_quantization_px": 0.5,
        "slice_max_abs_error_8bit": float(
            config.validation.slice_max_abs_error_8bit
        ),
    }


def frame_report_entry(
    *,
    frame_index: int,
    sequence_index: int,
    output_file: str,
    focus_distance_m: float,
    coc_radius_px: Any,
    invalid_depth_pixels: int,
    clamped_depth_pixels: int,
    blades: int,
    max_blur_radius_px: float,
    render_time_ms: float,
) -> dict[str, Any]:
    radius = np.asarray(coc_radius_px, dtype=np.float64)
    minimum = float(np.min(radius))
    maximum = float(np.max(radius))
    mean = float(np.mean(radius))
    return {
        "frame_index": int(frame_index),
        "sequence_index": int(sequence_index),
        "output_file": output_file,
        "focus_distance_m": float(focus_distance_m),
        "coc_min_px": minimum,
        "coc_max_px": maximum,
        "coc_mean_px": mean,
        "coc_min_diameter_px": 2.0 * minimum,
        "coc_max_diameter_px": 2.0 * maximum,
        "coc_mean_diameter_px": 2.0 * mean,
        "aperture_blades": int(blades),
        "max_blur_radius_px": float(max_blur_radius_px),
        "invalid_depth_pixels": int(invalid_depth_pixels),
        "clamped_depth_pixels": int(clamped_depth_pixels),
        "render_time_ms": float(render_time_ms),
    }


def build_report(
    *,
    config: PipelineConfig,
    width: int,
    height: int,
    frames: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    status: str,
    error: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": status,
        "completed": status == "completed",
        "error": error,
        "parameters": global_report_parameters(config, width, height),
        "warnings": warnings,
        "total_frames": len(frames),
        "expected_total_frames": int(config.frame_count),
        "frames": frames,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
