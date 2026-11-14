from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigurationError


@dataclass(frozen=True)
class LensConfig:
    focal_length_mm: float
    f_number: float
    sensor_width_mm: float
    near_plane_m: float
    far_plane_m: float
    max_blur_radius_px: float
    invalid_depth_fallback_m: float
    depth_tolerance_m: float


@dataclass(frozen=True)
class ApertureConfig:
    blades: int
    rotation_deg: float
    supersample: int


@dataclass(frozen=True)
class RenderConfig:
    depth_slices: int
    gather_rounds: int
    png_compression: int
    output_pattern: str


@dataclass(frozen=True)
class ValidationConfig:
    slice_max_abs_error_8bit: float


@dataclass(frozen=True)
class PipelineConfig:
    frame_rate: float
    frame_start: int
    frame_count: int
    color_pattern: str
    depth_pattern: str
    focus_track: str
    seed: int
    lens: LensConfig
    aperture: ApertureConfig
    render: RenderConfig
    validation: ValidationConfig
    source_path: Path | None = None


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be a JSON object")
    return value


def _number(
    mapping: dict[str, Any],
    key: str,
    name: str,
    *,
    default: float | None = None,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if key in mapping:
        value = mapping[key]
    elif default is not None:
        value = default
    else:
        raise ConfigurationError(f"missing required parameter {name}.{key}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name}.{key} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ConfigurationError(f"{name}.{key} must be finite")
    if minimum is not None:
        if strict_minimum and value <= minimum:
            raise ConfigurationError(f"{name}.{key} must be > {minimum:g}")
        if not strict_minimum and value < minimum:
            raise ConfigurationError(f"{name}.{key} must be >= {minimum:g}")
    return value


def _integer(
    mapping: dict[str, Any],
    key: str,
    name: str,
    *,
    default: int | None = None,
    minimum: int | None = None,
) -> int:
    if key in mapping:
        value = mapping[key]
    elif default is not None:
        value = default
    else:
        raise ConfigurationError(f"missing required parameter {name}.{key}")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name}.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{name}.{key} must be >= {minimum}")
    return int(value)


def _string(
    mapping: dict[str, Any], key: str, name: str, *, default: str | None = None
) -> str:
    if key in mapping:
        value = mapping[key]
    elif default is not None:
        value = default
    else:
        raise ConfigurationError(f"missing required parameter {name}.{key}")
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{name}.{key} must be a non-empty string")
    return value


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path).resolve()
    try:
        with path.open("r", encoding="utf-8") as handle:
            root = json.load(handle)
    except OSError as exc:
        raise ConfigurationError(f"cannot read config {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"invalid JSON in config {path}: {exc}") from exc
    root = _as_mapping(root, "config")

    lens_raw = _as_mapping(root.get("lens"), "lens")
    aperture_raw = _as_mapping(root.get("aperture", {}), "aperture")
    render_raw = _as_mapping(root.get("render", {}), "render")
    validation_raw = _as_mapping(root.get("validation", {}), "validation")

    focal_mm = _number(
        lens_raw, "focal_length_mm", "lens", minimum=0.0, strict_minimum=True
    )
    f_number = _number(
        lens_raw, "f_number", "lens", minimum=0.0, strict_minimum=True
    )
    sensor_width_mm = _number(
        lens_raw, "sensor_width_mm", "lens", minimum=0.0, strict_minimum=True
    )
    near_m = _number(
        lens_raw, "near_plane_m", "lens", minimum=0.0, strict_minimum=True
    )
    far_m = _number(
        lens_raw, "far_plane_m", "lens", minimum=0.0, strict_minimum=True
    )
    if far_m <= near_m:
        raise ConfigurationError("lens.far_plane_m must be greater than lens.near_plane_m")
    if near_m <= focal_mm / 1000.0:
        raise ConfigurationError(
            "lens.near_plane_m must be greater than the focal length in meters"
        )
    max_radius = _number(
        lens_raw, "max_blur_radius_px", "lens", minimum=0.0
    )
    if max_radius > 128.0:
        raise ConfigurationError("lens.max_blur_radius_px must be <= 128")
    fallback_m = _number(
        lens_raw,
        "invalid_depth_fallback_m",
        "lens",
        default=far_m,
        minimum=0.0,
        strict_minimum=True,
    )
    depth_tolerance = _number(
        lens_raw,
        "depth_tolerance_m",
        "lens",
        default=max(1e-6, (far_m - near_m) * 1e-4),
        minimum=0.0,
    )

    blades = _integer(aperture_raw, "blades", "aperture", default=0, minimum=0)
    rotation_deg = _number(
        aperture_raw, "rotation_deg", "aperture", default=0.0
    )
    supersample = _integer(
        aperture_raw, "supersample", "aperture", default=8, minimum=1
    )
    if supersample > 32:
        raise ConfigurationError("aperture.supersample must be <= 32")

    depth_slices = _integer(
        render_raw, "depth_slices", "render", default=8, minimum=1
    )
    gather_rounds = _integer(
        render_raw, "gather_rounds", "render", default=2, minimum=1
    )
    if depth_slices > 256:
        raise ConfigurationError("render.depth_slices must be <= 256")
    if gather_rounds > 32:
        raise ConfigurationError("render.gather_rounds must be <= 32")
    png_compression = _integer(
        render_raw, "png_compression", "render", default=3, minimum=0
    )
    if png_compression > 9:
        raise ConfigurationError("render.png_compression must be in [0, 9]")
    output_pattern = _string(
        render_raw,
        "output_pattern",
        "render",
        default="frame_{frame:04d}.png",
    )
    if "{frame" not in output_pattern and "{index" not in output_pattern:
        raise ConfigurationError(
            "render.output_pattern must contain {frame...} or {index...}"
        )

    frame_rate = _number(root, "frame_rate", "config", minimum=0.0, strict_minimum=True)
    frame_start = _integer(root, "frame_start", "config", default=0, minimum=0)
    frame_count = _integer(root, "frame_count", "config", minimum=1)
    color_pattern = _string(root, "color_pattern", "config")
    depth_pattern = _string(root, "depth_pattern", "config")
    for pattern, name in (
        (color_pattern, "color_pattern"),
        (depth_pattern, "depth_pattern"),
    ):
        if "{frame" not in pattern and "{index" not in pattern:
            raise ConfigurationError(
                f"config.{name} must contain {{frame...}} or {{index...}}"
            )
    focus_track = _string(root, "focus_track", "config")
    seed = _integer(root, "seed", "config", default=0, minimum=0)
    if seed > 2**63 - 1:
        raise ConfigurationError("config.seed must fit in signed 64-bit")

    slice_tolerance = _number(
        validation_raw,
        "slice_max_abs_error_8bit",
        "validation",
        default=2.0,
        minimum=0.0,
    )

    return PipelineConfig(
        frame_rate=frame_rate,
        frame_start=frame_start,
        frame_count=frame_count,
        color_pattern=color_pattern,
        depth_pattern=depth_pattern,
        focus_track=focus_track,
        seed=seed,
        lens=LensConfig(
            focal_length_mm=focal_mm,
            f_number=f_number,
            sensor_width_mm=sensor_width_mm,
            near_plane_m=near_m,
            far_plane_m=far_m,
            max_blur_radius_px=max_radius,
            invalid_depth_fallback_m=fallback_m,
            depth_tolerance_m=depth_tolerance,
        ),
        aperture=ApertureConfig(
            blades=blades, rotation_deg=rotation_deg, supersample=supersample
        ),
        render=RenderConfig(
            depth_slices=depth_slices,
            gather_rounds=gather_rounds,
            png_compression=png_compression,
            output_pattern=output_pattern,
        ),
        validation=ValidationConfig(slice_max_abs_error_8bit=slice_tolerance),
        source_path=path,
    )


def format_frame_name(pattern: str, sequence_index: int, frame_index: int) -> str:
    try:
        return pattern.format(frame=frame_index, index=sequence_index)
    except (KeyError, ValueError, IndexError) as exc:
        raise ConfigurationError(f"invalid frame pattern {pattern!r}: {exc}") from exc
