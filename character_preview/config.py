from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .errors import ConfigError


@dataclass(frozen=True)
class RenderConfig:
    width: int = 640
    height: int = 640
    shadow_map_size: int = 1024
    point_shadow_size: int = 256
    max_point_lights: int = 4
    directional_light_scale: float = 1.0
    point_light_scale: float = 1.0
    exposure: float = 0.8
    shadow_bias: float = 0.0015
    shadow_normal_bias: float = 0.025
    point_shadow_bias: float = 0.012
    point_shadow_normal_bias: float = 0.025
    frame_count: int = 3
    time_step: float = 0.25
    camera_orbit_speed_deg: float = 12.0
    output_dir: str = "output"


DEFAULTS = RenderConfig()
HARD_MAX_POINT_LIGHTS = 8

INT_RULES: dict[str, tuple[int, int]] = {
    "width": (64, 4096),
    "height": (64, 4096),
    "shadow_map_size": (256, 4096),
    "point_shadow_size": (64, 2048),
    "max_point_lights": (0, HARD_MAX_POINT_LIGHTS),
    "frame_count": (1, 10000),
}
FLOAT_RULES: dict[str, tuple[float, float]] = {
    "directional_light_scale": (0.0, 100.0),
    "point_light_scale": (0.0, 100.0),
    "exposure": (0.05, 16.0),
    "shadow_bias": (0.0, 0.05),
    "shadow_normal_bias": (0.0, 0.25),
    "point_shadow_bias": (0.0, 0.2),
    "point_shadow_normal_bias": (0.0, 0.25),
    "time_step": (0.0, 120.0),
    "camera_orbit_speed_deg": (-360.0, 360.0),
}


def _warn(warnings: list[str], field: str, reason: str, default: Any) -> None:
    warnings.append(f"config.{field}: {reason}; using default {default!r}")


def config_from_dict(raw: Any) -> tuple[RenderConfig, list[str]]:
    if not isinstance(raw, dict):
        raise ConfigError("E_CONFIG_SCHEMA", "config root must be a JSON object")
    warnings: list[str] = []
    field_names = list(RenderConfig.__dataclass_fields__)
    fields = set(field_names)
    for unknown in sorted(set(raw) - fields):
        warnings.append(f"config.{unknown}: unknown field ignored")

    values: dict[str, Any] = {}
    defaults = DEFAULTS
    for field in field_names:
        default = getattr(defaults, field)
        if field not in raw:
            _warn(warnings, field, "missing", default)
            values[field] = default
            continue
        value = raw[field]
        if field in INT_RULES:
            minimum, maximum = INT_RULES[field]
            if not isinstance(value, int) or isinstance(value, bool):
                _warn(warnings, field, "must be an integer", default)
                values[field] = default
            elif not minimum <= value <= maximum:
                _warn(warnings, field, f"out of range [{minimum}, {maximum}]", default)
                values[field] = default
            else:
                values[field] = value
        elif field in FLOAT_RULES:
            minimum, maximum = FLOAT_RULES[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                _warn(warnings, field, "must be numeric", default)
                values[field] = default
            elif not math.isfinite(float(value)):
                _warn(warnings, field, "must be finite", default)
                values[field] = default
            elif not minimum <= float(value) <= maximum:
                _warn(warnings, field, f"out of range [{minimum}, {maximum}]", default)
                values[field] = default
            else:
                values[field] = float(value)
        elif field == "output_dir":
            if not isinstance(value, str) or not value.strip():
                _warn(warnings, field, "must be a non-empty string", default)
                values[field] = default
            else:
                values[field] = value
        else:
            values[field] = value
    return RenderConfig(**values), warnings


def load_config(path: str | Path | None) -> tuple[RenderConfig, list[str]]:
    if path is None:
        return DEFAULTS, []
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError("E_CONFIG_READ", f"cannot read config {config_path}", str(exc)) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            "E_CONFIG_JSON",
            f"invalid JSON in {config_path}",
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}",
        ) from exc
    return config_from_dict(raw)


def with_overrides(config: RenderConfig, overrides: dict[str, Any]) -> RenderConfig:
    clean = {key: value for key, value in overrides.items() if value is not None}
    return replace(config, **clean)

