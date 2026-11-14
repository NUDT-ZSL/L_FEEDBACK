"""Bake configuration parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .errors import invalid_config, missing_file


@dataclass(frozen=True)
class DirectionalLight:
    direction: tuple[float, float, float]
    color: tuple[float, float, float]
    intensity: float


@dataclass(frozen=True)
class HemisphereLight:
    sky_color: tuple[float, float, float]
    ground_color: tuple[float, float, float]
    intensity: float


@dataclass(frozen=True)
class BakeConfig:
    resolution: int
    output_bits: int
    output_gamma: float
    flip_green: bool
    directional: DirectionalLight
    hemisphere: HemisphereLight
    view_direction: tuple[float, float, float]
    alpha_epsilon: float
    output_names: dict[str, str]
    input_dir: Path | None
    output_dir: Path | None


def _as_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise invalid_config(f"{where} must be a JSON object")
    return value


def _number(value: Any, where: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise invalid_config(f"{where} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise invalid_config(f"{where} must be finite")
    if minimum is not None and number < minimum:
        raise invalid_config(f"{where} must be >= {minimum}")
    return number


def _vector3(
    value: Any, where: str, *, nonnegative: bool = False
) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise invalid_config(f"{where} must be an array of three numbers")
    result = []
    for index, component in enumerate(value):
        number = _number(
            component,
            f"{where}[{index}]",
            minimum=0.0 if nonnegative else None,
        )
        result.append(number)
    return (result[0], result[1], result[2])


def _normalize_direction(
    value: Any, where: str
) -> tuple[float, float, float]:
    vector = np.asarray(_vector3(value, where), dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise invalid_config(f"{where} must not be a zero-length vector")
    normalized = vector / length
    return (float(normalized[0]), float(normalized[1]), float(normalized[2]))


def _lookup_object(
    document: dict[str, Any], keys: tuple[str, ...], default: dict[str, Any] | None = None
) -> dict[str, Any]:
    for key in keys:
        if key in document:
            return _as_object(document[key], key)
    if default is not None:
        return default
    raise invalid_config(f"missing required configuration object: {keys[0]}")


def _first(mapping: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _validate_output_name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise invalid_config(f"{where} must be a non-empty PNG filename")
    name = value.strip()
    if Path(name).name != name or "/" in name or "\\" in name:
        raise invalid_config(f"{where} must be a plain filename")
    if not name.lower().endswith(".png"):
        raise invalid_config(f"{where} must end with .png")
    return name


def load_config(path: str | Path) -> BakeConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise missing_file(f"Configuration file not found: {config_path}")
    if not config_path.is_file():
        raise missing_file(f"Configuration path is not a file: {config_path}")

    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise invalid_config(f"Cannot read configuration file {config_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise invalid_config(
            f"Configuration file {config_path} is not valid JSON: {exc}"
        ) from exc
    document = _as_object(document, "configuration")

    resolution_raw = _first(
        document, ("resolution", "output_resolution", "bake_resolution", "size"), 256
    )
    if isinstance(resolution_raw, bool) or not isinstance(resolution_raw, int):
        raise invalid_config("resolution must be an integer")
    resolution = int(resolution_raw)
    if resolution not in (128, 256, 512):
        raise invalid_config("resolution must be one of 128, 256, 512")

    bits_raw = _first(
        document,
        (
            "output_bits",
            "output_bit_depth",
            "outputBits",
            "outputBitDepth",
            "bit_depth",
            "bits",
        ),
        8,
    )
    if isinstance(bits_raw, bool) or not isinstance(bits_raw, int):
        raise invalid_config("output_bits must be an integer")
    output_bits = int(bits_raw)
    if output_bits not in (8, 16):
        raise invalid_config("output_bits must be 8 or 16")

    gamma = _number(
        _first(document, ("output_gamma", "outputGamma", "gamma"), 2.2),
        "output_gamma",
    )
    if not (math.isclose(gamma, 1.0, abs_tol=1e-12) or math.isclose(gamma, 2.2, abs_tol=1e-12)):
        raise invalid_config("output_gamma must be 1.0 or 2.2")
    gamma = 1.0 if math.isclose(gamma, 1.0, abs_tol=1e-12) else 2.2

    flip_green = _first(document, ("flip_green", "flipGreen"), False)
    if not isinstance(flip_green, bool):
        raise invalid_config("flip_green must be a boolean")

    directional_obj = _lookup_object(
        document,
        ("directional_light", "directionalLight", "directional", "light", "sun"),
    )
    direction_value = _first(
        directional_obj,
        ("direction", "direction_to_light", "directionToLight", "to_light", "toLight", "vector"),
    )
    if direction_value is None:
        raise invalid_config("directional_light.direction is required")
    directional = DirectionalLight(
        direction=_normalize_direction(
            direction_value, "directional_light.direction"
        ),
        color=_vector3(
            _first(directional_obj, ("color", "colour"), [1.0, 1.0, 1.0]),
            "directional_light.color",
            nonnegative=True,
        ),
        intensity=_number(
            _first(directional_obj, ("intensity", "strength", "radiance"), 1.0),
            "directional_light.intensity",
            minimum=0.0,
        ),
    )

    hemisphere_obj = _lookup_object(
        document,
        (
            "hemisphere_light",
            "hemisphereLight",
            "hemisphere",
            "ambient_light",
            "ambient",
            "environment",
            "hemi",
        ),
        default={},
    )
    common_color = _first(hemisphere_obj, ("color", "colour"))
    sky_value = _first(
        hemisphere_obj,
        ("sky_color", "skyColor", "sky", "color", "colour"),
        common_color,
    )
    ground_value = _first(
        hemisphere_obj,
        ("ground_color", "groundColor", "ground", "color", "colour"),
        common_color,
    )
    if sky_value is None:
        sky_value = [1.0, 1.0, 1.0]
    if ground_value is None:
        ground_value = sky_value
    hemisphere = HemisphereLight(
        sky_color=_vector3(
            sky_value, "hemisphere_light.sky_color", nonnegative=True
        ),
        ground_color=_vector3(
            ground_value, "hemisphere_light.ground_color", nonnegative=True
        ),
        intensity=_number(
            _first(hemisphere_obj, ("intensity", "strength"), 1.0),
            "hemisphere_light.intensity",
            minimum=0.0,
        ),
    )

    view_direction = _normalize_direction(
        _first(
            document,
            ("view_direction", "viewDirection", "camera_direction"),
            [0.0, 0.0, 1.0],
        ),
        "view_direction",
    )
    alpha_epsilon = _number(
        _first(
            document,
            ("alpha_epsilon", "alphaEpsilon", "roughness_epsilon"),
            1.0e-3,
        ),
        "alpha_epsilon",
        minimum=1.0e-8,
    )

    names_obj = _as_object(
        _first(
            document,
            ("output_names", "outputNames"),
            {
                "irradiance": "irradiance.png",
                "specular": "specular.png",
                "final_color": "final_color.png",
            },
        ),
        "output_names",
    )
    output_names = {
        "irradiance": _validate_output_name(
            _first(names_obj, ("irradiance", "diffuse", "irradiance_map")),
            "output_names.irradiance",
        ),
        "specular": _validate_output_name(
            _first(names_obj, ("specular", "specular_map", "specularMap")),
            "output_names.specular",
        ),
        "final_color": _validate_output_name(
            _first(
                names_obj,
                ("final_color", "finalColor", "final", "color", "color_map"),
            ),
            "output_names.final_color",
        ),
    }
    if len({name.casefold() for name in output_names.values()}) != 3:
        raise invalid_config("output image filenames must be distinct")

    input_value = _first(
        document,
        ("input_dir", "inputDir", "input_directory", "inputDirectory", "input"),
    )
    output_value = _first(
        document,
        ("output_dir", "outputDir", "output_directory", "outputDirectory", "output"),
    )
    input_dir = Path(input_value) if isinstance(input_value, str) else None
    output_dir = Path(output_value) if isinstance(output_value, str) else None
    if input_value is not None and input_dir is None:
        raise invalid_config("input_dir must be a string path")
    if output_value is not None and output_dir is None:
        raise invalid_config("output_dir must be a string path")

    return BakeConfig(
        resolution=resolution,
        output_bits=output_bits,
        output_gamma=gamma,
        flip_green=flip_green,
        directional=directional,
        hemisphere=hemisphere,
        view_direction=view_direction,
        alpha_epsilon=alpha_epsilon,
        output_names=output_names,
        input_dir=input_dir,
        output_dir=output_dir,
    )

