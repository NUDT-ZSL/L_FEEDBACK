"""Scene JSON loading and strict normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import MaterialError
from .util import (
    canonical_lights,
    ensure_finite,
    normalize_vector,
    stable_json_text,
)


DEFAULT_SCENE: dict[str, Any] = {
    "assets": {
        "defaults": {"roughness": 0.5, "metallic": 0.0},
    },
    "surface": {
        "geometry_normal": [0.0, 0.0, 1.0],
        "uv_axis_u": [1.0, 0.0, 0.0],
        "uv_axis_v": [0.0, -1.0, 0.0],
        "uv_origin": [0.0, 0.0, 0.0],
        "material_size": [2.0, 2.0],
        "normal_green_sign": 1.0,
        "seam_smoothing_radius": 1,
    },
    "environment": {"color": [0.72, 0.78, 0.92], "intensity": 0.25},
    "camera": {
        "projection": "perspective",
        "position": [0.0, 0.0, 3.2],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0],
        "fov_y_degrees": 35.0,
        "ortho_scale": 2.4,
        "resolution": [384, 384],
    },
    "baking": {
        "chunk_rows": 32,
        "view_direction": [0.0, 0.0, 1.0],
        "exposure": 1.0,
        "normal_strength": 1.0,
    },
    "validation": {
        "seed": 20261115,
        "max_samples": 2048,
        "stratified": True,
        "normal_length_tolerance": 0.03,
        "angle_threshold_deg": 32.0,
        "seam_angle_threshold_deg": 25.0,
        "reference_gain": 0.55,
        "reference_min_gradient": 0.004,
        "reference_tilt_min": 0.04,
        "seam_width": 2,
        "tile_size": 64,
        "hard_edge_color_threshold": 0.22,
        "max_entries": 512,
    },
    "output": {
        "bit_depth": 8,
        "preview_resolution": [384, 384],
        "filename_pattern": "{kind}.png",
        "background": [0.025, 0.030, 0.038],
    },
}


@dataclass(frozen=True)
class TextureSpec:
    path: Path | None
    label: str
    options: dict[str, Any]


@dataclass
class Scene:
    path: Path
    raw: dict[str, Any]
    assets: dict[str, TextureSpec]
    defaults: dict[str, float]
    surface: dict[str, Any]
    lights: list[dict[str, Any]]
    environment: dict[str, Any]
    camera: dict[str, Any]
    baking: dict[str, Any]
    validation: dict[str, Any]
    output: dict[str, Any]

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "assets": {
                label: (
                    None
                    if spec.path is None
                    else spec.path.resolve().as_posix()
                )
                for label, spec in self.assets.items()
            },
        }

    def stable_text(self) -> str:
        return stable_json_text(self.fingerprint_payload())


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MaterialError("E004", f"{field} must be an object")
    return value


def _vector(
    value: Any, length: int, field: str, *, normalize: bool = False
) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MaterialError("E004", f"{field} must be a numeric vector") from exc
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise MaterialError(
            "E004", f"{field} must contain exactly {length} finite values"
        )
    if normalize:
        length_value = float(np.linalg.norm(vector))
        if length_value <= 1e-12:
            raise MaterialError("E004", f"{field} must be non-zero")
        vector = vector / length_value
    return vector


def _positive_int(value: Any, field: str, *, maximum: int = 1_000_000) -> int:
    if isinstance(value, bool):
        raise MaterialError("E004", f"{field} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise MaterialError("E004", f"{field} must be an integer") from exc
    if integer != value or integer <= 0 or integer > maximum:
        raise MaterialError(
            "E004", f"{field} must be an integer in [1, {maximum}]"
        )
    return integer


def _non_negative_int(value: Any, field: str, *, maximum: int = 1_000_000) -> int:
    if isinstance(value, bool):
        raise MaterialError("E004", f"{field} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise MaterialError("E004", f"{field} must be an integer") from exc
    if integer != value or integer < 0 or integer > maximum:
        raise MaterialError(
            "E004", f"{field} must be an integer in [0, {maximum}]"
        )
    return integer


def _texture_spec(
    value: Any,
    label: str,
    base_dir: Path,
    *,
    required: bool,
) -> TextureSpec:
    if value is None:
        if required:
            raise MaterialError("E004", f"assets.{label} is required")
        return TextureSpec(None, label, {})
    if isinstance(value, str):
        options: dict[str, Any] = {}
        path_text = value
    elif isinstance(value, dict):
        options = dict(value)
        path_text = options.pop("path", None)
        if path_text is None:
            raise MaterialError("E004", f"assets.{label}.path is required")
    else:
        raise MaterialError(
            "E004", f"assets.{label} must be a path string or object"
        )
    if not isinstance(path_text, str) or not path_text.strip():
        raise MaterialError("E004", f"assets.{label}.path must be non-empty")
    relative = Path(path_text)
    path = relative if relative.is_absolute() else base_dir / relative
    return TextureSpec(path=path.resolve(), label=label, options=options)


def _merge_defaults(raw: dict[str, Any], section: str) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise MaterialError("E004", f"{section} must be an object")
    defaults = DEFAULT_SCENE[section]
    if defaults is None:
        return {}
    merged = dict(defaults) if isinstance(defaults, dict) else defaults
    if isinstance(defaults, dict) and isinstance(raw, dict):
        for key, value in raw.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                child = dict(merged[key])
                child.update(value)
                merged[key] = child
            else:
                merged[key] = value
        return merged
    return raw


def load_scene(path: Path) -> Scene:
    if not path.exists():
        raise MaterialError(
            "E001", f"scene JSON does not exist: {path}", asset="scene"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MaterialError("E008", f"cannot read scene JSON: {exc}", asset="scene") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MaterialError(
            "E004",
            f"invalid scene JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
        ) from exc
    if not isinstance(raw, dict):
        raise MaterialError("E004", "scene JSON root must be an object")
    schema_version = raw.get("schema_version", 1)
    if schema_version != 1:
        raise MaterialError(
            "E004", f"unsupported scene schema_version: {schema_version!r}"
        )

    assets_raw = raw.get("assets")
    if assets_raw is None:
        top_level_keys = {
            "base_color",
            "basecolor",
            "albedo",
            "normal",
            "normal_map",
            "roughness",
            "roughness_map",
            "metallic",
            "metallic_map",
            "defaults",
        }
        assets_raw = {
            key: value
            for key, value in raw.items()
            if key in top_level_keys
        }
    assets_raw = _mapping(assets_raw, "assets")
    aliases = {
        "basecolor": "base_color",
        "albedo": "base_color",
        "normal_map": "normal",
        "roughness_map": "roughness",
        "metallic_map": "metallic",
    }
    normalized_assets: dict[str, Any] = {}
    for key, value in assets_raw.items():
        target = aliases.get(key, key)
        if target not in normalized_assets:
            normalized_assets[target] = value
    assets_raw = normalized_assets
    if "base_color" not in assets_raw or "normal" not in assets_raw:
        raise MaterialError(
            "E004", "assets.base_color and assets.normal are required"
        )

    defaults_raw = assets_raw.get("defaults", raw.get("defaults", {}))
    defaults_raw = _mapping(defaults_raw, "assets.defaults")
    default_roughness = ensure_finite(
        defaults_raw.get("roughness", 0.5), "assets.defaults.roughness"
    )
    default_metallic = ensure_finite(
        defaults_raw.get("metallic", 0.0), "assets.defaults.metallic"
    )
    if not 0.0 <= default_roughness <= 1.0:
        raise MaterialError("E004", "assets.defaults.roughness must be in [0, 1]")
    if not 0.0 <= default_metallic <= 1.0:
        raise MaterialError("E004", "assets.defaults.metallic must be in [0, 1]")

    asset_specs = {
        "base_color": _texture_spec(
            assets_raw["base_color"], "base_color", path.parent, required=True
        ),
        "normal": _texture_spec(
            assets_raw["normal"], "normal", path.parent, required=True
        ),
        "roughness": _texture_spec(
            assets_raw.get("roughness"), "roughness", path.parent, required=False
        ),
        "metallic": _texture_spec(
            assets_raw.get("metallic"), "metallic", path.parent, required=False
        ),
    }

    base_options = asset_specs["base_color"].options
    base_space = str(base_options.get("color_space", "srgb")).lower()
    if base_space not in {"srgb", "linear"}:
        raise MaterialError(
            "E004", "assets.base_color.color_space must be 'srgb' or 'linear'"
        )
    base_options["color_space"] = base_space

    normal_options = asset_specs["normal"].options
    normal_space = str(normal_options.get("space", "tangent")).lower()
    if normal_space != "tangent":
        raise MaterialError(
            "E004",
            "assets.normal.space must be 'tangent'; object-space maps are rejected",
        )
    normal_options["space"] = normal_space
    green_sign = ensure_finite(
        normal_options.get("green_sign", 1.0), "assets.normal.green_sign"
    )
    if green_sign not in {-1.0, 1.0}:
        raise MaterialError("E004", "assets.normal.green_sign must be -1 or 1")
    normal_options["green_sign"] = green_sign

    for label in ("roughness", "metallic"):
        options = asset_specs[label].options
        if "channel" in options:
            channel = _non_negative_int(
                options["channel"], f"assets.{label}.channel", maximum=3
            )
            options["channel"] = channel
        if "min" in options or "max" in options:
            lower = ensure_finite(options.get("min", 0.0), f"assets.{label}.min")
            upper = ensure_finite(options.get("max", 1.0), f"assets.{label}.max")
            if not 0.0 <= lower <= upper <= 1.0:
                raise MaterialError(
                    "E004",
                    f"assets.{label} min/max must satisfy 0 <= min <= max <= 1",
                )
            options["min"] = lower
            options["max"] = upper

    surface = _merge_defaults(raw.get("surface", {}), "surface")
    surface["geometry_normal"] = _vector(
        surface["geometry_normal"], 3, "surface.geometry_normal", normalize=True
    ).tolist()
    surface["uv_axis_u"] = _vector(
        surface["uv_axis_u"], 3, "surface.uv_axis_u", normalize=True
    ).tolist()
    surface["uv_axis_v"] = _vector(
        surface["uv_axis_v"], 3, "surface.uv_axis_v", normalize=True
    ).tolist()
    surface["uv_origin"] = _vector(
        surface["uv_origin"], 3, "surface.uv_origin"
    ).tolist()
    material_size = _vector(
        surface.get("material_size", [2.0, 2.0]), 2, "surface.material_size"
    )
    if np.any(material_size <= 0.0):
        raise MaterialError("E004", "surface.material_size entries must be positive")
    surface["material_size"] = material_size.tolist()
    surface["normal_green_sign"] = ensure_finite(
        surface.get("normal_green_sign", 1.0), "surface.normal_green_sign"
    )
    if surface["normal_green_sign"] not in {-1.0, 1.0}:
        raise MaterialError(
            "E004", "surface.normal_green_sign must be -1 or 1"
        )
    surface["seam_smoothing_radius"] = _non_negative_int(
        surface.get("seam_smoothing_radius", 1),
        "surface.seam_smoothing_radius",
        maximum=64,
    )
    islands = surface.get("islands")
    if islands is not None and not isinstance(islands, list):
        raise MaterialError("E004", "surface.islands must be an array")

    lights_raw = raw.get("lights", [])
    if not isinstance(lights_raw, list) or not lights_raw:
        raise MaterialError("E004", "lights must be a non-empty array")
    lights: list[dict[str, Any]] = []
    for index, item in enumerate(lights_raw):
        item = _mapping(item, f"lights[{index}]")
        direction = _vector(
            item.get("direction", item.get("dir")),
            3,
            f"lights[{index}].direction",
            normalize=True,
        )
        intensity_value = item.get("intensity", item.get("radiance", 1.0))
        intensity = ensure_finite(intensity_value, f"lights[{index}].intensity")
        if intensity < 0.0:
            raise MaterialError("E004", f"lights[{index}].intensity must be >= 0")
        color = _vector(
            item.get("color", [1.0, 1.0, 1.0]),
            3,
            f"lights[{index}].color",
        )
        if np.any(color < 0.0):
            raise MaterialError("E004", f"lights[{index}].color must be non-negative")
        lights.append(
            {
                "id": str(item.get("id", "")),
                "direction": direction.tolist(),
                "intensity": intensity,
                "color": color.tolist(),
            }
        )
    lights = canonical_lights(lights)

    environment_source = raw.get("environment", raw.get("ambient", {}))
    if isinstance(environment_source, (int, float)) and not isinstance(
        environment_source, bool
    ):
        environment_source = {"intensity": environment_source}
    environment = _merge_defaults(environment_source, "environment")
    if "intensity" not in environment:
        environment["intensity"] = raw.get("ambient_intensity", 0.25)
    environment["intensity"] = ensure_finite(
        environment["intensity"], "environment.intensity"
    )
    if environment["intensity"] < 0.0:
        raise MaterialError("E004", "environment.intensity must be >= 0")
    environment_color = _vector(
        environment.get("color", [1.0, 1.0, 1.0]),
        3,
        "environment.color",
    )
    if np.any(environment_color < 0.0):
        raise MaterialError("E004", "environment.color must be non-negative")
    environment["color"] = environment_color.tolist()

    camera = _merge_defaults(raw.get("camera", {}), "camera")
    projection = str(camera.get("projection", "perspective")).lower()
    if projection not in {"perspective", "orthographic"}:
        raise MaterialError(
            "E004", "camera.projection must be 'perspective' or 'orthographic'"
        )
    camera["projection"] = projection
    camera["position"] = _vector(
        camera.get("position", [0.0, 0.0, 3.2]), 3, "camera.position"
    ).tolist()
    camera["target"] = _vector(
        camera.get("target", [0.0, 0.0, 0.0]), 3, "camera.target"
    ).tolist()
    camera["up"] = _vector(
        camera.get("up", [0.0, 1.0, 0.0]), 3, "camera.up", normalize=True
    ).tolist()
    if np.allclose(camera["position"], camera["target"]):
        raise MaterialError("E004", "camera.position must differ from camera.target")
    camera_forward = np.asarray(camera["target"]) - np.asarray(camera["position"])
    camera_forward /= max(float(np.linalg.norm(camera_forward)), 1e-12)
    if float(np.linalg.norm(np.cross(camera_forward, camera["up"]))) < 1e-8:
        raise MaterialError(
            "E004", "camera.up must not be parallel to the view direction"
        )
    camera["fov_y_degrees"] = ensure_finite(
        camera.get("fov_y_degrees", 35.0), "camera.fov_y_degrees"
    )
    if not 1.0 < camera["fov_y_degrees"] < 179.0 and projection == "perspective":
        raise MaterialError(
            "E004", "camera.fov_y_degrees must be in (1, 179)"
        )
    camera["ortho_scale"] = ensure_finite(
        camera.get("ortho_scale", 2.4), "camera.ortho_scale"
    )
    if camera["ortho_scale"] <= 0.0:
        raise MaterialError("E004", "camera.ortho_scale must be positive")
    camera_resolution = _vector(
        camera.get("resolution", [384, 384]), 2, "camera.resolution"
    )
    if np.any(camera_resolution <= 0.0):
        raise MaterialError("E004", "camera.resolution entries must be positive")
    camera["resolution"] = [int(camera_resolution[0]), int(camera_resolution[1])]

    baking = _merge_defaults(raw.get("baking", {}), "baking")
    baking["chunk_rows"] = _positive_int(
        baking.get("chunk_rows", 32), "baking.chunk_rows", maximum=16384
    )
    baking["view_direction"] = _vector(
        baking.get("view_direction", [0.0, 0.0, 1.0]),
        3,
        "baking.view_direction",
        normalize=True,
    ).tolist()
    baking["exposure"] = ensure_finite(
        baking.get("exposure", 1.0), "baking.exposure"
    )
    if baking["exposure"] <= 0.0:
        raise MaterialError("E004", "baking.exposure must be positive")
    baking["normal_strength"] = ensure_finite(
        baking.get("normal_strength", 1.0), "baking.normal_strength"
    )
    if baking["normal_strength"] < 0.0:
        raise MaterialError("E004", "baking.normal_strength must be non-negative")

    validation = _merge_defaults(raw.get("validation", {}), "validation")
    if isinstance(raw.get("validation", {}), dict):
        seed = raw["validation"].get("seed", validation["seed"])
    else:
        seed = validation["seed"]
    validation["seed"] = _non_negative_int(
        seed, "validation.seed", maximum=2**63 - 1
    )
    validation["max_samples"] = _positive_int(
        validation.get("max_samples", 2048),
        "validation.max_samples",
        maximum=10_000_000,
    )
    stratified = validation.get("stratified", True)
    if not isinstance(stratified, bool):
        raise MaterialError("E004", "validation.stratified must be boolean")
    validation["stratified"] = stratified
    validation["normal_length_tolerance"] = ensure_finite(
        validation.get("normal_length_tolerance", 0.03),
        "validation.normal_length_tolerance",
    )
    if validation["normal_length_tolerance"] < 0.0:
        raise MaterialError(
            "E004", "validation.normal_length_tolerance must be >= 0"
        )
    validation["angle_threshold_deg"] = ensure_finite(
        validation.get("angle_threshold_deg", 32.0),
        "validation.angle_threshold_deg",
    )
    validation["seam_angle_threshold_deg"] = ensure_finite(
        validation.get("seam_angle_threshold_deg", 25.0),
        "validation.seam_angle_threshold_deg",
    )
    for key in ("angle_threshold_deg", "seam_angle_threshold_deg"):
        if not 0.0 <= validation[key] <= 180.0:
            raise MaterialError("E004", f"validation.{key} must be in [0, 180]")
    validation["reference_gain"] = ensure_finite(
        validation.get("reference_gain", 0.55), "validation.reference_gain"
    )
    if validation["reference_gain"] < 0.0:
        raise MaterialError("E004", "validation.reference_gain must be >= 0")
    validation["reference_min_gradient"] = ensure_finite(
        validation.get("reference_min_gradient", 0.004),
        "validation.reference_min_gradient",
    )
    if validation["reference_min_gradient"] < 0.0:
        raise MaterialError(
            "E004", "validation.reference_min_gradient must be >= 0"
        )
    validation["reference_tilt_min"] = ensure_finite(
        validation.get("reference_tilt_min", 0.04),
        "validation.reference_tilt_min",
    )
    if not 0.0 <= validation["reference_tilt_min"] <= 1.0:
        raise MaterialError(
            "E004", "validation.reference_tilt_min must be in [0, 1]"
        )
    validation["seam_width"] = _non_negative_int(
        validation.get("seam_width", 2), "validation.seam_width", maximum=64
    )
    validation["tile_size"] = _positive_int(
        validation.get("tile_size", 64), "validation.tile_size", maximum=4096
    )
    validation["hard_edge_color_threshold"] = ensure_finite(
        validation.get("hard_edge_color_threshold", 0.22),
        "validation.hard_edge_color_threshold",
    )
    if validation["hard_edge_color_threshold"] < 0.0:
        raise MaterialError(
            "E004", "validation.hard_edge_color_threshold must be >= 0"
        )
    validation["max_entries"] = _positive_int(
        validation.get("max_entries", 512),
        "validation.max_entries",
        maximum=100_000,
    )

    output = _merge_defaults(raw.get("output", {}), "output")
    output["bit_depth"] = _positive_int(
        output.get("bit_depth", 8), "output.bit_depth", maximum=16
    )
    if output["bit_depth"] not in {8, 16}:
        raise MaterialError("E004", "output.bit_depth must be 8 or 16")
    preview_resolution = _vector(
        output.get("preview_resolution", camera["resolution"]),
        2,
        "output.preview_resolution",
    )
    if np.any(preview_resolution <= 0.0):
        raise MaterialError(
            "E004", "output.preview_resolution entries must be positive"
        )
    output["preview_resolution"] = [
        int(preview_resolution[0]),
        int(preview_resolution[1]),
    ]
    pattern = output.get("filename_pattern", "{kind}.png")
    if not isinstance(pattern, str) or pattern.count("{kind}") != 1:
        raise MaterialError(
            "E004", "output.filename_pattern must contain exactly one {kind}"
        )
    remaining_pattern = pattern.replace("{kind}", "")
    if "{" in remaining_pattern or "}" in remaining_pattern:
        raise MaterialError(
            "E004", "output.filename_pattern contains an unsupported placeholder"
        )
    output["filename_pattern"] = pattern
    background = _vector(
        output.get("background", [0.025, 0.030, 0.038]),
        3,
        "output.background",
    )
    if np.any(background < 0.0):
        raise MaterialError("E004", "output.background must be non-negative")
    output["background"] = background.tolist()

    return Scene(
        path=path.resolve(),
        raw=raw,
        assets=asset_specs,
        defaults={"roughness": default_roughness, "metallic": default_metallic},
        surface=surface,
        lights=lights,
        environment=environment,
        camera=camera,
        baking=baking,
        validation=validation,
        output=output,
    )
