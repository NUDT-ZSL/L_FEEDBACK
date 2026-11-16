from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import (
    EXIT_ASSET_OR_ENVIRONMENT,
    EXIT_SCENE,
    RenderError,
    scene_error,
)


Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Camera:
    position: Vec3
    target: Vec3
    up: Vec3
    vertical_fov_degrees: float
    near: float
    far: float


@dataclass(frozen=True)
class Light:
    direction: Vec3
    color: Vec3
    intensity: float


@dataclass(frozen=True)
class Sphere:
    center: Vec3
    radius: float


@dataclass(frozen=True)
class Material:
    base_color_texture: Path
    roughness_texture: Path
    normal_texture: Path | None
    base_color_tint: Vec3
    specular_strength: float
    normal_strength: float


@dataclass(frozen=True)
class Environment:
    sky_color: Vec3
    ground_color: Vec3
    intensity: float


@dataclass(frozen=True)
class CompositeWeights:
    base_color: float
    direct: float
    ambient: float


@dataclass(frozen=True)
class OutputConfig:
    width: int
    height: int
    background_color: Vec3
    texture_filter: str
    block_size: int
    block_axis: str
    block_order: str
    directory: Path
    weights: CompositeWeights


@dataclass(frozen=True)
class Scene:
    schema_version: int
    camera: Camera
    light: Light
    sphere: Sphere
    material: Material
    environment: Environment
    output: OutputConfig
    scene_path: Path


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(value: Any, path: str) -> float:
    if not _is_number(value):
        raise scene_error(path, "expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise scene_error(path, "expected a finite number")
    return result


def _integer(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise scene_error(path, "expected an integer")
    return value


def _string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise scene_error(path, "expected a string")
    if not allow_empty and not value.strip():
        raise scene_error(path, "must not be empty")
    return value


def _object(
    value: Any,
    path: str,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise scene_error(path, "expected an object")
    required_set = set(required)
    optional_set = set(optional)
    missing = sorted(required_set - set(value))
    unknown = sorted(set(value) - required_set - optional_set)
    if missing:
        raise scene_error(
            f"{path}.{missing[0]}",
            "missing required field",
        )
    if unknown:
        raise scene_error(
            f"{path}.{unknown[0]}",
            "unknown field",
        )
    return value


def _vec3(value: Any, path: str) -> Vec3:
    if not isinstance(value, list) or len(value) != 3:
        raise scene_error(path, "expected an array of three numbers")
    return (
        _number(value[0], f"{path}[0]"),
        _number(value[1], f"{path}[1]"),
        _number(value[2], f"{path}[2]"),
    )


def _vec3_unit(value: Any, path: str) -> Vec3:
    vector = _vec3(value, path)
    length = math.sqrt(sum(component * component for component in vector))
    if length <= 1.0e-12:
        raise scene_error(path, "must not be the zero vector")
    return tuple(component / length for component in vector)  # type: ignore[return-value]


def _color(value: Any, path: str) -> Vec3:
    color = _vec3(value, path)
    if any(component < 0.0 or component > 1.0 for component in color):
        raise scene_error(path, "color components must be in [0, 1]")
    return color


def _range(value: Any, path: str, low: float, high: float) -> float:
    number = _number(value, path)
    if number < low or number > high:
        raise scene_error(path, f"must be in [{low}, {high}]")
    return number


def _integer_range(value: Any, path: str, low: int, high: int) -> int:
    number = _integer(value, path)
    if number < low or number > high:
        raise scene_error(path, f"must be in [{low}, {high}]")
    return number


def _enum(value: Any, path: str, choices: tuple[str, ...]) -> str:
    text = _string(value, path)
    if text not in choices:
        raise scene_error(path, f"must be one of {list(choices)}")
    return text


def _resolve_asset_path(value: str, scene_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (scene_dir / path)


def _output_context(raw: Any, scene_path: Path) -> Path:
    if isinstance(raw, dict) and isinstance(raw.get("output"), dict):
        directory = raw["output"].get("directory")
        if isinstance(directory, str) and directory.strip():
            path = Path(directory)
            return path if path.is_absolute() else (scene_path.parent / path)
    return scene_path.parent / "render_output"


def _parse_scene(
    raw: Any,
    scene_path: Path,
    output_override: Path | None,
) -> Scene:
    root = _object(
        raw,
        "$",
        required=(
            "schema_version",
            "camera",
            "light",
            "geometry",
            "material",
            "environment",
            "output",
        ),
    )
    schema_version = _integer(root["schema_version"], "$.schema_version")
    if schema_version != 1:
        raise scene_error("$.schema_version", "only schema version 1 is supported")

    camera_obj = _object(
        root["camera"],
        "$.camera",
        required=(
            "position",
            "target",
            "up",
            "vertical_fov_degrees",
            "near",
            "far",
        ),
    )
    camera = Camera(
        position=_vec3(camera_obj["position"], "$.camera.position"),
        target=_vec3(camera_obj["target"], "$.camera.target"),
        up=_vec3_unit(camera_obj["up"], "$.camera.up"),
        vertical_fov_degrees=_range(
            camera_obj["vertical_fov_degrees"],
            "$.camera.vertical_fov_degrees",
            1.0,
            179.0,
        ),
        near=_number(camera_obj["near"], "$.camera.near"),
        far=_number(camera_obj["far"], "$.camera.far"),
    )
    light_obj = _object(
        root["light"],
        "$.light",
        required=("direction", "color", "intensity"),
    )
    light = Light(
        direction=_vec3_unit(light_obj["direction"], "$.light.direction"),
        color=_color(light_obj["color"], "$.light.color"),
        intensity=_range(light_obj["intensity"], "$.light.intensity", 0.0, 100.0),
    )

    geometry_obj = _object(
        root["geometry"],
        "$.geometry",
        required=("type", "center", "radius"),
    )
    if _enum(geometry_obj["type"], "$.geometry.type", ("sphere",)) != "sphere":
        raise AssertionError("unreachable")
    sphere = Sphere(
        center=_vec3(geometry_obj["center"], "$.geometry.center"),
        radius=_number(geometry_obj["radius"], "$.geometry.radius"),
    )
    if sphere.radius <= 0.0:
        raise scene_error("$.geometry.radius", "must be greater than zero")

    offset = tuple(camera.position[i] - sphere.center[i] for i in range(3))
    if sum(component * component for component in offset) < sphere.radius**2:
        raise RenderError(
            code="E_CAMERA_INSIDE_SPHERE",
            message="camera position is inside the sphere",
            exit_code=EXIT_ASSET_OR_ENVIRONMENT,
        )

    if camera.near <= 0.0:
        raise scene_error("$.camera.near", "must be greater than zero")
    if camera.far <= camera.near:
        raise scene_error("$.camera.far", "must be greater than camera.near")
    forward = tuple(camera.target[i] - camera.position[i] for i in range(3))
    if sum(component * component for component in forward) <= 1.0e-24:
        raise scene_error("$.camera.target", "must differ from camera.position")
    cross = (
        forward[1] * camera.up[2] - forward[2] * camera.up[1],
        forward[2] * camera.up[0] - forward[0] * camera.up[2],
        forward[0] * camera.up[1] - forward[1] * camera.up[0],
    )
    if sum(component * component for component in cross) <= 1.0e-24:
        raise scene_error("$.camera.up", "must not be parallel to the view direction")

    material_obj = _object(
        root["material"],
        "$.material",
        required=(
            "base_color_texture",
            "roughness_texture",
            "base_color_tint",
            "specular_strength",
            "normal_strength",
        ),
        optional=("normal_texture",),
    )
    normal_texture: Path | None = None
    if material_obj.get("normal_texture") is not None:
        normal_texture = _resolve_asset_path(
            _string(material_obj["normal_texture"], "$.material.normal_texture"),
            scene_path.parent,
        )
    material = Material(
        base_color_texture=_resolve_asset_path(
            _string(
                material_obj["base_color_texture"],
                "$.material.base_color_texture",
            ),
            scene_path.parent,
        ),
        roughness_texture=_resolve_asset_path(
            _string(
                material_obj["roughness_texture"],
                "$.material.roughness_texture",
            ),
            scene_path.parent,
        ),
        normal_texture=normal_texture,
        base_color_tint=_color(
            material_obj["base_color_tint"],
            "$.material.base_color_tint",
        ),
        specular_strength=_range(
            material_obj["specular_strength"],
            "$.material.specular_strength",
            0.0,
            1.0,
        ),
        normal_strength=_range(
            material_obj["normal_strength"],
            "$.material.normal_strength",
            0.0,
            4.0,
        ),
    )

    environment_obj = _object(
        root["environment"],
        "$.environment",
        required=("sky_color", "ground_color", "intensity"),
    )
    environment = Environment(
        sky_color=_color(environment_obj["sky_color"], "$.environment.sky_color"),
        ground_color=_color(
            environment_obj["ground_color"],
            "$.environment.ground_color",
        ),
        intensity=_range(
            environment_obj["intensity"],
            "$.environment.intensity",
            0.0,
            100.0,
        ),
    )

    output_obj = _object(
        root["output"],
        "$.output",
        required=(
            "width",
            "height",
            "background_color",
            "texture_filter",
            "block_size",
            "block_axis",
            "block_order",
            "directory",
            "weights",
        ),
    )
    width = _integer_range(output_obj["width"], "$.output.width", 64, 1024)
    height = _integer_range(output_obj["height"], "$.output.height", 64, 1024)
    block_size = _integer(output_obj["block_size"], "$.output.block_size")
    if block_size < 1:
        raise scene_error("$.output.block_size", "must be greater than zero")
    block_axis = _enum(
        output_obj["block_axis"],
        "$.output.block_axis",
        ("row", "column"),
    )
    block_order = _enum(
        output_obj["block_order"],
        "$.output.block_order",
        ("forward", "reverse"),
    )
    weights_obj = _object(
        output_obj["weights"],
        "$.output.weights",
        required=("base_color", "direct", "ambient"),
    )
    weights = CompositeWeights(
        base_color=_range(
            weights_obj["base_color"],
            "$.output.weights.base_color",
            0.0,
            1.0,
        ),
        direct=_range(
            weights_obj["direct"],
            "$.output.weights.direct",
            0.0,
            1.0,
        ),
        ambient=_range(
            weights_obj["ambient"],
            "$.output.weights.ambient",
            0.0,
            1.0,
        ),
    )
    weight_sum = weights.base_color + weights.direct + weights.ambient
    if abs(weight_sum - 1.0) > 1.0e-9:
        raise scene_error(
            "$.output.weights",
            "base_color + direct + ambient must equal 1",
        )

    if output_override is not None:
        output_directory = output_override
    else:
        directory_text = _string(output_obj["directory"], "$.output.directory")
        directory_path = Path(directory_text)
        output_directory = (
            directory_path
            if directory_path.is_absolute()
            else scene_path.parent / directory_path
        )

    output = OutputConfig(
        width=width,
        height=height,
        background_color=_color(
            output_obj["background_color"],
            "$.output.background_color",
        ),
        texture_filter=_enum(
            output_obj["texture_filter"],
            "$.output.texture_filter",
            ("nearest", "bilinear"),
        ),
        block_size=block_size,
        block_axis=block_axis,
        block_order=block_order,
        directory=output_directory,
        weights=weights,
    )

    if width % block_size != 0 or height % block_size != 0:
        raise RenderError(
            code="E_BLOCK_NOT_DIVISIBLE",
            message=(
                f"block_size {block_size} must divide output width {width} "
                f"and height {height}"
            ),
            exit_code=EXIT_ASSET_OR_ENVIRONMENT,
            details={
                "width": width,
                "height": height,
                "block_size": block_size,
            },
        )

    return Scene(
        schema_version=schema_version,
        camera=camera,
        light=light,
        sphere=sphere,
        material=material,
        environment=environment,
        output=output,
        scene_path=scene_path,
    )


def load_scene(scene_path: Path | str, output_override: Path | None = None) -> Scene:
    path = Path(scene_path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        error = RenderError(
            code="E_SCENE_IO",
            message=f"cannot read scene file {path}: {exc}",
            exit_code=EXIT_SCENE,
            report_dir=path.parent / "render_output",
        )
        raise error from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RenderError(
            code="E_SCENE_JSON",
            message=f"invalid JSON in {path}: {exc}",
            exit_code=EXIT_SCENE,
            report_dir=path.parent / "render_output",
        ) from exc

    context = (
        output_override
        if output_override is not None
        else _output_context(raw, path)
    )
    try:
        return _parse_scene(raw, path, output_override)
    except RenderError as exc:
        if exc.report_dir is None:
            exc.report_dir = context
        raise
