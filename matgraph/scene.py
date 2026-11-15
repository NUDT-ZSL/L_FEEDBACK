from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from .errors import scene_error
from .limits import DEFAULT_LIMITS, Limits
from .mathutil import as_vec3, normalize


@dataclass(frozen=True)
class Camera:
    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_degrees: float
    forward: np.ndarray
    right: np.ndarray
    true_up: np.ndarray


@dataclass(frozen=True)
class DirectionalLight:
    direction: np.ndarray
    color: np.ndarray


@dataclass(frozen=True)
class AmbientLight:
    sky: np.ndarray
    ground: np.ndarray


@dataclass(frozen=True)
class SphereObject:
    center: np.ndarray
    radius: float
    type: str = "sphere"


@dataclass(frozen=True)
class PlaneObject:
    point: np.ndarray
    normal: np.ndarray
    tangent: np.ndarray
    bitangent: np.ndarray
    size: np.ndarray
    type: str = "plane"


SceneObject = Union[SphereObject, PlaneObject]


@dataclass(frozen=True)
class Scene:
    name: str
    camera: Camera
    width: int
    height: int
    exposure: float
    background: np.ndarray
    ambient: AmbientLight
    lights: Sequence[DirectionalLight]
    objects: Sequence[SceneObject]


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise scene_error("E_SCENE_SCHEMA", f"{field} must be an object", field=field)
    return value


def _number(
    value: Any,
    field: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise scene_error("E_SCENE_VALUE", f"{field} must be a finite number", field=field)
    result = float(value)
    if not math.isfinite(result):
        raise scene_error("E_SCENE_VALUE", f"{field} must be finite", field=field)
    if minimum is not None and result < minimum:
        raise scene_error(
            "E_SCENE_VALUE",
            f"{field} must be >= {minimum:g}",
            field=field,
        )
    if maximum is not None and result > maximum:
        raise scene_error(
            "E_SCENE_VALUE",
            f"{field} must be <= {maximum:g}",
            field=field,
        )
    return result


def _integer(
    value: Any,
    field: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise scene_error("E_SCENE_VALUE", f"{field} must be an integer", field=field)
    result = int(value)
    if result < minimum or result > maximum:
        raise scene_error(
            "E_SCENE_VALUE",
            f"{field} must be within {minimum}..{maximum}",
            field=field,
        )
    return result


def _vec3(
    value: Any,
    field: str,
    *,
    nonnegative: bool = False,
    maximum: float = 1_000_000.0,
) -> np.ndarray:
    try:
        result = as_vec3(value, field)
    except ValueError as exc:
        raise scene_error("E_SCENE_VALUE", str(exc), field=field)
    if nonnegative and np.any(result < 0.0):
        raise scene_error(
            "E_SCENE_VALUE",
            f"{field} components must be non-negative",
            field=field,
        )
    if np.any(np.abs(result) > maximum):
        raise scene_error(
            "E_SCENE_VALUE",
            f"{field} components must have magnitude <= {maximum:g}",
            field=field,
        )
    return result


def _unit_vec3(value: Any, field: str) -> np.ndarray:
    vector = _vec3(value, field)
    try:
        return normalize(vector, field)
    except ValueError as exc:
        raise scene_error("E_SCENE_VALUE", str(exc), field=field)


def _vec2(value: Any, field: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 2:
        raise scene_error("E_SCENE_VALUE", f"{field} must contain 2 numbers", field=field)
    components = [
        _number(component, f"{field}[{index}]")
        for index, component in enumerate(value)
    ]
    return np.asarray(components, dtype=np.float64)


def _parse_camera(raw: Any) -> Camera:
    camera = _require_mapping(raw, "camera")
    position = _vec3(camera.get("position"), "camera.position")
    target = _vec3(camera.get("target"), "camera.target")
    up = _vec3(camera.get("up", [0.0, 1.0, 0.0]), "camera.up")
    fov = _number(
        camera.get("fov_degrees", 45.0),
        "camera.fov_degrees",
        minimum=1.0,
        maximum=179.0,
    )
    try:
        forward = normalize(target - position, "camera forward direction")
        right = normalize(np.cross(forward, up), "camera right direction")
        true_up = normalize(np.cross(right, forward), "camera up direction")
    except ValueError as exc:
        raise scene_error("E_SCENE_VALUE", str(exc), field="camera")
    return Camera(
        position=position,
        target=target,
        up=up,
        fov_degrees=fov,
        forward=forward,
        right=right,
        true_up=true_up,
    )


def _parse_lights(raw: Any, limits: Limits) -> List[DirectionalLight]:
    if not isinstance(raw, list) or not raw:
        raise scene_error(
            "E_SCENE_SCHEMA",
            "lights must be a non-empty array",
            field="lights",
        )
    if len(raw) > limits.max_lights:
        raise scene_error(
            "E_LIMIT",
            f"light count exceeds limit {limits.max_lights}",
            field="lights",
        )
    lights: List[DirectionalLight] = []
    for index, item in enumerate(raw):
        light = _require_mapping(item, f"lights[{index}]")
        if light.get("type", "directional") != "directional":
            raise scene_error(
                "E_SCENE_TYPE",
                "only directional lights are supported",
                field=f"lights[{index}].type",
            )
        direction = _unit_vec3(
            light.get("direction"),
            f"lights[{index}].direction",
        )
        color = _vec3(
            light.get("color"),
            f"lights[{index}].color",
            nonnegative=True,
            maximum=10_000.0,
        )
        lights.append(DirectionalLight(direction=direction, color=color))
    return lights


def _parse_ambient(raw: Any) -> AmbientLight:
    ambient = _require_mapping(raw, "render.ambient")
    return AmbientLight(
        sky=_vec3(
            ambient.get("sky"),
            "render.ambient.sky",
            nonnegative=True,
            maximum=10_000.0,
        ),
        ground=_vec3(
            ambient.get("ground"),
            "render.ambient.ground",
            nonnegative=True,
            maximum=10_000.0,
        ),
    )


def _parse_objects(raw: Any, limits: Limits) -> List[SceneObject]:
    if not isinstance(raw, list) or not raw:
        raise scene_error(
            "E_SCENE_SCHEMA",
            "objects must be a non-empty array",
            field="objects",
        )
    if len(raw) > limits.max_objects:
        raise scene_error(
            "E_LIMIT",
            f"object count exceeds limit {limits.max_objects}",
            field="objects",
        )
    objects: List[SceneObject] = []
    for index, item in enumerate(raw):
        obj = _require_mapping(item, f"objects[{index}]")
        obj_type = obj.get("type")
        if obj_type == "sphere":
            center = _vec3(obj.get("center"), f"objects[{index}].center")
            radius = _number(
                obj.get("radius"),
                f"objects[{index}].radius",
                minimum=1e-6,
                maximum=1000.0,
            )
            objects.append(SphereObject(center=center, radius=radius))
        elif obj_type == "plane":
            point = _vec3(obj.get("point"), f"objects[{index}].point")
            normal = _unit_vec3(obj.get("normal"), f"objects[{index}].normal")
            size = _vec2(obj.get("size"), f"objects[{index}].size")
            if np.any(size <= 0.0) or np.any(size > 1_000_000.0):
                raise scene_error(
                    "E_SCENE_VALUE",
                    "plane size components must be within 0..1e6",
                    field=f"objects[{index}].size",
                )
            axis = (
                np.asarray([0.0, 1.0, 0.0])
                if abs(float(normal[1])) < 0.9
                else np.asarray([1.0, 0.0, 0.0])
            )
            tangent = normalize(np.cross(axis, normal), "plane tangent")
            bitangent = normalize(np.cross(normal, tangent), "plane bitangent")
            objects.append(
                PlaneObject(
                    point=point,
                    normal=normal,
                    tangent=tangent,
                    bitangent=bitangent,
                    size=size,
                )
            )
        else:
            raise scene_error(
                "E_SCENE_TYPE",
                "object type must be 'sphere' or 'plane'",
                field=f"objects[{index}].type",
            )
    return objects


def parse_scene(
    raw: Mapping[str, Any],
    limits: Limits = DEFAULT_LIMITS,
    *,
    width_override: Optional[int] = None,
    height_override: Optional[int] = None,
) -> Scene:
    if not isinstance(raw, Mapping):
        raise scene_error("E_SCENE_ROOT", "scene root must be a JSON object")
    if raw.get("schema") != "scene/v1":
        raise scene_error("E_SCENE_SCHEMA", "scene schema must be 'scene/v1'", field="schema")
    name = raw.get("name", "")
    if not isinstance(name, str):
        raise scene_error("E_SCENE_SCHEMA", "scene name must be a string", field="name")
    camera = _parse_camera(raw.get("camera"))
    render = _require_mapping(raw.get("render"), "render")

    width = (
        _integer(width_override, "render.width", minimum=1, maximum=limits.max_width)
        if width_override is not None
        else _integer(
            render.get("width"),
            "render.width",
            minimum=1,
            maximum=limits.max_width,
        )
    )
    height = (
        _integer(height_override, "render.height", minimum=1, maximum=limits.max_height)
        if height_override is not None
        else _integer(
            render.get("height"),
            "render.height",
            minimum=1,
            maximum=limits.max_height,
        )
    )
    if width * height > limits.max_pixels:
        raise scene_error(
            "E_LIMIT",
            f"pixel count {width * height} exceeds limit {limits.max_pixels}",
            field="render",
        )
    exposure = _number(
        render.get("exposure", 1.0),
        "render.exposure",
        minimum=0.0,
        maximum=8.0,
    )
    background = _vec3(
        render.get("background", [0.015, 0.02, 0.03]),
        "render.background",
        nonnegative=True,
        maximum=10_000.0,
    )
    ambient = _parse_ambient(render.get("ambient"))
    lights = _parse_lights(raw.get("lights"), limits)
    objects = _parse_objects(raw.get("objects"), limits)
    return Scene(
        name=name,
        camera=camera,
        width=width,
        height=height,
        exposure=exposure,
        background=background,
        ambient=ambient,
        lights=tuple(lights),
        objects=tuple(objects),
    )


def load_scene(
    path: Path,
    limits: Limits = DEFAULT_LIMITS,
    *,
    width_override: Optional[int] = None,
    height_override: Optional[int] = None,
) -> Scene:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise scene_error("E_IO", f"cannot read scene: {type(exc).__name__}")
    if size > limits.max_json_bytes:
        raise scene_error("E_LIMIT", f"scene exceeds {limits.max_json_bytes} bytes")
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise scene_error(
            "E_INVALID_JSON",
            f"invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}",
        )
    except UnicodeError:
        raise scene_error("E_INVALID_JSON", "scene is not valid UTF-8")
    except OSError as exc:
        raise scene_error("E_IO", f"cannot read scene: {type(exc).__name__}")
    return parse_scene(
        raw,
        limits,
        width_override=width_override,
        height_override=height_override,
    )
