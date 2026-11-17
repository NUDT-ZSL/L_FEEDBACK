"""Scene schema, validation, and deterministic default-scene generation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import (
    ERROR_COATING_RANGE,
    ERROR_GEOMETRY,
    ERROR_IO_OR_JSON,
    ERROR_LIGHTS,
    ERROR_MATERIAL_TYPE,
    ERROR_RENDER_PARAM,
    ERROR_SCHEMA,
    SceneError,
)
from .geometry import (
    AABB,
    Plane,
    RectLight,
    Sphere,
    SphereLight,
    Triangle,
    Vec3,
    cross,
    dot,
    length,
    normalize,
    sub,
)
from .materials import BaseMaterial, Coating, Material
from .sampling import hash01

DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 256
DEFAULT_SEED = 158

IOR_MIN = 1.0001
IOR_MAX = 2.5
THICKNESS_MIN = 1.0e-6
THICKNESS_MAX = 0.05
SUPPORTED_AA = (1, 2, 4)


@dataclass(frozen=True)
class Camera:
    position: Vec3
    look_at: Vec3
    up: Vec3
    fov_degrees: float
    forward: Vec3
    right: Vec3
    true_up: Vec3


@dataclass(frozen=True)
class RenderConfig:
    samples_per_pixel: int = 2
    aa_samples: int = 1
    shadow_samples: int = 8
    max_depth: int = 5
    exposure: float = 1.0
    self_intersection_epsilon: float = 1.0e-4


@dataclass
class Scene:
    width: int
    height: int
    seed: int
    camera: Camera
    materials: Dict[str, Material]
    geometry: List[Any]
    lights: List[Any]
    render: RenderConfig
    background: Vec3
    description_hash: str
    description: Dict[str, Any] = field(repr=False)

    @property
    def pixel_count(self) -> int:
        return self.width * self.height


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(
    value: Any,
    path: str,
    code: int = ERROR_SCHEMA,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if not _is_number(value):
        raise SceneError(code, "%s must be a finite number" % path)
    result = float(value)
    if not math.isfinite(result):
        raise SceneError(code, "%s must be finite" % path)
    if minimum is not None and result < minimum:
        raise SceneError(code, "%s must be >= %s (got %s)" % (path, minimum, result))
    if maximum is not None and result > maximum:
        raise SceneError(code, "%s must be <= %s (got %s)" % (path, maximum, result))
    return result


def _integer(
    value: Any,
    path: str,
    code: int = ERROR_SCHEMA,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SceneError(code, "%s must be an integer" % path)
    if minimum is not None and value < minimum:
        raise SceneError(code, "%s must be >= %s (got %s)" % (path, minimum, value))
    if maximum is not None and value > maximum:
        raise SceneError(code, "%s must be <= %s (got %s)" % (path, maximum, value))
    return value


def _vec3(value: Any, path: str, code: int = ERROR_SCHEMA) -> Vec3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise SceneError(code, "%s must be a three-element array" % path)
    return (
        _number(value[0], path + "[0]", code),
        _number(value[1], path + "[1]", code),
        _number(value[2], path + "[2]", code),
    )


def _color(value: Any, path: str, code: int = ERROR_SCHEMA) -> Vec3:
    color = _vec3(value, path, code)
    for channel in color:
        if channel < 0.0 or channel > 1.0:
            raise SceneError(code, "%s channels must lie in [0,1] (got %r)" % (path, color))
    return color


def _required(mapping: Dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise SceneError(ERROR_SCHEMA, "missing required field %s.%s" % (path, key))
    return mapping[key]


def _parse_camera(data: Any) -> Camera:
    if not isinstance(data, dict):
        raise SceneError(ERROR_SCHEMA, "camera must be an object")
    position = _vec3(_required(data, "position", "camera"), "camera.position")
    look_at = _vec3(_required(data, "look_at", "camera"), "camera.look_at")
    up = _vec3(data.get("up", [0.0, 1.0, 0.0]), "camera.up")
    if length(sub(look_at, position)) <= 1.0e-9:
        raise SceneError(ERROR_SCHEMA, "camera.position and camera.look_at must differ")
    fov = _number(
        data.get("fov", 45.0), "camera.fov", ERROR_SCHEMA, minimum=1.0, maximum=179.0
    )
    forward = normalize(sub(look_at, position))
    up_length = length(up)
    if up_length <= 1.0e-9:
        raise SceneError(ERROR_SCHEMA, "camera.up must be non-zero")
    up_unit = (up[0] / up_length, up[1] / up_length, up[2] / up_length)
    right = cross(forward, up_unit)
    right_length = length(right)
    if right_length <= 1.0e-9:
        raise SceneError(ERROR_SCHEMA, "camera.up must not be parallel to the view direction")
    right = (right[0] / right_length, right[1] / right_length, right[2] / right_length)
    true_up = cross(right, forward)
    return Camera(position, look_at, up_unit, fov, forward, right, true_up)


def _parse_material(name: str, data: Any) -> Material:
    path = "materials.%s" % name
    if not isinstance(data, dict):
        raise SceneError(ERROR_SCHEMA, "%s must be an object" % path)
    kind = _required(data, "type", path)
    if not isinstance(kind, str):
        raise SceneError(ERROR_MATERIAL_TYPE, "%s.type must be a string" % path)
    if kind not in ("layered", "diffuse", "metal"):
        raise SceneError(ERROR_MATERIAL_TYPE, "unknown material type %r at %s.type" % (kind, path))

    base_data = data.get("base")
    if kind == "layered":
        if not isinstance(base_data, dict):
            raise SceneError(ERROR_SCHEMA, "missing or invalid %s.base" % path)
        base_kind = _required(base_data, "type", path + ".base")
        if not isinstance(base_kind, str):
            raise SceneError(ERROR_MATERIAL_TYPE, "%s.base.type must be a string" % path)
        if base_kind not in ("diffuse", "metal"):
            raise SceneError(
                ERROR_MATERIAL_TYPE, "unknown base material type %r at %s.base.type" % (base_kind, path)
            )
        base_color = _color(_required(base_data, "color", path + ".base"), path + ".base.color")
        roughness = _number(
            base_data.get("roughness", 0.0),
            path + ".base.roughness",
            ERROR_SCHEMA,
            minimum=0.0,
            maximum=1.0,
        )
        coating_data = data.get("coating")
        if not isinstance(coating_data, dict):
            raise SceneError(ERROR_SCHEMA, "missing or invalid %s.coating" % path)
        ior = _number(
            _required(coating_data, "ior", path + ".coating"), path + ".coating.ior", ERROR_SCHEMA
        )
        thickness = _number(
            _required(coating_data, "thickness", path + ".coating"),
            path + ".coating.thickness",
            ERROR_SCHEMA,
        )
        if ior < IOR_MIN or ior > IOR_MAX:
            raise SceneError(
                ERROR_COATING_RANGE,
                "%s.coating.ior must lie in [%s, %s] (got %s)"
                % (path, IOR_MIN, IOR_MAX, ior),
            )
        if thickness < THICKNESS_MIN or thickness > THICKNESS_MAX:
            raise SceneError(
                ERROR_COATING_RANGE,
                "%s.coating.thickness must lie in [%s, %s] (got %s)"
                % (path, THICKNESS_MIN, THICKNESS_MAX, thickness),
            )
        tint = _color(coating_data.get("tint", [1.0, 1.0, 1.0]), path + ".coating.tint")
        absorption = _vec3(
            coating_data.get("absorption", [0.0, 0.0, 0.0]), path + ".coating.absorption"
        )
        for channel in absorption:
            if channel < 0.0 or channel > 100.0:
                raise SceneError(
                    ERROR_SCHEMA,
                    "%s.coating.absorption channels must lie in [0,100]" % path,
                )
        return Material(
            name=name,
            kind=kind,
            base=BaseMaterial(base_kind, base_color, roughness),
            coating=Coating(ior, thickness, tint, absorption),
        )

    base_color = _color(_required(data, "color", path), path + ".color")
    roughness = _number(
        data.get("roughness", 0.0),
        path + ".roughness",
        ERROR_SCHEMA,
        minimum=0.0,
        maximum=1.0,
    )
    return Material(
        name=name,
        kind=kind,
        base=BaseMaterial(kind, base_color, roughness),
        coating=Coating(1.0, THICKNESS_MIN, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
    )


def _parse_materials(data: Any) -> Dict[str, Material]:
    if not isinstance(data, dict) or not data:
        raise SceneError(ERROR_SCHEMA, "materials must be a non-empty object")
    materials: Dict[str, Material] = {}
    for name, material_data in data.items():
        if not isinstance(name, str) or not name:
            raise SceneError(ERROR_SCHEMA, "material names must be non-empty strings")
        materials[name] = _parse_material(name, material_data)
    return materials


def _material_ref(data: Any, materials: Dict[str, Material], path: str) -> Material:
    if not isinstance(data, str):
        raise SceneError(ERROR_SCHEMA, "%s.material must be a material name" % path)
    if data not in materials:
        raise SceneError(ERROR_SCHEMA, "%s.material references unknown material %r" % (path, data))
    return materials[data]


def _parse_geometry(data: Any, materials: Dict[str, Material]) -> List[Any]:
    if not isinstance(data, list) or not data:
        raise SceneError(ERROR_GEOMETRY, "geometry must be a non-empty array")
    objects: List[Any] = []
    for index, primitive in enumerate(data):
        path = "geometry[%d]" % index
        if not isinstance(primitive, dict):
            raise SceneError(ERROR_GEOMETRY, "%s must be an object" % path)
        primitive_type = _required(primitive, "type", path)
        if not isinstance(primitive_type, str):
            raise SceneError(ERROR_GEOMETRY, "%s.type must be a string" % path)
        if primitive_type == "sphere":
            center = _vec3(_required(primitive, "center", path), path + ".center", ERROR_GEOMETRY)
            radius = _number(
                _required(primitive, "radius", path),
                path + ".radius",
                ERROR_GEOMETRY,
                minimum=1.0e-9,
            )
            objects.append(Sphere(center, radius, _material_ref(primitive.get("material"), materials, path)))
        elif primitive_type == "plane":
            point = _vec3(_required(primitive, "point", path), path + ".point", ERROR_GEOMETRY)
            normal = _vec3(_required(primitive, "normal", path), path + ".normal", ERROR_GEOMETRY)
            try:
                normal = normalize(normal)
            except ValueError as exc:
                raise SceneError(ERROR_GEOMETRY, "%s.normal is invalid: %s" % (path, exc)) from exc
            objects.append(Plane(point, normal, _material_ref(primitive.get("material"), materials, path)))
        elif primitive_type == "aabb":
            minimum = _vec3(_required(primitive, "min", path), path + ".min", ERROR_GEOMETRY)
            maximum = _vec3(_required(primitive, "max", path), path + ".max", ERROR_GEOMETRY)
            if any(minimum[axis] >= maximum[axis] for axis in range(3)):
                raise SceneError(
                    ERROR_GEOMETRY, "%s.min must be strictly smaller than %s.max" % (path, path)
                )
            objects.append(AABB(minimum, maximum, _material_ref(primitive.get("material"), materials, path)))
        elif primitive_type == "triangle":
            v0 = _vec3(_required(primitive, "v0", path), path + ".v0", ERROR_GEOMETRY)
            v1 = _vec3(_required(primitive, "v1", path), path + ".v1", ERROR_GEOMETRY)
            v2 = _vec3(_required(primitive, "v2", path), path + ".v2", ERROR_GEOMETRY)
            area_vector = cross(sub(v1, v0), sub(v2, v0))
            if length(area_vector) <= 1.0e-12:
                raise SceneError(ERROR_GEOMETRY, "%s is degenerate" % path)
            objects.append(Triangle(v0, v1, v2, _material_ref(primitive.get("material"), materials, path)))
        else:
            raise SceneError(ERROR_GEOMETRY, "unknown geometry type %r at %s.type" % (primitive_type, path))
    return objects


def _light_intensity(value: Any, path: str) -> Vec3:
    if _is_number(value):
        scalar = _number(value, path, ERROR_LIGHTS, minimum=0.0, maximum=1000.0)
        return (scalar, scalar, scalar)
    color = _vec3(value, path, ERROR_LIGHTS)
    for channel in color:
        if channel < 0.0 or channel > 1000.0:
            raise SceneError(ERROR_LIGHTS, "%s channels must lie in [0,1000]" % path)
    return color


def _parse_lights(data: Any) -> List[Any]:
    if not isinstance(data, list):
        raise SceneError(ERROR_LIGHTS, "lights must be an array")
    if not data:
        raise SceneError(ERROR_LIGHTS, "lights must not be empty")
    lights: List[Any] = []
    for index, light_data in enumerate(data):
        path = "lights[%d]" % index
        if not isinstance(light_data, dict):
            raise SceneError(ERROR_LIGHTS, "%s must be an object" % path)
        light_type = _required(light_data, "type", path)
        if not isinstance(light_type, str):
            raise SceneError(ERROR_LIGHTS, "%s.type must be a string" % path)
        if light_type == "rect":
            center = _vec3(_required(light_data, "position", path), path + ".position", ERROR_LIGHTS)
            right = _vec3(_required(light_data, "right", path), path + ".right", ERROR_LIGHTS)
            up = _vec3(_required(light_data, "up", path), path + ".up", ERROR_LIGHTS)
            if length(right) <= 1.0e-12 or length(up) <= 1.0e-12:
                raise SceneError(ERROR_LIGHTS, "%s.right and %s.up must be non-zero" % (path, path))
            try:
                right_unit = normalize(right)
                up_unit = normalize(up)
            except ValueError as exc:
                raise SceneError(ERROR_LIGHTS, "%s orientation is invalid: %s" % (path, exc)) from exc
            if abs(dot(right_unit, up_unit)) > 1.0e-4:
                raise SceneError(ERROR_LIGHTS, "%s.right and %s.up must be perpendicular" % (path, path))
            width = _number(
                _required(light_data, "width", path), path + ".width", ERROR_LIGHTS, minimum=1.0e-9
            )
            height = _number(
                _required(light_data, "height", path), path + ".height", ERROR_LIGHTS, minimum=1.0e-9
            )
            intensity = _light_intensity(
                _required(light_data, "intensity", path), path + ".intensity"
            )
            lights.append(
                RectLight(center, right_unit, up_unit, width, height, intensity, "light_%d" % index)
            )
        elif light_type == "sphere":
            center = _vec3(_required(light_data, "center", path), path + ".center", ERROR_LIGHTS)
            radius = _number(
                _required(light_data, "radius", path),
                path + ".radius",
                ERROR_LIGHTS,
                minimum=1.0e-9,
            )
            intensity = _light_intensity(
                _required(light_data, "intensity", path), path + ".intensity"
            )
            lights.append(SphereLight(center, radius, intensity, "light_%d" % index))
        else:
            raise SceneError(ERROR_LIGHTS, "unknown light type %r at %s.type" % (light_type, path))
    return lights


def _parse_render(data: Any) -> RenderConfig:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise SceneError(ERROR_RENDER_PARAM, "render must be an object")
    samples = _integer(
        data.get("samples_per_pixel", 2),
        "render.samples_per_pixel",
        ERROR_RENDER_PARAM,
        minimum=1,
        maximum=64,
    )
    aa = _integer(data.get("aa_samples", 1), "render.aa_samples", ERROR_RENDER_PARAM)
    if aa not in SUPPORTED_AA:
        raise SceneError(
            ERROR_RENDER_PARAM,
            "render.aa_samples must be one of %s (got %d)" % (SUPPORTED_AA, aa),
        )
    shadow = _integer(
        data.get("shadow_samples", 8),
        "render.shadow_samples",
        ERROR_RENDER_PARAM,
        minimum=1,
        maximum=256,
    )
    depth = _integer(
        data.get("max_depth", 5),
        "render.max_depth",
        ERROR_RENDER_PARAM,
        minimum=1,
        maximum=32,
    )
    exposure = _number(
        data.get("exposure", 1.0),
        "render.exposure",
        ERROR_RENDER_PARAM,
        minimum=1.0e-6,
        maximum=100.0,
    )
    epsilon = _number(
        data.get("self_intersection_epsilon", 1.0e-4),
        "render.self_intersection_epsilon",
        ERROR_RENDER_PARAM,
        minimum=1.0e-7,
        maximum=1.0e-2,
    )
    return RenderConfig(samples, aa, shadow, depth, exposure, epsilon)


def _canonical_hash(description: Dict[str, Any]) -> str:
    canonical = json.dumps(
        description, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_scene(description: Dict[str, Any]) -> Scene:
    """Validate a scene description and build immutable render objects."""

    if not isinstance(description, dict):
        raise SceneError(ERROR_SCHEMA, "scene root must be a JSON object")
    description = copy.deepcopy(description)
    width = _integer(
        description.get("width", DEFAULT_WIDTH), "width", ERROR_SCHEMA, minimum=1, maximum=4096
    )
    height = _integer(
        description.get("height", DEFAULT_HEIGHT), "height", ERROR_SCHEMA, minimum=1, maximum=4096
    )
    seed = _integer(
        description.get("seed", DEFAULT_SEED),
        "seed",
        ERROR_SCHEMA,
        minimum=-(1 << 63),
        maximum=(1 << 63) - 1,
    )
    camera = _parse_camera(_required(description, "camera", "scene"))
    materials = _parse_materials(_required(description, "materials", "scene"))
    geometry = _parse_geometry(_required(description, "geometry", "scene"), materials)
    lights = _parse_lights(_required(description, "lights", "scene"))
    render = _parse_render(description.get("render"))
    background = _color(description.get("background", [0.0, 0.0, 0.0]), "background")
    return Scene(
        width=width,
        height=height,
        seed=seed,
        camera=camera,
        materials=materials,
        geometry=geometry,
        lights=lights,
        render=render,
        background=background,
        description_hash=_canonical_hash(description),
        description=description,
    )


def load_scene(path: str) -> Scene:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise SceneError(ERROR_IO_OR_JSON, "cannot read scene %r: %s" % (path, exc)) from exc
    except json.JSONDecodeError as exc:
        raise SceneError(
            ERROR_IO_OR_JSON,
            "invalid JSON in %r at line %d column %d: %s"
            % (path, exc.lineno, exc.colno, exc.msg),
        ) from exc
    return parse_scene(data)


def create_default_scene(
    seed: int = DEFAULT_SEED,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> Dict[str, Any]:
    """Build the deterministic demo scene used when no JSON file is supplied."""

    seed = int(seed)
    sphere_slide = (hash01(seed, 101) - 0.5) * 0.12
    sphere_depth = (hash01(seed, 102) - 0.5) * 0.10
    light_scale = 0.92 + 0.16 * hash01(seed, 203)
    return {
        "width": int(width),
        "height": int(height),
        "seed": seed,
        "background": [0.020, 0.025, 0.036],
        "camera": {
            "position": [0.0, 1.45, 5.6],
            "look_at": [0.0, 0.78, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov": 43.0,
        },
        "materials": {
            "ground_coated": {
                "type": "layered",
                "base": {"type": "diffuse", "color": [0.44, 0.46, 0.50]},
                "coating": {
                    "ior": 1.36,
                    "thickness": 0.001,
                    "tint": [1.0, 1.0, 1.0],
                    "absorption": [0.0, 0.0, 0.0],
                },
            },
            "ceramic": {
                "type": "layered",
                "base": {"type": "diffuse", "color": [0.10, 0.30, 0.62]},
                "coating": {
                    "ior": 1.48,
                    "thickness": 0.004,
                    "tint": [0.98, 0.99, 1.0],
                    "absorption": [0.03, 0.01, 0.0],
                },
            },
            "gold": {
                "type": "layered",
                "base": {"type": "metal", "color": [0.90, 0.62, 0.20], "roughness": 0.055},
                "coating": {
                    "ior": 1.52,
                    "thickness": 0.001,
                    "tint": [1.0, 1.0, 1.0],
                    "absorption": [0.0, 0.0, 0.0],
                },
            },
            "pedestal": {
                "type": "layered",
                "base": {"type": "diffuse", "color": [0.25, 0.18, 0.13]},
                "coating": {
                    "ior": 1.40,
                    "thickness": 0.0015,
                    "tint": [1.0, 0.98, 0.95],
                    "absorption": [0.0, 0.0, 0.0],
                },
            },
            "backdrop": {
                "type": "layered",
                "base": {"type": "diffuse", "color": [0.26, 0.29, 0.35]},
                "coating": {
                    "ior": 1.30,
                    "thickness": 0.001,
                    "tint": [1.0, 1.0, 1.0],
                    "absorption": [0.0, 0.0, 0.0],
                },
            },
        },
        "geometry": [
            {
                "type": "plane",
                "point": [0.0, 0.0, 0.0],
                "normal": [0.0, 1.0, 0.0],
                "material": "ground_coated",
            },
            {
                "type": "aabb",
                "min": [-1.58, 0.0, -0.48],
                "max": [-0.52, 0.20, 0.50],
                "material": "pedestal",
            },
            {
                "type": "sphere",
                "center": [-1.05 + sphere_slide, 0.80, 0.02 + sphere_depth],
                "radius": 0.60,
                "material": "ceramic",
            },
            {
                "type": "sphere",
                "center": [0.68, 0.52, -0.02],
                "radius": 0.52,
                "material": "gold",
            },
            {
                "type": "triangle",
                "v0": [-2.9, 0.0, -1.82],
                "v1": [2.9, 0.0, -1.82],
                "v2": [0.0, 3.2, -1.95],
                "material": "backdrop",
            },
        ],
        "lights": [
            {
                "type": "rect",
                "position": [-1.35, 3.25, 1.45],
                "right": [1.0, 0.0, 0.0],
                "up": [0.0, 0.0, -1.0],
                "width": 5.60,
                "height": 4.80,
                "intensity": [
                    0.72 * light_scale,
                    0.64 * light_scale,
                    0.50 * light_scale,
                ],
            },
            {
                "type": "sphere",
                "center": [2.80, 2.40, 0.80],
                "radius": 0.55,
                "intensity": [0.80 * light_scale, 0.90 * light_scale, 1.05 * light_scale],
            },
        ],
        "render": {
            "samples_per_pixel": 8,
            "aa_samples": 1,
            "shadow_samples": 8,
            "max_depth": 5,
            "exposure": 1.35,
            "self_intersection_epsilon": 1.0e-4,
        },
    }

