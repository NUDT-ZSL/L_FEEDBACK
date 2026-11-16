"""Scene configuration loading, validation, hierarchy evaluation, and checksums."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import glm
import numpy as np
from PIL import Image

from .errors import ConfigError
from .mathutil import (
    glm_to_numpy,
    node_matrix,
    normalize,
    sha256_json,
    transform_points,
    vec3,
)
from .primitives import make_box, make_plane, make_sphere


MAX_LIGHTS = 16


@dataclass
class Mesh:
    name: str
    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    tangents: np.ndarray
    bitangent_signs: np.ndarray
    indices: np.ndarray

    @property
    def triangle_count(self) -> int:
        return int(self.indices.size // 3)


@dataclass
class Material:
    name: str
    albedo: np.ndarray
    metallic: float
    roughness: float
    normal_map: np.ndarray | None = None
    normal_strength: float = 1.0


@dataclass
class Light:
    name: str
    type: str
    color: np.ndarray
    intensity: float
    direction: np.ndarray
    casts_shadow: bool = True
    position: np.ndarray | None = None
    range: float = 20.0
    inner_deg: float = 15.0
    outer_deg: float = 30.0


@dataclass
class Camera:
    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_y_deg: float
    near: float
    far: float

    def matrices(self, aspect: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        view = glm.lookAt(vec3(self.position), vec3(self.target), vec3(self.up))
        projection = glm.perspective(
            glm.radians(float(self.fov_y_deg)), float(aspect), float(self.near), float(self.far)
        )
        return glm_to_numpy(view), glm_to_numpy(projection), glm_to_numpy(projection * view)


@dataclass
class ShadowSettings:
    enabled: bool = True
    directional_resolution: int = 512
    spot_resolution: int = 512
    point_resolution: int = 256
    bias: float = 0.0015
    normal_bias: float = 0.003
    filter_radius: int = 1
    filter_taps: int = 3


@dataclass
class RenderSettings:
    width: int
    height: int
    ssaa: int = 1
    exposure: float = 1.0
    tone_mapper: str = "aces"
    shadows: ShadowSettings = field(default_factory=ShadowSettings)


@dataclass
class IndirectSettings:
    enabled: bool = True
    probe_type: str = "hemisphere"
    sky_color: np.ndarray = field(default_factory=lambda: np.asarray([0.16, 0.22, 0.32], np.float32))
    ground_color: np.ndarray = field(default_factory=lambda: np.asarray([0.035, 0.03, 0.025], np.float32))
    intensity: float = 0.65


@dataclass
class Node:
    name: str
    mesh_name: str
    material_name: str
    parent_name: str | None
    translation: np.ndarray
    rotation_deg: np.ndarray
    scale: np.ndarray
    world_matrix: np.ndarray


@dataclass
class Scene:
    source_path: Path
    raw: dict[str, Any]
    camera: Camera
    meshes: Dict[str, Mesh]
    materials: Dict[str, Material]
    nodes: list[Node]
    lights: list[Light]
    rendering: RenderSettings
    indirect: IndirectSettings
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    checksums: dict[str, str]
    _node_matrices: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def bounds_center(self) -> np.ndarray:
        return (self.bounds_min + self.bounds_max) * 0.5

    @property
    def bounds_radius(self) -> float:
        return float(np.linalg.norm(self.bounds_max - self.bounds_min) * 0.5)

    @property
    def internal_size(self) -> tuple[int, int]:
        return self.rendering.width * self.rendering.ssaa, self.rendering.height * self.rendering.ssaa


def _error(path: str, message: str) -> ConfigError:
    return ConfigError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise _error(path, "expected a JSON object")
    return value


def _sequence(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise _error(path, "expected a JSON array")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(path, "expected a non-empty string")
    return value


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise _error(path, "expected true or false")
    return value


def _number(value: Any, path: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(path, "expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise _error(path, "expected a finite number")
    if minimum is not None and result < minimum:
        raise _error(path, f"must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise _error(path, f"must be <= {maximum}")
    return result


def _integer(value: Any, path: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(path, "expected an integer")
    if not minimum <= value <= maximum:
        raise _error(path, f"must be in [{minimum}, {maximum}]")
    return value


def _vec(value: Any, path: str, count: int, *, minimum: float | None = None) -> np.ndarray:
    items = _sequence(value, path)
    if len(items) != count:
        raise _error(path, f"expected {count} numbers")
    return np.asarray([_number(v, f"{path}[{i}]", minimum=minimum) for i, v in enumerate(items)], dtype=np.float64)


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise _error(f"{path}.{key}", "required field is missing")
    return mapping[key]


def _optional(mapping: Mapping[str, Any], key: str, default: Any) -> Any:
    return mapping[key] if key in mapping else default


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"{path}: scene file does not exist")
    if not path.is_file():
        raise ConfigError(f"{path}: scene path is not a regular file")
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read scene: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top-level JSON value must be an object")
    return raw


def _primitive_mesh(name: str, spec: Mapping[str, Any], path: str) -> Mesh:
    kind = _string(_required(spec, "type", path), f"{path}.type").lower()
    if kind == "plane":
        size = _number(_optional(spec, "size", 1.0), f"{path}.size", minimum=0.0001)
        uv_scale = _number(_optional(spec, "uv_scale", 1.0), f"{path}.uv_scale", minimum=0.0)
        data = make_plane(size, uv_scale)
    elif kind == "box":
        data = make_box(_number(_optional(spec, "size", 1.0), f"{path}.size", minimum=0.0001))
    elif kind == "sphere":
        rings = _integer(_optional(spec, "rings", 18), f"{path}.rings", minimum=3, maximum=256)
        segments = _integer(_optional(spec, "segments", 28), f"{path}.segments", minimum=3, maximum=512)
        data = make_sphere(
            _number(_optional(spec, "radius", 1.0), f"{path}.radius", minimum=0.0001), rings, segments
        )
    else:
        raise _error(f"{path}.type", "expected one of: plane, box, sphere")
    return _build_mesh(name, data["positions"], data["normals"], data["uvs"], data["indices"], None, path)


def _flatten_numeric(value: Any, path: str, width: int) -> np.ndarray:
    items = _sequence(value, path)
    if not items:
        raise _error(path, "must not be empty")
    if all(isinstance(item, list) for item in items):
        flattened: list[Any] = []
        for index, row in enumerate(items):
            row_values = _sequence(row, f"{path}[{index}]")
            flattened.extend(row_values)
    else:
        flattened = list(items)
    if len(flattened) % width:
        raise _error(path, f"flat array length must be divisible by {width}")
    array = np.asarray(
        [_number(item, f"{path}[{index}]") for index, item in enumerate(flattened)], dtype=np.float64
    )
    return array.reshape(-1, width)


def _compute_tangents(
    positions: np.ndarray, normals: np.ndarray, uvs: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    tangent_sum = np.zeros_like(positions, dtype=np.float64)
    bitangent_sum = np.zeros_like(positions, dtype=np.float64)
    triangles = indices.reshape(-1, 3)
    for triangle in triangles:
        i0, i1, i2 = (int(triangle[0]), int(triangle[1]), int(triangle[2]))
        edge1 = positions[i1].astype(np.float64) - positions[i0].astype(np.float64)
        edge2 = positions[i2].astype(np.float64) - positions[i0].astype(np.float64)
        duv1 = uvs[i1].astype(np.float64) - uvs[i0].astype(np.float64)
        duv2 = uvs[i2].astype(np.float64) - uvs[i0].astype(np.float64)
        determinant = duv1[0] * duv2[1] - duv1[1] * duv2[0]
        if abs(determinant) < 1e-12:
            continue
        inverse = 1.0 / determinant
        tangent = (edge1 * duv2[1] - edge2 * duv1[1]) * inverse
        bitangent = (edge2 * duv1[0] - edge1 * duv2[0]) * inverse
        for vertex in (i0, i1, i2):
            tangent_sum[vertex] += tangent
            bitangent_sum[vertex] += bitangent

    tangents = np.zeros_like(positions, dtype=np.float32)
    signs = np.ones(positions.shape[0], dtype=np.float32)
    for index in range(positions.shape[0]):
        normal = normals[index].astype(np.float64)
        tangent = tangent_sum[index]
        tangent = tangent - normal * float(np.dot(normal, tangent))
        if float(np.linalg.norm(tangent)) < 1e-10:
            axis = np.asarray([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.asarray([0.0, 1.0, 0.0])
            tangent = np.cross(normal, axis)
        tangent = normalize(tangent)
        tangents[index] = tangent.astype(np.float32)
        signs[index] = -1.0 if float(np.dot(np.cross(normal, tangent), bitangent_sum[index])) < 0.0 else 1.0
    return tangents, signs


def _build_mesh(
    name: str,
    positions_value: Any,
    normals_value: Any,
    uvs_value: Any,
    indices_value: Any,
    tangents_value: Any,
    path: str,
) -> Mesh:
    positions = _flatten_numeric(positions_value, f"{path}.positions", 3).astype(np.float32)
    count = positions.shape[0]
    if count < 3:
        raise _error(f"{path}.positions", "at least three vertices are required")
    if normals_value is None:
        normals = np.zeros_like(positions)
        normal_sum = np.zeros_like(positions, dtype=np.float64)
        indices_raw = _sequence(indices_value, f"{path}.indices")
    else:
        normals = _flatten_numeric(normals_value, f"{path}.normals", 3).astype(np.float32)
        if normals.shape[0] != count:
            raise _error(f"{path}.normals", "vertex count must match positions")
        normal_sum = None
        indices_raw = _sequence(indices_value, f"{path}.indices")
    if len(indices_raw) % 3 or not indices_raw:
        raise _error(f"{path}.indices", "must contain a non-empty multiple of three indices")
    indices = np.asarray(
        [_integer(index, f"{path}.indices[{i}]", minimum=0, maximum=count - 1) for i, index in enumerate(indices_raw)],
        dtype=np.uint32,
    )
    if normal_sum is not None:
        for triangle in indices.reshape(-1, 3):
            i0, i1, i2 = (int(triangle[0]), int(triangle[1]), int(triangle[2]))
            face = np.cross(positions[i1] - positions[i0], positions[i2] - positions[i0])
            normal_sum[i0] += face
            normal_sum[i1] += face
            normal_sum[i2] += face
        for index in range(count):
            normals[index] = normalize(normal_sum[index], fallback=np.asarray([0.0, 1.0, 0.0])).astype(np.float32)
    else:
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths < 1e-8):
            raise _error(f"{path}.normals", "normals must be non-zero")
        normals = (normals / lengths[:, None]).astype(np.float32)

    if uvs_value is None:
        uvs = np.zeros((count, 2), dtype=np.float32)
    else:
        uvs = _flatten_numeric(uvs_value, f"{path}.uvs", 2).astype(np.float32)
        if uvs.shape[0] != count:
            raise _error(f"{path}.uvs", "vertex count must match positions")

    if tangents_value is None:
        tangents, signs = _compute_tangents(positions, normals, uvs, indices)
    else:
        tangents = _flatten_numeric(tangents_value, f"{path}.tangents", 3).astype(np.float32)
        if tangents.shape[0] != count:
            raise _error(f"{path}.tangents", "vertex count must match positions")
        signs = np.ones(count, dtype=np.float32)
    return Mesh(name, positions, normals, uvs, tangents, signs, indices)


def _load_normal_map(value: Any, scene_dir: Path, path: str) -> np.ndarray | None:
    if value is None:
        return None
    spec = _mapping(value, path)
    if "file" in spec:
        relative = Path(_string(spec["file"], f"{path}.file"))
        file_path = relative if relative.is_absolute() else scene_dir / relative
        if not file_path.is_file():
            raise _error(f"{path}.file", f"normal map file does not exist: {file_path}")
        try:
            with Image.open(file_path) as image:
                rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        except OSError as exc:
            raise _error(f"{path}.file", f"cannot read normal map: {exc}") from exc
        return rgba

    width = _integer(_required(spec, "width", path), f"{path}.width", minimum=1, maximum=4096)
    height = _integer(_required(spec, "height", path), f"{path}.height", minimum=1, maximum=4096)
    channels = _integer(_optional(spec, "channels", 4), f"{path}.channels", minimum=3, maximum=4)
    data = _flatten_numeric(_required(spec, "data", path), f"{path}.data", channels)
    if data.shape[0] != width * height:
        raise _error(f"{path}.data", f"expected {width * height} pixels, got {data.shape[0]}")
    if np.any(data < 0.0) or np.any(data > 1.0):
        raise _error(f"{path}.data", "encoded normal-map values must be in [0, 1]")
    rgba = np.ones((height, width, 4), dtype=np.uint8)
    rgba[..., :channels] = np.rint(data.reshape(height, width, channels) * 255.0).astype(np.uint8)
    rgba[..., 3] = 255
    return rgba


def _parse_materials(raw: Mapping[str, Any]) -> Dict[str, Material]:
    result: Dict[str, Material] = {}
    for name, value in raw.items():
        path = f"materials.{name}"
        spec = _mapping(value, path)
        result[name] = Material(
            name=name,
            albedo=_vec(_required(spec, "albedo", path), f"{path}.albedo", 3, minimum=0.0),
            metallic=_number(_required(spec, "metallic", path), f"{path}.metallic", minimum=0.0, maximum=1.0),
            roughness=_number(
                _required(spec, "roughness", path), f"{path}.roughness", minimum=0.03, maximum=1.0
            ),
            normal_map=None,
            normal_strength=_number(
                _optional(spec, "normal_strength", 1.0), f"{path}.normal_strength", minimum=0.0, maximum=4.0
            ),
        )
    return result


def _parse_meshes(raw: Mapping[str, Any]) -> Dict[str, Mesh]:
    result: Dict[str, Mesh] = {}
    for name, value in raw.items():
        path = f"meshes.{name}"
        spec = _mapping(value, path)
        if "primitive" in spec:
            primitive = _mapping(spec["primitive"], f"{path}.primitive")
            result[name] = _primitive_mesh(name, primitive, f"{path}.primitive")
            continue
        positions = _required(spec, "positions", path)
        indices = _required(spec, "indices", path)
        result[name] = _build_mesh(
            name,
            positions,
            spec.get("normals"),
            spec.get("uvs"),
            indices,
            spec.get("tangents"),
            path,
        )
    return result


def _parse_camera(value: Any) -> Camera:
    spec = _mapping(value, "camera")
    position = _vec(_required(spec, "position", "camera"), "camera.position", 3)
    target = _vec(_required(spec, "target", "camera"), "camera.target", 3)
    up = _vec(_optional(spec, "up", [0.0, 1.0, 0.0]), "camera.up", 3)
    if float(np.linalg.norm(target - position)) < 1e-6:
        raise _error("camera.target", "must differ from camera.position")
    if float(np.linalg.norm(up)) < 1e-8:
        raise _error("camera.up", "must be a non-zero vector")
    if abs(float(np.dot(normalize(target - position), normalize(up)))) > 0.9999:
        raise _error("camera.up", "must not be parallel to the view direction")
    near = _number(_required(spec, "near", "camera"), "camera.near", minimum=0.0001)
    far = _number(_required(spec, "far", "camera"), "camera.far", minimum=near + 0.0001)
    return Camera(
        position=position,
        target=target,
        up=up,
        fov_y_deg=_number(
            _required(spec, "fov_y_deg", "camera"), "camera.fov_y_deg", minimum=1.0, maximum=179.0
        ),
        near=near,
        far=far,
    )


def _parse_lights(value: Any) -> list[Light]:
    lights_raw = _sequence(value, "lights")
    if not lights_raw:
        raise _error("lights", "at least one light is required")
    if len(lights_raw) > MAX_LIGHTS:
        raise _error("lights", f"at most {MAX_LIGHTS} lights are supported")
    lights: list[Light] = []
    names: set[str] = set()
    for index, item in enumerate(lights_raw):
        path = f"lights[{index}]"
        spec = _mapping(item, path)
        name = _string(_optional(spec, "name", f"light_{index}"), f"{path}.name")
        if name in names:
            raise _error(f"{path}.name", f"duplicate light name: {name}")
        names.add(name)
        light_type = _string(_required(spec, "type", path), f"{path}.type").lower()
        if light_type not in {"directional", "point", "spot"}:
            raise _error(f"{path}.type", "expected directional, point, or spot")
        direction = _vec(
            _required(spec, "direction", path) if light_type != "point" else _optional(spec, "direction", [0, -1, 0]),
            f"{path}.direction",
            3,
        )
        if float(np.linalg.norm(direction)) < 1e-8:
            raise _error(f"{path}.direction", "must be a non-zero vector")
        position = None
        if light_type in {"point", "spot"}:
            position = _vec(_required(spec, "position", path), f"{path}.position", 3)
        range_value = _number(
            _optional(spec, "range", 20.0), f"{path}.range", minimum=0.01
        )
        inner = _number(_optional(spec, "inner_deg", 15.0), f"{path}.inner_deg", minimum=0.0, maximum=89.0)
        outer = _number(_optional(spec, "outer_deg", 30.0), f"{path}.outer_deg", minimum=0.1, maximum=89.9)
        if light_type == "spot" and inner >= outer:
            raise _error(f"{path}.inner_deg", "must be smaller than outer_deg")
        lights.append(
            Light(
                name=name,
                type=light_type,
                color=_vec(_optional(spec, "color", [1, 1, 1]), f"{path}.color", 3, minimum=0.0),
                intensity=_number(
                    _required(spec, "intensity", path), f"{path}.intensity", minimum=0.0
                ),
                direction=normalize(direction),
                casts_shadow=_bool(_optional(spec, "casts_shadow", True), f"{path}.casts_shadow"),
                position=position,
                range=range_value,
                inner_deg=inner,
                outer_deg=outer,
            )
        )
    return lights


def _parse_shadows(value: Any) -> ShadowSettings:
    if value is None:
        return ShadowSettings()
    spec = _mapping(value, "rendering.shadows")
    taps = _integer(_optional(spec, "filter_taps", 3), "rendering.shadows.filter_taps", minimum=1, maximum=5)
    if taps not in {1, 3, 5}:
        raise _error("rendering.shadows.filter_taps", "must be 1, 3, or 5")
    return ShadowSettings(
        enabled=_bool(_optional(spec, "enabled", True), "rendering.shadows.enabled"),
        directional_resolution=_integer(
            _optional(spec, "directional_resolution", 512),
            "rendering.shadows.directional_resolution",
            minimum=64,
            maximum=4096,
        ),
        spot_resolution=_integer(
            _optional(spec, "spot_resolution", 512),
            "rendering.shadows.spot_resolution",
            minimum=64,
            maximum=4096,
        ),
        point_resolution=_integer(
            _optional(spec, "point_resolution", 256),
            "rendering.shadows.point_resolution",
            minimum=64,
            maximum=4096,
        ),
        bias=_number(_optional(spec, "bias", 0.0015), "rendering.shadows.bias", minimum=0.0, maximum=0.1),
        normal_bias=_number(
            _optional(spec, "normal_bias", 0.003), "rendering.shadows.normal_bias", minimum=0.0, maximum=0.1
        ),
        filter_radius=_integer(
            _optional(spec, "filter_radius", 1), "rendering.shadows.filter_radius", minimum=0, maximum=2
        ),
        filter_taps=taps,
    )


def _parse_rendering(value: Any) -> RenderSettings:
    spec = _mapping(value, "rendering")
    tone_mapper = _string(_optional(spec, "tone_mapper", "aces"), "rendering.tone_mapper").lower()
    if tone_mapper not in {"aces", "reinhard"}:
        raise _error("rendering.tone_mapper", "expected aces or reinhard")
    return RenderSettings(
        width=_integer(_required(spec, "width", "rendering"), "rendering.width", minimum=16, maximum=4096),
        height=_integer(_required(spec, "height", "rendering"), "rendering.height", minimum=16, maximum=4096),
        ssaa=_integer(_optional(spec, "ssaa", 1), "rendering.ssaa", minimum=1, maximum=4),
        exposure=_number(_optional(spec, "exposure", 1.0), "rendering.exposure", minimum=0.0001, maximum=1000.0),
        tone_mapper=tone_mapper,
        shadows=_parse_shadows(spec.get("shadows")),
    )


def _parse_indirect(value: Any) -> IndirectSettings:
    if value is None:
        return IndirectSettings()
    spec = _mapping(value, "indirect")
    probe = _mapping(_optional(spec, "probe", {}), "indirect.probe")
    probe_type = _string(_optional(probe, "type", "hemisphere"), "indirect.probe.type").lower()
    if probe_type != "hemisphere":
        raise _error("indirect.probe.type", "only hemisphere is supported")
    return IndirectSettings(
        enabled=_bool(_optional(spec, "enabled", True), "indirect.enabled"),
        probe_type=probe_type,
        sky_color=_vec(_optional(probe, "sky_color", [0.16, 0.22, 0.32]), "indirect.probe.sky_color", 3, minimum=0.0),
        ground_color=_vec(
            _optional(probe, "ground_color", [0.035, 0.03, 0.025]),
            "indirect.probe.ground_color",
            3,
            minimum=0.0,
        ),
        intensity=_number(_optional(spec, "intensity", 0.65), "indirect.intensity", minimum=0.0),
    )


def _evaluate_nodes(raw_nodes: Any, meshes: Mapping[str, Mesh], materials: Mapping[str, Material]) -> list[Node]:
    nodes_raw = _sequence(raw_nodes, "nodes")
    if not nodes_raw:
        raise _error("nodes", "at least one node is required")
    specs: dict[str, Mapping[str, Any]] = {}
    order: list[str] = []
    for index, value in enumerate(nodes_raw):
        path = f"nodes[{index}]"
        spec = _mapping(value, path)
        name = _string(_required(spec, "name", path), f"{path}.name")
        if name in specs:
            raise _error(f"{path}.name", f"duplicate node name: {name}")
        specs[name] = spec
        order.append(name)

    for name in order:
        spec = specs[name]
        mesh_name = _string(_required(spec, "mesh", f"nodes.{name}"), f"nodes.{name}.mesh")
        if mesh_name not in meshes and "mesh" not in spec:
            raise _error(f"nodes.{name}.mesh", f"unknown mesh: {mesh_name}")
        if mesh_name not in meshes:
            raise _error(f"nodes.{name}.mesh", f"unknown mesh: {mesh_name}")
        material_name = _string(_optional(spec, "material", "default"), f"nodes.{name}.material")
        if material_name not in materials:
            raise _error(f"nodes.{name}.material", f"unknown material: {material_name}")
        parent = spec.get("parent")
        if parent is not None and not isinstance(parent, str):
            raise _error(f"nodes.{name}.parent", "expected a node name or null")
        if parent is not None and parent not in specs:
            raise _error(f"nodes.{name}.parent", f"unknown parent node: {parent}")

    world: dict[str, np.ndarray] = {}
    visiting: set[str] = set()

    def evaluate(name: str) -> np.ndarray:
        if name in world:
            return world[name]
        if name in visiting:
            raise _error(f"nodes.{name}.parent", "cycle detected in scene graph")
        visiting.add(name)
        spec = specs[name]
        parent = spec.get("parent")
        local = node_matrix(
            _vec(_optional(spec, "translation", [0, 0, 0]), f"nodes.{name}.translation", 3),
            _vec(_optional(spec, "rotation_deg", [0, 0, 0]), f"nodes.{name}.rotation_deg", 3),
            _vec(_optional(spec, "scale", [1, 1, 1]), f"nodes.{name}.scale", 3),
        )
        if parent is None:
            matrix = local
        else:
            matrix = glm.mat4(glm_to_numpy(evaluate(parent))) * local
        world[name] = glm_to_numpy(matrix)
        visiting.remove(name)
        return world[name]

    nodes: list[Node] = []
    for name in order:
        spec = specs[name]
        matrix = evaluate(name)
        nodes.append(
            Node(
                name=name,
                mesh_name=str(_required(spec, "mesh", f"nodes.{name}")),
                material_name=str(_optional(spec, "material", "default")),
                parent_name=spec.get("parent"),
                translation=np.asarray(spec.get("translation", [0, 0, 0]), dtype=np.float64),
                rotation_deg=np.asarray(spec.get("rotation_deg", [0, 0, 0]), dtype=np.float64),
                scale=np.asarray(spec.get("scale", [1, 1, 1]), dtype=np.float64),
                world_matrix=matrix,
            )
        )
    return nodes


def _scene_bounds(meshes: Mapping[str, Mesh], nodes: Iterable[Node]) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.asarray([math.inf, math.inf, math.inf], dtype=np.float64)
    maximum = np.asarray([-math.inf, -math.inf, -math.inf], dtype=np.float64)
    found = False
    for node in nodes:
        mesh = meshes[node.mesh_name]
        transformed = transform_points(node.world_matrix, mesh.positions)
        minimum = np.minimum(minimum, transformed.min(axis=0))
        maximum = np.maximum(maximum, transformed.max(axis=0))
        found = True
    if not found:
        raise _error("nodes", "scene contains no drawable geometry")
    if np.any(maximum - minimum < 1e-8):
        maximum = minimum + np.asarray([1.0, 1.0, 1.0])
    return minimum.astype(np.float32), maximum.astype(np.float32)


def _mesh_payload(mesh: Mesh) -> dict[str, Any]:
    return {
        "positions": mesh.positions.astype(float).round(7).tolist(),
        "normals": mesh.normals.astype(float).round(7).tolist(),
        "uvs": mesh.uvs.astype(float).round(7).tolist(),
        "tangents": mesh.tangents.astype(float).round(7).tolist(),
        "bitangent_signs": mesh.bitangent_signs.astype(float).round(7).tolist(),
        "indices": mesh.indices.astype(int).tolist(),
    }


def _compute_checksums(
    camera: Camera,
    meshes: Mapping[str, Mesh],
    materials: Mapping[str, Material],
    nodes: list[Node],
    lights: list[Light],
    rendering: RenderSettings,
    indirect: IndirectSettings,
) -> dict[str, str]:
    geometry = {
        "meshes": {name: _mesh_payload(mesh) for name, mesh in sorted(meshes.items())},
        "nodes": [
            {
                "name": node.name,
                "mesh": node.mesh_name,
                "material": node.material_name,
                "parent": node.parent_name,
                "translation": node.translation.tolist(),
                "rotation_deg": node.rotation_deg.tolist(),
                "scale": node.scale.tolist(),
            }
            for node in nodes
        ],
        "camera": {
            "position": camera.position.tolist(),
            "target": camera.target.tolist(),
            "up": camera.up.tolist(),
            "fov_y_deg": camera.fov_y_deg,
            "near": camera.near,
            "far": camera.far,
        },
    }
    material_payload = {
        name: {
            "albedo": material.albedo.tolist(),
            "metallic": material.metallic,
            "roughness": material.roughness,
            "normal_strength": material.normal_strength,
            "normal_map_sha256": sha256_json(
                material.normal_map.astype(int).tolist() if material.normal_map is not None else None
            ),
        }
        for name, material in sorted(materials.items())
    }
    lighting_payload = [
        {
            "type": light.type,
            "color": light.color.tolist(),
            "intensity": light.intensity,
            "direction": light.direction.tolist(),
            "position": light.position.tolist() if light.position is not None else None,
            "range": light.range,
            "inner_deg": light.inner_deg,
            "outer_deg": light.outer_deg,
            "casts_shadow": light.casts_shadow,
        }
        for light in lights
    ]
    render_payload = {
        "width": rendering.width,
        "height": rendering.height,
        "ssaa": rendering.ssaa,
        "exposure": rendering.exposure,
        "tone_mapper": rendering.tone_mapper,
        "shadows": rendering.shadows.__dict__,
        "indirect": {
            "enabled": indirect.enabled,
            "probe_type": indirect.probe_type,
            "sky_color": indirect.sky_color.tolist(),
            "ground_color": indirect.ground_color.tolist(),
            "intensity": indirect.intensity,
        },
    }
    checksums = {
        "geometry": sha256_json(geometry),
        "materials": sha256_json(material_payload),
        "lighting": sha256_json(lighting_payload),
        "render_settings": sha256_json(render_payload),
    }
    checksums["scene"] = sha256_json(checksums)
    return checksums


def load_scene(path: str | os.PathLike[str]) -> Scene:
    source_path = Path(path).expanduser().resolve()
    raw = _load_json(source_path)
    version = _integer(_optional(raw, "version", 1), "version", minimum=1, maximum=1)
    del version
    camera = _parse_camera(_required(raw, "camera", "$"))
    materials_raw = _mapping(_required(raw, "materials", "$"), "materials")
    if not materials_raw:
        raise _error("materials", "at least one material is required")
    materials = _parse_materials(materials_raw)
    if "default" not in materials:
        materials["default"] = Material("default", np.asarray([0.8, 0.8, 0.8]), 0.0, 0.5)
    for name, material in materials.items():
        material_spec = _mapping(materials_raw.get(name, {}), f"materials.{name}")
        material.normal_map = _load_normal_map(
            material_spec.get("normal_map"), source_path.parent, f"materials.{name}.normal_map"
        )
    meshes = _parse_meshes(_mapping(_required(raw, "meshes", "$"), "meshes"))
    if not meshes:
        raise _error("meshes", "at least one mesh is required")
    nodes = _evaluate_nodes(_required(raw, "nodes", "$"), meshes, materials)
    lights = _parse_lights(_required(raw, "lights", "$"))
    rendering = _parse_rendering(_required(raw, "rendering", "$"))
    indirect = _parse_indirect(raw.get("indirect"))
    bounds_min, bounds_max = _scene_bounds(meshes, nodes)
    checksums = _compute_checksums(camera, meshes, materials, nodes, lights, rendering, indirect)
    return Scene(
        source_path=source_path,
        raw=raw,
        camera=camera,
        meshes=meshes,
        materials=materials,
        nodes=nodes,
        lights=lights,
        rendering=rendering,
        indirect=indirect,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        checksums=checksums,
    )


def refresh_checksums(scene: Scene) -> None:
    """Recompute checksums after programmatic changes to a loaded scene."""
    scene.checksums = _compute_checksums(
        scene.camera,
        scene.meshes,
        scene.materials,
        scene.nodes,
        scene.lights,
        scene.rendering,
        scene.indirect,
    )
