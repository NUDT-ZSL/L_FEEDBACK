from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ModelError
from .math3d import from_column_major, normalize, world_bounds

SUPPORTED_SCHEMA_VERSION = 1
MIN_NODES = 3


@dataclass(frozen=True)
class Material:
    name: str
    diffuse: np.ndarray
    specular: np.ndarray
    roughness: float
    metallic: float


@dataclass(frozen=True)
class MeshNode:
    name: str
    positions: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    transform: np.ndarray
    material: Material


@dataclass(frozen=True)
class DirectionalLight:
    name: str
    direction: np.ndarray
    color: np.ndarray
    intensity: float
    cast_shadow: bool


@dataclass(frozen=True)
class PointLight:
    name: str
    position: np.ndarray
    color: np.ndarray
    intensity: float
    range: float
    cast_shadow: bool
    source_index: int


@dataclass(frozen=True)
class Camera:
    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_deg: float


@dataclass(frozen=True)
class SceneModel:
    nodes: tuple[MeshNode, ...]
    materials: dict[str, Material]
    ambient: np.ndarray
    directional: DirectionalLight
    point_lights: tuple[PointLight, ...]
    camera: Camera
    units: str
    color_space: str

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return world_bounds((node.transform, node.positions) for node in self.nodes)


def _fail(code: str, message: str, detail: str | None = None) -> None:
    raise ModelError(code, message, detail)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _vector(value: Any, size: int, code: str, label: str, *, non_negative: bool = False) -> np.ndarray:
    if not isinstance(value, list) or len(value) != size:
        _fail(code, f"{label} must be an array of {size} numbers")
    if not all(_is_number(item) for item in value):
        _fail(code, f"{label} contains a non-finite or non-numeric value")
    result = np.asarray(value, dtype=np.float64)
    if non_negative and np.any(result < 0.0):
        _fail(code, f"{label} must be non-negative")
    return result


def _string(value: Any, code: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(code, f"{label} must be a non-empty string")
    return value.strip()


def _boolean(value: Any, code: str, label: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        _fail(code, f"{label} must be a boolean")
    return value


def _matrix(value: Any, label: str) -> np.ndarray:
    if isinstance(value, list) and len(value) == 4 and all(
        isinstance(row, list) and len(row) == 4 for row in value
    ):
        flat = [item for row in value for item in row]
        matrix = np.asarray(flat, dtype=np.float64).reshape(4, 4)
    elif isinstance(value, list) and len(value) == 16:
        flat = value
        matrix = from_column_major(flat)
    else:
        _fail("E_MODEL_TRANSFORM", f"{label} must have 16 column-major values or 4 rows")
    if not np.all(np.isfinite(matrix)):
        _fail("E_MODEL_TRANSFORM", f"{label} contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        _fail("E_MODEL_TRANSFORM", f"{label} must be an affine matrix")
    if abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-10:
        _fail("E_MODEL_TRANSFORM", f"{label} has a singular 3x3 basis")
    return matrix


def _parse_material(raw: Any, index: int) -> Material:
    if not isinstance(raw, dict):
        _fail("E_MODEL_MATERIAL", f"materials[{index}] must be an object")
    name = _string(raw.get("name"), "E_MODEL_MATERIAL", f"materials[{index}].name")
    diffuse = _vector(raw.get("diffuse"), 3, "E_MODEL_MATERIAL", f"material {name}.diffuse")
    specular = _vector(
        raw.get("specular"), 3, "E_MODEL_MATERIAL", f"material {name}.specular", non_negative=True
    )
    if np.any(diffuse < 0.0):
        _fail("E_MODEL_MATERIAL", f"material {name}.diffuse must be non-negative")
    roughness = raw.get("roughness")
    metallic = raw.get("metallic")
    if not _is_number(roughness) or not 0.0 <= float(roughness) <= 1.0:
        _fail("E_MODEL_MATERIAL", f"material {name}.roughness must be in [0, 1]")
    if not _is_number(metallic) or not 0.0 <= float(metallic) <= 1.0:
        _fail("E_MODEL_MATERIAL", f"material {name}.metallic must be in [0, 1]")
    return Material(
        name=name,
        diffuse=diffuse,
        specular=specular,
        roughness=float(roughness),
        metallic=float(metallic),
    )


def _parse_geometry(raw: Any, node_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label = f"node {node_name}.mesh"
    if not isinstance(raw, dict):
        _fail("E_MODEL_GEOMETRY", f"{label} must be an object")
    positions_raw = raw.get("positions")
    normals_raw = raw.get("normals")
    indices_raw = raw.get("indices")
    if not isinstance(positions_raw, list) or not positions_raw:
        _fail("E_MODEL_GEOMETRY", f"{label}.positions must be a non-empty array")
    if not isinstance(normals_raw, list) or len(normals_raw) != len(positions_raw):
        _fail("E_MODEL_GEOMETRY", f"{label}.normals must match positions")
    try:
        positions = np.asarray(positions_raw, dtype=np.float64)
        normals = np.asarray(normals_raw, dtype=np.float64)
    except (TypeError, ValueError):
        _fail("E_MODEL_GEOMETRY", f"{label} contains non-numeric vertex data")
    if positions.ndim == 1 and positions.size % 3 == 0:
        positions = positions.reshape(-1, 3)
    if normals.ndim == 1 and normals.size % 3 == 0:
        normals = normals.reshape(-1, 3)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 3:
        _fail("E_MODEL_GEOMETRY", f"{label}.positions must contain at least 3 XYZ vertices")
    if normals.shape != positions.shape:
        _fail("E_MODEL_GEOMETRY", f"{label}.normals must have the same shape as positions")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(normals)):
        _fail("E_MODEL_GEOMETRY", f"{label} contains non-finite values")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-6) or np.any(np.abs(lengths - 1.0) > 1e-3):
        _fail("E_MODEL_GEOMETRY", f"{label}.normals must be unit length")
    if not isinstance(indices_raw, list) or not indices_raw:
        _fail("E_MODEL_GEOMETRY", f"{label}.indices must be a non-empty array")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in indices_raw):
        _fail("E_MODEL_GEOMETRY", f"{label}.indices must contain integers")
    indices = np.asarray(indices_raw, dtype=np.uint32)
    if indices.size % 3 != 0:
        _fail("E_MODEL_GEOMETRY", f"{label}.indices length must be a multiple of 3")
    if int(indices.max()) >= positions.shape[0] or int(indices.min()) < 0:
        _fail("E_MODEL_GEOMETRY", f"{label}.indices reference a missing vertex")
    return positions, normals, indices


def _parse_point_light(raw: Any, index: int) -> PointLight:
    if not isinstance(raw, dict):
        _fail("E_MODEL_LIGHT", f"point_lights[{index}] must be an object")
    name = _string(raw.get("name"), "E_MODEL_LIGHT", f"point_lights[{index}].name")
    position = _vector(raw.get("position"), 3, "E_MODEL_LIGHT", f"point light {name}.position")
    color = _vector(
        raw.get("color"), 3, "E_MODEL_LIGHT", f"point light {name}.color", non_negative=True
    )
    intensity = raw.get("intensity")
    light_range = raw.get("range", 20.0)
    if not _is_number(intensity) or float(intensity) <= 0.0:
        _fail("E_MODEL_LIGHT", f"point light {name}.intensity must be positive")
    if not _is_number(light_range) or float(light_range) <= 0.0:
        _fail("E_MODEL_LIGHT", f"point light {name}.range must be positive")
    cast_shadow = _boolean(raw.get("cast_shadow"), "E_MODEL_LIGHT", f"point light {name}.cast_shadow", True)
    return PointLight(
        name=name,
        position=position,
        color=color,
        intensity=float(intensity),
        range=float(light_range),
        cast_shadow=cast_shadow,
        source_index=index,
    )


def model_from_dict(raw: Any) -> SceneModel:
    if not isinstance(raw, dict):
        _fail("E_MODEL_SCHEMA", "model root must be a JSON object")
    version = raw.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        _fail(
            "E_MODEL_VERSION",
            f"unsupported schema_version {version!r}",
            f"expected {SUPPORTED_SCHEMA_VERSION}",
        )
    units = raw.get("units", "meters")
    color_space = raw.get("color_space", "linear")
    if units != "meters":
        _fail("E_MODEL_SCHEMA", "only meters are supported", f"got {units!r}")
    if color_space != "linear":
        _fail("E_MODEL_SCHEMA", "material colors must be declared linear", f"got {color_space!r}")

    materials_raw = raw.get("materials")
    if not isinstance(materials_raw, list) or not materials_raw:
        _fail("E_MODEL_MATERIAL", "materials must be a non-empty array")
    materials: dict[str, Material] = {}
    for index, material_raw in enumerate(materials_raw):
        material = _parse_material(material_raw, index)
        if material.name in materials:
            _fail("E_MODEL_MATERIAL", "duplicate material name", material.name)
        materials[material.name] = material

    nodes_raw = raw.get("nodes")
    if not isinstance(nodes_raw, list):
        _fail("E_MODEL_SCHEMA", "nodes must be an array")
    if len(nodes_raw) < MIN_NODES:
        _fail("E_MODEL_NODE_COUNT", f"at least {MIN_NODES} mesh nodes are required", str(len(nodes_raw)))
    nodes: list[MeshNode] = []
    node_names: set[str] = set()
    for index, node_raw in enumerate(nodes_raw):
        if not isinstance(node_raw, dict):
            _fail("E_MODEL_SCHEMA", f"nodes[{index}] must be an object")
        name = _string(node_raw.get("name"), "E_MODEL_NAME", f"nodes[{index}].name")
        if name in node_names:
            _fail("E_MODEL_NAME", "duplicate mesh node name", name)
        node_names.add(name)
        material_name = _string(
            node_raw.get("material"), "E_MODEL_MATERIAL_REF", f"node {name}.material"
        )
        if material_name not in materials:
            _fail("E_MODEL_MATERIAL_REF", f"node {name} references an unknown material", material_name)
        positions, normals, indices = _parse_geometry(node_raw.get("mesh"), name)
        transform = _matrix(node_raw.get("transform"), f"node {name}.transform")
        nodes.append(
            MeshNode(
                name=name,
                positions=positions,
                normals=normals,
                indices=indices,
                transform=transform,
                material=materials[material_name],
            )
        )

    scene = raw.get("scene")
    if not isinstance(scene, dict):
        _fail("E_MODEL_SCHEMA", "scene must be an object")
    ambient = _vector(scene.get("ambient"), 3, "E_MODEL_LIGHT", "scene.ambient", non_negative=True)

    directional_raw = scene.get("directional")
    if not isinstance(directional_raw, dict):
        _fail("E_MODEL_LIGHT", "scene.directional must be an object")
    directional_name = _string(
        directional_raw.get("name", "directional"), "E_MODEL_LIGHT", "directional.name"
    )
    direction = _vector(
        directional_raw.get("direction"), 3, "E_MODEL_LIGHT", "directional.direction"
    )
    if np.linalg.norm(direction) < 1e-8:
        _fail("E_MODEL_LIGHT", "directional.direction must not be zero")
    direction = normalize(direction)
    directional_color = _vector(
        directional_raw.get("color"),
        3,
        "E_MODEL_LIGHT",
        "directional.color",
        non_negative=True,
    )
    directional_intensity = directional_raw.get("intensity")
    if not _is_number(directional_intensity) or float(directional_intensity) <= 0.0:
        _fail("E_MODEL_LIGHT", "directional.intensity must be positive")
    directional = DirectionalLight(
        name=directional_name,
        direction=direction,
        color=directional_color,
        intensity=float(directional_intensity),
        cast_shadow=_boolean(
            directional_raw.get("cast_shadow"),
            "E_MODEL_LIGHT",
            "directional.cast_shadow",
            True,
        ),
    )

    point_lights_raw = scene.get("point_lights", [])
    if not isinstance(point_lights_raw, list):
        _fail("E_MODEL_LIGHT", "scene.point_lights must be an array")
    point_lights = tuple(_parse_point_light(item, index) for index, item in enumerate(point_lights_raw))
    point_names: set[str] = set()
    for light in point_lights:
        if light.name in point_names:
            _fail("E_MODEL_LIGHT", "duplicate point light name", light.name)
        point_names.add(light.name)

    camera_raw = scene.get("camera")
    if not isinstance(camera_raw, dict):
        _fail("E_MODEL_SCHEMA", "scene.camera must be an object")
    camera_position = _vector(
        camera_raw.get("position"), 3, "E_MODEL_SCHEMA", "camera.position"
    )
    camera_target = _vector(camera_raw.get("target"), 3, "E_MODEL_SCHEMA", "camera.target")
    camera_up = _vector(camera_raw.get("up", [0, 1, 0]), 3, "E_MODEL_SCHEMA", "camera.up")
    if np.linalg.norm(camera_target - camera_position) < 1e-8:
        _fail("E_MODEL_SCHEMA", "camera target must differ from camera position")
    if np.linalg.norm(camera_up) < 1e-8:
        _fail("E_MODEL_SCHEMA", "camera up must not be zero")
    if np.linalg.norm(np.cross(camera_target - camera_position, camera_up)) < 1e-8:
        _fail("E_MODEL_SCHEMA", "camera up must not be parallel to the view direction")
    fov_deg = camera_raw.get("fov_deg")
    if not _is_number(fov_deg) or not 1.0 <= float(fov_deg) < 179.0:
        _fail("E_MODEL_SCHEMA", "camera.fov_deg must be between 1 and 179")
    camera = Camera(
        position=camera_position,
        target=camera_target,
        up=camera_up,
        fov_deg=float(fov_deg),
    )
    return SceneModel(
        nodes=tuple(nodes),
        materials=materials,
        ambient=ambient,
        directional=directional,
        point_lights=point_lights,
        camera=camera,
        units=units,
        color_space=color_space,
    )


def load_model(path: str | Path) -> SceneModel:
    model_path = Path(path)
    try:
        text = model_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelError("E_MODEL_READ", f"cannot read model {model_path}", str(exc)) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelError(
            "E_MODEL_JSON",
            f"invalid JSON in {model_path}",
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}",
        ) from exc
    return model_from_dict(raw)


def select_point_lights(
    lights: tuple[PointLight, ...], limit: int, target: np.ndarray
) -> tuple[PointLight, ...]:
    """Choose the most influential lights deterministically.

    Score is intensity / squared distance to the camera target, which combines
    brightness and distance. Name and source index break ties.
    """
    if limit <= 0:
        return ()
    target = np.asarray(target, dtype=np.float64)

    def key(light: PointLight) -> tuple[float, str, int]:
        distance_sq = max(float(np.dot(light.position - target, light.position - target)), 1e-6)
        score = light.intensity / distance_sq
        return (-score, light.name, light.source_index)

    eligible = list(lights)
    eligible.sort(key=key)
    return tuple(eligible[:limit])

