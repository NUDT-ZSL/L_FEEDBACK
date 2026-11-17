from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import SceneError
from .jsonutil import canonical_hash
from .math3d import F64, look_at, normalize


MAX_VERTICES = 5000
MAX_WIDTH = 1280
MAX_HEIGHT = 720
MAX_WEIGHTS = 4
MAX_SHADOW_RESOLUTION = 2048


@dataclass(frozen=True)
class Material:
    name: str
    diffuse: np.ndarray
    specular: np.ndarray
    shininess: float


@dataclass(frozen=True)
class Camera:
    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_y_degrees: float
    fov_y_radians: float
    near: float
    far: float
    right: np.ndarray
    true_up: np.ndarray
    forward: np.ndarray
    view: np.ndarray


@dataclass(frozen=True)
class DirectionalLight:
    direction: np.ndarray
    color: np.ndarray
    intensity: float
    cast_shadow: bool


@dataclass(frozen=True)
class PointLight:
    position: np.ndarray
    color: np.ndarray
    intensity: float
    falloff: float


@dataclass(frozen=True)
class ShadowConfig:
    resolution: int
    splits: np.ndarray
    bias: float
    slope_bias: float
    normal_offset: float
    extent_padding: float
    depth_padding: float

    @property
    def resolution_depth_bias(self) -> float:
        if self.resolution <= 1024:
            return 1.5 / float(self.resolution)
        return 0.5 / float(self.resolution)

    @property
    def normal_offset_scale(self) -> float:
        if self.resolution <= 1024:
            return math.sqrt(1024.0 / float(self.resolution))
        return 1.0


@dataclass(frozen=True)
class Bone:
    name: str
    parent: int


@dataclass(frozen=True)
class Scene:
    scene_id: str
    width: int
    height: int
    background: np.ndarray
    camera: Camera
    ambient: np.ndarray
    directional: DirectionalLight
    point: PointLight
    shadow: ShadowConfig
    materials: tuple[Material, ...]
    bones: tuple[Bone, ...]
    parents: np.ndarray
    inverse_bind: np.ndarray
    topo_order: tuple[int, ...]
    frames: tuple[int, ...]
    local_by_frame: dict[int, np.ndarray]
    frame_hashes: dict[int, str]
    vertices: np.ndarray
    normals: np.ndarray
    skin_indices: np.ndarray
    skin_weights: np.ndarray
    triangles: np.ndarray
    triangle_materials: np.ndarray
    frame_pattern: str
    stats_pattern: str
    manifest_name: str
    scene_hash: str
    structure_hash: str
    raw_scene: dict[str, Any]


def _path(parent: str, child: str) -> str:
    if parent == "$":
        return child
    return f"{parent}.{child}" if parent else child


def _expect_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SceneError("E_SCHEMA", path, "expected a JSON object")
    return value


def _require(obj: dict[str, Any], key: str, path: str) -> Any:
    field_path = _path(path, key)
    if key not in obj:
        raise SceneError("E_SCHEMA", field_path, "required field is missing")
    return obj[key]


def _as_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SceneError("E_SCHEMA", path, "expected an integer")
    return value


def _as_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SceneError("E_SCHEMA", path, "expected a number")
    result = float(value)
    if not math.isfinite(result):
        raise SceneError("E_SCHEMA", path, "expected a finite number")
    return result


def _as_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise SceneError("E_SCHEMA", path, "expected a non-empty string")
    return value


def _as_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise SceneError("E_SCHEMA", path, "expected a boolean")
    return value


def _vector(value: Any, path: str, length: int) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise SceneError("E_SCHEMA", path, f"expected an array of {length} numbers")
    result = np.empty(length, dtype=F64)
    for index, item in enumerate(value):
        result[index] = _as_float(item, f"{path}[{index}]")
    return result


def _matrix4(value: Any, path: str) -> np.ndarray:
    if isinstance(value, list):
        if len(value) == 16 and all(
            not isinstance(item, (list, dict)) for item in value
        ):
            flat = [_as_float(item, f"{path}[{index}]") for index, item in enumerate(value)]
            return np.asarray(flat, dtype=F64).reshape(4, 4)
        if len(value) == 4 and all(isinstance(row, list) for row in value):
            rows: list[list[float]] = []
            for row_index, row in enumerate(value):
                if len(row) != 4:
                    raise SceneError(
                        "E_MATRIX_DIM",
                        f"{path}[{row_index}]",
                        "matrix rows must contain exactly 4 values",
                    )
                rows.append(
                    [
                        _as_float(item, f"{path}[{row_index}][{column_index}]")
                        for column_index, item in enumerate(row)
                    ]
                )
            return np.asarray(rows, dtype=F64)
    raise SceneError(
        "E_MATRIX_DIM",
        path,
        "expected a 4x4 matrix as 16 values or four rows of four values",
    )


def _check_vector_range(
    vector: np.ndarray,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    if minimum is not None and np.any(vector < minimum):
        raise SceneError("E_SCHEMA", path, f"values must be >= {minimum}")
    if maximum is not None and np.any(vector > maximum):
        raise SceneError("E_SCHEMA", path, f"values must be <= {maximum}")


def _material(value: Any, index: int) -> Material:
    path = f"materials[{index}]"
    obj = _expect_object(value, path)
    name = obj.get("name", f"material_{index}")
    name = _as_string(name, f"{path}.name")
    diffuse = _vector(_require(obj, "diffuse", path), f"{path}.diffuse", 3)
    specular_value = obj.get("specular", [0.0, 0.0, 0.0])
    specular = _vector(specular_value, f"{path}.specular", 3)
    shininess = _as_float(_require(obj, "shininess", path), f"{path}.shininess")
    _check_vector_range(diffuse, f"{path}.diffuse", minimum=0.0)
    _check_vector_range(specular, f"{path}.specular", minimum=0.0)
    if shininess <= 0.0:
        raise SceneError("E_SCHEMA", f"{path}.shininess", "shininess must be positive")
    return Material(name=name, diffuse=diffuse, specular=specular, shininess=shininess)


def _camera(value: Any) -> Camera:
    obj = _expect_object(value, "camera")
    position = _vector(_require(obj, "position", "camera"), "camera.position", 3)
    target = _vector(_require(obj, "target", "camera"), "camera.target", 3)
    up_raw = obj.get("up", [0.0, 1.0, 0.0])
    up = normalize(_vector(up_raw, "camera.up", 3), "camera.up", code="E_CAMERA")
    fov = _as_float(
        _require(obj, "fov_y_degrees", "camera"), "camera.fov_y_degrees"
    )
    if not 1.0 <= fov <= 179.0:
        raise SceneError("E_CAMERA", "camera.fov_y_degrees", "fov must be in [1, 179]")
    near = _as_float(_require(obj, "near", "camera"), "camera.near")
    far = _as_float(_require(obj, "far", "camera"), "camera.far")
    if near <= 0.0 or far <= near:
        raise SceneError(
            "E_CAMERA", "camera.far", "camera requires 0 < near < far"
        )
    direction = target - position
    if float(np.linalg.norm(direction)) <= 1e-12:
        raise SceneError("E_CAMERA", "camera.target", "camera target equals position")
    forward = normalize(direction, "camera.target", code="E_CAMERA")
    right = np.cross(forward, up)
    if float(np.linalg.norm(right)) <= 1e-10:
        raise SceneError("E_CAMERA", "camera.up", "camera up is parallel to view direction")
    right = normalize(right)
    true_up = normalize(np.cross(right, forward))
    return Camera(
        position=position,
        target=target,
        up=up,
        fov_y_degrees=fov,
        fov_y_radians=math.radians(fov),
        near=near,
        far=far,
        right=right,
        true_up=true_up,
        forward=forward,
        view=look_at(position, target, up),
    )


def _directional(value: Any) -> DirectionalLight:
    obj = _expect_object(value, "lights.directional")
    direction = normalize(
        _vector(
            _require(obj, "direction", "lights.directional"),
            "lights.directional.direction",
            3,
        ),
        "lights.directional.direction",
    )
    color = _vector(
        obj.get("color", [1.0, 1.0, 1.0]), "lights.directional.color", 3
    )
    intensity = _as_float(
        obj.get("intensity", 1.0), "lights.directional.intensity"
    )
    cast_shadow = _as_bool(
        obj.get("cast_shadow", True), "lights.directional.cast_shadow"
    )
    _check_vector_range(color, "lights.directional.color", minimum=0.0)
    if intensity < 0.0:
        raise SceneError(
            "E_SCHEMA", "lights.directional.intensity", "intensity must be non-negative"
        )
    return DirectionalLight(
        direction=direction,
        color=color,
        intensity=intensity,
        cast_shadow=cast_shadow,
    )


def _point(value: Any, camera: Camera) -> PointLight:
    obj = _expect_object(value, "lights.point")
    position_value = obj.get("position", camera.position.tolist())
    position = _vector(position_value, "lights.point.position", 3)
    color = _vector(obj.get("color", [1.0, 1.0, 1.0]), "lights.point.color", 3)
    intensity = _as_float(obj.get("intensity", 0.0), "lights.point.intensity")
    falloff = _as_float(obj.get("falloff", 0.0), "lights.point.falloff")
    _check_vector_range(color, "lights.point.color", minimum=0.0)
    if intensity < 0.0:
        raise SceneError(
            "E_SCHEMA", "lights.point.intensity", "intensity must be non-negative"
        )
    if falloff < 0.0:
        raise SceneError(
            "E_SCHEMA", "lights.point.falloff", "falloff must be non-negative"
        )
    return PointLight(
        position=position,
        color=color,
        intensity=intensity,
        falloff=falloff,
    )


def _shadow(value: Any, camera: Camera) -> ShadowConfig:
    obj = _expect_object(value, "shadow")
    resolution = _as_int(obj.get("resolution", 1024), "shadow.resolution")
    if resolution <= 0 or resolution > MAX_SHADOW_RESOLUTION:
        raise SceneError(
            "E_LIMIT",
            "shadow.resolution",
            f"shadow resolution must be in [1, {MAX_SHADOW_RESOLUTION}]",
        )
    splits_value = obj.get("splits")
    if splits_value is None:
        distance = _as_float(
            obj.get("distance", camera.far), "shadow.distance"
        )
        if distance <= camera.near:
            raise SceneError(
                "E_SHADOW", "shadow.distance", "shadow distance must exceed camera near"
            )
        distance = min(distance, camera.far)
        lam = _as_float(obj.get("split_lambda", 0.5), "shadow.split_lambda")
        if not 0.0 <= lam <= 1.0:
            raise SceneError(
                "E_SHADOW", "shadow.split_lambda", "split_lambda must be in [0, 1]"
            )
        near = camera.near
        splits = []
        for index in range(3):
            ratio = (index + 1) / 3.0
            logarithmic = near * ((distance / near) ** ratio)
            uniform = near + (distance - near) * ratio
            splits.append(lam * logarithmic + (1.0 - lam) * uniform)
        splits_array = np.asarray(splits, dtype=F64)
    else:
        if not isinstance(splits_value, list) or len(splits_value) != 3:
            raise SceneError(
                "E_SHADOW",
                "shadow.splits",
                "exactly three cascade split distances are required",
            )
        splits_array = np.asarray(
            [
                _as_float(item, f"shadow.splits[{index}]")
                for index, item in enumerate(splits_value)
            ],
            dtype=F64,
        )
        if (
            np.any(splits_array <= camera.near)
            or np.any(np.diff(splits_array) <= 0.0)
        ):
            raise SceneError(
                "E_SHADOW",
                "shadow.splits",
                "cascade splits must exceed camera.near and be strictly increasing",
            )
    bias = _as_float(obj.get("bias", 0.001), "shadow.bias")
    slope_bias = _as_float(obj.get("slope_bias", 0.004), "shadow.slope_bias")
    normal_offset = _as_float(
        obj.get("normal_offset", 0.012), "shadow.normal_offset"
    )
    extent_padding = _as_float(
        obj.get("extent_padding", 1.12), "shadow.extent_padding"
    )
    depth_padding = _as_float(
        obj.get("depth_padding", 2.0), "shadow.depth_padding"
    )
    if min(bias, slope_bias, normal_offset, depth_padding) < 0.0:
        raise SceneError(
            "E_SHADOW", "shadow", "bias, offsets and padding must be non-negative"
        )
    if extent_padding < 1.0:
        raise SceneError(
            "E_SHADOW", "shadow.extent_padding", "extent_padding must be >= 1"
        )
    return ShadowConfig(
        resolution=resolution,
        splits=splits_array,
        bias=bias,
        slope_bias=slope_bias,
        normal_offset=normal_offset,
        extent_padding=extent_padding,
        depth_padding=depth_padding,
    )


def _clean_pattern(value: Any, path: str, extension: str, default: str) -> str:
    pattern = default if value is None else _as_string(value, path)
    if "/" in pattern or "\\" in pattern or pattern in {".", ".."}:
        raise SceneError("E_SCHEMA", path, "pattern must be a file name, not a path")
    try:
        example = pattern.format(frame=0)
    except (KeyError, IndexError, ValueError) as exc:
        raise SceneError(
            "E_SCHEMA", path, "pattern must accept a frame argument such as {frame:04d}"
        ) from exc
    if not example.endswith(extension):
        raise SceneError("E_SCHEMA", path, f"pattern must produce a {extension} file")
    return pattern


def _parse_bones(
    skeleton: dict[str, Any],
) -> tuple[tuple[Bone, ...], np.ndarray, np.ndarray, tuple[int, ...]]:
    bones_value = _require(skeleton, "bones", "skeleton")
    if not isinstance(bones_value, list) or not bones_value:
        raise SceneError("E_SCHEMA", "skeleton.bones", "expected a non-empty array")
    bones: list[Bone] = []
    parents = np.empty(len(bones_value), dtype=np.int32)
    inverse_bind = np.empty((len(bones_value), 4, 4), dtype=F64)
    names: set[str] = set()
    for index, value in enumerate(bones_value):
        path = f"skeleton.bones[{index}]"
        obj = _expect_object(value, path)
        name = _as_string(_require(obj, "name", path), f"{path}.name")
        if name in names:
            raise SceneError("E_SCHEMA", f"{path}.name", "bone names must be unique")
        names.add(name)
        parent = _as_int(_require(obj, "parent", path), f"{path}.parent")
        if parent < -1 or parent >= len(bones_value):
            raise SceneError(
                "E_BONE_PARENT",
                f"{path}.parent",
                "parent must be -1 or a valid bone index",
            )
        if parent == index:
            raise SceneError(
                "E_BONE_CYCLE", f"{path}.parent", "bone cannot parent itself"
            )
        inverse_bind[index] = _matrix4(
            _require(obj, "inverse_bind_matrix", path),
            f"{path}.inverse_bind_matrix",
        )
        parents[index] = parent
        bones.append(Bone(name=name, parent=parent))

    state = np.zeros(len(bones), dtype=np.int8)
    order: list[int] = []
    for start in range(len(bones)):
        if state[start] == 2:
            continue
        path: list[int] = []
        index = start
        while index >= 0 and state[index] == 0:
            state[index] = 1
            path.append(index)
            index = int(parents[index])
        if index >= 0 and state[index] == 1:
            raise SceneError(
                "E_BONE_CYCLE",
                f"skeleton.bones[{index}].parent",
                "bone parent chain contains a cycle",
            )
        for item in reversed(path):
            state[item] = 2
            order.append(item)
    return tuple(bones), parents, inverse_bind, tuple(order)


def _parse_mesh(
    mesh: dict[str, Any], bone_count: int, material_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices_raw = _require(mesh, "vertices", "mesh")
    if not isinstance(vertices_raw, list) or not vertices_raw:
        raise SceneError("E_SCHEMA", "mesh.vertices", "expected a non-empty array")
    if len(vertices_raw) > MAX_VERTICES:
        raise SceneError(
            "E_LIMIT",
            "mesh.vertices",
            f"at most {MAX_VERTICES} vertices are supported",
        )
    vertices = np.empty((len(vertices_raw), 3), dtype=F64)
    for index, value in enumerate(vertices_raw):
        vertices[index] = _vector(value, f"mesh.vertices[{index}]", 3)

    normals_raw = _require(mesh, "normals", "mesh")
    if not isinstance(normals_raw, list) or len(normals_raw) != len(vertices_raw):
        raise SceneError(
            "E_SCHEMA", "mesh.normals", "normals must match the vertex count"
        )
    normals = np.empty_like(vertices)
    for index, value in enumerate(normals_raw):
        path = f"mesh.normals[{index}]"
        normal = _vector(value, path, 3)
        if float(np.linalg.norm(normal)) <= 1e-12:
            raise SceneError("E_SCHEMA", path, "normal must be non-zero")
        normals[index] = normal / np.linalg.norm(normal)

    indices_raw = _require(mesh, "skin_indices", "mesh")
    weights_raw = _require(mesh, "skin_weights", "mesh")
    if not isinstance(indices_raw, list) or len(indices_raw) != len(vertices_raw):
        raise SceneError(
            "E_SCHEMA",
            "mesh.skin_indices",
            "skin_indices must contain one entry per vertex",
        )
    if not isinstance(weights_raw, list) or len(weights_raw) != len(vertices_raw):
        raise SceneError(
            "E_SCHEMA",
            "mesh.skin_weights",
            "skin_weights must contain one entry per vertex",
        )
    skin_indices = np.zeros((len(vertices_raw), MAX_WEIGHTS), dtype=np.int32)
    skin_weights = np.zeros((len(vertices_raw), MAX_WEIGHTS), dtype=F64)
    for vertex_index, (indices_value, weights_value) in enumerate(
        zip(indices_raw, weights_raw)
    ):
        indices_path = f"mesh.skin_indices[{vertex_index}]"
        weights_path = f"mesh.skin_weights[{vertex_index}]"
        if not isinstance(indices_value, list) or not isinstance(weights_value, list):
            raise SceneError(
                "E_SCHEMA", indices_path, "bone indices and weights must be arrays"
            )
        if len(indices_value) > MAX_WEIGHTS:
            raise SceneError(
                "E_WEIGHT_COUNT",
                indices_path,
                f"at most {MAX_WEIGHTS} bone influences are allowed",
            )
        if len(weights_value) > MAX_WEIGHTS:
            raise SceneError(
                "E_WEIGHT_COUNT",
                weights_path,
                f"at most {MAX_WEIGHTS} bone influences are allowed",
            )
        if len(indices_value) != len(weights_value):
            raise SceneError(
                "E_SCHEMA",
                weights_path,
                "bone index and weight counts must match",
            )
        if not indices_value:
            raise SceneError(
                "E_WEIGHT_COUNT",
                indices_path,
                "at least one bone influence is required",
            )
        weight_sum = 0.0
        for influence, (bone_value, weight_value) in enumerate(
            zip(indices_value, weights_value)
        ):
            bone_index = _as_int(bone_value, f"{indices_path}[{influence}]")
            if bone_index < 0 or bone_index >= bone_count:
                raise SceneError(
                    "E_WEIGHT_BONE_RANGE",
                    f"{indices_path}[{influence}]",
                    f"bone index must be in [0, {bone_count - 1}]",
                )
            weight = _as_float(weight_value, f"{weights_path}[{influence}]")
            if weight < 0.0:
                raise SceneError(
                    "E_WEIGHT_RANGE",
                    f"{weights_path}[{influence}]",
                    "skin weights must be non-negative",
                )
            skin_indices[vertex_index, influence] = bone_index
            skin_weights[vertex_index, influence] = weight
            weight_sum += weight
        if abs(weight_sum - 1.0) > 1e-6:
            raise SceneError(
                "E_WEIGHT_SUM",
                weights_path,
                f"skin weights must sum to 1.0 (got {weight_sum:.12g})",
            )

    triangles_raw = _require(mesh, "triangles", "mesh")
    if not isinstance(triangles_raw, list) or not triangles_raw:
        raise SceneError("E_SCHEMA", "mesh.triangles", "expected a non-empty array")
    triangles = np.empty((len(triangles_raw), 3), dtype=np.int32)
    triangle_materials = np.empty(len(triangles_raw), dtype=np.int32)
    for triangle_index, value in enumerate(triangles_raw):
        path = f"mesh.triangles[{triangle_index}]"
        obj = _expect_object(value, path)
        indices_value = _require(obj, "indices", path)
        if not isinstance(indices_value, list) or len(indices_value) != 3:
            raise SceneError(
                "E_SCHEMA", f"{path}.indices", "expected exactly three indices"
            )
        for corner, index_value in enumerate(indices_value):
            vertex_index = _as_int(index_value, f"{path}.indices[{corner}]")
            if vertex_index < 0 or vertex_index >= len(vertices_raw):
                raise SceneError(
                    "E_INDEX",
                    f"{path}.indices[{corner}]",
                    "triangle index is outside the vertex array",
                )
            triangles[triangle_index, corner] = vertex_index
        material_index = _as_int(
            _require(obj, "material", path), f"{path}.material"
        )
        if material_index < 0 or material_index >= material_count:
            raise SceneError(
                "E_MATERIAL_RANGE",
                f"{path}.material",
                "material slot is outside the material array",
            )
        triangle_materials[triangle_index] = material_index
    return (
        vertices,
        normals,
        skin_indices,
        skin_weights,
        triangles,
        triangle_materials,
    )


def _parse_animation(
    animation: dict[str, Any], bones_value: list[Any], bone_count: int
) -> tuple[tuple[int, ...], dict[int, np.ndarray]]:
    frames_raw = _require(animation, "frames", "animation")
    if not isinstance(frames_raw, list) or not frames_raw:
        raise SceneError("E_SCHEMA", "animation.frames", "expected a non-empty array")
    frames = [
        _as_int(value, f"animation.frames[{index}]")
        for index, value in enumerate(frames_raw)
    ]
    if any(frame < 0 for frame in frames):
        raise SceneError(
            "E_KEYFRAME", "animation.frames", "frame numbers must be non-negative"
        )
    if len(set(frames)) != len(frames):
        raise SceneError(
            "E_KEYFRAME", "animation.frames", "frame numbers must be unique"
        )
    frame_set = set(frames)
    local_by_frame: dict[int, np.ndarray] = {
        frame: np.empty((bone_count, 4, 4), dtype=F64) for frame in frames
    }
    seen: list[set[int]] = [set() for _ in range(bone_count)]
    for bone_index, value in enumerate(bones_value):
        path = f"skeleton.bones[{bone_index}]"
        obj = _expect_object(value, path)
        local_frames = _require(obj, "local_frames", path)
        if not isinstance(local_frames, list):
            raise SceneError(
                "E_SCHEMA", f"{path}.local_frames", "expected an array"
            )
        for entry_index, entry in enumerate(local_frames):
            entry_path = f"{path}.local_frames[{entry_index}]"
            entry_obj = _expect_object(entry, entry_path)
            frame = _as_int(_require(entry_obj, "frame", entry_path), f"{entry_path}.frame")
            if frame not in frame_set:
                raise SceneError(
                    "E_KEYFRAME",
                    f"{entry_path}.frame",
                    "local frame is not listed in animation.frames",
                )
            if frame in seen[bone_index]:
                raise SceneError(
                    "E_KEYFRAME",
                    f"{entry_path}.frame",
                    "duplicate local transform for this bone and frame",
                )
            seen[bone_index].add(frame)
            local_by_frame[frame][bone_index] = _matrix4(
                _require(entry_obj, "matrix", entry_path),
                f"{entry_path}.matrix",
            )
    for bone_index, bone_seen in enumerate(seen):
        missing = sorted(frame_set - bone_seen)
        if missing:
            raise SceneError(
                "E_KEYFRAME",
                f"skeleton.bones[{bone_index}].local_frames",
                f"missing local transform for frame {missing[0]}",
            )
    return tuple(sorted(frames)), local_by_frame


def _frame_hash(
    frame: int,
    bones: tuple[Bone, ...],
    local_matrices: np.ndarray,
) -> str:
    payload = {
        "frame": frame,
        "bones": [
            {
                "name": bone.name,
                "local": local_matrices[index].reshape(-1).tolist(),
            }
            for index, bone in enumerate(bones)
        ],
    }
    return canonical_hash(payload)


def _structure_hash(raw_scene: dict[str, Any]) -> str:
    structure = copy.deepcopy(raw_scene)
    if isinstance(structure.get("animation"), dict):
        structure["animation"].pop("frames", None)
    skeleton = structure.get("skeleton")
    if isinstance(skeleton, dict) and isinstance(skeleton.get("bones"), list):
        for bone in skeleton["bones"]:
            if isinstance(bone, dict):
                bone.pop("local_frames", None)
    return canonical_hash(structure)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _parse_json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite JSON number {value}")
    return result


def load_scene(path: Path) -> Scene:
    import json

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SceneError("E_IO", str(path), f"cannot read scene: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise SceneError("E_SCHEMA", "$", f"scene is not valid UTF-8: {exc}") from exc
    try:
        raw_scene = json.loads(
            raw_text,
            parse_constant=_reject_json_constant,
            parse_float=_parse_json_float,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise SceneError("E_SCHEMA", "$", f"invalid JSON: {exc}") from exc
    if not isinstance(raw_scene, dict):
        raise SceneError("E_SCHEMA", "$", "scene root must be an object")

    version = _as_int(
        _require(raw_scene, "schema_version", "$"), "schema_version"
    )
    if version != 1:
        raise SceneError("E_SCHEMA", "schema_version", "only schema_version 1 is supported")
    scene_id = _as_string(_require(raw_scene, "scene_id", "$"), "scene_id")

    output = _expect_object(_require(raw_scene, "output", "$"), "output")
    width = _as_int(_require(output, "width", "output"), "output.width")
    height = _as_int(_require(output, "height", "output"), "output.height")
    if width <= 0 or height <= 0:
        raise SceneError("E_SCHEMA", "output", "width and height must be positive")
    if width > MAX_WIDTH or height > MAX_HEIGHT:
        raise SceneError(
            "E_LIMIT",
            "output",
            f"output is limited to {MAX_WIDTH}x{MAX_HEIGHT}",
        )
    background = _vector(
        output.get("background", [0.03, 0.035, 0.045]),
        "output.background",
        3,
    )
    _check_vector_range(
        background,
        "output.background",
        minimum=0.0,
        maximum=1.0,
    )
    frame_pattern = _clean_pattern(
        output.get("frame_pattern"),
        "output.frame_pattern",
        ".png",
        "frame_{frame:04d}.png",
    )
    stats_pattern = _clean_pattern(
        output.get("stats_pattern"),
        "output.stats_pattern",
        ".json",
        "frame_{frame:04d}.json",
    )
    manifest_name = _clean_pattern(
        output.get("manifest_name"),
        "output.manifest_name",
        ".json",
        "manifest.json",
    )
    if manifest_name in {frame_pattern, stats_pattern}:
        raise SceneError(
            "E_SCHEMA", "output.manifest_name", "manifest name conflicts with output pattern"
        )

    camera = _camera(_require(raw_scene, "camera", "$"))
    lights = _expect_object(_require(raw_scene, "lights", "$"), "lights")
    ambient = _vector(
        lights.get("ambient", [0.05, 0.05, 0.05]), "lights.ambient", 3
    )
    _check_vector_range(ambient, "lights.ambient", minimum=0.0)
    directional = _directional(_require(lights, "directional", "lights"))
    point = _point(_require(lights, "point", "lights"), camera)
    shadow = _shadow(raw_scene.get("shadow", {}), camera)

    materials_raw = _require(raw_scene, "materials", "$")
    if not isinstance(materials_raw, list) or not materials_raw:
        raise SceneError("E_SCHEMA", "materials", "expected a non-empty material array")
    materials = tuple(
        _material(value, index) for index, value in enumerate(materials_raw)
    )

    skeleton = _expect_object(_require(raw_scene, "skeleton", "$"), "skeleton")
    bones, parents, inverse_bind, topo_order = _parse_bones(skeleton)
    bones_value = skeleton["bones"]
    animation = _expect_object(
        _require(raw_scene, "animation", "$"), "animation"
    )
    frames, local_by_frame = _parse_animation(
        animation, bones_value, len(bones)
    )
    mesh = _expect_object(_require(raw_scene, "mesh", "$"), "mesh")
    (
        vertices,
        normals,
        skin_indices,
        skin_weights,
        triangles,
        triangle_materials,
    ) = _parse_mesh(mesh, len(bones), len(materials))

    frame_hashes = {
        frame: _frame_hash(frame, bones, local_by_frame[frame]) for frame in frames
    }
    return Scene(
        scene_id=scene_id,
        width=width,
        height=height,
        background=background,
        camera=camera,
        ambient=ambient,
        directional=directional,
        point=point,
        shadow=shadow,
        materials=materials,
        bones=bones,
        parents=parents,
        inverse_bind=inverse_bind,
        topo_order=topo_order,
        frames=frames,
        local_by_frame=local_by_frame,
        frame_hashes=frame_hashes,
        vertices=vertices,
        normals=normals,
        skin_indices=skin_indices,
        skin_weights=skin_weights,
        triangles=triangles,
        triangle_materials=triangle_materials,
        frame_pattern=frame_pattern,
        stats_pattern=stats_pattern,
        manifest_name=manifest_name,
        scene_hash=canonical_hash(raw_scene),
        structure_hash=_structure_hash(raw_scene),
        raw_scene=raw_scene,
    )
