"""Parsing and strict validation for the self-describing scene document."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import SCENE_FORMAT, SCENE_VERSION
from .errors import (
    numeric,
    scene_hash,
    scene_io,
    scene_json,
    schema,
    timeline,
    unsupported_version,
)
from .math3d import mat4, quat_normalize, vec3


_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Bone:
    name: str
    parent: int
    inverse_bind: np.ndarray
    bind_local: np.ndarray


@dataclass(frozen=True)
class Keyframe:
    time: float
    translation: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class BoneTrack:
    bone: int
    keys: tuple[Keyframe, ...]


@dataclass(frozen=True)
class VertexCurve:
    vertex: int
    times: np.ndarray
    vectors: np.ndarray


@dataclass(frozen=True)
class BakeConfig:
    texture_sample_rate: float
    atlas_width: int
    atlas_height: int
    min_block_size: int
    padding: int
    error_tolerance: float


@dataclass
class Scene:
    source_path: Path
    raw_bytes: bytes
    input_file_sha256: str
    content_hash: str
    document: dict[str, Any]
    positions: np.ndarray
    normals: np.ndarray
    tangents: np.ndarray
    uvs: np.ndarray
    bone_weights: np.ndarray
    bone_indices: np.ndarray
    triangles: np.ndarray
    bones: tuple[Bone, ...]
    fps: float
    duration: float
    loop_mode: str
    tracks: dict[int, BoneTrack]
    curves: dict[int, VertexCurve]
    displacement_space: str
    displacement_mode: str
    bake: BakeConfig

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])


def canonical_json_bytes(document: dict[str, Any], *, drop_content_hash: bool = False) -> bytes:
    payload = dict(document)
    if drop_content_hash:
        payload.pop("contentHash", None)
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def compute_content_hash(document: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(document, drop_content_hash=True)).hexdigest()


def seal_scene(document: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(document)
    sealed["format"] = SCENE_FORMAT
    sealed["version"] = SCENE_VERSION
    sealed["hashAlgorithm"] = "sha256"
    sealed["contentHash"] = compute_content_hash(sealed)
    return sealed


def write_scene(path: Path, document: dict[str, Any]) -> None:
    sealed = seal_scene(document)
    text = json.dumps(sealed, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise schema(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise schema(f"{label} must be an array")
    return value


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise schema(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise numeric(f"{label} must be finite")
    return result


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise schema(f"{label} must be an integer")
    return int(value)


def _array(value: Any, shape: tuple[int, ...] | None, label: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise schema(f"{label} must be a rectangular numeric array") from exc
    if shape is not None and arr.shape != shape:
        raise schema(f"{label} must have shape {shape}, got {arr.shape}")
    if arr.ndim == 0:
        raise schema(f"{label} must be an array")
    if not np.all(np.isfinite(arr)):
        raise numeric(f"{label} contains a non-finite value")
    return arr


def _int_array(value: Any, shape: tuple[int, ...] | None, label: str) -> np.ndarray:
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise schema(f"{label} must be a rectangular integer array") from exc
    if shape is not None and arr.shape != shape:
        raise schema(f"{label} must have shape {shape}, got {arr.shape}")
    if arr.size == 0:
        raise schema(f"{label} must not be empty")
    if (
        not np.issubdtype(arr.dtype, np.number)
        or np.iscomplexobj(arr)
        or not np.all(np.isfinite(arr.astype(np.float64)))
    ):
        raise schema(f"{label} must contain finite numbers")
    if np.any(arr.astype(np.float64) != np.floor(arr.astype(np.float64))):
        raise schema(f"{label} must contain integers")
    return arr.astype(np.int64)


def _parse_matrix(value: Any, label: str) -> np.ndarray:
    return mat4(value, label)


def _parse_bones(mesh: dict[str, Any]) -> tuple[Bone, ...]:
    raw_bones = _require_list(mesh.get("skeleton"), "mesh.skeleton")
    if not raw_bones:
        raise schema("mesh.skeleton must contain at least one bone")
    bones: list[Bone] = []
    for i, raw in enumerate(raw_bones):
        item = _require_mapping(raw, f"mesh.skeleton[{i}]")
        name = item.get("name", f"bone_{i}")
        if not isinstance(name, str) or not name:
            raise schema(f"mesh.skeleton[{i}].name must be a non-empty string")
        parent = _require_int(item.get("parent", -1), f"mesh.skeleton[{i}].parent")
        if parent >= i:
            raise schema(f"mesh.skeleton[{i}].parent must refer to an earlier bone")
        if parent < -1:
            raise schema(f"mesh.skeleton[{i}].parent must be -1 or a valid bone index")
        inverse_bind = _parse_matrix(
            item.get("inverseBindMatrix"), f"mesh.skeleton[{i}].inverseBindMatrix"
        )
        bind_local = _parse_matrix(
            item.get("bindLocalMatrix"), f"mesh.skeleton[{i}].bindLocalMatrix"
        )
        if abs(float(np.linalg.det(inverse_bind))) < 1e-12:
            raise numeric(f"mesh.skeleton[{i}].inverseBindMatrix is singular")
        if abs(float(np.linalg.det(bind_local))) < 1e-12:
            raise numeric(f"mesh.skeleton[{i}].bindLocalMatrix is singular")
        bones.append(Bone(name=name, parent=parent, inverse_bind=inverse_bind, bind_local=bind_local))
    return tuple(bones)


def _parse_bone_bindings(
    mesh: dict[str, Any], vertex_count: int
) -> tuple[np.ndarray, np.ndarray]:
    raw_weights = _require_list(mesh.get("boneWeights"), "mesh.boneWeights")
    raw_indices = _require_list(mesh.get("boneIndices"), "mesh.boneIndices")
    if len(raw_weights) != vertex_count or len(raw_indices) != vertex_count:
        raise schema("boneWeights and boneIndices must contain one entry per vertex")
    weights = np.zeros((vertex_count, 4), dtype=np.float64)
    indices = np.zeros((vertex_count, 4), dtype=np.int64)
    for vertex in range(vertex_count):
        vertex_weights = _require_list(
            raw_weights[vertex], f"mesh.boneWeights[{vertex}]"
        )
        vertex_indices = _require_list(
            raw_indices[vertex], f"mesh.boneIndices[{vertex}]"
        )
        if not 1 <= len(vertex_weights) <= 4 or len(vertex_weights) != len(vertex_indices):
            raise schema(
                f"vertex {vertex} must have 1 to 4 matching bone weights and indices"
            )
        weights[vertex, : len(vertex_weights)] = _array(
            vertex_weights, (len(vertex_weights),), f"mesh.boneWeights[{vertex}]"
        )
        indices[vertex, : len(vertex_indices)] = _int_array(
            vertex_indices, (len(vertex_indices),), f"mesh.boneIndices[{vertex}]"
        )
    if np.any(weights < -1e-9):
        raise schema("mesh.boneWeights cannot contain negative values")
    weight_sums = weights.sum(axis=1)
    if np.any(weight_sums <= 1e-8):
        raise schema("each vertex must have a positive sum of bone weights")
    return weights / weight_sums[:, None], indices


def _parse_keyframes(raw_track: dict[str, Any], duration: float, label: str) -> tuple[Keyframe, ...]:
    raw_keys = _require_list(raw_track.get("keyframes"), f"{label}.keyframes")
    if not raw_keys:
        raise schema(f"{label}.keyframes must not be empty")
    keys: list[Keyframe] = []
    previous = -math.inf
    for i, raw in enumerate(raw_keys):
        item = _require_mapping(raw, f"{label}.keyframes[{i}]")
        t = _require_number(item.get("time"), f"{label}.keyframes[{i}].time")
        if t < -1e-9 or t > duration + 1e-9:
            raise timeline(f"{label}.keyframes[{i}].time is outside [0, duration]")
        t = min(duration, max(0.0, t))
        if t <= previous + 1e-9:
            raise timeline(f"{label}.keyframes must have strictly increasing times")
        previous = t
        translation = vec3(item.get("translation", [0.0, 0.0, 0.0]), f"{label}.keyframes[{i}].translation")
        rotation = quat_normalize(item.get("rotation", [0.0, 0.0, 0.0, 1.0]), f"{label}.keyframes[{i}].rotation")
        scale = vec3(item.get("scale", [1.0, 1.0, 1.0]), f"{label}.keyframes[{i}].scale")
        if np.any(np.abs(scale) < 1e-12):
            raise numeric(f"{label}.keyframes[{i}].scale must be non-zero")
        keys.append(Keyframe(t, translation, rotation, scale))
    return tuple(keys)


def _parse_displacement(
    raw: Any, vertex_count: int, duration: float, loop: bool
) -> tuple[dict[int, VertexCurve], str, str]:
    displacement = _require_mapping(raw, "displacement")
    space = displacement.get("space", "bind_normal_frame")
    if space not in {"bind_normal_frame", "normal"}:
        raise schema("displacement.space must be 'bind_normal_frame'")
    mode = displacement.get("integration", "absolute")
    if mode not in {"absolute", "delta"}:
        raise schema("displacement.integration must be 'absolute' or 'delta'")
    raw_curves = _require_list(displacement.get("vertexCurves"), "displacement.vertexCurves")
    curves: dict[int, VertexCurve] = {}
    for i, raw in enumerate(raw_curves):
        item = _require_mapping(raw, f"displacement.vertexCurves[{i}]")
        vertex = _require_int(item.get("vertex"), f"displacement.vertexCurves[{i}].vertex")
        if vertex < 0 or vertex >= vertex_count:
            raise schema(f"displacement.vertexCurves[{i}].vertex is out of range")
        if vertex in curves:
            raise schema(f"duplicate displacement curve for vertex {vertex}")
        raw_keys = _require_list(item.get("keys"), f"displacement.vertexCurves[{i}].keys")
        if not raw_keys:
            raise schema(f"displacement.vertexCurves[{i}].keys must not be empty")
        times: list[float] = []
        vectors: list[np.ndarray] = []
        previous = -math.inf
        for j, raw_key in enumerate(raw_keys):
            key = _require_mapping(raw_key, f"displacement.vertexCurves[{i}].keys[{j}]")
            t = _require_number(key.get("time"), f"displacement.vertexCurves[{i}].keys[{j}].time")
            if t < -1e-9 or t > duration + 1e-9:
                raise timeline(f"displacement curve key time {t} is outside [0, duration]")
            t = min(duration, max(0.0, t))
            if t <= previous + 1e-9:
                raise timeline(f"displacement curve keys for vertex {vertex} must be increasing")
            previous = t
            vector = vec3(
                key.get("offset"), f"displacement.vertexCurves[{i}].keys[{j}].offset"
            )
            times.append(t)
            vectors.append(vector)
        curves[vertex] = VertexCurve(
            vertex=vertex,
            times=np.asarray(times, dtype=np.float64),
            vectors=np.asarray(vectors, dtype=np.float64),
        )
    zero_times = np.array([0.0, duration], dtype=np.float64)
    zero_vectors = np.zeros((2, 3), dtype=np.float64)
    for vertex in range(vertex_count):
        if vertex not in curves:
            curves[vertex] = VertexCurve(vertex, zero_times.copy(), zero_vectors.copy())
    if loop:
        for vertex, curve in list(curves.items()):
            if curve.times[-1] >= duration - 1e-9:
                vectors = curve.vectors.copy()
                vectors[-1] = vectors[0]
                curves[vertex] = VertexCurve(vertex, curve.times.copy(), vectors)
    return curves, space, mode


def _parse_bake(raw: Any) -> BakeConfig:
    bake = _require_mapping(raw, "bake")
    rate = _require_number(bake.get("textureSampleRate"), "bake.textureSampleRate")
    if rate <= 0.0:
        raise schema("bake.textureSampleRate must be positive")
    width = _require_int(bake.get("atlasWidth"), "bake.atlasWidth")
    height = _require_int(bake.get("atlasHeight"), "bake.atlasHeight")
    if width <= 0 or height <= 0 or width > 16384 or height > 16384:
        raise schema("bake atlas dimensions must be in [1, 16384]")
    min_block = _require_int(bake.get("minBlockSize", 1), "bake.minBlockSize")
    if min_block < 1:
        raise schema("bake.minBlockSize must be at least 1")
    padding = _require_int(bake.get("padding", 1), "bake.padding")
    if padding < 1:
        raise schema("bake.padding must be at least 1 pixel")
    tolerance = _require_number(bake.get("errorTolerance"), "bake.errorTolerance")
    if tolerance < 0.0:
        raise schema("bake.errorTolerance must be non-negative")
    return BakeConfig(rate, width, height, min_block, padding, tolerance)


def parse_scene(raw_bytes: bytes, source_path: Path) -> Scene:
    try:
        document = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise scene_json(f"cannot parse scene JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise schema("scene root must be an object")
    if document.get("format") != SCENE_FORMAT:
        raise schema(f"scene.format must be '{SCENE_FORMAT}'")
    version = document.get("version")
    if not isinstance(version, str) or not version.startswith("1."):
        raise unsupported_version(
            f"scene version {version!r} is unsupported; expected major version 1"
        )
    if document.get("hashAlgorithm", "sha256") != "sha256":
        raise unsupported_version("scene.hashAlgorithm must be 'sha256'")
    supplied_hash = document.get("contentHash")
    if not isinstance(supplied_hash, str) or not _HASH_RE.match(supplied_hash):
        raise scene_hash("scene.contentHash must be a lowercase SHA-256 hex digest")
    try:
        computed_hash = compute_content_hash(document)
    except (TypeError, ValueError) as exc:
        raise schema(f"scene contains values that cannot be canonically hashed: {exc}") from exc
    if supplied_hash != computed_hash:
        raise scene_hash(
            f"scene content hash mismatch: declared {supplied_hash}, canonical {computed_hash}"
        )

    mesh = _require_mapping(document.get("mesh"), "mesh")
    positions = _array(mesh.get("positions"), None, "mesh.positions")
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise schema("mesh.positions must be a non-empty array of 3D points")
    vertex_count = positions.shape[0]
    normals = _array(mesh.get("normals"), (vertex_count, 3), "mesh.normals")
    tangents = _array(mesh.get("tangents"), (vertex_count, 4), "mesh.tangents")
    if np.any(np.abs(tangents[:, 3]) < 1e-12):
        raise schema("mesh.tangents handedness component must be non-zero")
    uvs = _array(mesh.get("uvs"), (vertex_count, 2), "mesh.uvs")
    weights, bone_indices = _parse_bone_bindings(mesh, vertex_count)
    bones = _parse_bones(mesh)
    if np.any(bone_indices < 0) or np.any(bone_indices >= len(bones)):
        raise schema("mesh.boneIndices contains an out-of-range bone index")
    triangles_raw = _require_list(mesh.get("triangles"), "mesh.triangles")
    if not triangles_raw:
        raise schema("mesh.triangles must not be empty")
    triangles = _int_array(triangles_raw, None, "mesh.triangles")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise schema("mesh.triangles must be an array of triangle index triples")
    if np.any(triangles < 0) or np.any(triangles >= vertex_count):
        raise schema("mesh.triangles contains an out-of-range vertex index")

    animation = _require_mapping(document.get("animation"), "animation")
    fps = _require_number(animation.get("fps"), "animation.fps")
    duration = _require_number(animation.get("duration"), "animation.duration")
    if fps <= 0.0 or duration <= 0.0:
        raise timeline("animation.fps and animation.duration must be positive")
    frame_product = fps * duration
    if abs(frame_product - round(frame_product)) > 1e-6:
        raise timeline(
            "animation duration times fps must be an integer so samples align to declared frames"
        )
    loop_mode = animation.get("loopMode")
    if loop_mode not in {"loop", "clamp"}:
        raise schema("animation.loopMode must be 'loop' or 'clamp'")
    raw_tracks = _require_list(animation.get("boneTracks"), "animation.boneTracks")
    tracks: dict[int, BoneTrack] = {}
    for i, raw in enumerate(raw_tracks):
        item = _require_mapping(raw, f"animation.boneTracks[{i}]")
        bone = _require_int(item.get("bone"), f"animation.boneTracks[{i}].bone")
        if bone < 0 or bone >= len(bones):
            raise schema(f"animation.boneTracks[{i}].bone is out of range")
        if bone in tracks:
            raise schema(f"duplicate animation track for bone {bone}")
        keys = _parse_keyframes(item, duration, f"animation.boneTracks[{i}]")
        tracks[bone] = BoneTrack(bone=bone, keys=keys)

    raw_curves = _require_mapping(document.get("displacement"), "displacement")
    curves, displacement_space, displacement_mode = _parse_displacement(
        raw_curves, vertex_count, duration, loop_mode == "loop"
    )
    bake = _parse_bake(document.get("bake"))

    return Scene(
        source_path=source_path,
        raw_bytes=raw_bytes,
        input_file_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        content_hash=computed_hash,
        document=document,
        positions=positions,
        normals=normals,
        tangents=tangents,
        uvs=uvs,
        bone_weights=weights,
        bone_indices=bone_indices,
        triangles=triangles,
        bones=bones,
        fps=fps,
        duration=duration,
        loop_mode=loop_mode,
        tracks=tracks,
        curves=curves,
        displacement_space=displacement_space,
        displacement_mode=displacement_mode,
        bake=bake,
    )


def load_scene(path: str | Path) -> Scene:
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise scene_io(f"cannot read scene file '{source}': {exc}") from exc
    return parse_scene(raw, source)

