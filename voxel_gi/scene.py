"""Scene schema, validation, deterministic hashes, and probe layout."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .errors import BusinessError, ErrorCode
from .sampler import disk_samples, normalize

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Material:
    material_id: str
    albedo: np.ndarray


@dataclass(frozen=True)
class Mesh:
    name: str
    material_id: str
    triangles: np.ndarray
    occluder: bool = True


@dataclass(frozen=True)
class Light:
    light_id: str
    kind: str
    position: np.ndarray
    color: np.ndarray
    radiant_power: float
    normal: Optional[np.ndarray] = None
    radius: float = 0.0
    sample_count: int = 1
    sample_points: Optional[np.ndarray] = None
    area: float = 0.0


@dataclass(frozen=True)
class ProbeSpec:
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    spacing: float
    up: np.ndarray


@dataclass(frozen=True)
class BakeConfig:
    voxel_resolution: int
    padding_voxels: int
    ray_count: int
    threads: int
    seed: int
    ambient_radiance: np.ndarray


@dataclass(frozen=True)
class Scene:
    schema_version: int
    scene_id: str
    seed: int
    materials: Mapping[str, Material]
    meshes: Tuple[Mesh, ...]
    lights: Tuple[Light, ...]
    probe_spec: ProbeSpec
    bake: BakeConfig

    @property
    def mesh_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        vertices = np.concatenate(
            [mesh.triangles.reshape(-1, 3) for mesh in self.meshes if mesh.occluder],
            axis=0,
        )
        return np.min(vertices, axis=0), np.max(vertices, axis=0)

    @property
    def total_light_power(self) -> float:
        return float(sum(light.radiant_power for light in self.lights))

    @property
    def albedo_upper_bound(self) -> float:
        values = [float(np.max(material.albedo)) for material in self.materials.values()]
        return max(values) if values else 0.0

    @property
    def energy_bound(self) -> float:
        return self.total_light_power * self.albedo_upper_bound


@dataclass(frozen=True)
class ProbeLayout:
    positions: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    spacing: float
    up: np.ndarray
    counts: Tuple[int, int, int]

    @property
    def probe_count(self) -> int:
        return int(self.positions.shape[0])


def _structured(code: str, message: str, **details: Any) -> BusinessError:
    return BusinessError(code, message, details)


def _as_float_array(
    value: Any,
    name: str,
    length: Optional[int] = None,
) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise _structured(
            ErrorCode.INVALID_SCHEMA,
            "{} must be a numeric array".format(name),
            field=name,
            reason=str(exc),
        ) from exc
    if length is not None and result.shape != (length,):
        raise _structured(
            ErrorCode.INVALID_SCHEMA,
            "{} must contain exactly {} numbers".format(name, length),
            field=name,
            shape=list(result.shape),
        )
    if not np.all(np.isfinite(result)):
        raise _structured(
            ErrorCode.INVALID_SCHEMA,
            "{} contains non-finite values".format(name),
            field=name,
        )
    return result


def _parse_triangle_mesh(
    mesh_json: Mapping[str, Any],
    mesh_index: int,
    materials: Mapping[str, Material],
) -> Mesh:
    name = str(mesh_json.get("name", "mesh_{}".format(mesh_index)))
    material_id = mesh_json.get("material")
    if material_id not in materials:
        raise _structured(
            ErrorCode.INVALID_MATERIAL,
            "Mesh references an unknown material",
            mesh=name,
            material=material_id,
        )
    triangles_json = mesh_json.get("triangles")
    if not isinstance(triangles_json, list) or not triangles_json:
        raise _structured(
            ErrorCode.INVALID_SCHEMA,
            "Mesh triangles must be a non-empty list",
            mesh=name,
            field="triangles",
        )
    try:
        triangles = np.asarray(triangles_json, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise _structured(
            ErrorCode.INVALID_SCHEMA,
            "Mesh triangles must be numeric triples",
            mesh=name,
            reason=str(exc),
        ) from exc
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise _structured(
            ErrorCode.INVALID_SCHEMA,
            "Mesh triangles must have shape (T, 3, 3)",
            mesh=name,
            shape=list(triangles.shape),
        )
    for triangle_index in range(triangles.shape[0]):
        triangle = triangles[triangle_index]
        if not np.all(np.isfinite(triangle)):
            raise _structured(
                ErrorCode.INVALID_MESH_NAN,
                "Mesh contains NaN or infinite vertex coordinates",
                mesh=name,
                triangle=int(triangle_index),
            )
        edge_a = triangle[1] - triangle[0]
        edge_b = triangle[2] - triangle[0]
        cross_length = float(np.linalg.norm(np.cross(edge_a, edge_b)))
        if cross_length <= 1.0e-12:
            raise _structured(
                ErrorCode.INVALID_MESH_DEGENERATE,
                "Mesh contains a degenerate triangle",
                mesh=name,
                triangle=int(triangle_index),
                cross_length=cross_length,
            )
    return Mesh(
        name=name,
        material_id=str(material_id),
        triangles=np.ascontiguousarray(triangles),
        occluder=bool(mesh_json.get("occluder", True)),
    )


def _parse_light(
    light_json: Mapping[str, Any],
    light_index: int,
    scene_seed: int,
) -> Light:
    light_id = str(light_json.get("id", "light_{}".format(light_index)))
    kind = str(light_json.get("type", "area")).lower()
    if kind not in ("area", "point"):
        raise _structured(
            ErrorCode.INVALID_LIGHT,
            "Light type must be 'area' or 'point'",
            light=light_id,
            field="type",
            value=kind,
        )
    try:
        position = _as_float_array(light_json.get("position"), "light.position", 3)
    except BusinessError as exc:
        raise _structured(
            ErrorCode.INVALID_LIGHT,
            "Light position must be three finite numbers",
            light=light_id,
            field="position",
        ) from exc
    try:
        color = _as_float_array(
            light_json.get("color", [1.0, 1.0, 1.0]), "light.color", 3
        )
    except BusinessError as exc:
        raise _structured(
            ErrorCode.INVALID_LIGHT,
            "Light color must be three finite numbers",
            light=light_id,
            field="color",
        ) from exc
    if np.any(color < 0.0) or np.any(color > 1.0):
        raise _structured(
            ErrorCode.INVALID_LIGHT,
            "Light color channels must be in [0, 1]",
            light=light_id,
            field="color",
        )
    try:
        power = float(light_json.get("radiant_power"))
    except (TypeError, ValueError):
        raise _structured(
            ErrorCode.INVALID_LIGHT,
            "Light radiant_power must be numeric",
            light=light_id,
            field="radiant_power",
        )
    if not math.isfinite(power) or power <= 0.0 or power > 1.0e12:
        raise _structured(
            ErrorCode.INVALID_LIGHT,
            "Light radiant_power must be finite, positive, and <= 1e12",
            light=light_id,
            field="radiant_power",
            value=power,
        )
    try:
        sample_count = int(light_json.get("samples", 1))
    except (TypeError, ValueError) as exc:
        raise _structured(
            ErrorCode.INVALID_LIGHT,
            "Light samples must be an integer",
            light=light_id,
            field="samples",
        ) from exc
    if sample_count < 1 or sample_count > 64:
        raise _structured(
            ErrorCode.INVALID_LIGHT,
            "Light samples must be in [1, 64]",
            light=light_id,
            field="samples",
            value=sample_count,
        )
    if kind == "point":
        return Light(
            light_id=light_id,
            kind=kind,
            position=position,
            color=color,
            radiant_power=power,
            sample_count=1,
            sample_points=position[None, :].copy(),
            area=0.0,
        )
    try:
        radius = float(light_json.get("radius", 0.0))
    except (TypeError, ValueError) as exc:
        raise _structured(
            ErrorCode.INVALID_LIGHT,
            "Area light radius must be numeric",
            light=light_id,
            field="radius",
        ) from exc
    if not math.isfinite(radius) or radius <= 0.0 or radius > 1.0e6:
        raise _structured(
            ErrorCode.INVALID_LIGHT,
            "Area light radius must be finite and positive",
            light=light_id,
            field="radius",
            value=radius,
        )
    try:
        normal = normalize(_as_float_array(light_json.get("normal"), "light.normal", 3))
    except BusinessError as exc:
        raise _structured(
            ErrorCode.INVALID_LIGHT,
            "Area light normal must be a finite three-vector",
            light=light_id,
            field="normal",
        ) from exc
    except ValueError as exc:
        raise _structured(
            ErrorCode.INVALID_LIGHT,
            "Area light normal must be finite and non-zero",
            light=light_id,
            field="normal",
        ) from exc
    offsets = disk_samples(radius, sample_count, normal, scene_seed, light_index)
    sample_points = np.round(position[None, :] + offsets, 12)
    return Light(
        light_id=light_id,
        kind=kind,
        position=position,
        color=color,
        radiant_power=power,
        normal=normal,
        radius=radius,
        sample_count=sample_count,
        sample_points=np.ascontiguousarray(sample_points),
        area=math.pi * radius * radius,
    )


def load_scene(path: Path) -> Scene:
    input_path = Path(path)
    try:
        text = input_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BusinessError(
            ErrorCode.IO_ERROR,
            "Unable to read scene description",
            {"path": str(input_path), "reason": str(exc)},
        ) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BusinessError(
            ErrorCode.INVALID_JSON,
            "Scene description is not valid JSON",
            {
                "path": str(input_path),
                "line": exc.lineno,
                "column": exc.colno,
                "reason": exc.msg,
            },
        ) from exc
    return parse_scene(raw)


def parse_scene(raw: Any) -> Scene:
    if not isinstance(raw, dict):
        raise _structured(ErrorCode.INVALID_SCHEMA, "Scene root must be an object")
    try:
        schema_version = int(raw.get("schema_version", SCHEMA_VERSION))
    except (TypeError, ValueError) as exc:
        raise _structured(
            ErrorCode.INVALID_SCHEMA,
            "schema_version must be an integer",
            field="schema_version",
        ) from exc
    if schema_version != SCHEMA_VERSION:
        raise _structured(
            ErrorCode.INVALID_SCHEMA,
            "Unsupported scene schema version",
            expected=SCHEMA_VERSION,
            actual=schema_version,
        )

    materials_json = raw.get("materials")
    if not isinstance(materials_json, list) or not materials_json:
        raise _structured(
            ErrorCode.INVALID_MATERIAL,
            "Scene must contain at least one material",
            field="materials",
        )
    materials: Dict[str, Material] = {}
    for index, material_json in enumerate(materials_json):
        if not isinstance(material_json, dict):
            raise _structured(
                ErrorCode.INVALID_MATERIAL,
                "Material entry must be an object",
                material_index=index,
            )
        material_id = str(material_json.get("id", "material_{}".format(index)))
        if material_id in materials:
            raise _structured(
                ErrorCode.INVALID_MATERIAL,
                "Duplicate material id",
                material=material_id,
            )
        albedo = _as_float_array(material_json.get("albedo"), "material.albedo", 3)
        if np.any(albedo < 0.0) or np.any(albedo > 1.0):
            raise _structured(
                ErrorCode.INVALID_MATERIAL,
                "Material albedo channels must be in [0, 1]",
                material=material_id,
                field="albedo",
            )
        materials[material_id] = Material(material_id=material_id, albedo=albedo)

    meshes_json = raw.get("meshes")
    if not isinstance(meshes_json, list) or not meshes_json:
        raise _structured(
            ErrorCode.INVALID_SCHEMA,
            "Scene must contain at least one mesh",
            field="meshes",
        )
    meshes = tuple(
        _parse_triangle_mesh(mesh_json, index, materials)
        for index, mesh_json in enumerate(meshes_json)
        if isinstance(mesh_json, dict)
    )
    if len(meshes) != len(meshes_json) or not any(mesh.occluder for mesh in meshes):
        raise _structured(
            ErrorCode.INVALID_SCHEMA,
            "Every mesh must be an object and at least one must be an occluder",
            field="meshes",
        )

    try:
        scene_seed = int(raw.get("seed", 0))
    except (TypeError, ValueError) as exc:
        raise _structured(
            ErrorCode.INVALID_BAKE_CONFIG,
            "Scene seed must be an integer",
            field="seed",
        ) from exc
    if scene_seed < 0:
        raise _structured(
            ErrorCode.INVALID_BAKE_CONFIG,
            "Scene seed must be non-negative",
            field="seed",
            value=scene_seed,
        )
    lights_json = raw.get("lights")
    if not isinstance(lights_json, list) or not lights_json:
        raise _structured(
            ErrorCode.INVALID_LIGHT,
            "Scene must contain at least one light",
            field="lights",
        )
    lights: List[Light] = []
    for index, light_json in enumerate(lights_json):
        if not isinstance(light_json, dict):
            raise _structured(
                ErrorCode.INVALID_LIGHT,
                "Light entry must be an object",
                light_index=index,
            )
        lights.append(_parse_light(light_json, index, scene_seed))

    scene_probe = parse_scene_probe_layout(raw, meshes)

    bake_json = raw.get("bake", {})
    if not isinstance(bake_json, dict):
        raise _structured(
            ErrorCode.INVALID_BAKE_CONFIG,
            "bake must be an object",
            field="bake",
        )
    try:
        resolution = int(bake_json.get("voxel_resolution", 32))
        padding = int(bake_json.get("padding_voxels", 1))
        ray_count = int(bake_json.get("ray_count", 256))
        threads = int(bake_json.get("threads", 4))
        bake_seed = int(bake_json.get("seed", scene_seed))
        ambient = _as_float_array(
            bake_json.get("ambient_radiance", [0.0, 0.0, 0.0]),
            "bake.ambient_radiance",
            3,
        )
    except (TypeError, ValueError) as exc:
        raise _structured(
            ErrorCode.INVALID_BAKE_CONFIG,
            "Bake parameters must have the required numeric types",
            reason=str(exc),
        ) from exc
    if resolution < 4 or resolution > 512:
        raise _structured(
            ErrorCode.INVALID_BAKE_CONFIG,
            "voxel_resolution must be in [4, 512]",
            field="voxel_resolution",
            value=resolution,
        )
    if padding < 0 or 2 * padding >= resolution:
        raise _structured(
            ErrorCode.INVALID_BAKE_CONFIG,
            "padding_voxels must satisfy 0 <= 2*padding < voxel_resolution",
            field="padding_voxels",
            value=padding,
        )
    if ray_count < 8 or ray_count > 65536:
        raise _structured(
            ErrorCode.INVALID_BAKE_CONFIG,
            "ray_count must be in [8, 65536]",
            field="ray_count",
            value=ray_count,
        )
    if threads < 1 or threads > 256:
        raise _structured(
            ErrorCode.INVALID_BAKE_CONFIG,
            "threads must be in [1, 256]",
            field="threads",
            value=threads,
        )
    if bake_seed < 0:
        raise _structured(
            ErrorCode.INVALID_BAKE_CONFIG,
            "bake.seed must be non-negative",
            field="seed",
            value=bake_seed,
        )
    if np.any(ambient < 0.0) or np.any(ambient > 1.0e6):
        raise _structured(
            ErrorCode.INVALID_BAKE_CONFIG,
            "ambient_radiance channels must be finite and in [0, 1e6]",
            field="ambient_radiance",
        )

    scene_id = str(raw.get("scene_id", "scene"))
    if not scene_id:
        raise _structured(ErrorCode.INVALID_SCHEMA, "scene_id must be non-empty")
    return Scene(
        schema_version=schema_version,
        scene_id=scene_id,
        seed=scene_seed,
        materials=materials,
        meshes=meshes,
        lights=tuple(lights),
        probe_spec=scene_probe,
        bake=BakeConfig(
            voxel_resolution=resolution,
            padding_voxels=padding,
            ray_count=ray_count,
            threads=threads,
            seed=bake_seed,
            ambient_radiance=ambient,
        ),
    )


def parse_scene_probe_layout(
    raw: Mapping[str, Any],
    meshes: Sequence[Mesh],
) -> ProbeSpec:
    layout_json = raw.get("probe_layout", {})
    if not isinstance(layout_json, dict):
        raise _structured(
            ErrorCode.INVALID_PROBE_BOUNDS,
            "probe_layout must be an object",
            field="probe_layout",
        )
    try:
        spacing = float(layout_json.get("spacing", 1.0))
    except (TypeError, ValueError):
        raise _structured(
            ErrorCode.INVALID_PROBE_SPACING,
            "Probe spacing must be numeric",
            field="spacing",
        )
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise _structured(
            ErrorCode.INVALID_PROBE_SPACING,
            "Probe spacing must be finite and positive",
            field="spacing",
            value=spacing,
        )
    occluder_vertices = np.concatenate(
        [mesh.triangles.reshape(-1, 3) for mesh in meshes if mesh.occluder],
        axis=0,
    )
    default_min = np.min(occluder_vertices, axis=0)
    default_max = np.max(occluder_vertices, axis=0)
    bounds_min = _as_float_array(
        layout_json.get("bounds_min", default_min.tolist()),
        "probe_layout.bounds_min",
        3,
    )
    bounds_max = _as_float_array(
        layout_json.get("bounds_max", default_max.tolist()),
        "probe_layout.bounds_max",
        3,
    )
    if np.any(bounds_max < bounds_min):
        raise _structured(
            ErrorCode.INVALID_PROBE_BOUNDS,
            "Probe bounds_max must be component-wise >= bounds_min",
            field="bounds",
        )
    try:
        up = normalize(
            _as_float_array(layout_json.get("up", [0.0, 1.0, 0.0]), "probe_layout.up", 3)
        )
    except ValueError as exc:
        raise _structured(
            ErrorCode.INVALID_PROBE_BOUNDS,
            "Probe up vector must be finite and non-zero",
            field="up",
        ) from exc
    return ProbeSpec(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        spacing=spacing,
        up=up,
    )


def build_probe_layout(scene: Scene) -> ProbeLayout:
    spec = scene.probe_spec
    extent = spec.bounds_max - spec.bounds_min
    counts = tuple(
        int(math.floor(float(extent[axis]) / spec.spacing + 1.0e-9)) + 1
        for axis in range(3)
    )
    axes: List[np.ndarray] = []
    for axis in range(3):
        count = counts[axis]
        span = float(count - 1) * spec.spacing
        start = float(spec.bounds_min[axis]) + 0.5 * (
            float(extent[axis]) - span
        )
        axes.append(start + np.arange(count, dtype=np.float64) * spec.spacing)
    xx, yy, zz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    positions = np.column_stack(
        (xx.reshape(-1), yy.reshape(-1), zz.reshape(-1))
    ).astype(np.float64)
    return ProbeLayout(
        positions=np.ascontiguousarray(positions),
        bounds_min=spec.bounds_min.copy(),
        bounds_max=spec.bounds_max.copy(),
        spacing=spec.spacing,
        up=spec.up.copy(),
        counts=(counts[0], counts[1], counts[2]),
    )


def _update_bytes(digest: "hashlib._Hash", payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "little", signed=False))
    digest.update(payload)


def compute_geometry_hash(scene: Scene) -> str:
    digest = hashlib.sha256()
    digest.update(b"VXGI-GEOMETRY-V1")
    for mesh in scene.meshes:
        _update_bytes(digest, mesh.name.encode("utf-8"))
        _update_bytes(digest, mesh.material_id.encode("utf-8"))
        _update_bytes(digest, b"1" if mesh.occluder else b"0")
        _update_bytes(
            digest,
            np.asarray(mesh.triangles, dtype="<f8").tobytes(order="C"),
        )
    for material_id in sorted(scene.materials):
        material = scene.materials[material_id]
        _update_bytes(digest, material_id.encode("utf-8"))
        _update_bytes(
            digest,
            np.asarray(material.albedo, dtype="<f8").tobytes(order="C"),
        )
    return digest.hexdigest()


def compute_lighting_hash(scene: Scene) -> str:
    digest = hashlib.sha256()
    digest.update(b"VXGI-LIGHTING-V1")
    for light in scene.lights:
        _update_bytes(digest, light.light_id.encode("utf-8"))
        _update_bytes(digest, light.kind.encode("utf-8"))
        _update_bytes(digest, np.asarray(light.position, dtype="<f8").tobytes())
        _update_bytes(digest, np.asarray(light.color, dtype="<f8").tobytes())
        _update_bytes(digest, np.asarray([light.radiant_power], dtype="<f8").tobytes())
        if light.normal is not None:
            _update_bytes(digest, np.asarray(light.normal, dtype="<f8").tobytes())
        _update_bytes(digest, np.asarray([light.radius], dtype="<f8").tobytes())
        _update_bytes(
            digest, np.asarray([light.sample_count], dtype="<i8").tobytes()
        )
        if light.sample_points is not None:
            _update_bytes(
                digest,
                np.asarray(light.sample_points, dtype="<f8").tobytes(order="C"),
            )
    return digest.hexdigest()


def compute_layout_hash(layout: ProbeLayout) -> str:
    digest = hashlib.sha256()
    digest.update(b"VXGI-PROBE-LAYOUT-V1")
    _update_bytes(
        digest, np.asarray(layout.positions, dtype="<f8").tobytes(order="C")
    )
    _update_bytes(
        digest, np.asarray(layout.bounds_min, dtype="<f8").tobytes(order="C")
    )
    _update_bytes(
        digest, np.asarray(layout.bounds_max, dtype="<f8").tobytes(order="C")
    )
    _update_bytes(
        digest,
        np.asarray(
            [layout.spacing, layout.counts[0], layout.counts[1], layout.counts[2]],
            dtype="<f8",
        ).tobytes(),
    )
    _update_bytes(digest, np.asarray(layout.up, dtype="<f8").tobytes())
    return digest.hexdigest()


def compute_scene_hash(
    scene: Scene,
    layout: ProbeLayout,
    geometry_hash: Optional[str] = None,
    layout_hash: Optional[str] = None,
    lighting_hash: Optional[str] = None,
) -> str:
    geom = geometry_hash or compute_geometry_hash(scene)
    lay = layout_hash or compute_layout_hash(layout)
    lights = lighting_hash or compute_lighting_hash(scene)
    digest = hashlib.sha256()
    digest.update(b"VXGI-SCENE-V1")
    for value in (geom, lay, lights):
        _update_bytes(digest, value.encode("ascii"))
    bake = scene.bake
    numeric = np.asarray(
        [
            float(bake.voxel_resolution),
            float(bake.padding_voxels),
            float(bake.ray_count),
            float(bake.seed),
        ],
        dtype="<f8",
    )
    _update_bytes(digest, numeric.tobytes())
    _update_bytes(
        digest,
        np.asarray(bake.ambient_radiance, dtype="<f8").tobytes(order="C"),
    )
    _update_bytes(
        digest,
        np.asarray(
            [scene.total_light_power, scene.albedo_upper_bound], dtype="<f8"
        ).tobytes(),
    )
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
