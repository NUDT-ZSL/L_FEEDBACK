from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .errors import SceneConfigError
from .linear import (
    finite_float,
    matrix_from_json,
    normalize,
    positive_int,
    vector3,
)


_LIGHT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass
class Camera:
    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_y_deg: float
    near: float
    far: float


@dataclass
class Light:
    light_id: str
    light_type: str
    color: np.ndarray
    intensity: float
    position: Optional[np.ndarray] = None
    direction: Optional[np.ndarray] = None
    range: Optional[float] = None
    inner_angle_deg: Optional[float] = None
    outer_angle_deg: Optional[float] = None

    def report_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.light_id,
            "type": self.light_type,
            "color": self.color.tolist(),
            "intensity": self.intensity,
        }
        if self.position is not None:
            result["position"] = self.position.tolist()
        if self.direction is not None:
            result["direction"] = self.direction.tolist()
        if self.range is not None:
            result["range"] = self.range
        if self.inner_angle_deg is not None:
            result["inner_angle_deg"] = self.inner_angle_deg
        if self.outer_angle_deg is not None:
            result["outer_angle_deg"] = self.outer_angle_deg
        return result


@dataclass
class Material:
    material_id: str
    base_color: np.ndarray
    specular_color: np.ndarray
    metallic: float
    roughness: float
    exponent_tangent: float
    exponent_bitangent: float
    anisotropy_rotation_deg: float
    anisotropy_axis: str
    anisotropy: float

    def report_dict(self) -> Dict[str, Any]:
        return {
            "id": self.material_id,
            "base_color": self.base_color.tolist(),
            "specular_color": self.specular_color.tolist(),
            "metallic": self.metallic,
            "roughness": self.roughness,
            "exponent_tangent": self.exponent_tangent,
            "exponent_bitangent": self.exponent_bitangent,
            "anisotropy_axis": self.anisotropy_axis,
            "anisotropy": self.anisotropy,
            "anisotropy_rotation_deg": self.anisotropy_rotation_deg,
        }


@dataclass
class MeshNode:
    name: str
    path: Path
    transform: np.ndarray
    positions: np.ndarray
    normals: np.ndarray
    tangents: np.ndarray
    bitangents: np.ndarray
    uvs: np.ndarray
    indices: np.ndarray
    material_indices: np.ndarray
    cast_shadow: bool


@dataclass
class ShadowSettings:
    resolution: int
    cascade_count: int
    split_lambda: float
    shadow_distance: float
    pcf_radius: int
    depth_bias: float
    slope_scale: float
    compare_tolerance: float


@dataclass
class Scene:
    source_path: Path
    source_sha256: str
    width: int
    height: int
    camera: Camera
    lights: List[Light]
    materials: List[Material]
    meshes: List[MeshNode]
    shadows: ShadowSettings
    raw_config: Dict[str, Any]


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        raise SceneConfigError("%s file does not exist: %s" % (label, path))
    except OSError as exc:
        raise SceneConfigError("cannot read %s file %s: %s" % (label, path, exc))
    except json.JSONDecodeError as exc:
        raise SceneConfigError(
            "invalid JSON in %s file %s at line %d column %d: %s"
            % (label, path, exc.lineno, exc.colno, exc.msg)
        )
    if not isinstance(value, dict):
        raise SceneConfigError("%s file root must be a JSON object: %s" % (label, path))
    return value


def _expect_object(value: Any, context: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise SceneConfigError("%s must be an object" % context)
    return value


def _expect_list(value: Any, context: str) -> List[Any]:
    if not isinstance(value, list):
        raise SceneConfigError("%s must be an array" % context)
    return value


def _resolve_path(base: Path, value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SceneConfigError("%s must be a non-empty path string" % context)
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _parse_camera(raw: Any) -> Camera:
    data = _expect_object(raw, "camera")
    try:
        position = vector3(data["position"], "camera.position")
        target = vector3(data["target"], "camera.target")
        up = normalize(vector3(data.get("up", [0.0, 1.0, 0.0]), "camera.up"))
        fov = finite_float(data.get("fov_y_deg", 45.0), "camera.fov_y_deg")
        near = finite_float(data.get("near", 0.1), "camera.near")
        far = finite_float(data.get("far", 100.0), "camera.far")
    except KeyError as exc:
        raise SceneConfigError("camera is missing required field '%s'" % exc.args[0])
    except ValueError as exc:
        raise SceneConfigError(str(exc))
    if fov <= 0.0 or fov >= 179.0:
        raise SceneConfigError("camera.fov_y_deg must be between 0 and 179")
    if near <= 0.0 or far <= near:
        raise SceneConfigError("camera requires 0 < near < far")
    if np.linalg.norm(target - position) < 1.0e-8:
        raise SceneConfigError("camera.position and camera.target must differ")
    if np.linalg.norm(up) < 1.0e-8:
        raise SceneConfigError("camera.up must be non-zero")
    return Camera(position, target, up, fov, near, far)


def _parse_material(raw: Any, index: int) -> Material:
    data = _expect_object(raw, "materials[%d]" % index)
    material_id = data.get("id", "material_%d" % index)
    if not isinstance(material_id, str) or not material_id:
        raise SceneConfigError("materials[%d].id must be a non-empty string" % index)
    try:
        base_color = vector3(data.get("base_color", [0.8, 0.8, 0.8]), "materials[%d].base_color" % index)
        specular_color = vector3(
            data.get("specular_color", [1.0, 1.0, 1.0]),
            "materials[%d].specular_color" % index,
        )
        metallic = finite_float(data.get("metallic", 0.0), "materials[%d].metallic" % index)
        roughness = finite_float(
            data.get("roughness", 0.45), "materials[%d].roughness" % index
        )
        anisotropy = finite_float(
            data.get("anisotropy", 0.0), "materials[%d].anisotropy" % index
        )
        rotation = finite_float(
            data.get("anisotropy_rotation_deg", 0.0),
            "materials[%d].anisotropy_rotation_deg" % index,
        )
        axis = data.get("anisotropy_axis", "tangent")
        if axis not in ("tangent", "bitangent"):
            raise ValueError(
                "materials[%d].anisotropy_axis must be 'tangent' or 'bitangent'" % index
            )
        base_exponent = finite_float(
            data.get("base_specular_exponent", 2.0 / max(roughness ** 4, 1.0e-4) - 2.0),
            "materials[%d].base_specular_exponent" % index,
        )
        auto_tangent = base_exponent * (1.0 - 0.8 * abs(anisotropy))
        auto_bitangent = base_exponent * (1.0 + 0.8 * abs(anisotropy))
        if axis == "bitangent":
            auto_tangent, auto_bitangent = auto_bitangent, auto_tangent
        if "tangent_specular_exponent" in data:
            exponent_t = finite_float(
                data["tangent_specular_exponent"],
                "materials[%d].tangent_specular_exponent" % index,
            )
        else:
            exponent_t = auto_tangent
        if "bitangent_specular_exponent" in data:
            exponent_b = finite_float(
                data["bitangent_specular_exponent"],
                "materials[%d].bitangent_specular_exponent" % index,
            )
        else:
            exponent_b = auto_bitangent
    except ValueError as exc:
        raise SceneConfigError(str(exc))
    if np.any(base_color < 0.0) or np.any(specular_color < 0.0):
        raise SceneConfigError("materials[%d] colors must be non-negative" % index)
    if not 0.0 <= metallic <= 1.0:
        raise SceneConfigError("materials[%d].metallic must be in [0, 1]" % index)
    if not 0.0 < roughness <= 1.0:
        raise SceneConfigError("materials[%d].roughness must be in (0, 1]" % index)
    if not -0.999 <= anisotropy <= 0.999:
        raise SceneConfigError("materials[%d].anisotropy must be in [-0.999, 0.999]" % index)
    if exponent_t < 1.0 or exponent_b < 1.0:
        raise SceneConfigError("materials[%d] specular exponents must be >= 1" % index)
    return Material(
        material_id=material_id,
        base_color=base_color,
        specular_color=specular_color,
        metallic=metallic,
        roughness=roughness,
        exponent_tangent=exponent_t,
        exponent_bitangent=exponent_b,
        anisotropy_rotation_deg=rotation,
        anisotropy_axis=axis,
        anisotropy=anisotropy,
    )


def _load_materials(path: Path) -> List[Material]:
    data = _read_json(path, "materials")
    raw_materials = data.get("materials")
    if not isinstance(raw_materials, list) or not raw_materials:
        raise SceneConfigError("materials file must contain a non-empty 'materials' array")
    materials = [_parse_material(item, index) for index, item in enumerate(raw_materials)]
    identifiers = [item.material_id for item in materials]
    if len(set(identifiers)) != len(identifiers):
        raise SceneConfigError("material ids must be unique")
    return materials


def _parse_light(raw: Any, index: int) -> Light:
    data = _expect_object(raw, "lights[%d]" % index)
    light_id = data.get("id")
    if not isinstance(light_id, str) or not _LIGHT_ID_PATTERN.match(light_id):
        raise SceneConfigError(
            "lights[%d].id must match [A-Za-z0-9_.-]+" % index
        )
    light_type = data.get("type")
    if light_type not in ("point", "directional", "spot"):
        raise SceneConfigError(
            "lights[%d].type must be point, directional, or spot" % index
        )
    try:
        color = vector3(data.get("color", [1.0, 1.0, 1.0]), "lights[%d].color" % index)
        intensity = finite_float(
            data.get("intensity", 1.0), "lights[%d].intensity" % index
        )
    except ValueError as exc:
        raise SceneConfigError(str(exc))
    if np.any(color < 0.0) or intensity < 0.0:
        raise SceneConfigError("lights[%d] color and intensity must be non-negative" % index)

    position = None
    direction = None
    light_range = None
    inner_angle = None
    outer_angle = None
    try:
        if light_type in ("point", "spot"):
            if "position" not in data:
                raise SceneConfigError("lights[%d].position is required for %s" % (index, light_type))
            position = vector3(data["position"], "lights[%d].position" % index)
            light_range = finite_float(data["range"], "lights[%d].range" % index)
            if light_range <= 0.0:
                raise SceneConfigError("lights[%d].range must be > 0" % index)
        if light_type in ("directional", "spot"):
            if "direction" not in data:
                raise SceneConfigError("lights[%d].direction is required for %s" % (index, light_type))
            direction = normalize(vector3(data["direction"], "lights[%d].direction" % index))
            if np.linalg.norm(vector3(data["direction"], "lights[%d].direction" % index)) < 1.0e-8:
                raise SceneConfigError("lights[%d].direction must be non-zero" % index)
        if light_type == "spot":
            inner_angle = finite_float(
                data.get("inner_angle_deg", 15.0), "lights[%d].inner_angle_deg" % index
            )
            outer_angle = finite_float(
                data.get("outer_angle_deg", 25.0), "lights[%d].outer_angle_deg" % index
            )
            if not 0.0 < inner_angle < outer_angle < 89.5:
                raise SceneConfigError(
                    "lights[%d] requires 0 < inner_angle_deg < outer_angle_deg < 89.5" % index
                )
    except KeyError as exc:
        raise SceneConfigError("lights[%d] is missing '%s'" % (index, exc.args[0]))
    return Light(
        light_id=light_id,
        light_type=light_type,
        color=color,
        intensity=intensity,
        position=position,
        direction=direction,
        range=light_range,
        inner_angle_deg=inner_angle,
        outer_angle_deg=outer_angle,
    )


def _load_mesh_arrays(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise SceneConfigError("mesh file does not exist: %s" % path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = ["positions", "normals", "tangents", "uvs", "indices", "material_indices"]
            missing = [name for name in required if name not in archive.files]
            if missing:
                raise SceneConfigError(
                    "mesh file %s is missing arrays: %s" % (path, ", ".join(missing))
                )
            result = {name: np.asarray(archive[name]) for name in archive.files}
    except SceneConfigError:
        raise
    except (OSError, ValueError) as exc:
        raise SceneConfigError("cannot load mesh file %s: %s" % (path, exc))

    positions = np.asarray(result["positions"], dtype=np.float64)
    normals = np.asarray(result["normals"], dtype=np.float64)
    tangents = np.asarray(result["tangents"], dtype=np.float64)
    uvs = np.asarray(result["uvs"], dtype=np.float64)
    indices = np.asarray(result["indices"])
    material_indices = np.asarray(result["material_indices"])
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise SceneConfigError("mesh %s positions must have shape (N, 3)" % path)
    if normals.shape != positions.shape:
        raise SceneConfigError("mesh %s normals must have the same shape as positions" % path)
    if tangents.shape != positions.shape:
        raise SceneConfigError("mesh %s tangents must have the same shape as positions" % path)
    if uvs.shape != (positions.shape[0], 2):
        raise SceneConfigError("mesh %s uvs must have shape (N, 2)" % path)
    if indices.ndim != 2 or indices.shape[1] != 3 or indices.shape[0] == 0:
        raise SceneConfigError("mesh %s indices must have shape (T, 3)" % path)
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(normals)):
        raise SceneConfigError("mesh %s contains non-finite positions or normals" % path)
    if not np.all(np.isfinite(tangents)) or not np.all(np.isfinite(uvs)):
        raise SceneConfigError("mesh %s contains non-finite tangents or uvs" % path)
    if np.any(indices < 0) or np.any(indices >= positions.shape[0]):
        raise SceneConfigError("mesh %s contains out-of-range vertex indices" % path)
    if material_indices.shape != (indices.shape[0],):
        raise SceneConfigError(
            "mesh %s material_indices must have shape (T,)" % path
        )
    if np.linalg.norm(normals, axis=1).min() < 1.0e-8:
        raise SceneConfigError("mesh %s contains zero-length normals" % path)
    if np.linalg.norm(tangents, axis=1).min() < 1.0e-8:
        raise SceneConfigError("mesh %s contains zero-length tangents" % path)
    if "bitangents" in result:
        bitangents = np.asarray(result["bitangents"], dtype=np.float64)
        if bitangents.shape != positions.shape:
            raise SceneConfigError("mesh %s bitangents must match positions shape" % path)
    else:
        bitangents = np.cross(normals, tangents)
    if np.linalg.norm(bitangents, axis=1).min() < 1.0e-8:
        raise SceneConfigError("mesh %s contains zero-length bitangents" % path)
    return {
        "positions": positions,
        "normals": normalize(normals),
        "tangents": normalize(tangents),
        "bitangents": normalize(bitangents),
        "uvs": uvs,
        "indices": indices.astype(np.int64),
        "material_indices": material_indices.astype(np.int64),
    }


def _parse_mesh(raw: Any, index: int, scene_dir: Path, materials: List[Material]) -> MeshNode:
    data = _expect_object(raw, "meshes[%d]" % index)
    name = data.get("name", "mesh_%d" % index)
    if not isinstance(name, str) or not name:
        raise SceneConfigError("meshes[%d].name must be a non-empty string" % index)
    try:
        path = _resolve_path(scene_dir, data.get("path"), "meshes[%d].path" % index)
    except SceneConfigError:
        raise
    if path.suffix.lower() != ".npz":
        raise SceneConfigError("meshes[%d].path must reference a .npz mesh" % index)
    arrays = _load_mesh_arrays(path)
    try:
        transform = matrix_from_json(data.get("transform"), "meshes[%d].transform" % index)
    except ValueError as exc:
        raise SceneConfigError(str(exc))
    material_indices = arrays["material_indices"].copy()
    if "material_index" in data:
        try:
            override = positive_int(data["material_index"], "meshes[%d].material_index" % index, 0)
        except ValueError as exc:
            raise SceneConfigError(str(exc))
        material_indices.fill(override)
    if material_indices.min(initial=0) < 0 or material_indices.max(initial=0) >= len(materials):
        raise SceneConfigError(
            "mesh '%s' references a material index outside [0, %d]"
            % (name, len(materials) - 1)
        )
    cast_shadow = data.get("cast_shadow", True)
    if not isinstance(cast_shadow, bool):
        raise SceneConfigError("meshes[%d].cast_shadow must be boolean" % index)
    return MeshNode(
        name=name,
        path=path,
        transform=transform,
        positions=arrays["positions"],
        normals=arrays["normals"],
        tangents=arrays["tangents"],
        bitangents=arrays["bitangents"],
        uvs=arrays["uvs"],
        indices=arrays["indices"],
        material_indices=material_indices,
        cast_shadow=cast_shadow,
    )


def load_scene(path: str, overrides: Optional[Dict[str, Any]] = None) -> Scene:
    scene_path = Path(path).expanduser().resolve()
    config = _read_json(scene_path, "scene")
    overrides = dict(overrides or {})
    try:
        width = positive_int(config.get("width", 640), "width")
        height = positive_int(config.get("height", 480), "height")
        if "width" in overrides:
            width = positive_int(overrides["width"], "--width")
        if "height" in overrides:
            height = positive_int(overrides["height"], "--height")

        camera = _parse_camera(config.get("camera"))
        materials_path = _resolve_path(scene_path.parent, config.get("materials"), "materials")
        materials = _load_materials(materials_path)
        raw_lights = config.get("lights")
        if not isinstance(raw_lights, list) or not raw_lights:
            raise SceneConfigError("scene.lights must be a non-empty array")
        lights = [_parse_light(item, index) for index, item in enumerate(raw_lights)]
        light_ids = [light.light_id for light in lights]
        if len(set(light_ids)) != len(light_ids):
            raise SceneConfigError("light ids must be unique")
        raw_meshes = config.get("meshes")
        if not isinstance(raw_meshes, list) or not raw_meshes:
            raise SceneConfigError("scene.meshes must be a non-empty array")
        meshes = [
            _parse_mesh(item, index, scene_path.parent, materials)
            for index, item in enumerate(raw_meshes)
        ]
        mesh_names = [mesh.name for mesh in meshes]
        if len(set(mesh_names)) != len(mesh_names):
            raise SceneConfigError("mesh names must be unique")

        shadow_raw = _expect_object(config.get("shadows", {}), "shadows")
        depth_raw = _expect_object(config.get("depth", {}), "depth")
        resolution = positive_int(
            overrides.get("shadow_resolution", shadow_raw.get("resolution", 512)),
            "shadows.resolution",
            8,
        )
        cascade_count = positive_int(
            overrides.get("cascade_count", shadow_raw.get("cascade_count", 2)),
            "shadows.cascade_count",
            2,
        )
        split_lambda = finite_float(
            shadow_raw.get("split_lambda", 0.5), "shadows.split_lambda"
        )
        if not 0.0 <= split_lambda <= 1.0:
            raise SceneConfigError("shadows.split_lambda must be in [0, 1]")
        shadow_distance = finite_float(
            shadow_raw.get("shadow_distance", camera.far),
            "shadows.shadow_distance",
        )
        if not camera.near < shadow_distance <= camera.far:
            raise SceneConfigError(
                "shadows.shadow_distance must satisfy camera.near < distance <= camera.far"
            )
        pcf_radius = positive_int(
            shadow_raw.get("pcf_radius", 1), "shadows.pcf_radius", 0
        )
        if pcf_radius > 4:
            raise SceneConfigError("shadows.pcf_radius must be <= 4")
        depth_bias = finite_float(
            overrides.get("depth_bias", depth_raw.get("bias", 0.0008)),
            "depth.bias",
        )
        slope_scale = finite_float(
            overrides.get("slope_scale", depth_raw.get("slope_scale", 1.0)),
            "depth.slope_scale",
        )
        compare_tolerance = finite_float(
            overrides.get(
                "depth_tolerance", depth_raw.get("compare_tolerance", 0.0015)
            ),
            "depth.compare_tolerance",
        )
        if depth_bias < 0.0:
            raise SceneConfigError("depth.bias must be non-negative")
        if slope_scale < 0.0:
            raise SceneConfigError("depth.slope_scale must be non-negative")
        if compare_tolerance < 0.0:
            raise SceneConfigError("depth.compare_tolerance must be non-negative")
    except KeyError as exc:
        raise SceneConfigError("scene is missing required field '%s'" % exc.args[0])
    except ValueError as exc:
        raise SceneConfigError(str(exc))

    try:
        digest = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise SceneConfigError("cannot hash scene file %s: %s" % (scene_path, exc))

    return Scene(
        source_path=scene_path,
        source_sha256=digest,
        width=width,
        height=height,
        camera=camera,
        lights=lights,
        materials=materials,
        meshes=meshes,
        shadows=ShadowSettings(
            resolution=resolution,
            cascade_count=cascade_count,
            split_lambda=split_lambda,
            shadow_distance=shadow_distance,
            pcf_radius=pcf_radius,
            depth_bias=depth_bias,
            slope_scale=slope_scale,
            compare_tolerance=compare_tolerance,
        ),
        raw_config=config,
    )
