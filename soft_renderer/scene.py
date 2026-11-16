from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .errors import SceneError
from .mathutil import normalize, normalize_single, srgb_to_linear


def _fail(path: str, message: str) -> "SceneError":
    return SceneError("%s: %s" % (path, message))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_float(value: Any, path: str) -> float:
    if not _is_number(value):
        raise _fail(path, "expected a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise _fail(path, "expected a finite number")
    return result


def _int(value: Any, path: str) -> int:
    if isinstance(value, bool):
        raise _fail(path, "expected an integer")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise _fail(path, "expected an integer")
    if isinstance(value, float) and not value.is_integer():
        raise _fail(path, "expected an integer")
    return result


def _vector(value: Any, length: int, path: str) -> np.ndarray:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != length:
        raise _fail(path, "expected a list of %d numbers" % length)
    result = np.asarray([_finite_float(item, "%s[%d]" % (path, i)) for i, item in enumerate(value)])
    return result.astype(np.float64)


def _color(value: Any, path: str, color_space: str) -> np.ndarray:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("#"):
            text = text[1:]
        if len(text) == 3:
            text = "".join(character * 2 for character in text)
        if len(text) != 6:
            raise _fail(path, "expected #RGB, #RRGGBB, or three numeric channels")
        try:
            channels = np.asarray(
                [int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)],
                dtype=np.float64,
            )
        except ValueError:
            raise _fail(path, "invalid hexadecimal color")
        channels /= 255.0
    else:
        channels = _vector(value, 3, path)
        if np.any(channels > 1.0) and np.all(channels >= 0.0) and np.all(channels <= 255.0):
            channels = channels / 255.0
    if not np.all(np.isfinite(channels)) or np.any(channels < 0.0):
        raise _fail(path, "color channels must be finite and non-negative")
    if color_space == "linear":
        linear = channels.astype(np.float64)
    elif color_space == "srgb":
        linear = srgb_to_linear(channels)
    else:
        raise _fail("color_space", "must be 'srgb' or 'linear'")
    return linear.astype(np.float64)


@dataclass
class Material:
    id: int
    name: str
    base_color: np.ndarray
    roughness: float
    metallic: float


@dataclass
class Camera:
    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_y_deg: float
    near: float
    far: float


@dataclass
class Attenuation:
    constant: float = 1.0
    linear: float = 0.0
    quadratic: float = 1.0
    range: Optional[float] = None


@dataclass
class Light:
    id: int
    name: str
    kind: str
    position: np.ndarray
    color: np.ndarray
    intensity: float
    attenuation: Attenuation
    direction: Optional[np.ndarray] = None
    inner_cos: float = 1.0
    outer_cos: float = 1.0
    shadow_bias: float = 0.0015
    cast_shadow: bool = True
    shadow_resolution: Optional[int] = None


@dataclass
class Scene:
    width: int
    height: int
    camera: Camera
    materials: Dict[int, Material]
    material_order: List[int]
    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    triangles: np.ndarray
    triangle_materials: np.ndarray
    lights: List[Light]
    background: np.ndarray
    source_sha256: str
    source_path: str
    ambient: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


def _mesh_vertex_data(mesh: Dict[str, Any], path: str) -> Tuple[np.ndarray, List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
    vertices_value = mesh.get("vertices")
    raw_normals = mesh.get("normals")
    raw_uvs = mesh.get("uvs", mesh.get("uv", mesh.get("texcoords")))

    positions: List[np.ndarray] = []
    normals: List[Optional[np.ndarray]] = []
    uvs: List[Optional[np.ndarray]] = []

    if isinstance(vertices_value, dict):
        positions_value = (
            vertices_value.get("positions")
            or vertices_value.get("position")
            or vertices_value.get("p")
        )
        raw_normals = (
            vertices_value.get("normals")
            if vertices_value.get("normals") is not None
            else vertices_value.get("normal", raw_normals)
        )
        raw_uvs = (
            vertices_value.get("uvs")
            if vertices_value.get("uvs") is not None
            else vertices_value.get("uv", raw_uvs)
        )
        if positions_value is None:
            raise _fail(path + ".vertices", "expected positions or a list of vertices")
        vertices_value = positions_value

    if vertices_value is None:
        positions_value = mesh.get("positions")
        if positions_value is None:
            raise _fail(path, "missing vertices/positions")
        vertices_value = positions_value

    if not isinstance(vertices_value, (list, tuple)):
        raise _fail(path + ".vertices", "expected a list")

    for index, vertex in enumerate(vertices_value):
        vertex_path = "%s.vertices[%d]" % (path, index)
        if isinstance(vertex, dict):
            position_value = (
                vertex.get("position")
                if vertex.get("position") is not None
                else vertex.get("p", vertex.get("vertex"))
            )
            if position_value is None:
                raise _fail(vertex_path, "missing position")
            positions.append(_vector(position_value, 3, vertex_path + ".position"))
            normal_value = vertex.get("normal", vertex.get("n"))
            normal = None if normal_value is None else _vector(normal_value, 3, vertex_path + ".normal")
            uv_value = vertex.get("uv", vertex.get("texcoord", vertex.get("tc")))
            uv = None if uv_value is None else _vector(uv_value, 2, vertex_path + ".uv")
            normals.append(normal)
            uvs.append(uv)
        else:
            positions.append(_vector(vertex, 3, vertex_path))
            normal = None
            uv = None
            if raw_normals is not None:
                if not isinstance(raw_normals, (list, tuple)) or index >= len(raw_normals):
                    raise _fail(path + ".normals", "must contain one normal per vertex")
                normal_value = raw_normals[index]
                if normal_value is not None:
                    normal = _vector(normal_value, 3, "%s.normals[%d]" % (path, index))
            if raw_uvs is not None:
                if not isinstance(raw_uvs, (list, tuple)) or index >= len(raw_uvs):
                    raise _fail(path + ".uvs", "must contain one UV per vertex")
                uv_value = raw_uvs[index]
                if uv_value is not None:
                    uv = _vector(uv_value, 2, "%s.uvs[%d]" % (path, index))
            normals.append(normal)
            uvs.append(uv)

    position_array = np.asarray(positions, dtype=np.float64)
    if position_array.shape != (len(positions), 3) or len(positions) == 0:
        raise _fail(path + ".vertices", "at least one complete vertex is required")
    return position_array, normals, uvs


def _mesh_triangles(
    mesh: Dict[str, Any],
    path: str,
    vertex_count: int,
    default_material: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    triangle_data = mesh.get("triangles")
    material_values = mesh.get("material_ids")
    if material_values is None:
        material_values = mesh.get("triangle_materials")
    if material_values is None:
        material_values = mesh.get("materials")

    if triangle_data is None and mesh.get("indices") is not None:
        indices = np.asarray(mesh["indices"], dtype=object).reshape(-1).tolist()
        if len(indices) % 3 != 0:
            raise _fail(path + ".indices", "length must be a multiple of three")
        triangle_data = [indices[i : i + 3] for i in range(0, len(indices), 3)]
        mesh["triangles"] = triangle_data

    if triangle_data is None:
        raise _fail(path, "missing triangles/indices")
    if not isinstance(triangle_data, (list, tuple)):
        raise _fail(path + ".triangles", "expected a list")

    if isinstance(material_values, list) and material_values and isinstance(material_values[0], dict):
        material_values = None

    triangle_indices: List[Tuple[int, int, int]] = []
    triangle_materials: List[int] = []
    for index, triangle in enumerate(triangle_data):
        triangle_path = "%s.triangles[%d]" % (path, index)
        explicit_material: Any = None
        if isinstance(triangle, dict):
            indices_value = triangle.get("indices")
            if indices_value is None:
                indices_value = triangle.get("vertices", triangle.get("v"))
            if indices_value is None:
                raise _fail(triangle_path, "missing indices")
            explicit_material = triangle.get("material")
            if explicit_material is None:
                explicit_material = triangle.get("material_id", triangle.get("mat"))
            triangle = indices_value
        if not isinstance(triangle, (list, tuple, np.ndarray)) or len(triangle) < 3:
            raise _fail(triangle_path, "expected three vertex indices")
        indices = [
            _int(triangle[0], triangle_path + "[0]"),
            _int(triangle[1], triangle_path + "[1]"),
            _int(triangle[2], triangle_path + "[2]"),
        ]
        for local_index, vertex_index in enumerate(indices):
            if vertex_index < 0 or vertex_index >= vertex_count:
                raise _fail(
                    "%s[%d]" % (triangle_path, local_index),
                    "vertex index %d is out of range [0, %d)" % (vertex_index, vertex_count),
                )
        material: Any = explicit_material
        if material is None and len(triangle) >= 4 and _is_number(triangle[3]):
            material = triangle[3]
        if material is None and material_values is not None:
            if not isinstance(material_values, (list, tuple)) or index >= len(material_values):
                raise _fail(path + ".material_ids", "must contain one material ID per triangle")
            material = material_values[index]
        if material is None:
            material = default_material
        if material is None:
            raise _fail(triangle_path, "missing material ID")
        triangle_indices.append((indices[0], indices[1], indices[2]))
        triangle_materials.append(_int(material, triangle_path + ".material"))

    if not triangle_indices:
        raise _fail(path + ".triangles", "at least one triangle is required")
    return (
        np.asarray(triangle_indices, dtype=np.int64),
        np.asarray(triangle_materials, dtype=np.int64),
    )


def _compute_normals(
    positions: np.ndarray,
    triangles: np.ndarray,
    supplied: Sequence[Optional[np.ndarray]],
) -> np.ndarray:
    accumulated = np.zeros_like(positions)
    for triangle in triangles:
        i0, i1, i2 = int(triangle[0]), int(triangle[1]), int(triangle[2])
        face = np.cross(positions[i1] - positions[i0], positions[i2] - positions[i0])
        accumulated[i0] += face
        accumulated[i1] += face
        accumulated[i2] += face
    computed = normalize(accumulated)
    result = np.zeros_like(positions)
    for index in range(len(positions)):
        if supplied[index] is not None:
            normal = np.asarray(supplied[index], dtype=np.float64)
            length = float(np.sqrt(np.dot(normal, normal)))
            if not np.isfinite(length) or length <= 1.0e-12:
                raise _fail("vertices[%d].normal" % index, "normal must have non-zero length")
            result[index] = normal / length
        elif np.linalg.norm(accumulated[index]) > 1.0e-12:
            result[index] = computed[index]
        else:
            result[index] = np.array([0.0, 1.0, 0.0])
    return result


def _parse_camera(document: Dict[str, Any]) -> Camera:
    camera_value = document.get("camera")
    if not isinstance(camera_value, dict):
        raise _fail("camera", "expected an object")
    if camera_value.get("position") is None:
        raise _fail("camera.position", "missing camera position")
    position = _vector(camera_value["position"], 3, "camera.position")
    target_value = camera_value.get("target", camera_value.get("look_at"))
    if target_value is None:
        raise _fail("camera.target", "missing camera target/look_at")
    target = _vector(target_value, 3, "camera.target")
    up = _vector(camera_value.get("up", [0.0, 1.0, 0.0]), 3, "camera.up")
    fov = _finite_float(
        camera_value.get("fov_y_deg", camera_value.get("fov_deg", camera_value.get("fov", 45.0))),
        "camera.fov_y_deg",
    )
    near = _finite_float(camera_value.get("near", 0.05), "camera.near")
    far = _finite_float(camera_value.get("far", 1000.0), "camera.far")
    if not (0.0 < fov < 180.0):
        raise _fail("camera.fov_y_deg", "must be between 0 and 180")
    if not (near > 0.0 and far > near):
        raise _fail("camera", "near must be positive and less than far")
    if np.linalg.norm(target - position) <= 1.0e-12:
        raise _fail("camera", "position and target must differ")
    forward = normalize_single(target - position)
    if abs(float(np.dot(forward, normalize_single(up)))) > 0.999999:
        raise _fail("camera.up", "must not be parallel to the view direction")
    return Camera(position, target, up, fov, near, far)


def _parse_lights(document: Dict[str, Any], color_space: str) -> List[Light]:
    light_values = document.get("lights", [])
    if not isinstance(light_values, list):
        raise _fail("lights", "expected a list")
    lights: List[Light] = []
    for index, value in enumerate(light_values):
        path = "lights[%d]" % index
        if not isinstance(value, dict):
            raise _fail(path, "expected an object")
        kind = str(value.get("type", value.get("kind", ""))).lower()
        if kind not in ("point", "spot"):
            raise _fail(path + ".type", "must be 'point' or 'spot'")
        if value.get("position") is None:
            raise _fail(path + ".position", "missing light position")
        position = _vector(value["position"], 3, path + ".position")
        color = _color(value.get("color", [1.0, 1.0, 1.0]), path + ".color", color_space)
        intensity = _finite_float(value.get("intensity", 1.0), path + ".intensity")
        if intensity < 0.0:
            raise _fail(path + ".intensity", "must be non-negative")

        attenuation_value = value.get(
            "attenuation",
            value.get("distance_attenuation", None),
        )
        range_value = value.get("range", value.get("max_distance"))
        if isinstance(attenuation_value, dict):
            constant = _finite_float(attenuation_value.get("constant", 1.0), path + ".attenuation.constant")
            linear = _finite_float(attenuation_value.get("linear", 0.0), path + ".attenuation.linear")
            quadratic = _finite_float(attenuation_value.get("quadratic", 1.0), path + ".attenuation.quadratic")
            if attenuation_value.get("range") is not None:
                range_value = attenuation_value["range"]
        elif attenuation_value is None:
            constant, linear, quadratic = 1.0, 0.0, 1.0
        else:
            quadratic = _finite_float(attenuation_value, path + ".attenuation")
            constant, linear = 1.0, 0.0
        if min(constant, linear, quadratic) < 0.0 or constant + linear + quadratic <= 0.0:
            raise _fail(path + ".attenuation", "coefficients must be non-negative and not all zero")
        attenuation_range = None
        if range_value is not None:
            attenuation_range = _finite_float(range_value, path + ".range")
            if attenuation_range <= 0.0:
                raise _fail(path + ".range", "must be positive")

        direction = None
        inner_cos, outer_cos = 1.0, 1.0
        if kind == "spot":
            direction_value = value.get("direction")
            if direction_value is None and value.get("target") is not None:
                direction_value = _vector(value["target"], 3, path + ".target") - position
            if direction_value is None:
                raise _fail(path + ".direction", "spot lights require direction or target")
            direction = normalize_single(_vector(direction_value, 3, path + ".direction"))
            outer_deg = _finite_float(
                value.get(
                    "outer_angle_deg",
                    value.get(
                        "outer_cone_angle_deg",
                        value.get(
                            "outer_cone_deg",
                            value.get(
                                "cone_angle_deg",
                                value.get("cone_angle", value.get("outer_angle", 30.0)),
                            ),
                        ),
                    ),
                ),
                path + ".outer_angle_deg",
            )
            inner_deg = _finite_float(
                value.get(
                    "inner_angle_deg",
                    value.get(
                        "inner_cone_angle_deg",
                        value.get(
                            "inner_cone_deg",
                            value.get("inner_cone_angle", value.get("inner_angle", outer_deg * 0.7)),
                        ),
                    ),
                ),
                path + ".inner_angle_deg",
            )
            if not (0.0 < inner_deg < outer_deg < 180.0):
                raise _fail(path, "spot angles must satisfy 0 < inner < outer < 180 degrees")
            inner_cos = float(np.cos(np.deg2rad(inner_deg)))
            outer_cos = float(np.cos(np.deg2rad(outer_deg)))

        shadow_bias = _finite_float(value.get("shadow_bias", 0.0015), path + ".shadow_bias")
        if shadow_bias < 0.0:
            raise _fail(path + ".shadow_bias", "must be non-negative")
        shadow_resolution = value.get("shadow_resolution")
        if shadow_resolution is not None:
            shadow_resolution = _int(shadow_resolution, path + ".shadow_resolution")
            if shadow_resolution <= 0:
                raise _fail(path + ".shadow_resolution", "must be positive")
        lights.append(
            Light(
                id=index,
                name=str(value.get("name", "light_%d" % index)),
                kind=kind,
                position=position,
                color=color,
                intensity=intensity,
                attenuation=Attenuation(constant, linear, quadratic, attenuation_range),
                direction=direction,
                inner_cos=inner_cos,
                outer_cos=outer_cos,
                shadow_bias=shadow_bias,
                cast_shadow=bool(value.get("cast_shadow", True)),
                shadow_resolution=shadow_resolution,
            )
        )
    return lights


def _canonical_material_ids(materials: List[Material]) -> List[int]:
    identifiers = [material.id for material in materials]
    if len(set(identifiers)) != len(identifiers):
        duplicate = next(identifier for identifier in identifiers if identifiers.count(identifier) > 1)
        raise _fail("materials", "duplicate material ID %d" % duplicate)
    return sorted(identifiers)


def load_scene(path: str) -> Scene:
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as error:
        raise SceneError("cannot read scene %r: %s" % (path, error))
    try:
        document = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise SceneError("scene %r is not UTF-8: %s" % (path, error))
    except json.JSONDecodeError as error:
        raise SceneError(
            "invalid JSON in %r at line %d column %d: %s"
            % (path, error.lineno, error.colno, error.msg)
        )
    if not isinstance(document, dict):
        raise _fail("scene", "top-level JSON value must be an object")
    color_space = str(document.get("color_space", "srgb")).lower()
    if color_space not in ("srgb", "linear"):
        raise _fail("color_space", "must be 'srgb' or 'linear'")

    image_value = document.get("image", {})
    if image_value is None:
        image_value = {}
    if not isinstance(image_value, dict):
        raise _fail("image", "expected an object")
    width_value = image_value.get("width", document.get("width"))
    height_value = image_value.get("height", document.get("height"))
    if width_value is None or height_value is None:
        raise _fail("image", "width and height are required")
    width = _int(width_value, "image.width")
    height = _int(height_value, "image.height")
    if not (1 <= width <= 16384 and 1 <= height <= 16384):
        raise _fail("image", "width and height must be in [1, 16384]")

    camera = _parse_camera(document)
    background = _color(document.get("background", [0.0, 0.0, 0.0]), "background", color_space)
    ambient = _color(document.get("ambient", [0.0, 0.0, 0.0]), "ambient", color_space)

    material_values = document.get("materials")
    if isinstance(material_values, dict):
        keyed_materials = []
        for key, value in material_values.items():
            if not isinstance(value, dict):
                raise _fail("materials.%s" % key, "expected an object")
            item = dict(value)
            if "id" not in item:
                try:
                    item["id"] = int(key)
                except (TypeError, ValueError):
                    raise _fail("materials.%s.id" % key, "must be an integer")
            keyed_materials.append(item)
        material_values = keyed_materials
    if not isinstance(material_values, list) or not material_values:
        raise _fail("materials", "expected a non-empty list")
    materials: List[Material] = []
    for index, value in enumerate(material_values):
        path_prefix = "materials[%d]" % index
        if not isinstance(value, dict):
            raise _fail(path_prefix, "expected an object")
        identifier = _int(value.get("id", index), path_prefix + ".id")
        base_color_value = value.get(
            "base_color",
            value.get(
                "baseColor",
                value.get(
                    "baseColorFactor",
                    value.get(
                        "base_color_factor",
                        value.get("albedo", value.get("diffuse", value.get("color"))),
                    ),
                ),
            ),
        )
        if base_color_value is None:
            raise _fail(path_prefix + ".base_color", "missing base color")
        base_color = _color(base_color_value, path_prefix + ".base_color", color_space)
        roughness = _finite_float(
            value.get(
                "roughness",
                value.get("roughnessFactor", value.get("roughness_factor", 0.5)),
            ),
            path_prefix + ".roughness",
        )
        metallic = _finite_float(
            value.get(
                "metallic",
                value.get("metallicFactor", value.get("metallic_factor", 0.0)),
            ),
            path_prefix + ".metallic",
        )
        if not (0.0 <= roughness <= 1.0):
            raise _fail(path_prefix + ".roughness", "must be in [0, 1]")
        if not (0.0 <= metallic <= 1.0):
            raise _fail(path_prefix + ".metallic", "must be in [0, 1]")
        materials.append(
            Material(
                id=identifier,
                name=str(value.get("name", "material_%d" % identifier)),
                base_color=base_color,
                roughness=roughness,
                metallic=metallic,
            )
        )
    material_order = _canonical_material_ids(materials)
    material_map = {material.id: material for material in materials}

    mesh_values = document.get("meshes")
    if mesh_values is None:
        mesh_values = document.get("geometry")
    if mesh_values is None:
        mesh_values = document.get("objects")
    if not isinstance(mesh_values, list) or not mesh_values:
        raise _fail("meshes", "expected a non-empty list")

    all_positions: List[np.ndarray] = []
    all_normals: List[np.ndarray] = []
    all_uvs: List[np.ndarray] = []
    all_triangles: List[np.ndarray] = []
    all_materials: List[np.ndarray] = []
    vertex_offset = 0

    for mesh_index, mesh in enumerate(mesh_values):
        mesh_path = "meshes[%d]" % mesh_index
        if not isinstance(mesh, dict):
            raise _fail(mesh_path, "expected an object")
        default_material = mesh.get("material", mesh.get("material_id"))
        if default_material is not None:
            default_material = _int(default_material, mesh_path + ".material")
        positions, supplied_normals, supplied_uvs = _mesh_vertex_data(mesh, mesh_path)
        triangles, triangle_materials = _mesh_triangles(
            mesh, mesh_path, len(positions), default_material
        )
        for material_id in triangle_materials:
            if int(material_id) not in material_map:
                raise _fail(
                    mesh_path + ".triangles",
                    "unknown material ID %d" % int(material_id),
                )
        normals = _compute_normals(positions, triangles, supplied_normals)
        uvs = np.asarray(
            [uv if uv is not None else np.zeros(2, dtype=np.float64) for uv in supplied_uvs],
            dtype=np.float64,
        )
        all_positions.append(positions)
        all_normals.append(normals)
        all_uvs.append(uvs)
        all_triangles.append(triangles + vertex_offset)
        all_materials.append(triangle_materials)
        vertex_offset += len(positions)

    scene = Scene(
        width=width,
        height=height,
        camera=camera,
        materials=material_map,
        material_order=material_order,
        positions=np.concatenate(all_positions, axis=0),
        normals=np.concatenate(all_normals, axis=0),
        uvs=np.concatenate(all_uvs, axis=0),
        triangles=np.concatenate(all_triangles, axis=0),
        triangle_materials=np.concatenate(all_materials, axis=0),
        lights=_parse_lights(document, color_space),
        background=background,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source_path=os.path.abspath(path),
        ambient=ambient,
    )
    return scene
