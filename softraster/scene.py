"""JSON scene parsing, business validation, and derived mesh data."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .errors import index_error, layer_limit_error, schema_error
from .math3d import PerspectiveCamera, normalize


@dataclass(frozen=True)
class ShadowSettings:
    enabled: bool = True
    map_size: int = 256
    radius: float = 2.0
    samples: int = 12
    bias: float = 0.002
    normal_bias: float = 0.02


@dataclass(frozen=True)
class Light:
    name: str
    kind: str
    color: np.ndarray
    intensity: float
    direction: np.ndarray | None
    position: np.ndarray | None
    range: float | None
    shadow: ShadowSettings


@dataclass(frozen=True)
class Layer:
    base_color: np.ndarray
    metallic: float
    roughness: float
    clearcoat: float
    normal_strength: float
    normal: np.ndarray
    opacity: float


@dataclass(frozen=True)
class Material:
    name: str
    layers: tuple[Layer, ...]


@dataclass(frozen=True)
class Mesh:
    name: str
    vertices: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    material_index: int
    double_sided: bool
    center: np.ndarray
    radius: float


@dataclass(frozen=True)
class RenderSettings:
    width: int
    height: int
    exposure: float
    max_layers: int


@dataclass(frozen=True)
class Scene:
    camera: PerspectiveCamera
    lights: tuple[Light, ...]
    materials: tuple[Material, ...]
    meshes: tuple[Mesh, ...]
    ambient_color: np.ndarray
    ambient_intensity: float
    background: np.ndarray
    render: RenderSettings
    center: np.ndarray
    radius: float


def load_scene(path: str | Path) -> Scene:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise schema_error(
            "scene file does not exist", path=str(path)
        ) from exc
    except OSError as exc:
        raise schema_error(
            "scene file could not be read", path=str(path), error=str(exc)
        ) from exc
    except json.JSONDecodeError as exc:
        raise schema_error(
            "scene file is not valid JSON",
            path=str(path),
            line=exc.lineno,
            column=exc.colno,
        ) from exc
    return parse_scene(payload, source=str(path))


def parse_scene(payload: Any, source: str = "<memory>") -> Scene:
    if not isinstance(payload, dict):
        raise schema_error("scene root must be a JSON object", source=source)

    has_wrapper = "scene" in payload
    if has_wrapper:
        scene_obj = payload["scene"]
        if not isinstance(scene_obj, dict):
            raise schema_error("'scene' must be a JSON object", source=source)
        render_obj = payload.get("render", scene_obj.get("render", {}))
    else:
        scene_obj = payload
        render_obj = scene_obj.get("render", {})
    if not isinstance(render_obj, dict):
        raise schema_error("'render' must be a JSON object", source=source)

    render = _parse_render(render_obj, source)
    camera = _parse_camera(_require(scene_obj, "camera", source), render, source)

    ambient_obj = _require(scene_obj, "ambient", source)
    ambient_color, ambient_intensity = _parse_ambient(ambient_obj, source)

    background = _optional_vec3(
        scene_obj.get("background", [0.0, 0.0, 0.0]),
        "background",
        source,
        nonnegative=True,
    )

    lights_obj = _require(scene_obj, "lights", source)
    if not isinstance(lights_obj, list) or not lights_obj:
        raise schema_error("'lights' must be a non-empty array", source=source)
    lights = tuple(
        _parse_light(item, index, source)
        for index, item in enumerate(lights_obj)
    )
    point_count = sum(light.kind == "point" for light in lights)
    directional_count = sum(light.kind == "directional" for light in lights)
    if point_count < 3 or directional_count < 1:
        raise schema_error(
            "scene requires at least 3 point lights and 1 directional light",
            source=source,
            point_lights=point_count,
            directional_lights=directional_count,
        )

    materials_obj = _require(scene_obj, "materials", source)
    if not isinstance(materials_obj, list) or not materials_obj:
        raise schema_error(
            "'materials' must be a non-empty array", source=source
        )
    materials = tuple(
        _parse_material(item, index, render.max_layers, source)
        for index, item in enumerate(materials_obj)
    )

    meshes_obj = _require(scene_obj, "meshes", source)
    if not isinstance(meshes_obj, list) or not meshes_obj:
        raise schema_error("'meshes' must be a non-empty array", source=source)
    meshes = tuple(
        _parse_mesh(item, index, len(materials), source)
        for index, item in enumerate(meshes_obj)
    )

    all_vertices = np.concatenate([mesh.vertices for mesh in meshes], axis=0)
    minimum = np.min(all_vertices, axis=0)
    maximum = np.max(all_vertices, axis=0)
    center = (minimum + maximum) * 0.5
    radius = float(np.max(np.linalg.norm(all_vertices - center, axis=1)))
    radius = max(radius, 1.0e-4)

    return Scene(
        camera=camera,
        lights=lights,
        materials=materials,
        meshes=meshes,
        ambient_color=ambient_color,
        ambient_intensity=ambient_intensity,
        background=background,
        render=render,
        center=center,
        radius=radius,
    )


def _parse_render(render_obj: dict[str, Any], source: str) -> RenderSettings:
    width = _as_int(_require(render_obj, "width", source), "render.width", source)
    height = _as_int(_require(render_obj, "height", source), "render.height", source)
    if width < 1 or height < 1 or width > 8192 or height > 8192:
        raise schema_error(
            "render width and height must be in [1, 8192]",
            source=source,
            width=width,
            height=height,
        )
    exposure = _as_float(
        render_obj.get("exposure", 1.0), "render.exposure", source
    )
    if exposure < 0.0:
        raise schema_error(
            "'render.exposure' must be non-negative", source=source
        )
    max_layers = _as_int(
        render_obj.get("max_layers", 8), "render.max_layers", source
    )
    if not 1 <= max_layers <= 32:
        raise schema_error(
            "'render.max_layers' must be in [1, 32]", source=source
        )
    return RenderSettings(
        width=width,
        height=height,
        exposure=exposure,
        max_layers=max_layers,
    )


def _parse_camera(
    camera_obj: Any, render: RenderSettings, source: str
) -> PerspectiveCamera:
    if not isinstance(camera_obj, dict):
        raise schema_error("'camera' must be a JSON object", source=source)
    position = _vec3(
        _require(camera_obj, "position", source),
        "camera.position",
        source,
    )
    target = _vec3(
        _require(camera_obj, "target", source), "camera.target", source
    )
    up = _vec3(_require(camera_obj, "up", source), "camera.up", source)
    if np.linalg.norm(target - position) <= 1.0e-9:
        raise schema_error(
            "camera position and target must differ", source=source
        )
    if np.linalg.norm(up) <= 1.0e-9:
        raise schema_error("camera.up must be non-zero", source=source)
    fov_y = _as_float(
        _require(camera_obj, "fov_y_deg", source), "camera.fov_y_deg", source
    )
    near = _as_float(
        _require(camera_obj, "near", source), "camera.near", source
    )
    far = _as_float(_require(camera_obj, "far", source), "camera.far", source)
    if not 0.0 < fov_y < 180.0:
        raise schema_error(
            "'camera.fov_y_deg' must be between 0 and 180", source=source
        )
    if not 0.0 < near < far:
        raise schema_error(
            "camera near/far must satisfy 0 < near < far", source=source
        )
    try:
        return PerspectiveCamera(
            eye=position,
            target=target,
            up_hint=up,
            fov_y_radians=math.radians(fov_y),
            near=near,
            far=far,
            width=render.width,
            height=render.height,
        )
    except ValueError as exc:
        raise schema_error(str(exc), source=source, field="camera.up") from exc


def _parse_ambient(ambient_obj: Any, source: str) -> tuple[np.ndarray, float]:
    if isinstance(ambient_obj, list):
        color = _vec3(ambient_obj, "ambient", source, nonnegative=True)
        return color, 1.0
    if not isinstance(ambient_obj, dict):
        raise schema_error(
            "'ambient' must be an RGB array or object", source=source
        )
    color = _vec3(
        _require(ambient_obj, "color", source),
        "ambient.color",
        source,
        nonnegative=True,
    )
    intensity = _as_float(
        ambient_obj.get("intensity", 1.0), "ambient.intensity", source
    )
    if intensity < 0.0:
        raise schema_error(
            "'ambient.intensity' must be non-negative", source=source
        )
    return color, intensity


def _parse_light(item: Any, index: int, source: str) -> Light:
    where = f"lights[{index}]"
    if not isinstance(item, dict):
        raise schema_error(f"{where} must be a JSON object", source=source)
    kind = _require(item, "type", source)
    if not isinstance(kind, str):
        raise schema_error(f"{where}.type must be a string", source=source)
    kind = kind.lower()
    if kind not in {"point", "directional"}:
        raise schema_error(
            f"{where}.type must be 'point' or 'directional'",
            source=source,
        )
    name = item.get("name", f"light_{index}")
    if not isinstance(name, str):
        raise schema_error(f"{where}.name must be a string", source=source)
    color = _vec3(
        _require(item, "color", source),
        f"{where}.color",
        source,
        nonnegative=True,
    )
    intensity = _as_float(
        _require(item, "intensity", source), f"{where}.intensity", source
    )
    if intensity < 0.0:
        raise schema_error(
            f"{where}.intensity must be non-negative", source=source
        )
    shadow = _parse_shadow(item.get("shadow", {}), where, source)
    if kind == "point":
        position = _vec3(
            _require(item, "position", source),
            f"{where}.position",
            source,
        )
        light_range = _as_float(
            _require(item, "range", source), f"{where}.range", source
        )
        if light_range <= 0.0:
            raise schema_error(
                f"{where}.range must be positive", source=source
            )
        return Light(
            name=name,
            kind=kind,
            color=color,
            intensity=intensity,
            direction=None,
            position=position,
            range=light_range,
            shadow=shadow,
        )
    direction = _vec3(
        _require(item, "direction", source),
        f"{where}.direction",
        source,
    )
    if float(np.linalg.norm(direction)) <= 1.0e-9:
        raise schema_error(
            f"{where}.direction must be non-zero", source=source
        )
    return Light(
        name=name,
        kind=kind,
        color=color,
        intensity=intensity,
        direction=normalize(direction),
        position=None,
        range=None,
        shadow=shadow,
    )


def _parse_shadow(
    shadow_obj: Any, light_where: str, source: str
) -> ShadowSettings:
    if not isinstance(shadow_obj, dict):
        raise schema_error(
            f"{light_where}.shadow must be a JSON object", source=source
        )
    enabled = shadow_obj.get("enabled", True)
    if not isinstance(enabled, bool):
        raise schema_error(
            f"{light_where}.shadow.enabled must be a boolean", source=source
        )
    map_size = _as_int(
        shadow_obj.get("map_size", 256),
        f"{light_where}.shadow.map_size",
        source,
    )
    if not 16 <= map_size <= 4096:
        raise schema_error(
            f"{light_where}.shadow.map_size must be in [16, 4096]",
            source=source,
        )
    radius = _as_float(
        shadow_obj.get("radius", 2.0),
        f"{light_where}.shadow.radius",
        source,
    )
    if radius < 0.0:
        raise schema_error(
            f"{light_where}.shadow.radius must be non-negative",
            source=source,
        )
    samples = _as_int(
        shadow_obj.get("samples", 12),
        f"{light_where}.shadow.samples",
        source,
    )
    if not 1 <= samples <= 64:
        raise schema_error(
            f"{light_where}.shadow.samples must be in [1, 64]",
            source=source,
        )
    bias = _as_float(
        shadow_obj.get("bias", 0.002),
        f"{light_where}.shadow.bias",
        source,
    )
    normal_bias = _as_float(
        shadow_obj.get("normal_bias", 0.02),
        f"{light_where}.shadow.normal_bias",
        source,
    )
    if bias < 0.0 or normal_bias < 0.0:
        raise schema_error(
            f"{light_where}.shadow bias values must be non-negative",
            source=source,
        )
    return ShadowSettings(
        enabled=enabled,
        map_size=map_size,
        radius=radius,
        samples=samples,
        bias=bias,
        normal_bias=normal_bias,
    )


def _parse_material(
    item: Any,
    index: int,
    max_layers: int,
    source: str,
) -> Material:
    where = f"materials[{index}]"
    if not isinstance(item, dict):
        raise schema_error(f"{where} must be a JSON object", source=source)
    layers_obj = _require(item, "layers", source)
    if not isinstance(layers_obj, list) or not layers_obj:
        raise schema_error(
            f"{where}.layers must be a non-empty array", source=source
        )
    if len(layers_obj) > max_layers:
        raise layer_limit_error(
            "material layer count exceeds the configured limit",
            material=index,
            layer_count=len(layers_obj),
            max_layers=max_layers,
        )
    name = item.get("name", f"material_{index}")
    if not isinstance(name, str):
        raise schema_error(f"{where}.name must be a string", source=source)
    layers = tuple(
        _parse_layer(layer, f"{where}.layers[{layer_index}]", source)
        for layer_index, layer in enumerate(layers_obj)
    )
    return Material(name=name, layers=layers)


def _parse_layer(item: Any, where: str, source: str) -> Layer:
    if not isinstance(item, dict):
        raise schema_error(f"{where} must be a JSON object", source=source)
    base_color = _vec3(
        _require(item, "base_color", source),
        f"{where}.base_color",
        source,
        nonnegative=True,
    )
    metallic = _unit_float(
        _require(item, "metallic", source), f"{where}.metallic", source
    )
    roughness = _unit_float(
        _require(item, "roughness", source),
        f"{where}.roughness",
        source,
    )
    if roughness <= 0.0:
        raise schema_error(
            f"{where}.roughness must be greater than zero", source=source
        )
    clearcoat = _unit_float(
        _require(item, "clearcoat", source),
        f"{where}.clearcoat",
        source,
    )
    normal_strength = _as_float(
        _require(item, "normal_strength", source),
        f"{where}.normal_strength",
        source,
    )
    if normal_strength < 0.0:
        raise schema_error(
            f"{where}.normal_strength must be non-negative", source=source
        )
    normal = _vec3(
        item.get("normal", [0.0, 0.0, 1.0]),
        f"{where}.normal",
        source,
    )
    if float(np.linalg.norm(normal)) <= 1.0e-9:
        raise schema_error(
            f"{where}.normal must be non-zero", source=source
        )
    opacity = _unit_float(
        item.get("opacity", 1.0), f"{where}.opacity", source
    )
    return Layer(
        base_color=base_color,
        metallic=metallic,
        roughness=roughness,
        clearcoat=clearcoat,
        normal_strength=normal_strength,
        normal=normal,
        opacity=opacity,
    )


def _parse_mesh(
    item: Any,
    index: int,
    material_count: int,
    source: str,
) -> Mesh:
    where = f"meshes[{index}]"
    if not isinstance(item, dict):
        raise schema_error(f"{where} must be a JSON object", source=source)
    name = item.get("name", f"mesh_{index}")
    if not isinstance(name, str):
        raise schema_error(f"{where}.name must be a string", source=source)
    material_value = _require(item, "material", source)
    material_index = _as_int(material_value, f"{where}.material", source)
    if material_index < 0 or material_index >= material_count:
        raise index_error(
            "mesh material index is out of range",
            mesh=index,
            material=material_index,
            material_count=material_count,
        )

    vertices_obj = _require(item, "vertices", source)
    vertices = _array2d(
        vertices_obj, f"{where}.vertices", source, columns=3, nonempty=True
    )
    if not np.all(np.isfinite(vertices)):
        raise schema_error(
            f"{where}.vertices must contain finite numbers", source=source
        )

    if "triangles" in item:
        indices_obj = item["triangles"]
    elif "indices" in item:
        indices_obj = item["indices"]
    else:
        raise schema_error(
            f"missing required field 'triangles' (or 'indices')", source=source
        )
    indices = _parse_triangles(indices_obj, where, source)
    max_index = indices.max(initial=-1)
    min_index = indices.min(initial=0)
    if min_index < 0 or max_index >= vertices.shape[0]:
        raise index_error(
            "mesh vertex index is out of range",
            mesh=index,
            vertex_index=int(min_index if min_index < 0 else max_index),
            vertex_count=int(vertices.shape[0]),
        )

    if "normals" in item:
        normals = _array2d(
            item["normals"],
            f"{where}.normals",
            source,
            columns=3,
            nonempty=True,
        )
        if normals.shape[0] != vertices.shape[0]:
            raise schema_error(
                f"{where}.normals must have one entry per vertex",
                source=source,
                vertex_count=int(vertices.shape[0]),
                normal_count=int(normals.shape[0]),
            )
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths <= 1.0e-9) or not np.all(np.isfinite(normals)):
            raise schema_error(
                f"{where}.normals must be finite and non-zero", source=source
            )
        normals = normals / lengths[:, None]
    else:
        normals = _compute_vertex_normals(vertices, indices)

    double_sided = item.get("double_sided", False)
    if not isinstance(double_sided, bool):
        raise schema_error(
            f"{where}.double_sided must be a boolean", source=source
        )

    minimum = np.min(vertices, axis=0)
    maximum = np.max(vertices, axis=0)
    center = (minimum + maximum) * 0.5
    radius = float(np.max(np.linalg.norm(vertices - center, axis=1)))
    radius = max(radius, 1.0e-6)
    return Mesh(
        name=name,
        vertices=vertices,
        normals=normals,
        indices=indices,
        material_index=material_index,
        double_sided=double_sided,
        center=center,
        radius=radius,
    )


def _parse_triangles(triangles_obj: Any, where: str, source: str) -> np.ndarray:
    if not isinstance(triangles_obj, list) or not triangles_obj:
        raise schema_error(
            f"{where}.triangles must be a non-empty array", source=source
        )
    nested = all(isinstance(row, list) for row in triangles_obj)
    if nested:
        for row_index, row in enumerate(triangles_obj):
            if len(row) != 3:
                raise schema_error(
                    f"{where}.triangles[{row_index}] must contain 3 indices",
                    source=source,
                )
        flat = [value for row in triangles_obj for value in row]
    else:
        if any(isinstance(value, list) for value in triangles_obj):
            raise schema_error(
                f"{where}.triangles cannot mix flat and nested indices",
                source=source,
            )
        flat = triangles_obj
    if len(flat) % 3 != 0:
        raise schema_error(
            f"{where}.triangles must contain a multiple of 3 indices",
            source=source,
        )
    parsed = [_as_int(value, f"{where}.triangles", source) for value in flat]
    return np.asarray(parsed, dtype=np.int64).reshape((-1, 3))


def _compute_vertex_normals(
    vertices: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    for face in indices:
        a, b, c = (vertices[int(index)] for index in face)
        face_normal = np.cross(b - a, c - a)
        normals[int(face[0])] += face_normal
        normals[int(face[1])] += face_normal
        normals[int(face[2])] += face_normal
    lengths = np.linalg.norm(normals, axis=1)
    degenerate = lengths <= 1.0e-12
    normals[~degenerate] /= lengths[~degenerate, None]
    normals[degenerate] = np.asarray([0.0, 1.0, 0.0])
    return normals


def _require(obj: dict[str, Any], key: str, source: str) -> Any:
    if key not in obj:
        raise schema_error(f"missing required field '{key}'", source=source)
    return obj[key]


def _as_int(value: Any, name: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise schema_error(f"'{name}' must be an integer", source=source)
    return int(value)


def _as_float(value: Any, name: str, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise schema_error(f"'{name}' must be a number", source=source)
    result = float(value)
    if not math.isfinite(result):
        raise schema_error(f"'{name}' must be finite", source=source)
    return result


def _unit_float(value: Any, name: str, source: str) -> float:
    result = _as_float(value, name, source)
    if not 0.0 <= result <= 1.0:
        raise schema_error(f"'{name}' must be in [0, 1]", source=source)
    return result


def _vec3(
    value: Any,
    name: str,
    source: str,
    *,
    nonnegative: bool = False,
) -> np.ndarray:
    result = _array2d(
        value, name, source, columns=3, nonempty=False, flatten=True
    )
    if result.shape != (3,):
        raise schema_error(f"'{name}' must contain exactly 3 numbers", source=source)
    if not np.all(np.isfinite(result)):
        raise schema_error(f"'{name}' must contain finite numbers", source=source)
    if nonnegative and np.any(result < 0.0):
        raise schema_error(f"'{name}' components must be non-negative", source=source)
    return result.astype(np.float64, copy=False)


def _optional_vec3(
    value: Any,
    name: str,
    source: str,
    *,
    nonnegative: bool = False,
) -> np.ndarray:
    return _vec3(value, name, source, nonnegative=nonnegative)


def _array2d(
    value: Any,
    name: str,
    source: str,
    *,
    columns: int,
    nonempty: bool,
    flatten: bool = False,
) -> np.ndarray:
    if not isinstance(value, list):
        raise schema_error(f"'{name}' must be a JSON array", source=source)
    if flatten:
        if len(value) != columns:
            raise schema_error(
                f"'{name}' must contain exactly {columns} numbers", source=source
            )
        if any(isinstance(item, list) for item in value):
            raise schema_error(f"'{name}' must be a flat array", source=source)
        array = np.asarray(
            [_as_float(item, name, source) for item in value],
            dtype=np.float64,
        )
        return array
    if nonempty and not value:
        raise schema_error(f"'{name}' must not be empty", source=source)
    if any(not isinstance(row, list) for row in value):
        raise schema_error(
            f"'{name}' must be an array of {columns}-element arrays",
            source=source,
        )
    rows: list[list[float]] = []
    for row_index, row in enumerate(value):
        if len(row) != columns:
            raise schema_error(
                f"'{name}[{row_index}]' must contain exactly {columns} numbers",
                source=source,
            )
        rows.append(
            [_as_float(item, f"{name}[{row_index}]", source) for item in row]
        )
    return np.asarray(rows, dtype=np.float64).reshape((-1, columns))


def iter_mesh_faces(mesh: Mesh) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    for face in mesh.indices:
        yield mesh.vertices[face], mesh.normals[face]
