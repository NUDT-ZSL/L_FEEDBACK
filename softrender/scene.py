"""Scene JSON parsing and strict validation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import BusinessError
from .math3d import (
    model_matrix,
    normalize,
    normalize_rows,
    normal_matrix,
    srgb_to_linear,
    vec3,
)
from .textures import Texture, load_texture


@dataclass(frozen=True)
class Camera:
    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_y: float
    near: float
    far: float
    width: int
    height: int
    background: np.ndarray
    ambient: np.ndarray
    exposure: float
    cull_mode: str


@dataclass(frozen=True)
class Material:
    name: str
    base_color: np.ndarray
    diffuse_map: Texture
    normal_map: Texture
    specular_map: Texture
    specular: float
    shininess: float
    opacity: float
    blend: str
    double_sided: bool
    emission: np.ndarray
    normal_strength: float

    @property
    def transparent(self) -> bool:
        return self.blend == "alpha" or self.opacity < 1.0


@dataclass(frozen=True)
class Primitive:
    material: str
    indices: np.ndarray


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    tangents: np.ndarray | None
    primitives: tuple[Primitive, ...]
    model: np.ndarray
    normal_transform: np.ndarray


@dataclass(frozen=True)
class ShadowSettings:
    enabled: bool
    resolution: int
    bias: float
    normal_bias: float
    slope_bias: float
    pcf_radius: int


@dataclass(frozen=True)
class Light:
    name: str
    kind: str
    color: np.ndarray
    intensity: float
    direction: np.ndarray | None
    position: np.ndarray | None
    light_range: float | None
    attenuation: tuple[float, float, float] | None
    shadow: ShadowSettings


@dataclass(frozen=True)
class DebugSettings:
    wireframe: bool
    wireframe_color: np.ndarray
    normals: bool
    normal_color: np.ndarray
    normal_length: float


@dataclass(frozen=True)
class Scene:
    camera: Camera
    mesh: Mesh
    materials: dict[str, Material]
    lights: tuple[Light, ...]
    debug: DebugSettings
    input_sha256: str


def _get(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise BusinessError("E_SCHEMA", f"missing required field '{key}'", f"{path}.{key}")
    return mapping[key]


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BusinessError("E_SCHEMA", "expected an object", path)
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise BusinessError("E_SCHEMA", "expected an array", path)
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BusinessError("E_SCHEMA", "expected a number", path)
    result = float(value)
    if not math.isfinite(result):
        raise BusinessError("E_VALUE", "number must be finite", path)
    return result


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BusinessError("E_SCHEMA", "expected an integer", path)
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise BusinessError("E_SCHEMA", "expected a boolean", path)
    return value


def _vector(value: Any, path: str, length: int = 3) -> np.ndarray:
    values = _array(value, path)
    if len(values) != length:
        raise BusinessError("E_VALUE", f"expected {length} numbers", path)
    return np.asarray([_number(item, f"{path}[{index}]") for index, item in enumerate(values)])


def _unit_vector(value: Any, path: str) -> np.ndarray:
    result = normalize(_vector(value, path))
    if float(np.linalg.norm(result)) <= 1e-8:
        raise BusinessError("E_VALUE", "direction vector must be non-zero", path)
    return result


def _color(value: Any, path: str) -> np.ndarray:
    color = _vector(value, path)
    if np.any(color < 0.0) or np.any(color > 1.0):
        raise BusinessError("E_VALUE", "color components must be in [0, 1]", path)
    return srgb_to_linear(color)


def _parse_camera(data: Any) -> Camera:
    path = "$.camera"
    camera = _object(data, path)
    position = _vector(_get(camera, "position", path), f"{path}.position")
    target = _vector(_get(camera, "target", path), f"{path}.target")
    up = _unit_vector(_get(camera, "up", path), f"{path}.up")
    if float(np.linalg.norm(target - position)) <= 1e-8:
        raise BusinessError("E_VALUE", "camera target must differ from position", f"{path}.target")
    forward = normalize(target - position)
    if abs(float(np.dot(forward, up))) > 0.999999:
        raise BusinessError("E_VALUE", "camera up vector must not be parallel to view", f"{path}.up")
    fov_y = _number(_get(camera, "fovY", path), f"{path}.fovY")
    if not (0.0 < fov_y < 179.0):
        raise BusinessError("E_VALUE", "fovY must be in (0, 179)", f"{path}.fovY")
    near = _number(_get(camera, "near", path), f"{path}.near")
    far = _number(_get(camera, "far", path), f"{path}.far")
    if near <= 0.0:
        raise BusinessError("E_VALUE", "near must be positive", f"{path}.near")
    if far <= near:
        raise BusinessError("E_VALUE", "far must be greater than near", f"{path}.far")
    width = _integer(_get(camera, "width", path), f"{path}.width")
    height = _integer(_get(camera, "height", path), f"{path}.height")
    if not (1 <= width <= 4096 and 1 <= height <= 4096):
        raise BusinessError("E_VALUE", "render dimensions must be in [1, 4096]", path)
    background = _color(camera.get("background", [0.018, 0.023, 0.032]), f"{path}.background")
    ambient = _color(camera.get("ambient", [0.055, 0.06, 0.075]), f"{path}.ambient")
    exposure = _number(camera.get("exposure", 1.0), f"{path}.exposure")
    if exposure <= 0.0 or exposure > 64.0:
        raise BusinessError("E_VALUE", "exposure must be in (0, 64]", f"{path}.exposure")
    cull_mode = camera.get("cullMode", "back")
    if cull_mode not in {"back", "none"}:
        raise BusinessError("E_VALUE", "cullMode must be 'back' or 'none'", f"{path}.cullMode")
    return Camera(
        position=position,
        target=target,
        up=up,
        fov_y=fov_y,
        near=near,
        far=far,
        width=width,
        height=height,
        background=background,
        ambient=ambient,
        exposure=exposure,
        cull_mode=cull_mode,
    )


def _compute_normals(vertices: np.ndarray, primitives: list[Primitive]) -> np.ndarray:
    normals = np.zeros_like(vertices)
    for primitive in primitives:
        triangles = vertices[primitive.indices]
        face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        for corner in range(3):
            np.add.at(normals, primitive.indices[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    normals[lengths <= 1e-12] = (0.0, 1.0, 0.0)
    return normalize_rows(normals)


def _parse_mesh(data: Any) -> Mesh:
    path = "$.mesh"
    mesh = _object(data, path)
    raw_vertices = _array(_get(mesh, "vertices", path), f"{path}.vertices")
    raw_uvs = _array(_get(mesh, "uvs", path), f"{path}.uvs")
    if len(raw_vertices) < 3:
        raise BusinessError("E_VALUE", "mesh needs at least three vertices", f"{path}.vertices")
    if len(raw_uvs) != len(raw_vertices):
        raise BusinessError("E_VALUE", "uvs length must match vertices", f"{path}.uvs")
    vertices = np.stack(
        [_vector(item, f"{path}.vertices[{index}]") for index, item in enumerate(raw_vertices)]
    )
    uvs = np.stack([_vector(item, f"{path}.uvs[{index}]", 2) for index, item in enumerate(raw_uvs)])

    raw_primitives = _array(_get(mesh, "primitives", path), f"{path}.primitives")
    if not raw_primitives:
        raise BusinessError("E_VALUE", "mesh needs at least one primitive", f"{path}.primitives")
    primitives: list[Primitive] = []
    total_indices = 0
    for primitive_index, raw_primitive in enumerate(raw_primitives):
        primitive_path = f"{path}.primitives[{primitive_index}]"
        primitive = _object(raw_primitive, primitive_path)
        material = _get(primitive, "material", primitive_path)
        if not isinstance(material, str) or not material:
            raise BusinessError("E_SCHEMA", "primitive material must be a string", f"{primitive_path}.material")
        raw_indices = _array(_get(primitive, "indices", primitive_path), f"{primitive_path}.indices")
        if not raw_indices or len(raw_indices) % 3 != 0:
            raise BusinessError("E_VALUE", "indices must be a non-empty multiple of three", f"{primitive_path}.indices")
        indices = np.asarray(
            [_integer(item, f"{primitive_path}.indices[{index}]") for index, item in enumerate(raw_indices)],
            dtype=np.int64,
        )
        if np.any(indices < 0) or np.any(indices >= len(vertices)):
            raise BusinessError("E_VALUE", "index out of vertex range", f"{primitive_path}.indices")
        total_indices += indices.size
        primitives.append(Primitive(material=material, indices=indices.reshape((-1, 3))))

    if "normals" in mesh:
        raw_normals = _array(mesh["normals"], f"{path}.normals")
        if len(raw_normals) != len(vertices):
            raise BusinessError("E_VALUE", "normals length must match vertices", f"{path}.normals")
        normals = np.stack(
            [_vector(item, f"{path}.normals[{index}]") for index, item in enumerate(raw_normals)]
        )
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths <= 1e-8):
            raise BusinessError("E_VALUE", "normals must be non-zero", f"{path}.normals")
        normals = normalize_rows(normals)
    else:
        normals = _compute_normals(vertices, primitives)

    tangents: np.ndarray | None = None
    if "tangents" in mesh:
        raw_tangents = _array(mesh["tangents"], f"{path}.tangents")
        if len(raw_tangents) != len(vertices):
            raise BusinessError("E_VALUE", "tangents length must match vertices", f"{path}.tangents")
        tangent_rows: list[np.ndarray] = []
        for index, item in enumerate(raw_tangents):
            raw = _array(item, f"{path}.tangents[{index}]")
            if len(raw) not in {3, 4}:
                raise BusinessError("E_VALUE", "tangent must have three or four components", f"{path}.tangents[{index}]")
            tangent = np.asarray([_number(v, f"{path}.tangents[{index}]") for v in raw])
            if len(tangent) == 3:
                tangent = np.append(tangent, 1.0)
            if float(np.linalg.norm(tangent[:3])) <= 1e-8:
                raise BusinessError("E_VALUE", "tangent must be non-zero", f"{path}.tangents[{index}]")
            if abs(abs(tangent[3]) - 1.0) > 1e-6:
                raise BusinessError("E_VALUE", "tangent handedness must be -1 or 1", f"{path}.tangents[{index}]")
            tangent_rows.append(tangent)
        tangents = np.stack(tangent_rows)

    transform_path = f"{path}.transform"
    transform = _object(mesh.get("transform", {}), transform_path)
    offset = _vector(transform.get("translation", [0.0, 0.0, 0.0]), f"{transform_path}.translation")
    rotation = _vector(transform.get("rotationDegrees", [0.0, 0.0, 0.0]), f"{transform_path}.rotationDegrees")
    scale = _vector(transform.get("scale", [1.0, 1.0, 1.0]), f"{transform_path}.scale")
    if np.any(np.abs(scale) <= 1e-9):
        raise BusinessError("E_VALUE", "mesh scale components must be non-zero", f"{transform_path}.scale")
    model = model_matrix(offset, rotation, scale)
    return Mesh(
        vertices=vertices,
        normals=normals,
        uvs=uvs,
        tangents=tangents,
        primitives=tuple(primitives),
        model=model,
        normal_transform=normal_matrix(model),
    )


def _parse_materials(data: Any) -> dict[str, Material]:
    path = "$.materials"
    raw_materials = _array(data, path)
    if not raw_materials:
        raise BusinessError("E_VALUE", "scene needs at least one material", path)
    materials: dict[str, Material] = {}
    for index, raw_material in enumerate(raw_materials):
        material_path = f"{path}[{index}]"
        material = _object(raw_material, material_path)
        name = _get(material, "name", material_path)
        if not isinstance(name, str) or not name:
            raise BusinessError("E_SCHEMA", "material name must be a non-empty string", f"{material_path}.name")
        if name in materials:
            raise BusinessError("E_VALUE", f"duplicate material name: {name}", f"{material_path}.name")
        base_color = _color(_get(material, "baseColor", material_path), f"{material_path}.baseColor")
        diffuse_map = load_texture(
            _get(material, "diffuseMap", material_path), "diffuse", f"{material_path}.diffuseMap"
        )
        normal_map = load_texture(
            _get(material, "normalMap", material_path), "normal", f"{material_path}.normalMap"
        )
        specular_map = load_texture(
            _get(material, "specularMap", material_path), "specular", f"{material_path}.specularMap"
        )
        specular = _number(material.get("specular", 0.5), f"{material_path}.specular")
        shininess = _number(material.get("shininess", 32.0), f"{material_path}.shininess")
        opacity = _number(material.get("opacity", 1.0), f"{material_path}.opacity")
        normal_strength = _number(material.get("normalStrength", 1.0), f"{material_path}.normalStrength")
        if specular < 0.0 or specular > 16.0:
            raise BusinessError("E_VALUE", "specular must be in [0, 16]", f"{material_path}.specular")
        if shininess <= 0.0 or shininess > 4096.0:
            raise BusinessError("E_VALUE", "shininess must be in (0, 4096]", f"{material_path}.shininess")
        if opacity < 0.0 or opacity > 1.0:
            raise BusinessError("E_VALUE", "opacity must be in [0, 1]", f"{material_path}.opacity")
        if normal_strength < 0.0 or normal_strength > 8.0:
            raise BusinessError("E_VALUE", "normalStrength must be in [0, 8]", f"{material_path}.normalStrength")
        blend = material.get("blend", "opaque")
        if blend not in {"opaque", "alpha"}:
            raise BusinessError("E_VALUE", "blend must be opaque or alpha", f"{material_path}.blend")
        if opacity < 1.0 and blend == "opaque":
            raise BusinessError("E_VALUE", "opacity < 1 requires blend='alpha'", f"{material_path}.opacity")
        double_sided = _boolean(material.get("doubleSided", False), f"{material_path}.doubleSided")
        emission = _color(material.get("emission", [0.0, 0.0, 0.0]), f"{material_path}.emission")
        materials[name] = Material(
            name=name,
            base_color=base_color,
            diffuse_map=diffuse_map,
            normal_map=normal_map,
            specular_map=specular_map,
            specular=specular,
            shininess=shininess,
            opacity=opacity,
            blend=blend,
            double_sided=double_sided,
            emission=emission,
            normal_strength=normal_strength,
        )
    return materials


def _parse_shadow(data: Any, path: str) -> ShadowSettings:
    shadow = _object(data, path)
    enabled = _boolean(shadow.get("enabled", True), f"{path}.enabled")
    resolution = _integer(shadow.get("resolution", 512), f"{path}.resolution")
    if not (32 <= resolution <= 2048):
        raise BusinessError("E_VALUE", "shadow resolution must be in [32, 2048]", f"{path}.resolution")
    bias = _number(shadow.get("bias", 0.0015), f"{path}.bias")
    normal_bias = _number(shadow.get("normalBias", 0.015), f"{path}.normalBias")
    slope_bias = _number(shadow.get("slopeBias", 1.5), f"{path}.slopeBias")
    pcf_radius = _integer(shadow.get("pcfRadius", 1), f"{path}.pcfRadius")
    if bias < 0.0 or bias > 0.25:
        raise BusinessError("E_VALUE", "shadow bias must be in [0, 0.25]", f"{path}.bias")
    if normal_bias < 0.0 or normal_bias > 1.0:
        raise BusinessError("E_VALUE", "normalBias must be in [0, 1]", f"{path}.normalBias")
    if slope_bias < 0.0 or slope_bias > 32.0:
        raise BusinessError("E_VALUE", "slopeBias must be in [0, 32]", f"{path}.slopeBias")
    if not (0 <= pcf_radius <= 4):
        raise BusinessError("E_VALUE", "pcfRadius must be in [0, 4]", f"{path}.pcfRadius")
    return ShadowSettings(
        enabled=enabled,
        resolution=resolution,
        bias=bias,
        normal_bias=normal_bias,
        slope_bias=slope_bias,
        pcf_radius=pcf_radius,
    )


def _parse_lights(data: Any) -> tuple[Light, ...]:
    path = "$.lights"
    raw_lights = _array(data, path)
    if len(raw_lights) < 2:
        raise BusinessError("E_VALUE", "scene needs at least two lights", path)
    if len(raw_lights) > 8:
        raise BusinessError("E_VALUE", "at most eight lights are supported", path)
    lights: list[Light] = []
    seen_names: set[str] = set()
    for index, raw_light in enumerate(raw_lights):
        light_path = f"{path}[{index}]"
        light = _object(raw_light, light_path)
        name = _get(light, "name", light_path)
        if not isinstance(name, str) or not name:
            raise BusinessError("E_SCHEMA", "light name must be a non-empty string", f"{light_path}.name")
        if name in seen_names:
            raise BusinessError("E_VALUE", f"duplicate light name: {name}", f"{light_path}.name")
        seen_names.add(name)
        kind = _get(light, "type", light_path)
        if kind not in {"directional", "point"}:
            raise BusinessError("E_VALUE", "light type must be directional or point", f"{light_path}.type")
        color = _color(_get(light, "color", light_path), f"{light_path}.color")
        intensity = _number(light.get("intensity", 1.0), f"{light_path}.intensity")
        if intensity < 0.0 or intensity > 100.0:
            raise BusinessError("E_VALUE", "intensity must be in [0, 100]", f"{light_path}.intensity")
        direction: np.ndarray | None = None
        position: np.ndarray | None = None
        light_range: float | None = None
        attenuation: tuple[float, float, float] | None = None
        if kind == "directional":
            direction = _unit_vector(_get(light, "direction", light_path), f"{light_path}.direction")
        else:
            position = _vector(_get(light, "position", light_path), f"{light_path}.position")
            light_range = _number(light.get("range", 20.0), f"{light_path}.range")
            if light_range <= 0.0:
                raise BusinessError("E_VALUE", "point light range must be positive", f"{light_path}.range")
            attenuation_data = _object(light.get("attenuation", {}), f"{light_path}.attenuation")
            constant = _number(attenuation_data.get("constant", 1.0), f"{light_path}.attenuation.constant")
            linear = _number(attenuation_data.get("linear", 0.0), f"{light_path}.attenuation.linear")
            quadratic = _number(attenuation_data.get("quadratic", 1.2), f"{light_path}.attenuation.quadratic")
            if constant <= 0.0 or linear < 0.0 or quadratic < 0.0:
                raise BusinessError(
                    "E_VALUE",
                    "attenuation constant must be positive and other terms non-negative",
                    f"{light_path}.attenuation",
                )
            attenuation = (constant, linear, quadratic)
        shadow = _parse_shadow(light.get("shadow", {}), f"{light_path}.shadow")
        lights.append(
            Light(
                name=name,
                kind=kind,
                color=color,
                intensity=intensity,
                direction=direction,
                position=position,
                light_range=light_range,
                attenuation=attenuation,
                shadow=shadow,
            )
        )
    return tuple(lights)


def _parse_debug(data: Any) -> DebugSettings:
    path = "$.debug"
    debug = _object(data if data is not None else {}, path)
    wireframe = _boolean(debug.get("wireframe", False), f"{path}.wireframe")
    normals = _boolean(debug.get("normals", False), f"{path}.normals")
    wireframe_color = _color(debug.get("wireframeColor", [0.1, 1.0, 0.3]), f"{path}.wireframeColor")
    normal_color = _color(debug.get("normalColor", [1.0, 0.2, 0.8]), f"{path}.normalColor")
    normal_length = _number(debug.get("normalLength", 0.12), f"{path}.normalLength")
    if normal_length <= 0.0 or normal_length > 10.0:
        raise BusinessError("E_VALUE", "normalLength must be in (0, 10]", f"{path}.normalLength")
    return DebugSettings(
        wireframe=wireframe,
        wireframe_color=wireframe_color,
        normals=normals,
        normal_color=normal_color,
        normal_length=normal_length,
    )


def parse_scene(data: Any, input_sha256: str = "") -> Scene:
    root = _object(data, "$")
    version = _integer(root.get("schemaVersion", 1), "$.schemaVersion")
    if version != 1:
        raise BusinessError("E_VALUE", "only schemaVersion 1 is supported", "$.schemaVersion")
    camera = _parse_camera(_get(root, "camera", "$"))
    mesh = _parse_mesh(_get(root, "mesh", "$"))
    materials = _parse_materials(_get(root, "materials", "$"))
    for primitive_index, primitive in enumerate(mesh.primitives):
        if primitive.material not in materials:
            raise BusinessError(
                "E_VALUE",
                f"primitive references unknown material: {primitive.material}",
                f"$.mesh.primitives[{primitive_index}].material",
            )
    lights = _parse_lights(_get(root, "lights", "$"))
    debug = _parse_debug(root.get("debug"))
    return Scene(
        camera=camera,
        mesh=mesh,
        materials=materials,
        lights=lights,
        debug=debug,
        input_sha256=input_sha256,
    )


def load_scene(path: str | Path) -> Scene:
    scene_path = Path(path)
    try:
        payload = scene_path.read_bytes()
    except OSError as exc:
        raise BusinessError("E_INPUT_READ", f"cannot read scene file: {exc}", str(scene_path)) from exc
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BusinessError("E_JSON_PARSE", f"invalid UTF-8 JSON: {exc}", str(scene_path)) from exc
    digest = hashlib.sha256(payload).hexdigest()
    return parse_scene(data, digest)
