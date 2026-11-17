from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import RenderError
from .math3d import look_at, model_matrix, perspective

MAX_LIGHTS = 32
MAX_MATERIALS = 32
REQUIRED_ATTACHMENTS = (
    "albedo_roughness",
    "normal_side",
    "linear_depth",
    "material_id",
)
TONE_CURVES = ("linear", "reinhard", "aces", "filmic")


@dataclass(frozen=True)
class Camera:
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    fov_y_deg: float
    near: float
    far: float

    @property
    def view(self) -> np.ndarray:
        return look_at(self.position, self.target, self.up)

    @property
    def inv_view(self) -> np.ndarray:
        return np.linalg.inv(self.view)


@dataclass(frozen=True)
class Light:
    position: tuple[float, float, float]
    color: tuple[float, float, float]
    intensity: float
    radius: float


@dataclass(frozen=True)
class Material:
    id: int
    name: str
    base_color: tuple[float, float, float]
    roughness: float
    pattern: str
    texture_size: int
    uv_scale: tuple[float, float]
    back_roughness_scale: float
    back_roughness_bias: float
    normal_strength: float


@dataclass(frozen=True)
class RenderConfig:
    width: int
    height: int
    tile_size: int
    attachments: tuple[str, ...]
    background: tuple[float, float, float]
    ambient_strength: float
    tone_curve: str
    exposure: float
    dof_enabled: bool
    focus_distance: float
    aperture: float
    focal_length_mm: float
    sensor_width_mm: float
    max_blur_radius: float
    reference_check: bool
    reference_threshold: float


@dataclass(frozen=True)
class ObjectSpec:
    index: int
    type: str
    material_id: int
    material_name: str
    center: tuple[float, float, float]
    rotation_deg: tuple[float, float, float]
    radius: float | None
    size: tuple[float, float, float] | None
    open_box: bool

    @property
    def model(self) -> np.ndarray:
        return model_matrix(np.asarray(self.center), np.asarray(self.rotation_deg))


@dataclass(frozen=True)
class Scene:
    raw: dict[str, Any]
    camera: Camera
    lights: tuple[Light, ...]
    materials: tuple[Material, ...]
    objects: tuple[ObjectSpec, ...]
    render: RenderConfig
    scene_sha256: str

    @property
    def projection(self) -> np.ndarray:
        return perspective(
            self.camera.fov_y_deg,
            self.render.width / self.render.height,
            self.camera.near,
            self.camera.far,
        )

    @property
    def inv_projection(self) -> np.ndarray:
        return np.linalg.inv(self.projection)


@dataclass
class MeshData:
    name: str
    vertices: np.ndarray
    indices: np.ndarray

    @property
    def vertex_bytes(self) -> bytes:
        return np.asarray(self.vertices, dtype="<f4").tobytes(order="C")

    @property
    def index_bytes(self) -> bytes:
        return np.asarray(self.indices, dtype="<u4").tobytes(order="C")


@dataclass(frozen=True)
class TextureData:
    albedo: bytes
    roughness: bytes
    normal: bytes
    size: int


@dataclass
class AssetBundle:
    meshes: list[MeshData]
    textures: dict[int, TextureData]
    geometry_sha256: str
    texture_sha256: str
    combined_sha256: str


def default_scene_dict() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "seed": 194,
        "camera": {
            "position": [4.8, 3.6, 7.8],
            "target": [0.0, 1.05, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov_y_deg": 48.0,
            "near": 0.1,
            "far": 60.0,
        },
        "render": {
            "width": 320,
            "height": 180,
            "tile_size": 16,
            "background": [0.012, 0.018, 0.035],
            "ambient_strength": 0.035,
            "g_buffer": {"attachments": list(REQUIRED_ATTACHMENTS)},
            "tone_mapping": {
                "curve": "aces",
                "exposure": 1.2,
                "srgb": True,
            },
            "dof": {
                "enabled": True,
                "focus_distance": 6.8,
                "aperture": 1.1,
                "focal_length_mm": 100.0,
                "sensor_width_mm": 36.0,
                "max_blur_radius": 6.0,
            },
            "reference_check": {"enabled": True, "abs_threshold": 0.01},
        },
        "materials": [
            {
                "id": 1,
                "name": "floor_tile",
                "base_color": [0.32, 0.34, 0.36],
                "roughness": 0.82,
                "pattern": "tiles",
                "texture_size": 128,
                "uv_scale": [4.0, 4.0],
                "back_roughness_scale": 1.05,
                "back_roughness_bias": 0.02,
                "normal_strength": 0.55,
            },
            {
                "id": 2,
                "name": "blue_painted",
                "base_color": [0.17, 0.42, 0.78],
                "roughness": 0.36,
                "pattern": "bands",
                "texture_size": 96,
                "uv_scale": [2.0, 2.0],
                "back_roughness_scale": 1.3,
                "back_roughness_bias": 0.04,
                "normal_strength": 0.8,
            },
            {
                "id": 3,
                "name": "red_clay",
                "base_color": [0.76, 0.21, 0.15],
                "roughness": 0.31,
                "pattern": "speckle",
                "texture_size": 96,
                "uv_scale": [2.5, 2.5],
                "back_roughness_scale": 1.45,
                "back_roughness_bias": 0.06,
                "normal_strength": 0.9,
            },
            {
                "id": 4,
                "name": "brushed_gold",
                "base_color": [0.78, 0.53, 0.16],
                "roughness": 0.24,
                "pattern": "stripes",
                "texture_size": 96,
                "uv_scale": [3.0, 1.0],
                "back_roughness_scale": 1.35,
                "back_roughness_bias": 0.03,
                "normal_strength": 0.7,
            },
            {
                "id": 5,
                "name": "green_matte",
                "base_color": [0.18, 0.58, 0.34],
                "roughness": 0.52,
                "pattern": "speckle",
                "texture_size": 96,
                "uv_scale": [2.0, 2.0],
                "back_roughness_scale": 1.2,
                "back_roughness_bias": 0.05,
                "normal_strength": 0.6,
            },
            {
                "id": 6,
                "name": "violet_two_sided",
                "base_color": [0.50, 0.25, 0.68],
                "roughness": 0.4,
                "pattern": "bands",
                "texture_size": 96,
                "uv_scale": [2.0, 2.0],
                "back_roughness_scale": 1.8,
                "back_roughness_bias": 0.08,
                "normal_strength": 0.9,
            },
        ],
        "lights": [
            {
                "position": [4.2, 5.5, 4.5],
                "color": [1.0, 0.82, 0.67],
                "intensity": 88.0,
                "radius": 11.0,
            },
            {
                "position": [-4.5, 3.8, 3.5],
                "color": [0.42, 0.62, 1.0],
                "intensity": 58.0,
                "radius": 10.0,
            },
            {
                "position": [1.0, 4.8, -4.5],
                "color": [0.92, 0.55, 1.0],
                "intensity": 72.0,
                "radius": 9.0,
            },
            {
                "position": [-1.6, 2.2, 1.0],
                "color": [1.0, 0.32, 0.16],
                "intensity": 18.0,
                "radius": 3.8,
            },
            {
                "position": [2.0, 1.8, 1.2],
                "color": [0.2, 0.7, 1.0],
                "intensity": 16.0,
                "radius": 3.5,
            },
            {
                "position": [0.0, 6.8, 0.0],
                "color": [1.0, 0.96, 0.9],
                "intensity": 42.0,
                "radius": 8.0,
            },
        ],
        "objects": [
            {"type": "floor", "material": "floor_tile", "size": [12.0, 12.0]},
            {
                "type": "sphere",
                "material": "blue_painted",
                "center": [-1.8, 1.1, 0.0],
                "radius": 1.1,
            },
            {
                "type": "sphere",
                "material": "red_clay",
                "center": [0.25, 0.85, -0.5],
                "radius": 0.85,
            },
            {
                "type": "sphere",
                "material": "brushed_gold",
                "center": [1.9, 1.25, 0.3],
                "radius": 1.25,
            },
            {
                "type": "sphere",
                "material": "green_matte",
                "center": [0.65, 0.42, 1.55],
                "radius": 0.42,
            },
            {
                "type": "box",
                "material": "violet_two_sided",
                "center": [-2.45, 0.7, 1.45],
                "size": [1.5, 1.4, 1.25],
                "rotation_deg": [0.0, 17.0, 0.0],
                "open": True,
            },
            {
                "type": "box",
                "material": "floor_tile",
                "center": [2.45, 0.55, -1.0],
                "size": [1.1, 1.1, 1.5],
                "rotation_deg": [0.0, -24.0, 0.0],
            },
        ],
    }


def write_default_scene(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(default_scene_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_scene_file(path: str | Path) -> tuple[dict[str, Any], Scene]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise RenderError(
            "E_SCENE_IO",
            f"cannot read scene file: {path}",
            {"path": str(path), "reason": str(exc)},
        ) from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RenderError(
            "E_SCENE_JSON",
            "scene file is not valid JSON",
            {"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    return data, parse_scene(data)


def _scene_error(message: str, **details: Any) -> RenderError:
    return RenderError("E_SCENE_SCHEMA", message, details)


def _dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _scene_error(f"{path} must be an object", path=path)
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise _scene_error(f"{path} must be an array", path=path)
    return value


def _number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _scene_error(f"{path} must be a number", path=path)
    result = float(value)
    if not math.isfinite(result):
        raise _scene_error(f"{path} must be finite", path=path)
    if positive and result <= 0.0:
        raise _scene_error(f"{path} must be > 0", path=path)
    return result


def _vec(value: Any, path: str, length: int, *, positive: bool = False) -> tuple[float, ...]:
    values = _list(value, path)
    if len(values) != length:
        raise _scene_error(f"{path} must contain {length} numbers", path=path)
    return tuple(_number(item, f"{path}[{i}]", positive=positive) for i, item in enumerate(values))


def parse_scene(raw: dict[str, Any]) -> Scene:
    raw = _dict(raw, "$")
    if raw.get("schema_version") != 1:
        raise _scene_error("unsupported schema_version", expected=1, actual=raw.get("schema_version"))

    camera_data = _dict(raw.get("camera"), "camera")
    camera = Camera(
        position=_vec(camera_data.get("position"), "camera.position", 3),
        target=_vec(camera_data.get("target"), "camera.target", 3),
        up=_vec(camera_data.get("up"), "camera.up", 3),
        fov_y_deg=_number(camera_data.get("fov_y_deg"), "camera.fov_y_deg", positive=True),
        near=_number(camera_data.get("near"), "camera.near", positive=True),
        far=_number(camera_data.get("far"), "camera.far", positive=True),
    )
    if camera.far <= camera.near:
        raise _scene_error("camera.far must be greater than camera.near")
    if camera.fov_y_deg >= 180.0:
        raise _scene_error("camera.fov_y_deg must be < 180")
    if np.linalg.norm(np.asarray(camera.target) - np.asarray(camera.position)) <= 1e-9:
        raise _scene_error("camera target must differ from camera position")
    try:
        look_at(camera.position, camera.target, camera.up)
    except ValueError as exc:
        raise _scene_error("camera.up must not be parallel to the view direction") from exc

    render_data = _dict(raw.get("render"), "render")
    width = _number(render_data.get("width"), "render.width", positive=True)
    height = _number(render_data.get("height"), "render.height", positive=True)
    tile_size = _number(render_data.get("tile_size"), "render.tile_size", positive=True)
    if int(width) != width or int(height) != height or int(tile_size) != tile_size:
        raise _scene_error("render width, height and tile_size must be integers")
    width_i, height_i, tile_i = int(width), int(height), int(tile_size)
    if width_i > 4096 or height_i > 4096:
        raise _scene_error("render resolution exceeds the 4096x4096 safety limit")
    if tile_i > 256:
        raise _scene_error("render.tile_size must be <= 256")

    if "g_buffer" not in render_data:
        raise RenderError(
            "E_MISSING_ATTACHMENT",
            "scene does not declare a G-Buffer",
            {"missing": list(REQUIRED_ATTACHMENTS)},
        )
    gbuffer_data = _dict(render_data.get("g_buffer"), "render.g_buffer")
    if "attachments" not in gbuffer_data:
        raise RenderError(
            "E_MISSING_ATTACHMENT",
            "scene does not declare G-Buffer attachments",
            {"missing": list(REQUIRED_ATTACHMENTS)},
        )
    attachments = _list(gbuffer_data.get("attachments"), "render.g_buffer.attachments")
    if any(not isinstance(item, str) for item in attachments):
        raise _scene_error("g-buffer attachment names must be strings")
    missing = [name for name in REQUIRED_ATTACHMENTS if name not in attachments]
    if missing:
        raise RenderError(
            "E_MISSING_ATTACHMENT",
            "scene requests an incomplete G-Buffer",
            {"missing": missing, "required": list(REQUIRED_ATTACHMENTS), "actual": attachments},
        )
    unknown = [name for name in attachments if name not in REQUIRED_ATTACHMENTS]
    if unknown:
        raise _scene_error("unknown G-Buffer attachment", unknown=unknown)
    if len(set(attachments)) != len(attachments):
        raise _scene_error("G-Buffer attachment names must be unique")

    tone_data = _dict(render_data.get("tone_mapping"), "render.tone_mapping")
    tone_curve = tone_data.get("curve", "aces")
    if tone_curve not in TONE_CURVES:
        raise _scene_error("unsupported tone mapping curve", allowed=list(TONE_CURVES), actual=tone_curve)
    if tone_data.get("srgb", True) is not True:
        raise _scene_error("render.tone_mapping.srgb must be true for PNG output")

    dof_data = _dict(render_data.get("dof", {}), "render.dof")
    reference_data = _dict(render_data.get("reference_check", {}), "render.reference_check")

    render = RenderConfig(
        width=width_i,
        height=height_i,
        tile_size=tile_i,
        attachments=tuple(attachments),
        background=_vec(render_data.get("background"), "render.background", 3),
        ambient_strength=_number(render_data.get("ambient_strength"), "render.ambient_strength"),
        tone_curve=str(tone_curve),
        exposure=_number(tone_data.get("exposure"), "render.tone_mapping.exposure"),
        dof_enabled=bool(dof_data.get("enabled", False)),
        focus_distance=_number(
            dof_data.get("focus_distance", 1.0),
            "render.dof.focus_distance",
            positive=True,
        ),
        aperture=_number(dof_data.get("aperture", 2.8), "render.dof.aperture", positive=True),
        focal_length_mm=_number(
            dof_data.get("focal_length_mm", 50.0),
            "render.dof.focal_length_mm",
            positive=True,
        ),
        sensor_width_mm=_number(
            dof_data.get("sensor_width_mm", 36.0),
            "render.dof.sensor_width_mm",
            positive=True,
        ),
        max_blur_radius=_number(
            dof_data.get("max_blur_radius", 0.0),
            "render.dof.max_blur_radius",
        ),
        reference_check=bool(reference_data.get("enabled", False)),
        reference_threshold=_number(
            reference_data.get("abs_threshold", 0.01),
            "render.reference_check.abs_threshold",
            positive=True,
        ),
    )
    if render.ambient_strength < 0.0:
        raise _scene_error("render.ambient_strength must be >= 0")
    if render.exposure < 0.0:
        raise _scene_error("render.tone_mapping.exposure must be >= 0")
    if render.max_blur_radius < 0.0:
        raise _scene_error("render.dof.max_blur_radius must be >= 0")
    if render.focal_length_mm >= 1000.0:
        raise _scene_error("render.dof.focal_length_mm is outside the supported range")

    material_items = _list(raw.get("materials"), "materials")
    if not material_items or len(material_items) > MAX_MATERIALS:
        raise _scene_error(f"materials must contain 1..{MAX_MATERIALS} entries")
    materials: list[Material] = []
    material_ids: set[int] = set()
    material_names: set[str] = set()
    for index, item in enumerate(material_items):
        path = f"materials[{index}]"
        data = _dict(item, path)
        material_id = data.get("id")
        if isinstance(material_id, bool) or not isinstance(material_id, int):
            raise _scene_error(f"{path}.id must be an integer")
        if material_id < 1 or material_id > MAX_MATERIALS:
            raise _scene_error(f"{path}.id must be in 1..{MAX_MATERIALS}")
        if material_id in material_ids:
            raise _scene_error("duplicate material id", material_id=material_id)
        material_ids.add(material_id)
        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise _scene_error(f"{path}.name must be a non-empty string")
        if name in material_names:
            raise _scene_error("duplicate material name", name=name)
        material_names.add(name)
        texture_size = _number(data.get("texture_size", 96), f"{path}.texture_size", positive=True)
        if int(texture_size) != texture_size or not 8 <= int(texture_size) <= 512:
            raise _scene_error(f"{path}.texture_size must be an integer in 8..512")
        material = Material(
            id=material_id,
            name=name,
            base_color=_vec(data.get("base_color"), f"{path}.base_color", 3),
            roughness=_number(data.get("roughness"), f"{path}.roughness"),
            pattern=str(data.get("pattern", "speckle")),
            texture_size=int(texture_size),
            uv_scale=_vec(data.get("uv_scale", [1.0, 1.0]), f"{path}.uv_scale", 2),
            back_roughness_scale=_number(
                data.get("back_roughness_scale", 1.0),
                f"{path}.back_roughness_scale",
            ),
            back_roughness_bias=_number(
                data.get("back_roughness_bias", 0.0),
                f"{path}.back_roughness_bias",
            ),
            normal_strength=_number(
                data.get("normal_strength", 1.0),
                f"{path}.normal_strength",
            ),
        )
        if not all(0.0 <= channel <= 1.0 for channel in material.base_color):
            raise _scene_error(f"{path}.base_color channels must be in [0, 1]")
        if not 0.0 <= material.roughness <= 1.0:
            raise _scene_error(f"{path}.roughness must be in [0, 1]")
        if any(scale <= 0.0 for scale in material.uv_scale):
            raise _scene_error(f"{path}.uv_scale values must be > 0")
        if material.back_roughness_scale < 0.0:
            raise _scene_error(f"{path}.back_roughness_scale must be >= 0")
        if material.normal_strength < 0.0:
            raise _scene_error(f"{path}.normal_strength must be >= 0")
        materials.append(material)
    materials.sort(key=lambda item: item.id)
    if [item.id for item in materials] != list(range(1, len(materials) + 1)):
        raise _scene_error("material ids must be dense and start at 1")
    material_by_name = {item.name: item for item in materials}

    light_items = _list(raw.get("lights"), "lights")
    if not light_items or len(light_items) > MAX_LIGHTS:
        raise _scene_error(f"lights must contain 1..{MAX_LIGHTS} entries")
    lights: list[Light] = []
    for index, item in enumerate(light_items):
        path = f"lights[{index}]"
        data = _dict(item, path)
        try:
            position = _vec(data.get("position"), f"{path}.position", 3)
            color = _vec(data.get("color"), f"{path}.color", 3)
            intensity = _number(
                data.get("intensity"), f"{path}.intensity", positive=True
            )
            radius = _number(data.get("radius"), f"{path}.radius")
        except RenderError as exc:
            raise RenderError(
                "E_BAD_LIGHT",
                "light has an invalid bounding sphere or light property",
                {
                    "light_index": index,
                    "reason": exc.message,
                    "schema_details": exc.details,
                },
            ) from exc
        if radius <= 0.0:
            raise RenderError(
                "E_BAD_LIGHT",
                "light bounding sphere radius must be > 0",
                {"light_index": index, "radius": radius},
            )
        if not all(0.0 <= channel <= 8.0 for channel in color):
            raise RenderError(
                "E_BAD_LIGHT",
                "light color channels are outside the supported range [0, 8]",
                {"light_index": index, "color": color},
            )
        lights.append(Light(position=position, color=color, intensity=intensity, radius=radius))

    object_items = _list(raw.get("objects"), "objects")
    if not object_items:
        raise _scene_error("objects must not be empty")
    objects: list[ObjectSpec] = []
    for index, item in enumerate(object_items):
        path = f"objects[{index}]"
        data = _dict(item, path)
        object_type = data.get("type")
        if object_type not in ("floor", "sphere", "box"):
            raise _scene_error(f"{path}.type must be floor, sphere or box")
        material_name = data.get("material")
        if not isinstance(material_name, str) or material_name not in material_by_name:
            raise _scene_error(f"{path}.material does not name a declared material")
        material = material_by_name[material_name]
        center_value = data.get("center", [0.0, 0.0, 0.0])
        center = _vec(center_value, f"{path}.center", 3)
        rotation = _vec(data.get("rotation_deg", [0.0, 0.0, 0.0]), f"{path}.rotation_deg", 3)
        radius: float | None = None
        size: tuple[float, float, float] | None = None
        if object_type == "sphere":
            radius = _number(data.get("radius"), f"{path}.radius")
            if radius <= 0.0:
                raise RenderError(
                    "E_ZERO_AREA_GEOMETRY",
                    "sphere radius must be > 0",
                    {"object_index": index, "type": object_type, "radius": radius},
                )
        elif object_type == "box":
            size_vec = _vec(data.get("size"), f"{path}.size", 3)
            if any(component <= 0.0 for component in size_vec):
                raise RenderError(
                    "E_ZERO_AREA_GEOMETRY",
                    "box dimensions must all be > 0",
                    {"object_index": index, "type": object_type, "size": size_vec},
                )
            size = size_vec
        else:
            floor_size = _vec(data.get("size"), f"{path}.size", 2)
            if any(component <= 0.0 for component in floor_size):
                raise RenderError(
                    "E_ZERO_AREA_GEOMETRY",
                    "floor dimensions must both be > 0",
                    {"object_index": index, "type": object_type, "size": floor_size},
                )
            size = (floor_size[0], 0.0, floor_size[1])
        objects.append(
            ObjectSpec(
                index=index,
                type=str(object_type),
                material_id=material.id,
                material_name=material.name,
                center=center,
                rotation_deg=rotation,
                radius=radius,
                size=size,
                open_box=bool(data.get("open", False)),
            )
        )

    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    scene_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return Scene(
        raw=raw,
        camera=camera,
        lights=tuple(lights),
        materials=tuple(materials),
        objects=tuple(objects),
        render=render,
        scene_sha256=scene_sha,
    )


def _hash_update(hasher: Any, label: str, payload: bytes) -> None:
    label_bytes = label.encode("utf-8")
    hasher.update(len(label_bytes).to_bytes(4, "little"))
    hasher.update(label_bytes)
    hasher.update(len(payload).to_bytes(8, "little"))
    hasher.update(payload)


def _orient_triangles(vertices: np.ndarray, indices: np.ndarray, name: str) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.uint32).reshape(-1, 3).copy()
    positions = np.asarray(vertices[:, 0:3], dtype=np.float64)
    normals = np.asarray(vertices[:, 3:6], dtype=np.float64)
    for triangle_index, triangle in enumerate(indices):
        p0, p1, p2 = positions[triangle]
        area_vector = np.cross(p1 - p0, p2 - p0)
        area = float(np.linalg.norm(area_vector))
        if area <= 1e-10:
            raise RenderError(
                "E_ZERO_AREA_GEOMETRY",
                "generated mesh contains a zero-area triangle",
                {"mesh": name, "triangle": triangle_index},
            )
        normal_hint = normals[triangle[0]] + normals[triangle[1]] + normals[triangle[2]]
        if float(np.dot(area_vector, normal_hint)) < 0.0:
            indices[triangle_index, 1], indices[triangle_index, 2] = (
                indices[triangle_index, 2],
                indices[triangle_index, 1],
            )
    return indices.reshape(-1)


def _pack_vertices(
    positions: list[tuple[float, float, float]],
    normals: list[tuple[float, float, float]],
    tangents: list[tuple[float, float, float]],
    tangent_signs: list[float],
    uvs: list[tuple[float, float]],
) -> np.ndarray:
    return np.asarray(
        [
            (*position, *normal, *tangent, sign, *uv)
            for position, normal, tangent, sign, uv in zip(
                positions,
                normals,
                tangents,
                tangent_signs,
                uvs,
                strict=True,
            )
        ],
        dtype=np.float32,
    )


def make_sphere_mesh(name: str, radius: float, slices: int = 32, stacks: int = 16) -> MeshData:
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    tangents: list[tuple[float, float, float]] = []
    tangent_signs: list[float] = []
    uvs: list[tuple[float, float]] = []
    indices: list[int] = []

    positions.append((0.0, radius, 0.0))
    normals.append((0.0, 1.0, 0.0))
    tangents.append((1.0, 0.0, 0.0))
    tangent_signs.append(1.0)
    uvs.append((0.5, 1.0))
    top_index = 0

    rings: list[list[int]] = []
    for stack in range(1, stacks):
        phi = math.pi * stack / stacks
        y = math.cos(phi)
        ring_radius = math.sin(phi)
        ring: list[int] = []
        for slice_index in range(slices + 1):
            theta = 2.0 * math.pi * slice_index / slices
            cos_theta, sin_theta = math.cos(theta), math.sin(theta)
            normal = (ring_radius * cos_theta, y, ring_radius * sin_theta)
            positions.append((normal[0] * radius, normal[1] * radius, normal[2] * radius))
            normals.append(normal)
            tangents.append((-sin_theta, 0.0, cos_theta))
            tangent_signs.append(1.0)
            uvs.append((slice_index / slices, 1.0 - stack / stacks))
            ring.append(len(positions) - 1)
        rings.append(ring)

    positions.append((0.0, -radius, 0.0))
    normals.append((0.0, -1.0, 0.0))
    tangents.append((1.0, 0.0, 0.0))
    tangent_signs.append(1.0)
    uvs.append((0.5, 0.0))
    bottom_index = len(positions) - 1

    first_ring = rings[0]
    for slice_index in range(slices):
        indices.extend(
            [top_index, first_ring[slice_index], first_ring[slice_index + 1]]
        )
    for ring_index in range(len(rings) - 1):
        lower = rings[ring_index]
        upper = rings[ring_index + 1]
        for slice_index in range(slices):
            a, b = lower[slice_index], lower[slice_index + 1]
            c, d = upper[slice_index], upper[slice_index + 1]
            indices.extend([a, b, d, a, d, c])
    last_ring = rings[-1]
    for slice_index in range(slices):
        indices.extend(
            [bottom_index, last_ring[slice_index + 1], last_ring[slice_index]]
        )

    vertices = _pack_vertices(
        positions,
        normals,
        tangents,
        tangent_signs,
        uvs,
    )
    oriented = _orient_triangles(vertices, np.asarray(indices, dtype=np.uint32), name)
    return MeshData(name=name, vertices=vertices, indices=oriented)


def make_floor_mesh(name: str, width: float, depth: float) -> MeshData:
    hx, hz = width * 0.5, depth * 0.5
    positions = [(-hx, 0.0, -hz), (-hx, 0.0, hz), (hx, 0.0, hz), (hx, 0.0, -hz)]
    normals = [(0.0, 1.0, 0.0)] * 4
    tangents = [(1.0, 0.0, 0.0)] * 4
    signs = [1.0] * 4
    uvs = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
    vertices = _pack_vertices(positions, normals, tangents, signs, uvs)
    indices = _orient_triangles(
        vertices,
        np.asarray([0, 1, 2, 0, 2, 3], dtype=np.uint32),
        name,
    )
    return MeshData(name=name, vertices=vertices, indices=indices)


def make_box_mesh(
    name: str,
    size: tuple[float, float, float],
    *,
    open_box: bool = False,
) -> MeshData:
    hx, hy, hz = (component * 0.5 for component in size)
    faces = [
        (
            "px",
            (1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
            [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)],
        ),
        (
            "nx",
            (-1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            [(-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz), (-hx, -hy, -hz)],
        ),
        (
            "py",
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            [(-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz)],
        ),
        (
            "ny",
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            [(-hx, -hy, hz), (-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz)],
        ),
        (
            "pz",
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)],
        ),
        (
            "nz",
            (0.0, 0.0, -1.0),
            (-1.0, 0.0, 0.0),
            [(hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz)],
        ),
    ]
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    tangents: list[tuple[float, float, float]] = []
    signs: list[float] = []
    uvs: list[tuple[float, float]] = []
    indices: list[int] = []
    for face_name, normal, tangent, corners in faces:
        if open_box and face_name == "py":
            continue
        base = len(positions)
        positions.extend(corners)
        normals.extend([normal] * 4)
        tangents.extend([tangent] * 4)
        signs.extend([1.0] * 4)
        uvs.extend([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    vertices = _pack_vertices(positions, normals, tangents, signs, uvs)
    oriented = _orient_triangles(vertices, np.asarray(indices, dtype=np.uint32), name)
    return MeshData(name=name, vertices=vertices, indices=oriented)


def make_object_mesh(scene: Scene, obj: ObjectSpec) -> MeshData:
    name = f"object_{obj.index}_{obj.type}"
    if obj.type == "sphere":
        assert obj.radius is not None
        return make_sphere_mesh(name, obj.radius)
    if obj.type == "box":
        assert obj.size is not None
        return make_box_mesh(name, obj.size, open_box=obj.open_box)
    assert obj.size is not None
    return make_floor_mesh(name, obj.size[0], obj.size[2])


def _integer_noise(x: np.ndarray, y: np.ndarray, seed: int, salt: int) -> np.ndarray:
    xx = x.astype(np.uint64)
    yy = y.astype(np.uint64)
    value = (
        xx * np.uint64(374761393)
        + yy * np.uint64(668265263)
        + np.uint64((seed * 1442695040888963407 + salt * 2654435761) & 0xFFFFFFFF)
    ) & np.uint64(0xFFFFFFFF)
    value = (value ^ (value >> np.uint64(13))) * np.uint64(1274126177)
    value &= np.uint64(0xFFFFFFFF)
    value ^= value >> np.uint64(16)
    return (value & np.uint64(0xFFFF)).astype(np.float64) / 65535.0


def make_material_textures(material: Material, seed: int) -> TextureData:
    size = material.texture_size
    coordinates = (np.arange(size, dtype=np.float64) + 0.5) / size
    u, v = np.meshgrid(coordinates, coordinates)
    ix, iy = np.meshgrid(np.arange(size), np.arange(size))
    noise = _integer_noise(ix, iy, seed + material.id * 37, 11)
    wave = 0.5 + 0.5 * np.sin(
        2.0 * math.pi * (u * (2.0 + material.id) + v * (3.0 + material.id * 0.25))
    )
    if material.pattern == "tiles":
        checker = ((np.floor(u * 8.0) + np.floor(v * 8.0)) % 2.0) * 2.0 - 1.0
        modulation = 1.0 + 0.10 * checker + 0.035 * (noise - 0.5)
        roughness_pattern = 0.12 * checker + 0.08 * (noise - 0.5)
    elif material.pattern == "bands":
        modulation = 1.0 + 0.08 * np.sin(2.0 * math.pi * (v * 7.0 + 0.2 * noise))
        roughness_pattern = 0.10 * (wave - 0.5)
    elif material.pattern == "stripes":
        modulation = 1.0 + 0.13 * np.sin(2.0 * math.pi * u * 18.0)
        roughness_pattern = 0.09 * np.cos(2.0 * math.pi * u * 18.0)
    else:
        modulation = 0.92 + 0.16 * noise
        roughness_pattern = 0.15 * (noise - 0.5)

    base = np.asarray(material.base_color, dtype=np.float64)
    tint = np.stack(
        [
            modulation * (0.96 + 0.04 * wave),
            modulation,
            modulation * (1.04 - 0.04 * wave),
        ],
        axis=-1,
    )
    albedo = np.clip(base[None, None, :] * tint, 0.0, 1.0)
    alpha = np.ones((size, size, 1), dtype=np.float64)
    albedo_rgba = np.concatenate([albedo, alpha], axis=-1)

    roughness = np.clip(
        material.roughness + roughness_pattern + 0.025 * (wave - 0.5),
        0.04,
        1.0,
    )
    roughness_rg = np.repeat(roughness[..., None], 3, axis=-1)

    height = (
        0.08 * np.sin(2.0 * math.pi * (u * 3.0 + v * 2.0))
        + 0.04 * np.sin(2.0 * math.pi * (u * 7.0 - v * 5.0))
        + 0.02 * np.cos(2.0 * math.pi * (u * 11.0 + v * 9.0))
    )
    du, dv = np.gradient(height)
    normal = np.stack([-du * 2.0, -dv * 2.0, np.ones_like(height)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)

    return TextureData(
        albedo=np.round(albedo_rgba * 255.0).astype(np.uint8).tobytes(order="C"),
        roughness=np.round(roughness_rg * 255.0).astype(np.uint8).tobytes(order="C"),
        normal=np.round((normal * 0.5 + 0.5) * 255.0).astype(np.uint8).tobytes(order="C"),
        size=size,
    )


def build_assets(scene: Scene) -> AssetBundle:
    meshes: list[MeshData] = []
    geometry_hasher = hashlib.sha256()
    for obj in scene.objects:
        mesh = make_object_mesh(scene, obj)
        meshes.append(mesh)
        _hash_update(geometry_hasher, mesh.name, mesh.vertex_bytes)
        _hash_update(geometry_hasher, mesh.name + ".indices", mesh.index_bytes)

    textures: dict[int, TextureData] = {}
    texture_hasher = hashlib.sha256()
    seed = int(scene.raw.get("seed", 0))
    for material in scene.materials:
        texture = make_material_textures(material, seed)
        textures[material.id] = texture
        _hash_update(texture_hasher, f"{material.id}.albedo", texture.albedo)
        _hash_update(texture_hasher, f"{material.id}.roughness", texture.roughness)
        _hash_update(texture_hasher, f"{material.id}.normal", texture.normal)

    combined = hashlib.sha256()
    _hash_update(combined, "scene", scene.scene_sha256.encode("ascii"))
    _hash_update(combined, "geometry", geometry_hasher.digest())
    _hash_update(combined, "textures", texture_hasher.digest())
    return AssetBundle(
        meshes=meshes,
        textures=textures,
        geometry_sha256=geometry_hasher.hexdigest(),
        texture_sha256=texture_hasher.hexdigest(),
        combined_sha256=combined.hexdigest(),
    )

