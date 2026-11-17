"""Material JSON and scene JSON loading + validation."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .errors import AssetError
from . import math3d


def _read_json(path: str, what: str) -> dict:
    if not os.path.isfile(path):
        raise AssetError(f"{what} file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise AssetError(f"{path}: invalid JSON: {exc}")
    if not isinstance(data, dict):
        raise AssetError(f"{path}: top-level JSON value must be an object")
    return data


def _finite_number(value, name: str, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssetError(f"{path}: '{name}' must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise AssetError(f"{path}: '{name}' must be finite")
    return value


def _vec3(value, name: str, path: str, lo=None, hi=None) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise AssetError(f"{path}: '{name}' must be an array of 3 numbers")
    out = np.array(
        [_finite_number(v, f"{name}[{i}]", path) for i, v in enumerate(value)],
        dtype=np.float64,
    )
    if lo is not None and np.any(out < lo):
        raise AssetError(f"{path}: '{name}' components must be >= {lo}")
    if hi is not None and np.any(out > hi):
        raise AssetError(f"{path}: '{name}' components must be <= {hi}")
    return out


@dataclass
class Material:
    albedo: np.ndarray
    roughness: float
    normal_map: str | None  # absolute path or None


def load_material(path: str) -> Material:
    data = _read_json(path, "material")
    if "albedo" not in data:
        raise AssetError(f"{path}: missing required field 'albedo'")
    if "roughness" not in data:
        raise AssetError(f"{path}: missing required field 'roughness'")
    albedo = _vec3(data["albedo"], "albedo", path, lo=0.0, hi=1.0)
    roughness = _finite_number(data["roughness"], "roughness", path)
    if not (0.01 <= roughness <= 1.0):
        raise AssetError(f"{path}: 'roughness' must be in [0.01, 1.0]")
    normal_map = data.get("normal_map")
    if normal_map is not None:
        if not isinstance(normal_map, str) or not normal_map:
            raise AssetError(f"{path}: 'normal_map' must be a non-empty string path")
        normal_map = os.path.normpath(os.path.join(os.path.dirname(path), normal_map))
        if not os.path.isfile(normal_map):
            raise AssetError(f"{path}: normal map file not found: {normal_map}")
    return Material(albedo=albedo, roughness=roughness, normal_map=normal_map)


@dataclass
class ModelSpec:
    obj_path: str
    material_path: str
    transform: np.ndarray  # 4x4


@dataclass
class DirLight:
    direction: np.ndarray  # unit vector, direction the light travels
    radiance: np.ndarray   # color * intensity


@dataclass
class PointLight:
    position: np.ndarray
    radiance: np.ndarray   # color * intensity (before 1/d^2 falloff)
    min_distance: float


@dataclass
class Scene:
    models: list = field(default_factory=list)
    dir_lights: list = field(default_factory=list)
    point_lights: list = field(default_factory=list)
    ambient: np.ndarray = field(default_factory=lambda: np.array([0.03, 0.03, 0.035]))


def _parse_transform(value, path: str) -> np.ndarray:
    if value is None:
        return math3d.identity()
    if not isinstance(value, dict):
        raise AssetError(f"{path}: 'transform' must be an object")
    m = math3d.identity()
    if "scale" in value:
        s = value["scale"]
        if isinstance(s, (int, float)) and not isinstance(s, bool):
            s = _finite_number(s, "transform.scale", path)
            if s <= 0:
                raise AssetError(f"{path}: 'transform.scale' must be positive")
            m = math3d.uniform_scale(s) @ m
        else:
            sv = _vec3(s, "transform.scale", path, lo=1e-6)
            m = np.diag([sv[0], sv[1], sv[2], 1.0]) @ m
    if "rotate_y_deg" in value:
        m = math3d.rotation_y(_finite_number(value["rotate_y_deg"], "transform.rotate_y_deg", path)) @ m
    if "translate" in value:
        m = math3d.translation(_vec3(value["translate"], "transform.translate", path)) @ m
    return m


def load_scene(path: str) -> Scene:
    data = _read_json(path, "scene")
    base = os.path.dirname(os.path.abspath(path))

    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise AssetError(f"{path}: 'models' must be a non-empty array")
    scene = Scene()
    for i, entry in enumerate(models):
        if not isinstance(entry, dict):
            raise AssetError(f"{path}: models[{i}] must be an object")
        for key in ("obj", "material"):
            if key not in entry or not isinstance(entry[key], str) or not entry[key]:
                raise AssetError(f"{path}: models[{i}].{key} must be a non-empty string")
        obj_path = os.path.normpath(os.path.join(base, entry["obj"]))
        mat_path = os.path.normpath(os.path.join(base, entry["material"]))
        if not os.path.isfile(obj_path):
            raise AssetError(f"{path}: models[{i}] OBJ not found: {obj_path}")
        if not os.path.isfile(mat_path):
            raise AssetError(f"{path}: models[{i}] material not found: {mat_path}")
        transform = _parse_transform(entry.get("transform"), path)
        scene.models.append(ModelSpec(obj_path=obj_path, material_path=mat_path, transform=transform))

    lights = data.get("lights")
    if not isinstance(lights, dict):
        raise AssetError(f"{path}: 'lights' must be an object")
    dirs = lights.get("directional")
    points = lights.get("point")
    if not isinstance(dirs, list) or len(dirs) != 2:
        raise AssetError(f"{path}: exactly 2 directional lights are required")
    if not isinstance(points, list) or len(points) != 1:
        raise AssetError(f"{path}: exactly 1 point light is required")
    for i, spec in enumerate(dirs):
        if not isinstance(spec, dict):
            raise AssetError(f"{path}: lights.directional[{i}] must be an object")
        for key in ("direction", "color", "intensity"):
            if key not in spec:
                raise AssetError(f"{path}: lights.directional[{i}] missing '{key}'")
        direction = _vec3(spec["direction"], "direction", path)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            raise AssetError(f"{path}: lights.directional[{i}] direction is zero")
        color = _vec3(spec["color"], "color", path, lo=0.0)
        intensity = _finite_number(spec["intensity"], "intensity", path)
        if intensity <= 0:
            raise AssetError(f"{path}: lights.directional[{i}] intensity must be > 0")
        scene.dir_lights.append(DirLight(direction=direction / norm, radiance=color * intensity))
    for i, spec in enumerate(points):
        if not isinstance(spec, dict):
            raise AssetError(f"{path}: lights.point[{i}] must be an object")
        for key in ("position", "color", "intensity"):
            if key not in spec:
                raise AssetError(f"{path}: lights.point[{i}] missing '{key}'")
        position = _vec3(spec["position"], "position", path)
        color = _vec3(spec["color"], "color", path, lo=0.0)
        intensity = _finite_number(spec["intensity"], "intensity", path)
        if intensity <= 0:
            raise AssetError(f"{path}: lights.point[{i}] intensity must be > 0")
        min_distance = spec.get("min_distance", 0.05)
        min_distance = _finite_number(min_distance, "min_distance", path)
        if min_distance <= 0:
            raise AssetError(f"{path}: lights.point[{i}] min_distance must be > 0")
        scene.point_lights.append(
            PointLight(position=position, radiance=color * intensity, min_distance=min_distance)
        )

    ambient = data.get("ambient")
    if ambient is not None:
        scene.ambient = _vec3(ambient, "ambient", path, lo=0.0)
    return scene
