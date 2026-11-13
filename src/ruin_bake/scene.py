"""Deterministic procedural scene construction and stable content hashing."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any, Iterable

from .canonical import (
    SerializationError,
    canonical_json,
    rgb,
    sha256_canonical,
    vec3,
)
from .math3d import Vec3, mat_mul_vec, normalize, rotation_matrix_deg

SCENE_SCHEMA = "ruin-scene-v1"


@dataclass(frozen=True, slots=True)
class BoxGeometry:
    object_id: int
    center: Vec3
    half_size: Vec3
    rotation_deg: Vec3
    albedo: Vec3
    emission: Vec3

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "center": list(self.center),
            "half_size": list(self.half_size),
            "rotation_deg": list(self.rotation_deg),
            "albedo": list(self.albedo),
            "emission": list(self.emission),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BoxGeometry":
        if not isinstance(value, dict):
            raise SerializationError("geometry record must be an object")
        object_id = value.get("object_id")
        if isinstance(object_id, bool) or not isinstance(object_id, int) or object_id < 0:
            raise SerializationError("geometry.object_id must be a non-negative integer")
        half_size = vec3(value.get("half_size"), "geometry.half_size")
        if any(component <= 0.0 for component in half_size):
            raise SerializationError("geometry.half_size components must be positive")
        return cls(
            object_id=object_id,
            center=vec3(value.get("center"), "geometry.center"),
            half_size=half_size,
            rotation_deg=vec3(value.get("rotation_deg"), "geometry.rotation_deg"),
            albedo=rgb(value.get("albedo"), "geometry.albedo"),
            emission=rgb(value.get("emission"), "geometry.emission"),
        )

    def content_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def content_hash(self) -> str:
        return sha256_canonical(self.content_dict())


@dataclass(frozen=True, slots=True)
class DirectionalLight:
    light_id: str
    direction: Vec3
    radiance: Vec3

    def to_dict(self) -> dict[str, Any]:
        return {
            "light_id": self.light_id,
            "direction": list(self.direction),
            "radiance": list(self.radiance),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DirectionalLight":
        if not isinstance(value, dict):
            raise SerializationError("directional light must be an object")
        light_id = value.get("light_id")
        if not isinstance(light_id, str) or not light_id:
            raise SerializationError("directional.light_id must be a non-empty string")
        direction = vec3(value.get("direction"), "directional.direction")
        if sum(component * component for component in direction) <= 1.0e-24:
            raise SerializationError("directional.direction must be non-zero")
        return cls(light_id=light_id, direction=direction, radiance=rgb(value.get("radiance"), "directional.radiance"))

    def content_hash(self) -> str:
        return sha256_canonical(self.to_dict())


@dataclass(frozen=True, slots=True)
class PointLight:
    light_id: str
    position: Vec3
    radiance: Vec3

    def to_dict(self) -> dict[str, Any]:
        return {
            "light_id": self.light_id,
            "position": list(self.position),
            "radiance": list(self.radiance),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PointLight":
        if not isinstance(value, dict):
            raise SerializationError("point light must be an object")
        light_id = value.get("light_id")
        if not isinstance(light_id, str) or not light_id:
            raise SerializationError("point_light.light_id must be a non-empty string")
        return cls(
            light_id=light_id,
            position=vec3(value.get("position"), "point_light.position"),
            radiance=rgb(value.get("radiance"), "point_light.radiance"),
        )

    def content_hash(self) -> str:
        return sha256_canonical(self.to_dict())


@dataclass(frozen=True, slots=True)
class SceneGeneratorConfig:
    seed: int = 20261114
    bounds_min: Vec3 = (-10.0, -8.0, 0.0)
    bounds_max: Vec3 = (10.0, 8.0, 6.0)
    extra_boxes: int = 12
    point_light_count: int = 4
    emissive_box_count: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("scene seed must be an integer")
        if not isinstance(self.bounds_min, (list, tuple)) or len(self.bounds_min) != 3:
            raise ValueError("bounds_min must contain three numbers")
        if not isinstance(self.bounds_max, (list, tuple)) or len(self.bounds_max) != 3:
            raise ValueError("bounds_max must contain three numbers")
        object.__setattr__(
            self,
            "bounds_min",
            tuple(float(component) for component in self.bounds_min),
        )
        object.__setattr__(
            self,
            "bounds_max",
            tuple(float(component) for component in self.bounds_max),
        )
        for name in ("extra_boxes", "point_light_count", "emissive_box_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.extra_boxes <= 128:
            raise ValueError("extra_boxes must be between 0 and 128")
        if not 0 <= self.point_light_count <= 4:
            raise ValueError("point_light_count must be between 0 and 4")
        if not 0 <= self.emissive_box_count <= self.extra_boxes:
            raise ValueError("emissive_box_count must be between 0 and extra_boxes")
        for axis in range(3):
            if not math.isfinite(self.bounds_min[axis]) or not math.isfinite(self.bounds_max[axis]):
                raise ValueError("scene bounds must be finite")
            if self.bounds_max[axis] - self.bounds_min[axis] < 0.5:
                raise ValueError("each scene bound axis must span at least 0.5")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
            "extra_boxes": self.extra_boxes,
            "point_light_count": self.point_light_count,
            "emissive_box_count": self.emissive_box_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "SceneGeneratorConfig":
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise SerializationError("scene generator config must be an object")
        allowed = {
            "seed",
            "bounds_min",
            "bounds_max",
            "extra_boxes",
            "point_light_count",
            "emissive_box_count",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise SerializationError(f"unknown scene generator keys: {unknown}")
        defaults = cls()
        return cls(
            seed=value.get("seed", defaults.seed),
            bounds_min=vec3(value.get("bounds_min", defaults.bounds_min), "bounds_min"),
            bounds_max=vec3(value.get("bounds_max", defaults.bounds_max), "bounds_max"),
            extra_boxes=value.get("extra_boxes", defaults.extra_boxes),
            point_light_count=value.get("point_light_count", defaults.point_light_count),
            emissive_box_count=value.get("emissive_box_count", defaults.emissive_box_count),
        )


@dataclass(frozen=True, slots=True)
class Scene:
    seed: int
    generation: dict[str, Any]
    bounds_min: Vec3
    bounds_max: Vec3
    geometry: tuple[BoxGeometry, ...]
    directional: DirectionalLight
    point_lights: tuple[PointLight, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.bounds_min, (list, tuple)) or len(self.bounds_min) != 3:
            raise ValueError("scene bounds_min must contain three numbers")
        if not isinstance(self.bounds_max, (list, tuple)) or len(self.bounds_max) != 3:
            raise ValueError("scene bounds_max must contain three numbers")
        object.__setattr__(
            self,
            "bounds_min",
            tuple(float(component) for component in self.bounds_min),
        )
        object.__setattr__(
            self,
            "bounds_max",
            tuple(float(component) for component in self.bounds_max),
        )
        for axis in range(3):
            if not math.isfinite(self.bounds_min[axis]) or not math.isfinite(self.bounds_max[axis]):
                raise ValueError("scene bounds must be finite")
            if self.bounds_min[axis] >= self.bounds_max[axis]:
                raise ValueError("scene bounds_min must be strictly below bounds_max")
        ids = [box.object_id for box in self.geometry]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("geometry object ids must be unique and sorted")
        light_ids = [light.light_id for light in self.point_lights]
        if light_ids != sorted(light_ids) or len(light_ids) != len(set(light_ids)):
            raise ValueError("point light ids must be unique and sorted")
        if len(light_ids) > 4:
            raise ValueError("a scene may contain at most four point lights")
        if self.directional.light_id in light_ids:
            raise ValueError("directional and point light ids must be distinct")

    def content_dict(self) -> dict[str, Any]:
        return {
            "schema": SCENE_SCHEMA,
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
            "geometry": [box.to_dict() for box in self.geometry],
            "directional": self.directional.to_dict(),
            "point_lights": [light.to_dict() for light in self.point_lights],
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.content_dict()
        value["seed"] = self.seed
        value["generation"] = dict(self.generation)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Scene":
        if not isinstance(value, dict):
            raise SerializationError("scene must be an object")
        if value.get("schema") != SCENE_SCHEMA:
            raise SerializationError(f"unsupported scene schema: {value.get('schema')!r}")
        geometry_value = value.get("geometry")
        point_value = value.get("point_lights")
        if not isinstance(geometry_value, list) or not isinstance(point_value, list):
            raise SerializationError("scene geometry and point_lights must be arrays")
        seed = value.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SerializationError("scene.seed must be an integer")
        generation = value.get("generation")
        if not isinstance(generation, dict):
            raise SerializationError("scene.generation must be an object")
        geometry = tuple(sorted((BoxGeometry.from_dict(item) for item in geometry_value), key=lambda box: box.object_id))
        point_lights = tuple(sorted((PointLight.from_dict(item) for item in point_value), key=lambda light: light.light_id))
        scene = cls(
            seed=seed,
            generation=dict(generation),
            bounds_min=vec3(value.get("bounds_min"), "scene.bounds_min"),
            bounds_max=vec3(value.get("bounds_max"), "scene.bounds_max"),
            geometry=geometry,
            directional=DirectionalLight.from_dict(value.get("directional")),
            point_lights=point_lights,
        )
        for axis in range(3):
            if scene.bounds_min[axis] >= scene.bounds_max[axis]:
                raise SerializationError("scene bounds_min must be strictly below bounds_max")
        return scene

    @property
    def scene_hash(self) -> str:
        return sha256_canonical(self.content_dict())

    def geometry_hashes(self) -> dict[str, str]:
        return {str(box.object_id): box.content_hash() for box in self.geometry}

    def light_hashes(self) -> dict[str, str]:
        result = {self.directional.light_id: self.directional.content_hash()}
        for light in self.point_lights:
            result[light.light_id] = light.content_hash()
        return result

    def box_tuples(self) -> tuple[tuple[int, Vec3, Vec3, Vec3], ...]:
        return tuple(
            (box.object_id, box.center, box.half_size, box.rotation_deg)
            for box in self.geometry
        )

    def all_light_hashes(self) -> dict[str, str]:
        return self.light_hashes()


def _support_extent(half_size: Vec3, rotation_deg: Vec3) -> Vec3:
    matrix = rotation_matrix_deg(rotation_deg)
    return (
        abs(matrix[0]) * half_size[0] + abs(matrix[1]) * half_size[1] + abs(matrix[2]) * half_size[2],
        abs(matrix[3]) * half_size[0] + abs(matrix[4]) * half_size[1] + abs(matrix[5]) * half_size[2],
        abs(matrix[6]) * half_size[0] + abs(matrix[7]) * half_size[1] + abs(matrix[8]) * half_size[2],
    )


def _world_support_extent(box: BoxGeometry) -> Vec3:
    return _support_extent(box.half_size, box.rotation_deg)


def _clamp_center(value: float, low: float, high: float) -> float:
    if low > high:
        return (low + high) * 0.5
    return min(max(value, low), high)


def generate_scene(config: SceneGeneratorConfig | None = None, **overrides: Any) -> Scene:
    """Generate a deterministic ruin scene.

    The same `SceneGeneratorConfig` always performs exactly the same sequence
    of Mersenne-Twister operations.  No wall-clock time, process id, hash
    randomization, temporary directory, or thread scheduling affects output.
    """

    if config is None:
        config = SceneGeneratorConfig()
        if overrides:
            config = replace(config, **overrides)
    elif not isinstance(config, SceneGeneratorConfig):
        raise TypeError("config must be a SceneGeneratorConfig")
    elif overrides:
        config = replace(config, **overrides)

    rng = random.Random(config.seed)
    bounds_min = config.bounds_min
    bounds_max = config.bounds_max
    span_x = bounds_max[0] - bounds_min[0]
    span_y = bounds_max[1] - bounds_min[1]
    span_z = bounds_max[2] - bounds_min[2]

    floor_half = (span_x * 0.5, span_y * 0.5, 0.125)
    geometry: list[BoxGeometry] = [
        BoxGeometry(
            object_id=0,
            center=((bounds_min[0] + bounds_max[0]) * 0.5, (bounds_min[1] + bounds_max[1]) * 0.5, bounds_min[2] - floor_half[2] + 0.25),
            half_size=floor_half,
            rotation_deg=(0.0, 0.0, 0.0),
            albedo=(0.46, 0.43, 0.38),
            emission=(0.0, 0.0, 0.0),
        )
    ]

    box_scale = min(1.0, span_x / 8.0, span_y / 8.0, span_z / 6.0)
    box_scale = max(box_scale, 0.05)

    for index in range(config.extra_boxes):
        object_id = index + 1
        rotated = index % 4 != 0
        rotation = (
            rng.uniform(-18.0, 18.0) if rotated else 0.0,
            rng.uniform(-12.0, 12.0) if rotated else 0.0,
            rng.uniform(-70.0, 70.0) if rotated else 0.0,
        )
        half_size = (
            rng.uniform(0.25, 1.55) * box_scale,
            rng.uniform(0.25, 1.55) * box_scale,
            rng.uniform(0.25, 1.65) * box_scale,
        )
        _ = rng.uniform(0.0, 1.0)  # Stable draw count even if formula changes.
        center_x = rng.uniform(bounds_min[0] + 1.4, bounds_max[0] - 1.4)
        center_y = rng.uniform(bounds_min[1] + 1.4, bounds_max[1] - 1.4)
        max_z = max(1.0, span_z - 1.2)
        center_z = rng.uniform(0.35 + half_size[2], max_z)
        support = _support_extent(half_size, rotation)
        center_x = _clamp_center(
            center_x,
            bounds_min[0] + support[0],
            bounds_max[0] - support[0],
        )
        center_y = _clamp_center(
            center_y,
            bounds_min[1] + support[1],
            bounds_max[1] - support[1],
        )
        center_z = _clamp_center(
            center_z,
            bounds_min[2] + support[2],
            bounds_max[2] - support[2],
        )
        emissive = index >= config.extra_boxes - config.emissive_box_count
        if emissive:
            albedo = (rng.uniform(0.05, 0.18), rng.uniform(0.04, 0.14), rng.uniform(0.03, 0.12))
            emission = (rng.uniform(0.8, 1.8), rng.uniform(0.35, 1.0), rng.uniform(0.10, 0.35))
        else:
            albedo = (rng.uniform(0.12, 0.72), rng.uniform(0.10, 0.68), rng.uniform(0.08, 0.62))
            emission = (0.0, 0.0, 0.0)
        geometry.append(
            BoxGeometry(
                object_id=object_id,
                center=(center_x, center_y, center_z),
                half_size=half_size,
                rotation_deg=rotation,
                albedo=albedo,
                emission=emission,
            )
        )

    azimuth = rng.uniform(0.0, math.tau)
    elevation = math.radians(rng.uniform(38.0, 68.0))
    direction = normalize(
        (
            math.cos(azimuth) * math.cos(elevation),
            math.sin(azimuth) * math.cos(elevation),
            -math.sin(elevation),
        ),
        "directional light direction",
    )
    directional = DirectionalLight(
        light_id="dir-0",
        direction=direction,
        radiance=(rng.uniform(1.9, 2.8), rng.uniform(1.8, 2.6), rng.uniform(1.5, 2.3)),
    )

    point_lights: list[PointLight] = []
    for index in range(config.point_light_count):
        point_lights.append(
            PointLight(
                light_id=f"point-{index}",
                position=(
                    rng.uniform(bounds_min[0] + 1.0, bounds_max[0] - 1.0),
                    rng.uniform(bounds_min[1] + 1.0, bounds_max[1] - 1.0),
                    rng.uniform(2.0, max(2.1, bounds_max[2] - 0.8)),
                ),
                radiance=(rng.uniform(16.0, 36.0), rng.uniform(13.0, 31.0), rng.uniform(9.0, 25.0)),
            )
        )

    return Scene(
        seed=config.seed,
        generation=config.to_dict(),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        geometry=tuple(sorted(geometry, key=lambda box: box.object_id)),
        directional=directional,
        point_lights=tuple(sorted(point_lights, key=lambda light: light.light_id)),
    )


def _expected_light_id(light_id: str) -> None:
    if not isinstance(light_id, str) or not light_id:
        raise SerializationError("light id must be a non-empty string")


def apply_scene_patch(scene: Scene, patch: dict[str, Any] | None) -> Scene:
    """Apply a deterministic JSON-friendly scene patch.

    Supported keys:
      geometry_updates, geometry_add, geometry_remove,
      directional_update, point_light_updates, point_light_add,
      point_light_remove.
    """

    if patch in (None, {}):
        return scene
    if not isinstance(patch, dict):
        raise SerializationError("scene patch must be an object")
    allowed = {
        "geometry_updates",
        "geometry_add",
        "geometry_remove",
        "directional_update",
        "point_light_updates",
        "point_light_add",
        "point_light_remove",
    }
    unknown = sorted(set(patch) - allowed)
    if unknown:
        raise SerializationError(f"unknown scene patch keys: {unknown}")

    geometry_by_id = {box.object_id: box.to_dict() for box in scene.geometry}
    updates = patch.get("geometry_updates", {})
    if updates is not None:
        if not isinstance(updates, dict):
            raise SerializationError("geometry_updates must be an object keyed by object id")
        for key, fields in updates.items():
            try:
                object_id = int(key)
            except (TypeError, ValueError) as exc:
                raise SerializationError(f"invalid geometry update id: {key!r}") from exc
            if object_id not in geometry_by_id:
                raise SerializationError(f"geometry update references unknown object id {object_id}")
            if not isinstance(fields, dict):
                raise SerializationError("geometry update fields must be an object")
            merged = dict(geometry_by_id[object_id])
            for field, value in fields.items():
                if field == "object_id":
                    raise SerializationError("geometry_updates cannot change object_id")
                if field not in merged:
                    raise SerializationError(f"unknown geometry field: {field}")
                merged[field] = value
            geometry_by_id[object_id] = BoxGeometry.from_dict(merged).to_dict()

    removed = patch.get("geometry_remove", [])
    if not isinstance(removed, list):
        raise SerializationError("geometry_remove must be an array")
    for object_id in removed:
        if isinstance(object_id, bool) or not isinstance(object_id, int):
            raise SerializationError("geometry_remove entries must be integers")
        if object_id == 0:
            raise SerializationError("the floor object 0 cannot be removed")
        if object_id not in geometry_by_id:
            raise SerializationError(f"geometry_remove references unknown object id {object_id}")
        del geometry_by_id[object_id]

    additions = patch.get("geometry_add", [])
    if not isinstance(additions, list):
        raise SerializationError("geometry_add must be an array")
    next_id = max(geometry_by_id, default=-1) + 1
    for item in additions:
        if not isinstance(item, dict):
            raise SerializationError("geometry_add entries must be objects")
        candidate = dict(item)
        if "object_id" not in candidate:
            candidate["object_id"] = next_id
            next_id += 1
        else:
            candidate_id = candidate["object_id"]
            if isinstance(candidate_id, bool) or not isinstance(candidate_id, int):
                raise SerializationError("new geometry object_id must be an integer")
            if candidate_id in geometry_by_id:
                raise SerializationError(f"geometry_add reuses object id {candidate_id}")
            next_id = max(next_id, candidate_id + 1)
        box = BoxGeometry.from_dict(candidate)
        geometry_by_id[box.object_id] = box.to_dict()

    directional_dict = scene.directional.to_dict()
    directional_update = patch.get("directional_update")
    if directional_update is not None:
        if not isinstance(directional_update, dict):
            raise SerializationError("directional_update must be an object")
        for field, value in directional_update.items():
            if field == "light_id":
                raise SerializationError("directional_update cannot change light_id")
            if field not in directional_dict:
                raise SerializationError(f"unknown directional light field: {field}")
            directional_dict[field] = value
    directional = DirectionalLight.from_dict(directional_dict)

    points_by_id = {light.light_id: light.to_dict() for light in scene.point_lights}
    light_updates = patch.get("point_light_updates", {})
    if light_updates is not None:
        if not isinstance(light_updates, dict):
            raise SerializationError("point_light_updates must be an object keyed by light id")
        for light_id, fields in light_updates.items():
            _expected_light_id(light_id)
            if light_id not in points_by_id:
                raise SerializationError(f"point light update references unknown id {light_id!r}")
            if not isinstance(fields, dict):
                raise SerializationError("point light update fields must be an object")
            merged = dict(points_by_id[light_id])
            for field, value in fields.items():
                if field == "light_id":
                    raise SerializationError("point_light_updates cannot change light_id")
                if field not in merged:
                    raise SerializationError(f"unknown point light field: {field}")
                merged[field] = value
            points_by_id[light_id] = PointLight.from_dict(merged).to_dict()

    light_removed = patch.get("point_light_remove", [])
    if not isinstance(light_removed, list):
        raise SerializationError("point_light_remove must be an array")
    for light_id in light_removed:
        _expected_light_id(light_id)
        if light_id not in points_by_id:
            raise SerializationError(f"point_light_remove references unknown id {light_id!r}")
        del points_by_id[light_id]

    light_add = patch.get("point_light_add", [])
    if not isinstance(light_add, list):
        raise SerializationError("point_light_add must be an array")
    for item in light_add:
        if not isinstance(item, dict):
            raise SerializationError("point_light_add entries must be objects")
        light = PointLight.from_dict(item)
        if light.light_id == directional.light_id:
            raise SerializationError("point_light_add collides with the directional light id")
        if light.light_id in points_by_id:
            raise SerializationError(f"point_light_add reuses id {light.light_id!r}")
        if len(points_by_id) >= 4:
            raise SerializationError("a scene may contain at most four point lights")
        points_by_id[light.light_id] = light.to_dict()
    if len(points_by_id) > 4:
        raise SerializationError("a scene may contain at most four point lights")

    geometry = tuple(sorted((BoxGeometry.from_dict(value) for value in geometry_by_id.values()), key=lambda box: box.object_id))
    point_lights = tuple(sorted((PointLight.from_dict(value) for value in points_by_id.values()), key=lambda light: light.light_id))
    return Scene(
        seed=scene.seed,
        generation=dict(scene.generation),
        bounds_min=scene.bounds_min,
        bounds_max=scene.bounds_max,
        geometry=geometry,
        directional=directional,
        point_lights=point_lights,
    )


def changed_component_ids(old_scene: Scene, new_scene: Scene) -> tuple[set[int], set[str]]:
    old_geometry = old_scene.geometry_hashes()
    new_geometry = new_scene.geometry_hashes()
    geometry_ids: set[int] = set()
    for key in set(old_geometry) | set(new_geometry):
        if old_geometry.get(key) != new_geometry.get(key):
            geometry_ids.add(int(key))

    old_lights = old_scene.light_hashes()
    new_lights = new_scene.light_hashes()
    light_ids: set[str] = set()
    for key in set(old_lights) | set(new_lights):
        if old_lights.get(key) != new_lights.get(key):
            light_ids.add(key)
    return geometry_ids, light_ids


def geometry_change_kinds(old_scene: Scene, new_scene: Scene) -> tuple[set[int], set[int]]:
    """Return (material-only ids, shape ids).

    Shape changes can alter hits and shadow rays.  Albedo/emission-only changes
    can only alter radiance when the probe's ray actually hits that box.
    """

    old_boxes = {box.object_id: box for box in old_scene.geometry}
    new_boxes = {box.object_id: box for box in new_scene.geometry}
    material_ids: set[int] = set()
    shape_ids: set[int] = set()
    for object_id in set(old_boxes) | set(new_boxes):
        old = old_boxes.get(object_id)
        new = new_boxes.get(object_id)
        if old is None or new is None:
            shape_ids.add(object_id)
            material_ids.add(object_id)
            continue
        if (
            old.center != new.center
            or old.half_size != new.half_size
            or old.rotation_deg != new.rotation_deg
        ):
            shape_ids.add(object_id)
        if old.albedo != new.albedo or old.emission != new.emission:
            material_ids.add(object_id)
    return material_ids, shape_ids


def scene_json(scene: Scene) -> str:
    return canonical_json(scene.to_dict())

