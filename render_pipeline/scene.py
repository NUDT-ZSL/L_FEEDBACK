"""Scene parsing, validation and deterministic signatures."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .errors import E_SCENE_INVALID, PipelineError
from .math3d import Projection, Vec3, cross, make_basis, normalize, sub


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _vector(value: Any, length: int = 3) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    converted = tuple(_number(item) for item in value)
    if any(item is None for item in converted):
        return None
    return tuple(float(item) for item in converted if item is not None)


def _vec3(value: Any, default: Vec3 | None = None) -> Vec3:
    parsed = _vector(value)
    if parsed is None:
        if default is not None:
            return default
        raise PipelineError(E_SCENE_INVALID, f"expected a finite 3D vector, got {value!r}", "scene")
    return (parsed[0], parsed[1], parsed[2])


def _positive_int(value: Any, name: str, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PipelineError(E_SCENE_INVALID, f"{name} must be a positive integer", "scene")
    return value


def _nonnegative_float(value: Any, name: str, default: float) -> float:
    if value is None:
        return default
    converted = _number(value)
    if converted is None or converted < 0.0:
        raise PipelineError(E_SCENE_INVALID, f"{name} must be a finite non-negative number", "scene")
    return converted


def _odd_kernel(value: Any, name: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PipelineError(E_SCENE_INVALID, f"{name} must be a positive integer", "scene")
    return value if value % 2 == 1 else value + 1


@dataclass(frozen=True)
class TileSpec:
    index: int
    x: int
    y: int
    width: int
    height: int

    @property
    def tile_id(self) -> str:
        return f"tile-{self.x:04d}-{self.y:04d}"

    def as_dict(self) -> dict[str, int | str]:
        return {
            "index": self.index,
            "id": self.tile_id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class Material:
    index: int
    material_id: str
    albedo: Vec3
    emissive: Vec3
    roughness: float
    metallic: float
    specular: float
    shininess: float
    problem: str | None = None

    def definition(self) -> dict[str, Any]:
        return {
            "id": self.material_id,
            "albedo": list(self.albedo),
            "emissive": list(self.emissive),
            "roughness": self.roughness,
            "metallic": self.metallic,
            "specular": self.specular,
            "shininess": self.shininess,
            "problem": self.problem,
        }

    def signature(self) -> str:
        return digest(self.definition())


@dataclass(frozen=True)
class Triangle:
    index: int
    positions: tuple[Vec3, Vec3, Vec3]
    normals: tuple[Vec3, Vec3, Vec3]
    material_index: int
    material_error: str | None = None

    def bounds(self) -> tuple[Vec3, Vec3]:
        xs = (self.positions[0][0], self.positions[1][0], self.positions[2][0])
        ys = (self.positions[0][1], self.positions[1][1], self.positions[2][1])
        zs = (self.positions[0][2], self.positions[1][2], self.positions[2][2])
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


@dataclass(frozen=True)
class Light:
    index: int
    light_id: str
    kind: str
    color: Vec3
    intensity: float
    position: Vec3
    direction: Vec3
    range: float | None
    resolution: int | None
    pcf_kernel: int | None
    bias: float | None
    normal_bias: float | None

    def definition(self) -> dict[str, Any]:
        return {
            "id": self.light_id,
            "type": self.kind,
            "color": list(self.color),
            "intensity": self.intensity,
            "position": list(self.position),
            "direction": list(self.direction),
            "range": self.range,
            "resolution": self.resolution,
            "pcf_kernel": self.pcf_kernel,
            "bias": self.bias,
            "normal_bias": self.normal_bias,
        }

    def color_definition(self) -> dict[str, Any]:
        return {
            "color": list(self.color),
            "intensity": self.intensity,
        }

    def shadow_definition(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "position": list(self.position),
            "direction": list(self.direction),
            "range": self.range,
            "resolution": self.resolution,
            "pcf_kernel": self.pcf_kernel,
            "bias": self.bias,
            "normal_bias": self.normal_bias,
        }

    def color_signature(self) -> str:
        return digest(self.color_definition())

    def shadow_signature(self) -> str:
        return digest(self.shadow_definition())

    def state(self) -> dict[str, Any]:
        return self.definition()


@dataclass(frozen=True)
class ShadowConfig:
    quality: str
    directional_resolution: int
    point_resolution: int
    pcf_kernel: int
    bias: float
    normal_bias: float
    depth_epsilon: float

    def effective_resolution(self, light: Light) -> int:
        if light.resolution is not None:
            base_resolution = light.resolution
        elif light.kind == "point":
            base_resolution = self.point_resolution
        else:
            base_resolution = self.directional_resolution
        if self.quality == "fast":
            return max(1, base_resolution // 2)
        return base_resolution

    def effective_pcf_kernel(self, light: Light) -> int:
        if self.quality == "fast":
            return 1
        if light.pcf_kernel is not None:
            return light.pcf_kernel if light.pcf_kernel % 2 == 1 else light.pcf_kernel + 1
        return self.pcf_kernel

    def effective_bias(self, light: Light) -> float:
        return self.bias if light.bias is None else light.bias

    def effective_normal_bias(self, light: Light) -> float:
        return self.normal_bias if light.normal_bias is None else light.normal_bias

    def as_dict(self) -> dict[str, Any]:
        return {
            "quality": self.quality,
            "directional_resolution": self.directional_resolution,
            "point_resolution": self.point_resolution,
            "pcf_kernel": self.pcf_kernel,
            "bias": self.bias,
            "normal_bias": self.normal_bias,
            "depth_epsilon": self.depth_epsilon,
        }


@dataclass(frozen=True)
class Scene:
    scene_id: str
    version: int
    width: int
    height: int
    tile_width: int
    tile_height: int
    camera: Projection
    background: Vec3
    ambient: float
    materials: tuple[Material, ...]
    triangles: tuple[Triangle, ...]
    lights: tuple[Light, ...]
    shadow: ShadowConfig
    memory_limit_bytes: int | None
    bounds_min: Vec3
    bounds_max: Vec3
    source_path: str | None = None

    @property
    def tiles(self) -> tuple[TileSpec, ...]:
        tiles: list[TileSpec] = []
        index = 0
        y = 0
        while y < self.height:
            x = 0
            tile_height = min(self.tile_height, self.height - y)
            while x < self.width:
                tile_width = min(self.tile_width, self.width - x)
                tiles.append(TileSpec(index, x, y, tile_width, tile_height))
                index += 1
                x += tile_width
            y += tile_height
        return tuple(tiles)

    def structural_definition(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "version": self.version,
            "frame": {"width": self.width, "height": self.height},
            "tile": {"width": self.tile_width, "height": self.tile_height},
            "camera": {
                "eye": list(self.camera.basis.eye),
                "right": list(self.camera.basis.right),
                "up": list(self.camera.basis.up),
                "forward": list(self.camera.basis.forward),
                "kind": self.camera.projection_kind,
                "near": self.camera.near,
                "far": self.camera.far,
                "tan_half_fov_y": self.camera.tan_half_fov_y,
                "aspect": self.camera.aspect,
                "ortho_half_width": self.camera.ortho_half_width,
                "ortho_half_height": self.camera.ortho_half_height,
            },
            "background": list(self.background),
            "ambient": self.ambient,
            "materials": [material.material_id for material in self.materials],
            "triangles": [
                {
                    "positions": [list(position) for position in triangle.positions],
                    "normals": [list(normal) for normal in triangle.normals],
                    "material": triangle.material_index,
                    "material_error": triangle.material_error,
                }
                for triangle in self.triangles
            ],
            "lights": [{"id": light.light_id, "type": light.kind} for light in self.lights],
            "shadow": self.shadow.as_dict(),
        }

    def material_signatures(self) -> dict[str, str]:
        return {material.material_id: material.signature() for material in self.materials}

    def light_color_signatures(self) -> dict[str, str]:
        return {light.light_id: light.color_signature() for light in self.lights}

    def light_shadow_signatures(self) -> dict[str, str]:
        return {light.light_id: light.shadow_signature() for light in self.lights}

    def light_signatures(self) -> dict[str, str]:
        return {light.light_id: digest(light.definition()) for light in self.lights}

    def base_signature(self) -> str:
        return digest(self.structural_definition())

    def summary(self) -> str:
        payload = {
            "structural": self.structural_definition(),
            "materials": [material.definition() for material in self.materials],
            "lights": [light.definition() for light in self.lights],
            "memory_limit_bytes": self.memory_limit_bytes,
        }
        return digest(payload)

    def material(self, index: int) -> Material | None:
        if 0 <= index < len(self.materials):
            return self.materials[index]
        return None

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "version": self.version,
            "summary": self.summary(),
            "base_signature": self.base_signature(),
            "materials": self.material_signatures(),
            "light_colors": self.light_color_signatures(),
            "light_shadows": self.light_shadow_signatures(),
            "lights": {light.light_id: light.state() for light in self.lights},
        }


def _parse_material(raw: Any, index: int, fallback_id: str | None = None) -> Material:
    if not isinstance(raw, dict):
        raise PipelineError(E_SCENE_INVALID, f"material {index} must be an object", "scene")
    material_id = str(raw.get("id", raw.get("name", fallback_id or f"material-{index}")))
    problems: list[str] = []
    albedo_raw = raw.get(
        "albedo",
        raw.get("diffuse", raw.get("diffuse_albedo", raw.get("color", [0.8, 0.8, 0.8]))),
    )
    albedo = _vector(albedo_raw)
    if albedo is None:
        albedo = (0.8, 0.8, 0.8)
        problems.append("albedo must be three finite numbers")
    emissive = _vector(raw.get("emissive", [0.0, 0.0, 0.0]))
    if emissive is None:
        emissive = (0.0, 0.0, 0.0)
        problems.append("emissive must be three finite numbers")
    roughness = _number(raw.get("roughness", 0.5))
    metallic = _number(raw.get("metallic", 0.0))
    specular = _number(raw.get("specular", 0.25))
    shininess = _number(raw.get("shininess", 32.0))
    if roughness is None or not 0.0 <= roughness <= 1.0:
        problems.append("roughness must be in [0, 1]")
        roughness = 0.5 if roughness is None else roughness
    if metallic is None or not 0.0 <= metallic <= 1.0:
        problems.append("metallic must be in [0, 1]")
        metallic = 0.0 if metallic is None else metallic
    if specular is None or not 0.0 <= specular <= 1.0:
        problems.append("specular must be in [0, 1]")
        specular = 0.25 if specular is None else specular
    if shininess is None or shininess <= 0.0:
        problems.append("shininess must be positive")
        shininess = 32.0 if shininess is None else shininess
    for component in albedo:
        if component < 0.0 or component > 1.0:
            problems.append("albedo components must be in [0, 1]")
            break
    for component in emissive:
        if component < 0.0 or component > 1.0:
            problems.append("emissive components must be in [0, 1]")
            break
    return Material(
        index=index,
        material_id=material_id,
        albedo=(albedo[0], albedo[1], albedo[2]),
        emissive=(emissive[0], emissive[1], emissive[2]),
        roughness=float(roughness),
        metallic=float(metallic),
        specular=float(specular),
        shininess=float(shininess),
        problem="; ".join(problems) if problems else None,
    )


def _parse_material_table(raw_scene: dict[str, Any]) -> tuple[tuple[Material, ...], dict[str, int]]:
    raw_materials = raw_scene.get("materials", [])
    materials: list[Material] = []
    if isinstance(raw_materials, dict):
        for material_id in sorted(raw_materials):
            raw_material = dict(raw_materials[material_id])
            raw_material.setdefault("id", material_id)
            materials.append(_parse_material(raw_material, len(materials), material_id))
    elif isinstance(raw_materials, list):
        for index, raw_material in enumerate(raw_materials):
            materials.append(_parse_material(raw_material, index))
    elif raw_materials is None:
        materials.append(_parse_material({}, 0, "default"))
    else:
        raise PipelineError(E_SCENE_INVALID, "materials must be a list or object", "scene")
    if not materials:
        materials.append(_parse_material({}, 0, "default"))
    lookup: dict[str, int] = {}
    for material in materials:
        if material.material_id in lookup:
            raise PipelineError(E_SCENE_INVALID, f"duplicate material id {material.material_id!r}", "scene")
        lookup[material.material_id] = material.index
    return tuple(materials), lookup


def _resolve_material(value: Any, lookup: dict[str, int], default_index: int | None) -> tuple[int, str | None]:
    if value is None:
        if default_index is None:
            return -1, "triangle has no material"
        return default_index, None
    if isinstance(value, int) and not isinstance(value, bool):
        if 0 <= value < len(lookup):
            return value, None
        return -1, f"material index {value} is out of range"
    material_id = str(value)
    if material_id not in lookup:
        return -1, f"unknown material {material_id!r}"
    return lookup[material_id], None


def _triangle_from_raw(
    raw: Any,
    index: int,
    lookup: dict[str, int],
    default_material: int | None,
) -> Triangle:
    if isinstance(raw, (list, tuple)):
        if len(raw) not in (3, 4):
            raise PipelineError(
                E_SCENE_INVALID,
                f"triangle {index} array must contain three vertices and optional material",
                "scene",
            )
        material_value = raw[3] if len(raw) == 4 else None
        raw = {"positions": list(raw[:3])}
        if material_value is not None:
            raw["material"] = material_value
    if not isinstance(raw, dict):
        raise PipelineError(E_SCENE_INVALID, f"triangle {index} must be an object", "scene")
    if "positions" in raw:
        positions_raw = raw["positions"]
    elif "vertices" in raw:
        positions_raw = raw["vertices"]
    elif all(key in raw for key in ("a", "b", "c")):
        positions_raw = [raw["a"], raw["b"], raw["c"]]
    else:
        raise PipelineError(E_SCENE_INVALID, f"triangle {index} has no vertices", "scene")
    if not isinstance(positions_raw, (list, tuple)) or len(positions_raw) != 3:
        raise PipelineError(E_SCENE_INVALID, f"triangle {index} must have three vertices", "scene")
    positions = tuple(_vec3(item) for item in positions_raw)
    normals_raw = raw.get("normals")
    if normals_raw is None and all(key in raw for key in ("normal_a", "normal_b", "normal_c")):
        normals_raw = [raw["normal_a"], raw["normal_b"], raw["normal_c"]]
    if normals_raw is None:
        geometric = cross(sub(positions[1], positions[0]), sub(positions[2], positions[0]))
        flat_normal = normalize(geometric)
        normals = (flat_normal, flat_normal, flat_normal)
    else:
        if not isinstance(normals_raw, (list, tuple)) or len(normals_raw) != 3:
            raise PipelineError(E_SCENE_INVALID, f"triangle {index} normals must have three entries", "scene")
        normals = tuple(normalize(_vec3(item)) for item in normals_raw)
    material_index, material_error = _resolve_material(
        raw.get(
            "material",
            raw.get("material_id", raw.get("material_index")),
        ),
        lookup,
        default_material,
    )
    return Triangle(
        index=index,
        positions=(positions[0], positions[1], positions[2]),
        normals=(normals[0], normals[1], normals[2]),
        material_index=material_index,
        material_error=material_error,
    )


def _mesh_triangles(raw_mesh: Any, lookup: dict[str, int], start_index: int) -> list[Triangle]:
    if not isinstance(raw_mesh, dict):
        raise PipelineError(E_SCENE_INVALID, "mesh must be an object", "scene")
    material_value = raw_mesh.get("material", raw_mesh.get("material_id"))
    default_material, material_error = _resolve_material(material_value, lookup, None)
    if "triangles" in raw_mesh:
        raw_triangles = raw_mesh["triangles"]
        if not isinstance(raw_triangles, list):
            raise PipelineError(E_SCENE_INVALID, "mesh triangles must be a list", "scene")
        parsed: list[Triangle] = []
        for offset, raw_triangle in enumerate(raw_triangles):
            triangle = _triangle_from_raw(raw_triangle, start_index + offset, lookup, default_material)
            if triangle.material_error is None and material_error is not None:
                triangle = Triangle(
                    triangle.index,
                    triangle.positions,
                    triangle.normals,
                    triangle.material_index,
                    material_error,
                )
            parsed.append(triangle)
        return parsed

    vertices_raw = raw_mesh.get("vertices")
    indices_raw = raw_mesh.get("indices", raw_mesh.get("faces"))
    if not isinstance(vertices_raw, list):
        raise PipelineError(E_SCENE_INVALID, "mesh needs triangles or a vertices list", "scene")
    vertices = tuple(_vec3(vertex) for vertex in vertices_raw)
    if indices_raw is None:
        if len(vertices) % 3 != 0:
            raise PipelineError(E_SCENE_INVALID, "mesh vertex count must be divisible by 3", "scene")
        index_groups: Iterable[Any] = (range(0, len(vertices), 3),)
    else:
        if not isinstance(indices_raw, list) or len(indices_raw) % 3 != 0:
            raise PipelineError(E_SCENE_INVALID, "mesh indices must be a list divisible by 3", "scene")
        index_groups = (indices_raw,)
    normals_raw = raw_mesh.get("normals")
    vertex_normals: tuple[Vec3, ...] | None = None
    if normals_raw is not None:
        if not isinstance(normals_raw, list) or len(normals_raw) != len(vertices):
            raise PipelineError(E_SCENE_INVALID, "mesh normals must match vertex count", "scene")
        vertex_normals = tuple(normalize(_vec3(normal)) for normal in normals_raw)
    parsed = []
    for group in index_groups:
        values = list(group)
        for offset in range(0, len(values), 3):
            raw_indices = values[offset : offset + 3]
            if any(isinstance(item, bool) or not isinstance(item, int) for item in raw_indices):
                raise PipelineError(E_SCENE_INVALID, "mesh indices must be integers", "scene")
            if any(item < 0 or item >= len(vertices) for item in raw_indices):
                raise PipelineError(E_SCENE_INVALID, "mesh index out of range", "scene")
            positions = tuple(vertices[item] for item in raw_indices)
            if vertex_normals is None:
                flat_normal = normalize(cross(sub(positions[1], positions[0]), sub(positions[2], positions[0])))
                normals = (flat_normal, flat_normal, flat_normal)
            else:
                normals = tuple(vertex_normals[item] for item in raw_indices)
            parsed.append(
                Triangle(
                    index=start_index + len(parsed),
                    positions=(positions[0], positions[1], positions[2]),
                    normals=(normals[0], normals[1], normals[2]),
                    material_index=default_material,
                    material_error=material_error,
                )
            )
    return parsed


def _parse_triangles(raw_scene: dict[str, Any], lookup: dict[str, int]) -> tuple[Triangle, ...]:
    triangles: list[Triangle] = []
    default_material = 0 if lookup else None
    top_level = raw_scene.get("triangles", [])
    if top_level is not None:
        if not isinstance(top_level, list):
            raise PipelineError(E_SCENE_INVALID, "triangles must be a list", "scene")
        for raw_triangle in top_level:
            triangles.append(
                _triangle_from_raw(raw_triangle, len(triangles), lookup, default_material)
            )
    meshes = raw_scene.get(
        "meshes", raw_scene.get("objects", raw_scene.get("mesh", []))
    )
    if meshes is not None:
        if isinstance(meshes, dict):
            meshes = [meshes]
        if not isinstance(meshes, list):
            raise PipelineError(E_SCENE_INVALID, "meshes must be a list", "scene")
        for raw_mesh in meshes:
            triangles.extend(_mesh_triangles(raw_mesh, lookup, len(triangles)))
    return tuple(triangles)


def _parse_light(raw: Any, index: int, fallback_id: str | None = None) -> Light:
    if not isinstance(raw, dict):
        raise PipelineError(E_SCENE_INVALID, f"light {index} must be an object", "scene")
    light_id = str(raw.get("id", raw.get("name", fallback_id or f"light-{index}")))
    kind = str(raw.get("type", raw.get("kind", "directional"))).lower()
    if kind in {"sun", "directional", "directional_light", "dir"}:
        kind = "directional"
    elif kind in {"point", "point_light", "omni", "lamp"}:
        kind = "point"
    else:
        raise PipelineError(E_SCENE_INVALID, f"light {light_id!r} has unsupported type {kind!r}", "scene")
    color = _vec3(raw.get("color", raw.get("rgb", [1.0, 1.0, 1.0])))
    intensity = _number(raw.get("intensity", 1.0))
    if intensity is None or intensity < 0.0:
        raise PipelineError(E_SCENE_INVALID, f"light {light_id!r} intensity must be non-negative", "scene")
    position = _vec3(raw.get("position", [0.0, 0.0, 0.0]))
    direction = normalize(_vec3(raw.get("direction", [0.0, -1.0, 0.0])))
    range_value = raw.get("range")
    light_range: float | None
    if range_value is None:
        light_range = None
    else:
        converted_range = _number(range_value)
        if converted_range is None or converted_range <= 0.0:
            raise PipelineError(E_SCENE_INVALID, f"light {light_id!r} range must be positive", "scene")
        light_range = converted_range
    resolution_raw = raw.get("resolution", raw.get("shadow_resolution"))
    resolution = None if resolution_raw is None else _positive_int(resolution_raw, f"light {light_id!r} resolution")
    pcf_raw = raw.get("pcf_kernel", raw.get("pcf_size"))
    pcf_kernel = None if pcf_raw is None else _odd_kernel(pcf_raw, f"light {light_id!r} pcf kernel", 1)
    bias_raw = raw.get("bias", raw.get("shadow_bias"))
    bias = None if bias_raw is None else _nonnegative_float(bias_raw, f"light {light_id!r} bias", 0.0)
    normal_bias_raw = raw.get("normal_bias")
    normal_bias = (
        None
        if normal_bias_raw is None
        else _nonnegative_float(normal_bias_raw, f"light {light_id!r} normal bias", 0.0)
    )
    return Light(
        index=index,
        light_id=light_id,
        kind=kind,
        color=color,
        intensity=float(intensity),
        position=position,
        direction=direction,
        range=light_range,
        resolution=resolution,
        pcf_kernel=pcf_kernel,
        bias=bias,
        normal_bias=normal_bias,
    )


def _parse_lights(raw_scene: dict[str, Any]) -> tuple[Light, ...]:
    raw_lights = raw_scene.get("lights", [])
    lights: list[Light] = []
    if isinstance(raw_lights, dict):
        for light_id in sorted(raw_lights):
            raw_light = dict(raw_lights[light_id])
            raw_light.setdefault("id", light_id)
            lights.append(_parse_light(raw_light, len(lights), light_id))
    elif isinstance(raw_lights, list):
        for index, raw_light in enumerate(raw_lights):
            lights.append(_parse_light(raw_light, index))
    elif raw_lights is None:
        lights = []
    else:
        raise PipelineError(E_SCENE_INVALID, "lights must be a list or object", "scene")
    seen: set[str] = set()
    for light in lights:
        if light.light_id in seen:
            raise PipelineError(E_SCENE_INVALID, f"duplicate light id {light.light_id!r}", "scene")
        seen.add(light.light_id)
    return tuple(lights)


def _parse_shadow(raw_scene: dict[str, Any]) -> ShadowConfig:
    raw_shadow = raw_scene.get("shadow", {})
    if raw_shadow is None:
        raw_shadow = {}
    if not isinstance(raw_shadow, dict):
        raise PipelineError(E_SCENE_INVALID, "shadow must be an object", "scene")
    quality = str(raw_scene.get("quality", raw_shadow.get("quality", "high"))).lower()
    if quality in {"hq", "full", "high_quality"}:
        quality = "high"
    if quality in {"low", "single", "fast_mode"}:
        quality = "fast"
    if quality not in {"high", "fast"}:
        raise PipelineError(E_SCENE_INVALID, "quality must be 'high' or 'fast'", "scene")
    shared_resolution_raw = raw_scene.get(
        "shadow_map_resolution",
        raw_scene.get(
            "shadow_resolution",
            raw_shadow.get("resolution", raw_shadow.get("map_resolution")),
        ),
    )
    directional_raw = raw_shadow.get("directional_resolution", shared_resolution_raw)
    point_raw = raw_shadow.get("point_resolution", shared_resolution_raw)
    directional_resolution = _positive_int(directional_raw, "directional shadow resolution", 256)
    point_resolution = _positive_int(point_raw, "point shadow resolution", 128)
    pcf_raw = raw_scene.get("pcf_kernel", raw_shadow.get("pcf_kernel", raw_shadow.get("pcf_size")))
    pcf_kernel = _odd_kernel(pcf_raw, "pcf kernel", 5)
    bias = _nonnegative_float(
        raw_scene.get("shadow_bias", raw_shadow.get("bias", raw_shadow.get("depth_bias"))),
        "shadow bias",
        0.002,
    )
    normal_bias = _nonnegative_float(
        raw_scene.get("normal_bias", raw_shadow.get("normal_bias")),
        "normal bias",
        0.01,
    )
    depth_epsilon = _nonnegative_float(raw_shadow.get("depth_epsilon"), "depth epsilon", 1.0e-5)
    return ShadowConfig(
        quality=quality,
        directional_resolution=directional_resolution,
        point_resolution=point_resolution,
        pcf_kernel=pcf_kernel,
        bias=bias,
        normal_bias=normal_bias,
        depth_epsilon=depth_epsilon,
    )


def _parse_camera(raw_scene: dict[str, Any], width: int, height: int) -> Projection:
    raw_camera = raw_scene.get("camera", {})
    if raw_camera is None:
        raw_camera = {}
    if not isinstance(raw_camera, dict):
        raise PipelineError(E_SCENE_INVALID, "camera must be an object", "scene")
    eye = _vec3(raw_camera.get("position", raw_camera.get("eye", [0.0, 1.2, 5.0])))
    target = _vec3(raw_camera.get("target", raw_camera.get("look_at", [0.0, 0.6, 0.0])))
    up_hint = normalize(_vec3(raw_camera.get("up", [0.0, 1.0, 0.0])))
    basis = make_basis(eye, target, up_hint)
    kind = str(raw_camera.get("projection", raw_camera.get("type", "perspective"))).lower()
    if kind in {"ortho", "orthographic"}:
        kind = "orthographic"
    elif kind in {"perspective", "persp"}:
        kind = "perspective"
    else:
        raise PipelineError(E_SCENE_INVALID, "camera projection must be perspective or orthographic", "scene")
    near = _nonnegative_float(raw_camera.get("near"), "camera near", 0.05)
    far = _nonnegative_float(raw_camera.get("far"), "camera far", 100.0)
    if near <= 0.0 or far <= near:
        raise PipelineError(E_SCENE_INVALID, "camera requires 0 < near < far", "scene")
    fov_raw = raw_camera.get(
        "fov_y_degrees", raw_camera.get("fov_y", raw_camera.get("fov", 60.0))
    )
    fov_degrees = _number(fov_raw)
    if fov_degrees is None or not 1.0 <= fov_degrees < 179.0:
        raise PipelineError(E_SCENE_INVALID, "camera fov must be in (0, 180) degrees", "scene")
    aspect = float(width) / float(height)
    tan_half = math.tan(math.radians(fov_degrees) * 0.5)
    ortho_height = _number(raw_camera.get("ortho_height", raw_camera.get("orthographic_scale", 2.0)))
    if ortho_height is None or ortho_height <= 0.0:
        raise PipelineError(E_SCENE_INVALID, "ortho_height must be positive", "scene")
    return Projection(
        width=width,
        height=height,
        basis=basis,
        projection_kind=kind,
        near=near,
        far=far,
        tan_half_fov_y=tan_half,
        aspect=aspect,
        ortho_half_width=ortho_height * aspect * 0.5,
        ortho_half_height=ortho_height * 0.5,
    )


def load_scene(path_or_data: str | Path | dict[str, Any]) -> Scene:
    if isinstance(path_or_data, dict):
        raw_scene = path_or_data
        source_path = None
    else:
        source_path = str(path_or_data)
        try:
            raw_scene = json.loads(Path(path_or_data).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PipelineError(E_SCENE_INVALID, f"cannot read scene: {error}", "scene") from error
    if not isinstance(raw_scene, dict):
        raise PipelineError(E_SCENE_INVALID, "scene root must be an object", "scene")
    raw_frame = raw_scene.get("frame", {})
    if raw_frame is None:
        raw_frame = {}
    if isinstance(raw_frame, (list, tuple)) and len(raw_frame) == 2:
        raw_frame = {"width": raw_frame[0], "height": raw_frame[1]}
    if not isinstance(raw_frame, dict):
        raise PipelineError(E_SCENE_INVALID, "frame must be an object", "scene")
    raw_resolution = raw_scene.get("resolution", raw_scene.get("frame_size"))
    if isinstance(raw_resolution, (list, tuple)) and len(raw_resolution) == 2:
        resolution_width, resolution_height = raw_resolution
    elif isinstance(raw_resolution, dict):
        resolution_width = raw_resolution.get("width")
        resolution_height = raw_resolution.get("height")
    else:
        resolution_width, resolution_height = None, None
    width = _positive_int(
        raw_scene.get(
            "width",
            raw_scene.get("frame_width", raw_frame.get("width", resolution_width)),
        ),
        "frame width",
    )
    height = _positive_int(
        raw_scene.get(
            "height",
            raw_scene.get("frame_height", raw_frame.get("height", resolution_height)),
        ),
        "frame height",
    )
    raw_tile = raw_scene.get("tile", raw_scene.get("tile_size", {}))
    if raw_tile is None:
        raw_tile = {}
    if isinstance(raw_tile, int) and not isinstance(raw_tile, bool):
        raw_tile = (raw_tile, raw_tile)
    if isinstance(raw_tile, (list, tuple)):
        if len(raw_tile) != 2:
            raise PipelineError(E_SCENE_INVALID, "tile list must contain width and height", "scene")
        tile_width_raw, tile_height_raw = raw_tile
    elif isinstance(raw_tile, dict):
        tile_width_raw = raw_scene.get("tile_width", raw_tile.get("width", 32))
        tile_height_raw = raw_scene.get("tile_height", raw_tile.get("height", 32))
    else:
        raise PipelineError(E_SCENE_INVALID, "tile must be an object or [width, height]", "scene")
    tile_width = min(width, _positive_int(tile_width_raw, "tile width"))
    tile_height = min(height, _positive_int(tile_height_raw, "tile height"))
    materials, lookup = _parse_material_table(raw_scene)
    triangles = _parse_triangles(raw_scene, lookup)
    lights = _parse_lights(raw_scene)
    shadow = _parse_shadow(raw_scene)
    camera = _parse_camera(raw_scene, width, height)
    background = _vec3(raw_scene.get("background", [0.015, 0.018, 0.025]))
    ambient_raw = raw_scene.get("ambient", 0.06)
    if isinstance(ambient_raw, list):
        ambient_vector = _vec3(ambient_raw)
        ambient = (ambient_vector[0] + ambient_vector[1] + ambient_vector[2]) / 3.0
    else:
        ambient_number = _number(ambient_raw)
        if ambient_number is None or ambient_number < 0.0:
            raise PipelineError(E_SCENE_INVALID, "ambient must be a non-negative number", "scene")
        ambient = ambient_number
    memory_limit_raw = raw_scene.get("memory_limit_bytes", raw_scene.get("tile_memory_limit_bytes"))
    memory_limit: int | None
    if memory_limit_raw is None:
        memory_limit_mb_raw = raw_scene.get(
            "memory_limit_mb", raw_scene.get("tile_memory_limit_mb")
        )
        if memory_limit_mb_raw is None:
            memory_limit = None
        else:
            memory_limit_mb = _number(memory_limit_mb_raw)
            if memory_limit_mb is None or memory_limit_mb <= 0.0:
                raise PipelineError(E_SCENE_INVALID, "memory limit MB must be positive", "scene")
            memory_limit = int(memory_limit_mb * 1024.0 * 1024.0)
    else:
        memory_limit = _positive_int(memory_limit_raw, "memory limit")
    scene_box = raw_scene.get("bounds")
    if scene_box is not None:
        if not isinstance(scene_box, dict):
            raise PipelineError(E_SCENE_INVALID, "bounds must be an object", "scene")
        bounds_min = _vec3(scene_box.get("min"))
        bounds_max = _vec3(scene_box.get("max"))
    elif triangles:
        all_positions = [position for triangle in triangles for position in triangle.positions]
        bounds_min = (
            min(position[0] for position in all_positions),
            min(position[1] for position in all_positions),
            min(position[2] for position in all_positions),
        )
        bounds_max = (
            max(position[0] for position in all_positions),
            max(position[1] for position in all_positions),
            max(position[2] for position in all_positions),
        )
    else:
        center = camera.basis.eye
        bounds_min = (center[0] - 0.5, center[1] - 0.5, center[2] - 0.5)
        bounds_max = (center[0] + 0.5, center[1] + 0.5, center[2] + 0.5)
    scene_id = str(raw_scene.get("scene_id", raw_scene.get("id", "scene")))
    version_raw = raw_scene.get("version", 1)
    if isinstance(version_raw, bool) or not isinstance(version_raw, int) or version_raw < 0:
        raise PipelineError(E_SCENE_INVALID, "scene version must be a non-negative integer", "scene")
    return Scene(
        scene_id=scene_id,
        version=version_raw,
        width=width,
        height=height,
        tile_width=tile_width,
        tile_height=tile_height,
        camera=camera,
        background=background,
        ambient=ambient,
        materials=materials,
        triangles=triangles,
        lights=lights,
        shadow=shadow,
        memory_limit_bytes=memory_limit,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        source_path=source_path,
    )
