from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .diagnostics import Diagnostic


SCHEMA_VERSIONS = {"1.0", "1.1"}
CANONICAL_LAYOUT = (
    "world_position",
    "world_normal",
    "albedo",
    "metallic_roughness",
)
PASS_KINDS = ("depth_prepass", "normal", "deferred")
PASS_ORDER = {name: index for index, name in enumerate(PASS_KINDS)}
UNIFORM_TYPES = {
    "float",
    "int",
    "uint",
    "bool",
    "vec2",
    "vec3",
    "vec4",
    "ivec2",
    "ivec3",
    "ivec4",
    "uvec2",
    "uvec3",
    "uvec4",
    "bvec2",
    "bvec3",
    "bvec4",
    "mat2",
    "mat3",
    "mat4",
}
UNIFORM_ALIASES = {
    "scalar": "float",
    "boolean": "bool",
    "vector2": "vec2",
    "vector3": "vec3",
    "vector4": "vec4",
    "matrix2": "mat2",
    "matrix3": "mat3",
    "matrix4": "mat4",
}
BUILTIN_MESHES = {"cube", "plane", "zero_normal_plane"}
BLEND_FACTORS = {
    "zero": 0,
    "one": 1,
    "src_color": 768,
    "one_minus_src_color": 769,
    "src_alpha": 770,
    "one_minus_src_alpha": 771,
    "dst_alpha": 772,
    "one_minus_dst_alpha": 773,
    "dst_color": 774,
    "one_minus_dst_color": 775,
    "src_alpha_saturate": 776,
}
BLEND_EQUATIONS = {
    "add": 32774,
    "subtract": 32778,
    "reverse_subtract": 32779,
    "min": 32775,
    "max": 32776,
}
DEPTH_FUNCS = {
    "never": "0",
    "less": "<",
    "equal": "==",
    "lequal": "<=",
    "greater": ">",
    "notequal": "!=",
    "gequal": ">=",
    "always": "1",
}
VERSION_DEFAULTS = {
    "1.0": {
        "width": 128,
        "height": 96,
        "background": [0.015, 0.018, 0.025, 1.0],
        "normal_fallback": [0.0, 0.0, 1.0],
        "normal_tolerance": 2.0e-4,
        "absolute": 5.0e-4,
        "relative": 2.0e-3,
        "unused_declared": "warn",
    },
    "1.1": {
        "width": 160,
        "height": 120,
        "background": [0.015, 0.018, 0.025, 1.0],
        "normal_fallback": [0.0, 0.0, 1.0],
        "normal_tolerance": 1.0e-4,
        "absolute": 1.0e-4,
        "relative": 1.0e-3,
        "unused_declared": "warn",
    },
}


class ManifestJsonError(ValueError):
    pass


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestJsonError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _diag(
    diagnostics: list[Diagnostic],
    severity: str,
    code: str,
    message: str,
    path: str | None = None,
    **kwargs: Any,
) -> None:
    diagnostics.append(
        Diagnostic(
            severity=severity,
            code=code,
            message=message,
            path=path,
            **kwargs,
        )
    )


def _unknown_fields(
    value: dict[str, Any],
    allowed: set[str],
    path: str,
    diagnostics: list[Diagnostic],
    retained: list[dict[str, Any]],
) -> None:
    for key, item in value.items():
        if key in allowed:
            continue
        field_path = f"{path}.{key}" if path else key
        retained.append({"path": field_path, "value": copy.deepcopy(item)})
        _diag(
            diagnostics,
            "warning",
            "unknown_field",
            f"Unknown field retained without interpretation: {field_path}",
            field_path,
        )


def _value_at_path(root: Any, path: str) -> tuple[bool, Any]:
    current = root
    token = ""
    index = 0
    while index < len(path):
        character = path[index]
        if character == ".":
            if token:
                if not isinstance(current, dict) or token not in current:
                    return False, None
                current = current[token]
                token = ""
        elif character == "[":
            if token:
                if not isinstance(current, dict) or token not in current:
                    return False, None
                current = current[token]
                token = ""
            end = path.find("]", index)
            if end < 0:
                return False, None
            try:
                array_index = int(path[index + 1 : end])
            except ValueError:
                return False, None
            if not isinstance(current, list) or array_index < 0 or array_index >= len(current):
                return False, None
            current = current[array_index]
            index = end
        else:
            token += character
        index += 1
    if token:
        if not isinstance(current, dict) or token not in current:
            return False, None
        current = current[token]
    return True, current


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(
    value: Any,
    path: str,
    diagnostics: list[Diagnostic],
    default: float | None = None,
) -> float | None:
    if value is None:
        return default
    if not _is_number(value) or not math.isfinite(float(value)):
        _diag(diagnostics, "error", "invalid_number", f"Expected a finite number at {path}", path)
        return default
    return float(value)


def _integer(
    value: Any,
    path: str,
    diagnostics: list[Diagnostic],
    default: int | None = None,
) -> int | None:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        _diag(diagnostics, "error", "invalid_integer", f"Expected an integer at {path}", path)
        return default
    return value


def _boolean(
    value: Any,
    path: str,
    diagnostics: list[Diagnostic],
    default: bool | None = None,
) -> bool | None:
    if value is None:
        return default
    if not isinstance(value, bool):
        _diag(diagnostics, "error", "invalid_boolean", f"Expected a boolean at {path}", path)
        return default
    return value


def _vector(
    value: Any,
    size: int,
    path: str,
    diagnostics: list[Diagnostic],
    default: list[float] | None = None,
) -> list[float] | None:
    if value is None:
        return copy.deepcopy(default)
    if not isinstance(value, list) or len(value) != size:
        _diag(
            diagnostics,
            "error",
            "invalid_vector",
            f"Expected a {size}-element numeric array at {path}",
            path,
        )
        return copy.deepcopy(default)
    result: list[float] = []
    for index, item in enumerate(value):
        number = _number(item, f"{path}[{index}]", diagnostics)
        if number is None:
            return copy.deepcopy(default)
        result.append(number)
    return result


def _normalize_type(type_name: Any, path: str, diagnostics: list[Diagnostic]) -> str | None:
    if not isinstance(type_name, str):
        _diag(diagnostics, "error", "invalid_uniform_type", f"Uniform type at {path} must be a string", path)
        return None
    normalized = UNIFORM_ALIASES.get(type_name.strip().lower(), type_name.strip().lower())
    if normalized not in UNIFORM_TYPES:
        _diag(
            diagnostics,
            "error",
            "unsupported_uniform_type",
            f"Unsupported uniform type '{type_name}' at {path}",
            path,
        )
        return None
    return normalized


def _normalize_uniform_value(
    type_name: str,
    value: Any,
    path: str,
    diagnostics: list[Diagnostic],
) -> Any:
    if type_name in {"float", "int", "uint"}:
        number = _number(value, path, diagnostics)
        if number is None:
            return 0.0
        if type_name in {"int", "uint"}:
            if not float(number).is_integer():
                _diag(diagnostics, "error", "invalid_uniform_value", f"Expected an integer value at {path}", path)
                return 0
            converted = int(number)
            if type_name == "uint" and converted < 0:
                _diag(diagnostics, "error", "invalid_uniform_value", f"Unsigned value is negative at {path}", path)
                return 0
            return converted
        return number
    if type_name == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        _diag(diagnostics, "error", "invalid_uniform_value", f"Expected a boolean value at {path}", path)
        return False
    if type_name.startswith("vec") or type_name.startswith("ivec") or type_name.startswith("uvec") or type_name.startswith("bvec"):
        size = int(type_name[-1])
        if type_name.startswith("bvec"):
            if not isinstance(value, list) or len(value) != size:
                _diag(
                    diagnostics,
                    "error",
                    "invalid_uniform_value",
                    f"Expected a {size}-element boolean array at {path}",
                    path,
                )
                return [False] * size
            converted_bools: list[bool] = []
            for index, item in enumerate(value):
                if isinstance(item, bool):
                    converted_bools.append(item)
                elif isinstance(item, int) and item in (0, 1):
                    converted_bools.append(bool(item))
                else:
                    _diag(
                        diagnostics,
                        "error",
                        "invalid_uniform_value",
                        f"Expected a boolean component at {path}[{index}]",
                        f"{path}[{index}]",
                    )
                    converted_bools.append(False)
            return converted_bools
        vector = _vector(value, size, path, diagnostics, [0.0] * size)
        if vector is None:
            vector = [0.0] * size
        if type_name.startswith("ivec") or type_name.startswith("uvec"):
            converted = []
            for index, item in enumerate(vector):
                if not item.is_integer():
                    _diag(
                        diagnostics,
                        "error",
                        "invalid_uniform_value",
                        f"Expected an integer component at {path}[{index}]",
                        f"{path}[{index}]",
                    )
                    item = 0.0
                integer = int(item)
                if type_name.startswith("uvec") and integer < 0:
                    _diag(
                        diagnostics,
                        "error",
                        "invalid_uniform_value",
                        f"Unsigned component is negative at {path}[{index}]",
                        f"{path}[{index}]",
                    )
                    integer = 0
                converted.append(integer)
            return converted
        return vector
    if type_name.startswith("mat"):
        size = int(type_name[-1])
        if not isinstance(value, list) or len(value) != size or any(not isinstance(row, list) for row in value):
            _diag(
                diagnostics,
                "error",
                "invalid_uniform_value",
                f"Expected a {size}x{size} row-major matrix at {path}",
                path,
            )
            return [[1.0 if row == column else 0.0 for column in range(size)] for row in range(size)]
        matrix: list[list[float]] = []
        for row_index, row in enumerate(value):
            if len(row) != size:
                _diag(
                    diagnostics,
                    "error",
                    "invalid_uniform_value",
                    f"Expected {size} columns at {path}[{row_index}]",
                    f"{path}[{row_index}]",
                )
                matrix.append([0.0] * size)
                continue
            matrix.append(
                [
                    _number(component, f"{path}[{row_index}][{column_index}]", diagnostics, 0.0)
                    for column_index, component in enumerate(row)
                ]
            )
        return matrix
    _diag(diagnostics, "error", "unsupported_uniform_type", f"No value parser for {type_name}", path)
    return value


@dataclass
class UniformSpec:
    name: str
    type: str
    value: Any
    path: str
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "value": copy.deepcopy(self.value), "path": self.path}


@dataclass
class BlendSpec:
    enabled: bool = False
    src_factor: str = "one"
    dst_factor: str = "zero"
    equation: str = "add"

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "src_factor": self.src_factor,
            "dst_factor": self.dst_factor,
            "equation": self.equation,
        }


@dataclass
class DepthSpec:
    test: bool = True
    write: bool = True
    func: str = "lequal"

    def to_dict(self) -> dict[str, Any]:
        return {"test": self.test, "write": self.write, "func": self.func}


@dataclass
class MaterialRefSpec:
    name: str
    path: str
    channel: str
    uniforms: dict[str, UniformSpec] = field(default_factory=dict)


@dataclass
class ChannelSpec:
    name: str
    kind: str
    materials: list[MaterialRefSpec]
    layout: list[str]
    path: str
    declared_index: int

    def material_names(self) -> list[str]:
        return [item.name for item in self.materials]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "declared_index": self.declared_index,
            "materials": self.material_names(),
            "layout": list(self.layout),
            "path": self.path,
        }


@dataclass
class MaterialSpec:
    name: str
    vertex_shader: str
    fragment_shader: str
    vertex_path: Path
    fragment_path: Path
    uniforms: dict[str, UniformSpec]
    blend: BlendSpec
    depth: DepthSpec
    passes: list[str]
    raw: dict[str, Any]
    path: str
    channel_names: list[str] = field(default_factory=list)
    config_valid: bool = True
    renderable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "channels": list(self.channel_names),
            "vertex_shader": self.vertex_shader,
            "fragment_shader": self.fragment_shader,
            "uniforms": {name: spec.to_dict() for name, spec in self.uniforms.items()},
            "blend": self.blend.to_dict(),
            "depth": self.depth.to_dict(),
            "passes": list(self.passes),
            "config_valid": self.config_valid,
        }


@dataclass
class CameraSpec:
    position: list[float]
    target: list[float]
    up: list[float]
    fov_y_deg: float
    near: float
    far: float
    path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": list(self.position),
            "target": list(self.target),
            "up": list(self.up),
            "fov_y_deg": self.fov_y_deg,
            "near": self.near,
            "far": self.far,
        }


@dataclass
class LightSpec:
    name: str
    type: str
    position: list[float]
    direction: list[float]
    color: list[float]
    intensity: float
    radius: float
    enabled: bool
    path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "position": list(self.position),
            "direction": list(self.direction),
            "color": list(self.color),
            "intensity": self.intensity,
            "radius": self.radius,
            "enabled": self.enabled,
        }


@dataclass
class ObjectSpec:
    name: str
    mesh: str
    material: str
    position: list[float]
    rotation: list[float]
    scale: list[float]
    path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mesh": self.mesh,
            "material": self.material,
            "position": list(self.position),
            "rotation": list(self.rotation),
            "scale": list(self.scale),
        }


@dataclass
class SceneSpec:
    name: str
    camera: CameraSpec
    lights: list[LightSpec]
    objects: list[ObjectSpec]
    path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "camera": self.camera.to_dict(),
            "lights": [item.to_dict() for item in self.lights],
            "objects": [item.to_dict() for item in self.objects],
        }


@dataclass
class RenderSettings:
    width: int
    height: int
    background: list[float]
    normal_fallback: list[float]
    normal_tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "background": list(self.background),
            "normal_fallback": list(self.normal_fallback),
            "normal_tolerance": self.normal_tolerance,
        }


@dataclass
class ComparisonSettings:
    absolute: float
    relative: float
    channels: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "absolute": self.absolute,
            "relative": self.relative,
            "channels": list(self.channels),
        }


@dataclass
class ParsedManifest:
    path: Path
    raw: dict[str, Any] | None
    schema_version: str
    render: RenderSettings
    comparison: ComparisonSettings
    uniform_policy: dict[str, str]
    materials: dict[str, MaterialSpec]
    channels: list[ChannelSpec]
    scenes: list[SceneSpec]
    diagnostics: list[Diagnostic]
    unknown_fields: list[dict[str, Any]]
    manifest_sha256: str | None

    def ordered_channels(self) -> list[ChannelSpec]:
        return sorted(self.channels, key=lambda item: (PASS_ORDER.get(item.kind, 99), item.declared_index))

    def errors(self) -> list[Diagnostic]:
        return [item for item in self.diagnostics if item.severity == "error"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_sha256": self.manifest_sha256,
            "render": self.render.to_dict(),
            "comparison": self.comparison.to_dict(),
            "uniform_policy": dict(self.uniform_policy),
            "materials": [item.to_dict() for item in self.materials.values()],
            "channels": [item.to_dict() for item in self.channels],
            "scenes": [item.to_dict() for item in self.scenes],
            "unknown_fields": copy.deepcopy(self.unknown_fields),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


def _version_string(value: Any) -> str | None:
    if value is None:
        return "1.0"
    if isinstance(value, str):
        normalized = value.strip()
        try:
            return f"{float(normalized):.1f}"
        except ValueError:
            return normalized
    if _is_number(value):
        return f"{float(value):.1f}"
    return None


def _parse_uniform_spec(
    name: str,
    raw: Any,
    path: str,
    diagnostics: list[Diagnostic],
    forced_type: str | None = None,
) -> UniformSpec | None:
    if not isinstance(name, str) or not name:
        _diag(diagnostics, "error", "invalid_uniform_name", f"Invalid uniform name at {path}", path)
        return None
    if not isinstance(raw, dict):
        _diag(
            diagnostics,
            "error",
            "invalid_uniform_declaration",
            f"Uniform {name} must be an object with type and value at {path}",
            path,
        )
        return None
    _unknown_fields(raw, {"type", "value"}, path, diagnostics, [])
    type_name = _normalize_type(raw.get("type"), f"{path}.type", diagnostics)
    if type_name is None:
        return None
    if forced_type is not None and type_name != forced_type:
        _diag(
            diagnostics,
            "error",
            "uniform_override_type_conflict",
            f"Override type {type_name} at {path} conflicts with declared type {forced_type}",
            f"{path}.type",
        )
        return None
    if "value" not in raw:
        _diag(diagnostics, "error", "missing_uniform_value", f"Uniform {name} has no value at {path}", path)
        return None
    value = _normalize_uniform_value(type_name, raw["value"], f"{path}.value", diagnostics)
    return UniformSpec(name=name, type=type_name, value=value, path=path, sources=[path])


def _infer_uniform_type(value: Any) -> str | None:
    if isinstance(value, bool):
        return "bool"
    if _is_number(value):
        return "float"
    if isinstance(value, list) and len(value) in (2, 3, 4) and all(_is_number(item) for item in value):
        return f"vec{len(value)}"
    if (
        isinstance(value, list)
        and len(value) in (2, 3, 4)
        and all(isinstance(row, list) and len(row) == len(value) for row in value)
    ):
        return f"mat{len(value)}"
    return None


def _parse_override(
    name: str,
    raw: Any,
    path: str,
    diagnostics: list[Diagnostic],
    base_type: str | None,
) -> UniformSpec | None:
    if isinstance(raw, dict) and ("type" in raw or "value" in raw):
        return _parse_uniform_spec(name, raw, path, diagnostics, forced_type=base_type)
    inferred = base_type or _infer_uniform_type(raw)
    if inferred is None:
        _diag(
            diagnostics,
            "error",
            "ambiguous_uniform_override",
            f"Cannot infer uniform type for override at {path}",
            path,
        )
        return None
    normalized = _normalize_uniform_value(inferred, raw, path, diagnostics)
    return UniformSpec(name=name, type=inferred, value=normalized, path=path, sources=[path])


def _parse_blend(
    raw: Any,
    path: str,
    diagnostics: list[Diagnostic],
) -> BlendSpec:
    result = BlendSpec()
    if raw is None:
        return result
    if not isinstance(raw, dict):
        _diag(diagnostics, "error", "invalid_blend", f"Blend state at {path} must be an object", path)
        return result
    _unknown_fields(raw, {"enabled", "src_factor", "dst_factor", "equation"}, path, diagnostics, [])
    result.enabled = bool(_boolean(raw.get("enabled"), f"{path}.enabled", diagnostics, False))
    src = raw.get("src_factor", "one")
    dst = raw.get("dst_factor", "zero")
    equation = raw.get("equation", "add")
    if not isinstance(src, str) or src.strip().lower() not in BLEND_FACTORS:
        _diag(diagnostics, "error", "unsupported_blend_factor", f"Unsupported source blend factor at {path}.src_factor", f"{path}.src_factor")
    else:
        result.src_factor = src.strip().lower()
    if not isinstance(dst, str) or dst.strip().lower() not in BLEND_FACTORS:
        _diag(diagnostics, "error", "unsupported_blend_factor", f"Unsupported destination blend factor at {path}.dst_factor", f"{path}.dst_factor")
    else:
        result.dst_factor = dst.strip().lower()
    if result.dst_factor == "src_alpha_saturate":
        _diag(
            diagnostics,
            "error",
            "invalid_blend_combination",
            "SRC_ALPHA_SATURATE cannot be used as a destination factor",
            f"{path}.dst_factor",
        )
    if not isinstance(equation, str) or equation.strip().lower() not in BLEND_EQUATIONS:
        _diag(diagnostics, "error", "unsupported_blend_equation", f"Unsupported blend equation at {path}.equation", f"{path}.equation")
    else:
        result.equation = equation.strip().lower()
    return result


def _parse_depth(
    raw: Any,
    path: str,
    diagnostics: list[Diagnostic],
) -> DepthSpec:
    result = DepthSpec()
    if raw is None:
        return result
    if not isinstance(raw, dict):
        _diag(diagnostics, "error", "invalid_depth", f"Depth state at {path} must be an object", path)
        return result
    _unknown_fields(raw, {"test", "write", "func"}, path, diagnostics, [])
    result.test = bool(_boolean(raw.get("test"), f"{path}.test", diagnostics, True))
    result.write = bool(_boolean(raw.get("write"), f"{path}.write", diagnostics, True))
    func = raw.get("func", "lequal")
    if not isinstance(func, str) or func.strip().lower() not in DEPTH_FUNCS:
        _diag(diagnostics, "error", "unsupported_depth_func", f"Unsupported depth function at {path}.func", f"{path}.func")
    else:
        result.func = func.strip().lower()
    return result


def _parse_passes(raw: Any, path: str, diagnostics: list[Diagnostic]) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        result = [str(item) for item in raw if isinstance(item, str)]
        if len(result) != len(raw):
            _diag(diagnostics, "error", "invalid_passes", f"Material passes at {path} must be pass-name strings", path)
    elif isinstance(raw, dict):
        result = [str(key) for key, enabled in raw.items() if enabled]
    else:
        _diag(diagnostics, "error", "invalid_passes", f"Material passes at {path} must be a list or object", path)
        return []
    for index, kind in enumerate(result):
        if kind not in PASS_KINDS:
            _diag(diagnostics, "error", "unknown_pass_kind", f"Unknown pass kind '{kind}' at {path}[{index}]", f"{path}[{index}]")
    return [item for item in result if item in PASS_KINDS]


def _parse_material(
    raw: Any,
    index: int,
    path: str,
    manifest_dir: Path,
    diagnostics: list[Diagnostic],
) -> MaterialSpec | None:
    if not isinstance(raw, dict):
        _diag(diagnostics, "error", "invalid_material", f"Material at {path} must be an object", path)
        return None
    _unknown_fields(
        raw,
        {"name", "vertex_shader", "fragment_shader", "uniforms", "blend", "depth", "passes", "description"},
        path,
        diagnostics,
        [],
    )
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        _diag(diagnostics, "error", "invalid_material_name", f"Material at {path} needs a non-empty name", f"{path}.name")
        return None
    name = name.strip()
    vertex = raw.get("vertex_shader")
    fragment = raw.get("fragment_shader")
    if not isinstance(vertex, str) or not vertex:
        _diag(diagnostics, "error", "missing_shader", f"Material {name} has no vertex_shader", f"{path}.vertex_shader", material=name)
        vertex = ""
    if not isinstance(fragment, str) or not fragment:
        _diag(diagnostics, "error", "missing_shader", f"Material {name} has no fragment_shader", f"{path}.fragment_shader", material=name)
        fragment = ""
    vertex_path = (manifest_dir / vertex).resolve() if vertex else manifest_dir / "__missing__.vert"
    fragment_path = (manifest_dir / fragment).resolve() if fragment else manifest_dir / "__missing__.frag"
    for stage, shader_path, relative in (
        ("vertex", vertex_path, vertex),
        ("fragment", fragment_path, fragment),
    ):
        if relative and not shader_path.is_file():
            _diag(
                diagnostics,
                "error",
                "shader_file_missing",
                f"{stage.capitalize()} shader file does not exist: {relative}",
                f"{path}.{stage}_shader",
                material=name,
                stage=stage,
            )
    uniforms: dict[str, UniformSpec] = {}
    raw_uniforms = raw.get("uniforms", {})
    if not isinstance(raw_uniforms, dict):
        _diag(diagnostics, "error", "invalid_uniforms", f"uniforms at {path} must be an object", f"{path}.uniforms", material=name)
        raw_uniforms = {}
    for uniform_name, uniform_raw in raw_uniforms.items():
        uniform = _parse_uniform_spec(
            uniform_name,
            uniform_raw,
            f"{path}.uniforms.{uniform_name}",
            diagnostics,
            forced_type=None,
        )
        if uniform is not None:
            uniforms[uniform_name] = uniform
    blend = _parse_blend(raw.get("blend"), f"{path}.blend", diagnostics)
    depth = _parse_depth(raw.get("depth"), f"{path}.depth", diagnostics)
    if blend.enabled and depth.write:
        _diag(
            diagnostics,
            "error",
            "blend_with_depth_write",
            f"Material {name} enables blending while depth.write is true",
            f"{path}.depth.write",
            material=name,
        )
    material = MaterialSpec(
        name=name,
        vertex_shader=vertex,
        fragment_shader=fragment,
        vertex_path=vertex_path,
        fragment_path=fragment_path,
        uniforms=uniforms,
        blend=blend,
        depth=depth,
        passes=_parse_passes(raw.get("passes"), f"{path}.passes", diagnostics),
        raw=raw,
        path=path,
    )
    for uniform_name in ("u_metallic", "u_roughness"):
        if uniform_name not in uniforms:
            continue
        uniform = uniforms[uniform_name]
        if uniform.type != "float":
            continue
        value = float(uniform.value)
        if value < 0.0 or value > 1.0:
            _diag(
                diagnostics,
                "error",
                f"{uniform_name[2:]}_out_of_range",
                f"{uniform_name}={value} in material {name} must be in [0, 1]",
                uniform.path,
                material=name,
                details={"value": value, "minimum": 0.0, "maximum": 1.0},
            )
    return material


def _parse_materials(
    raw: Any,
    manifest_dir: Path,
    diagnostics: list[Diagnostic],
) -> dict[str, MaterialSpec]:
    result: dict[str, MaterialSpec] = {}
    if raw is None:
        _diag(diagnostics, "error", "missing_materials", "Manifest has no materials", "materials")
        return result
    if isinstance(raw, dict):
        entries: list[tuple[str, Any]] = [(f"materials.{key}", value) for key, value in raw.items()]
        for entry_path, value in entries:
            if isinstance(value, dict) and "name" not in value:
                value = dict(value)
                value["name"] = entry_path.rsplit(".", 1)[-1]
    elif isinstance(raw, list):
        entries = [(f"materials[{index}]", value) for index, value in enumerate(raw)]
    else:
        _diag(diagnostics, "error", "invalid_materials", "materials must be an array or object", "materials")
        return result
    for entry_path, value in entries:
        material = _parse_material(value, 0, entry_path, manifest_dir, diagnostics)
        if material is None:
            continue
        if material.name in result:
            _diag(
                diagnostics,
                "error",
                "duplicate_material",
                f"Material name '{material.name}' is declared more than once",
                entry_path,
                material=material.name,
                details={"existing_path": result[material.name].path, "conflicting_path": entry_path},
            )
            result[material.name].config_valid = False
            continue
        result[material.name] = material
    return result


def _parse_channel_ref(
    raw: Any,
    path: str,
    channel: str,
    materials: dict[str, MaterialSpec],
    diagnostics: list[Diagnostic],
) -> MaterialRefSpec | None:
    if isinstance(raw, str):
        name = raw
        ref_path = path
        override_raw: dict[str, Any] = {}
    elif isinstance(raw, dict):
        name = raw.get("name")
        ref_path = path
        if not isinstance(name, str) or not name:
            _diag(diagnostics, "error", "invalid_material_ref", f"Material reference at {path} needs a name", f"{path}.name", channels=(channel,))
            return None
        _unknown_fields(raw, {"name", "uniforms"}, path, diagnostics, [])
        override_raw = raw.get("uniforms", {})
        if not isinstance(override_raw, dict):
            _diag(diagnostics, "error", "invalid_uniform_override", f"uniform overrides at {path} must be an object", f"{path}.uniforms", channels=(channel,))
            override_raw = {}
    else:
        _diag(diagnostics, "error", "invalid_material_ref", f"Material reference at {path} must be a name or object", path, channels=(channel,))
        return None
    if name not in materials:
        _diag(
            diagnostics,
            "error",
            "unknown_material_reference",
            f"Channel {channel} references unknown material '{name}'",
            path,
            material=name,
            channels=(channel,),
        )
        return None
    base = materials[name]
    overrides: dict[str, UniformSpec] = {}
    for uniform_name, value in override_raw.items():
        base_type = base.uniforms[uniform_name].type if uniform_name in base.uniforms else None
        uniform = _parse_override(
            uniform_name,
            value,
            f"{ref_path}.uniforms.{uniform_name}",
            diagnostics,
            base_type,
        )
        if uniform is not None:
            overrides[uniform_name] = uniform
    return MaterialRefSpec(name=name, path=path, channel=channel, uniforms=overrides)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _resolve_channel_overrides(
    materials: dict[str, MaterialSpec],
    channels: list[ChannelSpec],
    diagnostics: list[Diagnostic],
) -> None:
    by_material: dict[str, list[tuple[ChannelSpec, MaterialRefSpec]]] = {}
    for channel in channels:
        for ref in channel.materials:
            by_material.setdefault(ref.name, []).append((channel, ref))
    for name, refs in by_material.items():
        material = materials[name]
        candidates: dict[str, list[tuple[str, UniformSpec]]] = {}
        for uniform_name, spec in material.uniforms.items():
            candidates.setdefault(uniform_name, []).append((spec.path, spec))
        for _, ref in refs:
            for uniform_name, spec in ref.uniforms.items():
                candidates.setdefault(uniform_name, []).append((spec.path, spec))
        for uniform_name, values in candidates.items():
            fingerprints = {
                _canonical_json({"type": item.type, "value": item.value})
                for _, item in values
            }
            if len(fingerprints) > 1:
                sources = [source for source, _ in values]
                _diag(
                    diagnostics,
                    "error",
                    "channel_uniform_conflict",
                    f"Conflicting declarations for {name}.{uniform_name} across channel references",
                    values[0][1].path,
                    material=name,
                    details={"sources": sources},
                )
                material.config_valid = False
                continue
            chosen = values[-1][1]
            merged = copy.deepcopy(chosen)
            merged.sources = [source for source, _ in values]
            material.uniforms[uniform_name] = merged


def _parse_channel(
    raw: Any,
    index: int,
    path: str,
    materials: dict[str, MaterialSpec],
    diagnostics: list[Diagnostic],
) -> ChannelSpec | None:
    if not isinstance(raw, dict):
        _diag(diagnostics, "error", "invalid_channel", f"Channel at {path} must be an object", path)
        return None
    _unknown_fields(raw, {"name", "kind", "materials", "layout"}, path, diagnostics, [])
    kind = raw.get("kind", raw.get("name"))
    if not isinstance(kind, str) or kind not in PASS_KINDS:
        _diag(diagnostics, "error", "unknown_channel_kind", f"Unsupported channel kind at {path}", f"{path}.kind")
        return None
    name = raw.get("name", kind)
    if not isinstance(name, str) or not name:
        _diag(diagnostics, "error", "invalid_channel_name", f"Channel name at {path} must be a string", f"{path}.name")
        name = kind
    raw_materials = raw.get("materials")
    if not isinstance(raw_materials, list):
        _diag(diagnostics, "error", "invalid_channel_materials", f"Channel {name} materials must be an array", f"{path}.materials")
        raw_materials = []
    refs: list[MaterialRefSpec] = []
    seen: set[str] = set()
    for ref_index, ref_raw in enumerate(raw_materials):
        ref = _parse_channel_ref(
            ref_raw,
            f"{path}.materials[{ref_index}]",
            name,
            materials,
            diagnostics,
        )
        if ref is None:
            continue
        if ref.name in seen:
            _diag(
                diagnostics,
                "error",
                "duplicate_channel_material",
                f"Material '{ref.name}' is listed more than once in channel {name}",
                f"{path}.materials[{ref_index}]",
                material=ref.name,
                channels=(name,),
            )
            continue
        seen.add(ref.name)
        refs.append(ref)
    raw_layout = raw.get("layout")
    if raw_layout is None:
        layout = list(CANONICAL_LAYOUT)
    elif isinstance(raw_layout, list) and all(isinstance(item, str) for item in raw_layout):
        layout = list(raw_layout)
    elif isinstance(raw_layout, dict):
        ordered = [raw_layout.get(str(item)) for item in range(len(raw_layout))]
        if not all(isinstance(item, str) for item in ordered):
            _diag(diagnostics, "error", "invalid_channel_layout", f"Channel {name} layout has invalid slots", f"{path}.layout")
            layout = list(CANONICAL_LAYOUT)
        else:
            layout = [str(item) for item in ordered]
    else:
        _diag(diagnostics, "error", "invalid_channel_layout", f"Channel {name} layout must be an array or slot object", f"{path}.layout")
        layout = list(CANONICAL_LAYOUT)
    if tuple(layout) != CANONICAL_LAYOUT:
        _diag(
            diagnostics,
            "error",
            "gbuffer_layout_mismatch",
            f"Channel {name} layout {layout} does not match canonical {list(CANONICAL_LAYOUT)}",
            f"{path}.layout",
            channels=(name,),
            details={"expected": list(CANONICAL_LAYOUT), "actual": layout},
        )
    return ChannelSpec(
        name=name,
        kind=kind,
        materials=refs,
        layout=layout,
        path=path,
        declared_index=index,
    )


def _parse_channels(
    raw: Any,
    materials: dict[str, MaterialSpec],
    diagnostics: list[Diagnostic],
) -> list[ChannelSpec]:
    channels: list[ChannelSpec] = []
    if raw is not None:
        if not isinstance(raw, list):
            _diag(diagnostics, "error", "invalid_channels", "channels must be an array", "channels")
            return channels
        for index, item in enumerate(raw):
            channel = _parse_channel(item, index, f"channels[{index}]", materials, diagnostics)
            if channel is not None:
                channels.append(channel)
    else:
        assigned: dict[str, list[str]] = {kind: [] for kind in PASS_KINDS}
        for material in materials.values():
            passes = material.passes or list(PASS_KINDS)
            for kind in passes:
                assigned[kind].append(material.name)
        for index, kind in enumerate(PASS_KINDS):
            refs = [
                MaterialRefSpec(name=name, path=f"materials.{name}.passes", channel=kind)
                for name in assigned[kind]
            ]
            channels.append(
                ChannelSpec(
                    name=kind,
                    kind=kind,
                    materials=refs,
                    layout=list(CANONICAL_LAYOUT),
                    path="derived",
                    declared_index=index,
                )
            )
    seen_kinds: dict[str, ChannelSpec] = {}
    for channel in channels:
        if channel.kind in seen_kinds:
            _diag(
                diagnostics,
                "error",
                "duplicate_channel_kind",
                f"Channel kind '{channel.kind}' is declared more than once",
                channel.path,
                channels=(channel.name,),
                details={"existing_path": seen_kinds[channel.kind].path},
            )
        else:
            seen_kinds[channel.kind] = channel
    for kind in PASS_KINDS:
        if kind not in seen_kinds:
            _diag(diagnostics, "error", "missing_channel", f"Required channel '{kind}' is missing", "channels")
    if len(seen_kinds) == len(PASS_KINDS):
        sets = {kind: {ref.name for ref in seen_kinds[kind].materials} for kind in PASS_KINDS}
        baseline = sets[PASS_KINDS[0]]
        for kind in PASS_KINDS[1:]:
            if sets[kind] != baseline:
                only_baseline = sorted(baseline - sets[kind])
                only_current = sorted(sets[kind] - baseline)
                _diag(
                    diagnostics,
                    "error",
                    "channel_material_mismatch",
                    f"Channel {kind} material set differs from {PASS_KINDS[0]}",
                    seen_kinds[kind].path,
                    channels=(seen_kinds[kind].name,),
                    details={
                        "baseline_channel": PASS_KINDS[0],
                        "only_in_baseline": only_baseline,
                        "only_in_current": only_current,
                    },
                )
    for channel in channels:
        for ref in channel.materials:
            if ref.name in materials and channel.name not in materials[ref.name].channel_names:
                materials[ref.name].channel_names.append(channel.name)
    for material in materials.values():
        material.channel_names.sort(key=lambda name: min(
            (channel.declared_index for channel in channels if channel.name == name),
            default=9999,
        ))
    _resolve_channel_overrides(materials, channels, diagnostics)
    return channels


def _parse_camera(raw: Any, path: str, diagnostics: list[Diagnostic]) -> CameraSpec:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        _diag(diagnostics, "error", "invalid_camera", f"Camera at {path} must be an object", path)
        raw = {}
    _unknown_fields(raw, {"position", "target", "up", "fov_y_deg", "near", "far"}, path, diagnostics, [])
    position = _vector(raw.get("position"), 3, f"{path}.position", diagnostics, [0.0, 0.0, 5.0])
    target = _vector(raw.get("target"), 3, f"{path}.target", diagnostics, [0.0, 0.0, 0.0])
    up = _vector(raw.get("up"), 3, f"{path}.up", diagnostics, [0.0, 1.0, 0.0])
    fov = _number(raw.get("fov_y_deg"), f"{path}.fov_y_deg", diagnostics, 45.0)
    near = _number(raw.get("near"), f"{path}.near", diagnostics, 0.1)
    far = _number(raw.get("far"), f"{path}.far", diagnostics, 100.0)
    assert position is not None and target is not None and up is not None
    assert fov is not None and near is not None and far is not None
    if fov <= 0.0 or fov >= 180.0:
        _diag(diagnostics, "error", "invalid_camera_fov", f"Camera FOV must be in (0, 180) at {path}.fov_y_deg", f"{path}.fov_y_deg")
    if near <= 0.0 or far <= near:
        _diag(diagnostics, "error", "invalid_camera_range", f"Camera requires 0 < near < far at {path}", path)
    if position == target or up == [0.0, 0.0, 0.0]:
        _diag(diagnostics, "error", "degenerate_camera", f"Camera basis is degenerate at {path}", path)
    return CameraSpec(position, target, up, fov, near, far, path)


def _parse_light(raw: Any, index: int, path: str, diagnostics: list[Diagnostic]) -> LightSpec | None:
    if not isinstance(raw, dict):
        _diag(diagnostics, "error", "invalid_light", f"Light at {path} must be an object", path)
        return None
    _unknown_fields(
        raw,
        {"name", "type", "position", "direction", "color", "intensity", "radius", "enabled"},
        path,
        diagnostics,
        [],
    )
    name = raw.get("name", f"light_{index}")
    if not isinstance(name, str) or not name:
        _diag(diagnostics, "error", "invalid_light_name", f"Light name at {path} must be a string", f"{path}.name")
        name = f"light_{index}"
    light_type = raw.get("type", "point")
    if not isinstance(light_type, str) or light_type.strip().lower() not in {"point", "directional"}:
        _diag(diagnostics, "error", "unsupported_light_type", f"Unsupported light type at {path}.type", f"{path}.type")
        light_type = "point"
    else:
        light_type = light_type.strip().lower()
    position = _vector(raw.get("position"), 3, f"{path}.position", diagnostics, [0.0, 0.0, 5.0])
    direction = _vector(raw.get("direction"), 3, f"{path}.direction", diagnostics, [0.0, 0.0, -1.0])
    color = _vector(raw.get("color"), 3, f"{path}.color", diagnostics, [1.0, 1.0, 1.0])
    intensity = _number(raw.get("intensity"), f"{path}.intensity", diagnostics, 1.0)
    radius = _number(raw.get("radius"), f"{path}.radius", diagnostics, 20.0)
    enabled = _boolean(raw.get("enabled"), f"{path}.enabled", diagnostics, True)
    assert position is not None and direction is not None and color is not None
    assert intensity is not None and radius is not None and enabled is not None
    if intensity < 0.0:
        _diag(diagnostics, "error", "negative_light_intensity", f"Light intensity at {path}.intensity must be non-negative", f"{path}.intensity")
    if radius <= 0.0:
        _diag(diagnostics, "error", "invalid_light_radius", f"Light radius at {path}.radius must be positive", f"{path}.radius")
    if light_type == "directional" and direction == [0.0, 0.0, 0.0]:
        _diag(diagnostics, "error", "degenerate_light_direction", f"Directional light at {path} has zero direction", f"{path}.direction")
    return LightSpec(name, light_type, position, direction, color, intensity, radius, enabled, path)


def _default_camera(path: str) -> CameraSpec:
    return CameraSpec([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 45.0, 0.1, 100.0, path)


def _default_lights(path: str) -> list[LightSpec]:
    return [
        LightSpec(
            name="key",
            type="directional",
            position=[0.0, 0.0, 0.0],
            direction=[0.35, -0.8, -0.5],
            color=[1.0, 0.96, 0.90],
            intensity=1.0,
            radius=10000.0,
            enabled=True,
            path=f"{path}[0]",
        ),
        LightSpec(
            name="fill",
            type="point",
            position=[-3.0, 2.0, 4.0],
            direction=[0.0, 0.0, -1.0],
            color=[0.25, 0.42, 0.9],
            intensity=0.6,
            radius=12.0,
            enabled=True,
            path=f"{path}[1]",
        ),
    ]


def _auto_objects(material_names: list[str], path: str) -> list[ObjectSpec]:
    count = max(len(material_names), 1)
    spacing = 1.15
    result: list[ObjectSpec] = []
    for index, material in enumerate(material_names):
        x = (index - (count - 1) / 2.0) * spacing
        result.append(
            ObjectSpec(
                name=f"auto_{index}",
                mesh="cube",
                material=material,
                position=[x, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                scale=[0.42, 0.42, 0.42],
                path=f"{path}[{index}]",
            )
        )
    return result


def _parse_object(
    raw: Any,
    index: int,
    path: str,
    material_names: set[str],
    channel_material_names: set[str],
    diagnostics: list[Diagnostic],
) -> ObjectSpec | None:
    if not isinstance(raw, dict):
        _diag(diagnostics, "error", "invalid_object", f"Scene object at {path} must be an object", path)
        return None
    _unknown_fields(raw, {"name", "mesh", "material", "position", "rotation", "scale"}, path, diagnostics, [])
    name = raw.get("name", f"object_{index}")
    if not isinstance(name, str) or not name:
        _diag(diagnostics, "error", "invalid_object_name", f"Object name at {path} must be a string", f"{path}.name")
        name = f"object_{index}"
    mesh = raw.get("mesh", "cube")
    if not isinstance(mesh, str) or mesh not in BUILTIN_MESHES:
        _diag(diagnostics, "error", "unsupported_mesh", f"Unsupported built-in mesh '{mesh}' at {path}.mesh", f"{path}.mesh")
        mesh = "cube"
    material = raw.get("material")
    if not isinstance(material, str) or material not in material_names:
        _diag(
            diagnostics,
            "error",
            "unknown_object_material",
            f"Object {name} references unknown material '{material}'",
            f"{path}.material",
        )
        return None
    if material not in channel_material_names:
        _diag(
            diagnostics,
            "warning",
            "material_not_in_channels",
            f"Object {name} uses material '{material}' which is absent from render channels",
            f"{path}.material",
            material=material,
        )
    position = _vector(raw.get("position"), 3, f"{path}.position", diagnostics, [0.0, 0.0, 0.0])
    rotation = _vector(raw.get("rotation"), 3, f"{path}.rotation", diagnostics, [0.0, 0.0, 0.0])
    scale = _vector(raw.get("scale"), 3, f"{path}.scale", diagnostics, [1.0, 1.0, 1.0])
    assert position is not None and rotation is not None and scale is not None
    if any(abs(component) < 1.0e-8 for component in scale):
        _diag(diagnostics, "error", "degenerate_object_scale", f"Object {name} has a zero scale component", f"{path}.scale")
    return ObjectSpec(name, mesh, material, position, rotation, scale, path)


def _parse_scenes(
    raw: Any,
    materials: dict[str, MaterialSpec],
    channels: list[ChannelSpec],
    diagnostics: list[Diagnostic],
) -> list[SceneSpec]:
    channel_material_names = {
        ref.name for channel in channels for ref in channel.materials if ref.name in materials
    }
    if raw is None:
        raw = [{}]
    if not isinstance(raw, list):
        _diag(diagnostics, "error", "invalid_scenes", "scenes must be an array", "scenes")
        raw = [{}]
    scenes: list[SceneSpec] = []
    names: set[str] = set()
    for scene_index, scene_raw in enumerate(raw):
        path = f"scenes[{scene_index}]"
        if not isinstance(scene_raw, dict):
            _diag(diagnostics, "error", "invalid_scene", f"Scene at {path} must be an object", path)
            continue
        _unknown_fields(scene_raw, {"name", "camera", "lights", "objects"}, path, diagnostics, [])
        name = scene_raw.get("name", f"scene_{scene_index}")
        if not isinstance(name, str) or not name:
            _diag(diagnostics, "error", "invalid_scene_name", f"Scene name at {path} must be a string", f"{path}.name")
            name = f"scene_{scene_index}"
        if name in names:
            _diag(diagnostics, "error", "duplicate_scene", f"Scene name '{name}' is duplicated", f"{path}.name")
            name = f"{name}_{scene_index}"
        names.add(name)
        camera = _parse_camera(scene_raw.get("camera"), f"{path}.camera", diagnostics)
        raw_lights = scene_raw.get("lights")
        if raw_lights is None:
            lights = _default_lights(f"{path}.lights")
        elif not isinstance(raw_lights, list):
            _diag(diagnostics, "error", "invalid_lights", f"lights at {path} must be an array", f"{path}.lights")
            lights = _default_lights(f"{path}.lights")
        else:
            lights = []
            for light_index, light_raw in enumerate(raw_lights):
                light = _parse_light(light_raw, light_index, f"{path}.lights[{light_index}]", diagnostics)
                if light is not None:
                    lights.append(light)
        if len(lights) > 4:
            _diag(
                diagnostics,
                "error",
                "too_many_lights",
                f"Scene {name} has {len(lights)} lights; the shader contract supports at most 4",
                f"{path}.lights",
            )
        raw_objects = scene_raw.get("objects")
        if raw_objects is None:
            objects = _auto_objects(sorted(channel_material_names), f"{path}.objects")
        elif not isinstance(raw_objects, list):
            _diag(diagnostics, "error", "invalid_objects", f"objects at {path} must be an array", f"{path}.objects")
            objects = []
        else:
            objects = []
            object_names: set[str] = set()
            for object_index, object_raw in enumerate(raw_objects):
                object_spec = _parse_object(
                    object_raw,
                    object_index,
                    f"{path}.objects[{object_index}]",
                    set(materials),
                    channel_material_names,
                    diagnostics,
                )
                if object_spec is None:
                    continue
                if object_spec.name in object_names:
                    _diag(
                        diagnostics,
                        "error",
                        "duplicate_object",
                        f"Object name '{object_spec.name}' is duplicated in scene {name}",
                        object_spec.path,
                    )
                    continue
                object_names.add(object_spec.name)
                objects.append(object_spec)
        if not objects:
            _diag(diagnostics, "warning", "empty_scene", f"Scene {name} contains no renderable objects", path)
        scenes.append(SceneSpec(name=name, camera=camera, lights=lights, objects=objects, path=path))
    return scenes


def parse_manifest(path: str | Path) -> ParsedManifest:
    manifest_path = Path(path).resolve()
    defaults = VERSION_DEFAULTS["1.0"]
    diagnostics: list[Diagnostic] = []
    unknown_fields: list[dict[str, Any]] = []
    raw: dict[str, Any] | None = None
    digest: str | None = None
    try:
        text = manifest_path.read_text(encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    except OSError as exc:
        _diag(
            diagnostics,
            "error",
            "manifest_read_error",
            f"Cannot read manifest: {exc}",
            str(manifest_path),
        )
        text = ""
    if text:
        try:
            raw_value = json.loads(
                text,
                object_pairs_hook=_json_pairs,
                parse_constant=lambda value: (_ for _ in ()).throw(ManifestJsonError(f"non-finite JSON number: {value}")),
            )
            if not isinstance(raw_value, dict):
                raise ManifestJsonError("manifest root must be a JSON object")
            raw = raw_value
        except (json.JSONDecodeError, ManifestJsonError) as exc:
            _diag(diagnostics, "error", "manifest_json_error", f"Invalid manifest JSON: {exc}", str(manifest_path))
    if raw is None:
        raw = {}
    _unknown_fields(
        raw,
        {
            "schema_version",
            "render",
            "comparison",
            "uniform_policy",
            "materials",
            "channels",
            "passes",
            "scenes",
        },
        "",
        diagnostics,
        unknown_fields,
    )
    version = _version_string(raw.get("schema_version"))
    if version is None:
        _diag(diagnostics, "error", "invalid_schema_version", "schema_version must be a string or number", "schema_version")
        version = "1.0"
    if raw.get("schema_version") is None:
        _diag(
            diagnostics,
            "warning",
            "schema_version_missing",
            "schema_version is missing; version 1.0 defaults are applied",
            "schema_version",
        )
    if version not in SCHEMA_VERSIONS:
        _diag(
            diagnostics,
            "error",
            "unsupported_schema_version",
            f"Unsupported schema_version '{version}'; supported versions are {sorted(SCHEMA_VERSIONS)}",
            "schema_version",
            details={"supported": sorted(SCHEMA_VERSIONS)},
        )
        version = "1.1"
    defaults = VERSION_DEFAULTS[version]
    render_raw = raw.get("render", {})
    if not isinstance(render_raw, dict):
        _diag(diagnostics, "error", "invalid_render", "render must be an object", "render")
        render_raw = {}
    _unknown_fields(
        render_raw,
        {"width", "height", "background", "normal_fallback", "normal_tolerance"},
        "render",
        diagnostics,
        unknown_fields,
    )
    width = _integer(render_raw.get("width"), "render.width", diagnostics, defaults["width"])
    height = _integer(render_raw.get("height"), "render.height", diagnostics, defaults["height"])
    assert width is not None and height is not None
    if width <= 0 or height <= 0 or width > 4096 or height > 4096:
        _diag(diagnostics, "error", "invalid_render_size", "render width/height must be in [1, 4096]", "render")
        width = max(1, min(4096, width))
        height = max(1, min(4096, height))
    background = _vector(render_raw.get("background"), 4, "render.background", diagnostics, defaults["background"])
    normal_fallback = _vector(
        render_raw.get("normal_fallback"),
        3,
        "render.normal_fallback",
        diagnostics,
        defaults["normal_fallback"],
    )
    normal_tolerance = _number(
        render_raw.get("normal_tolerance"),
        "render.normal_tolerance",
        diagnostics,
        defaults["normal_tolerance"],
    )
    assert background is not None and normal_fallback is not None and normal_tolerance is not None
    if normal_tolerance < 0.0:
        _diag(diagnostics, "error", "invalid_normal_tolerance", "normal_tolerance must be non-negative", "render.normal_tolerance")
    fallback_length = math.sqrt(sum(component * component for component in normal_fallback))
    if abs(fallback_length - 1.0) > normal_tolerance:
        _diag(
            diagnostics,
            "error",
            "invalid_normal_fallback",
            f"normal_fallback must be a unit vector (length={fallback_length:.9g})",
            "render.normal_fallback",
        )
    render = RenderSettings(width, height, background, normal_fallback, normal_tolerance)
    comparison_raw = raw.get("comparison", {})
    if not isinstance(comparison_raw, dict):
        _diag(diagnostics, "error", "invalid_comparison", "comparison must be an object", "comparison")
        comparison_raw = {}
    _unknown_fields(
        comparison_raw,
        {"absolute", "relative", "channels"},
        "comparison",
        diagnostics,
        unknown_fields,
    )
    absolute = _number(comparison_raw.get("absolute"), "comparison.absolute", diagnostics, defaults["absolute"])
    relative = _number(comparison_raw.get("relative"), "comparison.relative", diagnostics, defaults["relative"])
    assert absolute is not None and relative is not None
    if absolute < 0.0:
        _diag(diagnostics, "error", "invalid_absolute_threshold", "comparison.absolute must be non-negative", "comparison.absolute")
    if relative < 0.0:
        _diag(diagnostics, "error", "invalid_relative_threshold", "comparison.relative must be non-negative", "comparison.relative")
    channels_raw = comparison_raw.get("channels", ["R", "G", "B"])
    comparison_channels: list[str] = []
    if not isinstance(channels_raw, list):
        _diag(diagnostics, "error", "invalid_comparison_channels", "comparison.channels must be an array", "comparison.channels")
    else:
        for index, item in enumerate(channels_raw):
            normalized = item.upper() if isinstance(item, str) else ""
            if normalized not in {"R", "G", "B", "A"}:
                _diag(
                    diagnostics,
                    "error",
                    "invalid_comparison_channel",
                    f"Unsupported comparison channel at comparison.channels[{index}]",
                    f"comparison.channels[{index}]",
                )
            elif normalized not in comparison_channels:
                comparison_channels.append(normalized)
    if not comparison_channels:
        comparison_channels = ["R", "G", "B"]
    comparison = ComparisonSettings(absolute, relative, comparison_channels)
    policy_raw = raw.get("uniform_policy", {})
    if not isinstance(policy_raw, dict):
        _diag(diagnostics, "error", "invalid_uniform_policy", "uniform_policy must be an object", "uniform_policy")
        policy_raw = {}
    _unknown_fields(policy_raw, {"unused_declared"}, "uniform_policy", diagnostics, unknown_fields)
    unused_policy = policy_raw.get("unused_declared", defaults["unused_declared"])
    if unused_policy not in {"warn", "error", "ignore"}:
        _diag(
            diagnostics,
            "error",
            "invalid_unused_uniform_policy",
            "uniform_policy.unused_declared must be warn, error, or ignore",
            "uniform_policy.unused_declared",
        )
        unused_policy = "warn"
    uniform_policy = {"unused_declared": str(unused_policy)}
    materials = _parse_materials(raw.get("materials"), manifest_path.parent, diagnostics)
    raw_channels = raw.get("channels", raw.get("passes"))
    channels = _parse_channels(raw_channels, materials, diagnostics)
    scenes = _parse_scenes(raw.get("scenes"), materials, channels, diagnostics)
    retained_paths = {item["path"] for item in unknown_fields}
    for item in diagnostics:
        if item.code != "unknown_field" or item.path is None or item.path in retained_paths:
            continue
        found, value = _value_at_path(raw, item.path)
        if found:
            unknown_fields.append({"path": item.path, "value": copy.deepcopy(value)})
            retained_paths.add(item.path)
    return ParsedManifest(
        path=manifest_path,
        raw=raw,
        schema_version=version,
        render=render,
        comparison=comparison,
        uniform_policy=uniform_policy,
        materials=materials,
        channels=channels,
        scenes=scenes,
        diagnostics=diagnostics,
        unknown_fields=unknown_fields,
        manifest_sha256=digest,
    )
