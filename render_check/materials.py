"""Material description parsing and validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import material_error


REQUIRED_TEXTURE_FORMATS = frozenset({"RGBA8", "RGBA32F"})
REQUIRED_UNIFORM_NAMES = (
    "base_color",
    "metallic",
    "roughness",
    "emissive_strength",
    "uv_scale",
)


@dataclass(frozen=True)
class TextureSpec:
    path: Path
    format: str
    unit: int


@dataclass(frozen=True)
class Material:
    name: str
    vertex_shader: Path
    fragment_shader: Path
    vertex_source: str
    fragment_source: str
    texture: TextureSpec
    uniforms: dict[str, Any]
    source_path: Path
    description: str


@dataclass(frozen=True)
class TextureData:
    array: Any
    declared_format: str
    actual_format: str
    actual_description: str
    width: int
    height: int
    dtype: str


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_object(value: Any, field: str, material_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise material_error(
            f"material '{material_name}': field '{field}' must be an object",
            field=field,
            material=material_name,
            expected="object",
            actual=type(value).__name__,
        )
    return value


def _require_string(
    obj: dict[str, Any],
    field: str,
    material_name: str,
    *,
    allow_empty: bool = False,
) -> str:
    if field not in obj:
        raise material_error(
            f"material '{material_name}': missing required field '{field}'",
            field=field,
            material=material_name,
        )
    value = obj[field]
    if not isinstance(value, str):
        raise material_error(
            f"material '{material_name}': field '{field}' must be a string",
            field=field,
            material=material_name,
            expected="string",
            actual=type(value).__name__,
        )
    if not allow_empty and not value.strip():
        raise material_error(
            f"material '{material_name}': field '{field}' must not be empty",
            field=field,
            material=material_name,
        )
    return value


def _resolve_file(raw_path: str, base_dir: Path, field: str, material_name: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    try:
        path = path.resolve()
    except OSError:
        path = path.absolute()
    if not path.exists():
        raise material_error(
            f"material '{material_name}': referenced file for '{field}' does not exist: {path}",
            field=field,
            material=material_name,
            path=str(path),
        )
    if not path.is_file():
        raise material_error(
            f"material '{material_name}': referenced path for '{field}' is not a file: {path}",
            field=field,
            material=material_name,
            path=str(path),
        )
    return path


def _read_shader(path: Path, field: str, material_name: str) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise material_error(
            f"material '{material_name}': cannot read {field} '{path}': {exc}",
            field=field,
            material=material_name,
            path=str(path),
        ) from exc


def _validate_vector(
    value: Any,
    field: str,
    material_name: str,
    *,
    length: int,
    minimum: float,
    maximum: float,
) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise material_error(
            f"material '{material_name}': uniform '{field}' must be a {length}-element number array",
            field=field,
            material=material_name,
            expected=f"array[{length}]",
            actual=type(value).__name__,
        )
    result: list[float] = []
    for index, item in enumerate(value):
        if not _is_number(item) or not math.isfinite(float(item)):
            raise material_error(
                f"material '{material_name}': uniform '{field}[{index}]' must be a finite number",
                field=field,
                material=material_name,
            )
        number = float(item)
        if number < minimum or number > maximum:
            raise material_error(
                f"material '{material_name}': uniform '{field}[{index}]' must be in "
                f"[{minimum}, {maximum}]",
                field=field,
                material=material_name,
                actual=number,
            )
        result.append(number)
    return result


def _validate_scalar(
    value: Any,
    field: str,
    material_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise material_error(
            f"material '{material_name}': uniform '{field}' must be a finite number",
            field=field,
            material=material_name,
            expected="number",
            actual=type(value).__name__,
        )
    number = float(value)
    if minimum is not None and number < minimum:
        raise material_error(
            f"material '{material_name}': uniform '{field}' must be >= {minimum}",
            field=field,
            material=material_name,
            actual=number,
        )
    if maximum is not None and number > maximum:
        raise material_error(
            f"material '{material_name}': uniform '{field}' must be <= {maximum}",
            field=field,
            material=material_name,
            actual=number,
        )
    return number


def _validate_uniform_value(value: Any, field: str, material_name: str) -> Any:
    if _is_number(value):
        number = float(value)
        if not math.isfinite(number):
            raise material_error(
                f"material '{material_name}': uniform '{field}' must be finite",
                field=field,
                material=material_name,
            )
        return number
    if isinstance(value, list):
        if not value:
            raise material_error(
                f"material '{material_name}': uniform '{field}' must not be an empty array",
                field=field,
                material=material_name,
            )
        for index, item in enumerate(value):
            if not _is_number(item) or not math.isfinite(float(item)):
                raise material_error(
                    f"material '{material_name}': uniform '{field}[{index}]' must be "
                    "a finite number",
                    field=field,
                    material=material_name,
                )
        return [float(item) for item in value]
    raise material_error(
        f"material '{material_name}': uniform '{field}' must be a number or number array",
        field=field,
        material=material_name,
        expected="number | array<number>",
        actual=type(value).__name__,
    )


def _parse_material(
    item: Any,
    index: int,
    base_dir: Path,
    source_path: Path,
) -> Material:
    item = _require_object(item, f"materials[{index}]", f"materials[{index}]")
    material_name = _require_string(item, "name", f"materials[{index}]").strip()
    identity = f"materials[{index}] ({material_name})"

    vertex_shader = _resolve_file(
        _require_string(item, "vertex_shader", identity),
        base_dir,
        "vertex_shader",
        material_name,
    )
    fragment_shader = _resolve_file(
        _require_string(item, "fragment_shader", identity),
        base_dir,
        "fragment_shader",
        material_name,
    )

    if "texture" not in item:
        raise material_error(
            f"material '{material_name}': missing required field 'texture'",
            field="texture",
            material=material_name,
        )
    texture_obj = _require_object(item["texture"], "texture", material_name)
    texture_path = _resolve_file(
        _require_string(texture_obj, "path", material_name),
        base_dir,
        "texture.path",
        material_name,
    )
    texture_format = _require_string(texture_obj, "format", material_name)
    texture_unit = texture_obj.get("unit", 0)
    if not isinstance(texture_unit, int) or isinstance(texture_unit, bool):
        raise material_error(
            f"material '{material_name}': texture.unit must be an integer",
            field="texture.unit",
            material=material_name,
            expected="integer",
            actual=type(texture_unit).__name__,
        )
    if texture_unit < 0 or texture_unit > 15:
        raise material_error(
            f"material '{material_name}': texture.unit must be in [0, 15]",
            field="texture.unit",
            material=material_name,
            actual=texture_unit,
        )

    if "uniforms" not in item:
        raise material_error(
            f"material '{material_name}': missing required field 'uniforms'",
            field="uniforms",
            material=material_name,
        )
    raw_uniforms = _require_object(item["uniforms"], "uniforms", material_name)
    uniforms: dict[str, Any] = {}
    for key, value in raw_uniforms.items():
        if not isinstance(key, str) or not key:
            raise material_error(
                f"material '{material_name}': uniform names must be non-empty strings",
                field="uniforms",
                material=material_name,
            )
        uniforms[key] = _validate_uniform_value(value, key, material_name)

    for required in REQUIRED_UNIFORM_NAMES:
        if required not in uniforms:
            raise material_error(
                f"material '{material_name}': missing uniform default '{required}'",
                field=f"uniforms.{required}",
                material=material_name,
            )

    uniforms["base_color"] = _validate_vector(
        uniforms["base_color"],
        "base_color",
        material_name,
        length=3,
        minimum=0.0,
        maximum=1.0,
    )
    uniforms["metallic"] = _validate_scalar(
        uniforms["metallic"], "metallic", material_name, minimum=0.0, maximum=1.0
    )
    uniforms["roughness"] = _validate_scalar(
        uniforms["roughness"], "roughness", material_name, minimum=0.0, maximum=1.0
    )
    uniforms["emissive_strength"] = _validate_scalar(
        uniforms["emissive_strength"],
        "emissive_strength",
        material_name,
        minimum=0.0,
    )
    uniforms["uv_scale"] = _validate_scalar(
        uniforms["uv_scale"], "uv_scale", material_name, minimum=0.0
    )

    description = item.get("description", "")
    if not isinstance(description, str):
        raise material_error(
            f"material '{material_name}': field 'description' must be a string",
            field="description",
            material=material_name,
            expected="string",
            actual=type(description).__name__,
        )

    return Material(
        name=material_name,
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader,
        vertex_source=_read_shader(vertex_shader, "vertex_shader", material_name),
        fragment_source=_read_shader(fragment_shader, "fragment_shader", material_name),
        texture=TextureSpec(path=texture_path, format=texture_format, unit=texture_unit),
        uniforms=uniforms,
        source_path=source_path,
        description=description,
    )


def load_materials(path: Path | str) -> list[Material]:
    source_path = Path(path)
    try:
        source_path = source_path.resolve()
    except OSError:
        source_path = source_path.absolute()
    if not source_path.exists():
        raise material_error(
            f"material description does not exist: {source_path}",
            path=str(source_path),
        )
    if not source_path.is_file():
        raise material_error(
            f"material description is not a file: {source_path}",
            path=str(source_path),
        )
    try:
        raw = json.loads(source_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise material_error(
            f"cannot parse material description '{source_path}': {exc}",
            path=str(source_path),
        ) from exc
    if not isinstance(raw, list):
        raise material_error(
            "material description must be a JSON array",
            path=str(source_path),
            expected="array",
            actual=type(raw).__name__,
        )
    if not raw:
        raise material_error(
            "material description must contain at least one material",
            path=str(source_path),
        )

    materials: list[Material] = []
    names: set[str] = set()
    base_dir = source_path.parent
    for index, item in enumerate(raw):
        material = _parse_material(item, index, base_dir, source_path)
        if material.name in names:
            raise material_error(
                f"duplicate material name: '{material.name}'",
                material=material.name,
                field="name",
            )
        names.add(material.name)
        materials.append(material)
    return materials

