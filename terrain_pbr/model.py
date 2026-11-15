"""Scene JSON parsing and validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import InputValidationError


MAX_COORDINATE = 1_000_000_000.0
MAX_CANVAS_DIMENSION = 8_192
MAX_CANVAS_PIXELS = 16_777_216
MAX_GRID_SIZE = 4_096
VALID_LODS = ("high", "medium", "low")


@dataclass(frozen=True)
class ShapeSpec:
    id: str
    kind: str
    levels: dict[str, tuple[tuple[float, float], ...]]
    width: float
    closed: bool
    merge_distance: float
    simplify_tolerance: float
    source_index: int


@dataclass(frozen=True)
class LayerSpec:
    id: str
    color: tuple[int, int, int]
    alpha: float
    shapes: tuple[ShapeSpec, ...]
    order: int


@dataclass(frozen=True)
class SceneSpec:
    name: str
    layers: tuple[LayerSpec, ...]
    background: tuple[int, int, int, int] = (0, 0, 0, 0)
    normal_strength: float = 1.0
    source_path: Path = field(default_factory=Path)


def _invalid(message: str, path: str, **details: Any) -> None:
    raise InputValidationError(message, field=path, details=details)


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _invalid("Expected a JSON object.", path)
    return value


def _require_list(value: Any, path: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        _invalid("Expected a JSON array.", path)
    if nonempty and not value:
        _invalid("Array must not be empty.", path)
    return value


def _require_number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _invalid("Expected a numeric value.", path)
    number = float(value)
    if not math.isfinite(number):
        _invalid("Coordinate or numeric value must be finite.", path)
    if minimum is not None and number < minimum:
        _invalid(f"Value must be >= {minimum}.", path, minimum=minimum, actual=number)
    return number


def _require_int(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _invalid("Expected an integer.", path)
    if minimum is not None and value < minimum:
        _invalid(f"Value must be >= {minimum}.", path, minimum=minimum, actual=value)
    return value


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _invalid("Expected a non-empty string.", path)
    return value


def _parse_points(value: Any, path: str) -> tuple[tuple[float, float], ...]:
    items = _require_list(value, path, nonempty=True)
    points: list[tuple[float, float]] = []
    for index, item in enumerate(items):
        item_path = f"{path}[{index}]"
        pair = _require_list(item, item_path)
        if len(pair) != 2:
            _invalid("A point must contain exactly [x, y].", item_path)
        x = _require_number(pair[0], f"{item_path}[0]")
        y = _require_number(pair[1], f"{item_path}[1]")
        if abs(x) > MAX_COORDINATE or abs(y) > MAX_COORDINATE:
            _invalid(
                f"Coordinate magnitude must be <= {MAX_COORDINATE:g}.",
                item_path,
            )
        points.append((x, y))
    return tuple(points)


def _same_point(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return abs(a[0] - b[0]) <= 1e-12 and abs(a[1] - b[1]) <= 1e-12


def _normalize_level_points(
    points: tuple[tuple[float, float], ...],
    *,
    kind: str,
    path: str,
) -> tuple[tuple[float, float], ...]:
    deduplicated: list[tuple[float, float]] = []
    for point in points:
        if not deduplicated or not _same_point(deduplicated[-1], point):
            deduplicated.append(point)

    if kind == "polygon":
        if len(deduplicated) > 1 and _same_point(deduplicated[0], deduplicated[-1]):
            deduplicated.pop()
        if len(deduplicated) < 3:
            _invalid("A polygon needs at least three distinct vertices.", path)
    else:
        if len(deduplicated) < 2:
            _invalid("A polyline needs at least two distinct vertices.", path)
    return tuple(deduplicated)


def _parse_color(value: Any, path: str) -> tuple[int, int, int]:
    items = _require_list(value, path)
    if len(items) != 3:
        _invalid("Layer color must be [r, g, b].", path)
    channels: list[int] = []
    for index, item in enumerate(items):
        channel = _require_int(item, f"{path}[{index}]", minimum=0)
        if channel > 255:
            _invalid("Color channel must be <= 255.", f"{path}[{index}]")
        channels.append(channel)
    return channels[0], channels[1], channels[2]


def _parse_shape(raw: Any, path: str, index: int) -> ShapeSpec:
    data = _require_mapping(raw, path)
    kind = _require_string(data.get("kind"), f"{path}.kind")
    if kind not in ("polygon", "polyline"):
        _invalid("Shape kind must be 'polygon' or 'polyline'.", f"{path}.kind")

    shape_id = data.get("id", f"shape-{index}")
    shape_id = _require_string(shape_id, f"{path}.id")

    geometry = _require_mapping(data.get("geometry"), f"{path}.geometry")
    levels: dict[str, tuple[tuple[float, float], ...]] = {}
    for lod in VALID_LODS:
        level_path = f"{path}.geometry.{lod}"
        if lod not in geometry:
            _invalid(f"Missing required geometry level '{lod}'.", level_path)
        points = _parse_points(geometry[lod], level_path)
        levels[lod] = _normalize_level_points(points, kind=kind, path=level_path)

    width = 2.0
    closed = False
    if kind == "polyline":
        if "width" in data:
            width = _require_number(data["width"], f"{path}.width", minimum=0.0)
            if width <= 0.0:
                _invalid("Polyline width must be > 0.", f"{path}.width")
        if "closed" in data:
            if not isinstance(data["closed"], bool):
                _invalid("closed must be a boolean.", f"{path}.closed")
            closed = data["closed"]

    simplify = data.get("simplify", {})
    simplify = _require_mapping(simplify, f"{path}.simplify")
    merge_distance = _require_number(
        simplify.get("merge_distance", 0.75),
        f"{path}.simplify.merge_distance",
        minimum=0.0,
    )
    simplify_tolerance = _require_number(
        simplify.get("tolerance", 1.0),
        f"{path}.simplify.tolerance",
        minimum=0.0,
    )

    return ShapeSpec(
        id=shape_id,
        kind=kind,
        levels=levels,
        width=width,
        closed=closed,
        merge_distance=merge_distance,
        simplify_tolerance=simplify_tolerance,
        source_index=index,
    )


def _parse_layer(raw: Any, path: str, order: int) -> LayerSpec:
    data = _require_mapping(raw, path)
    layer_id = _require_string(data.get("id"), f"{path}.id")
    color = _parse_color(data.get("color"), f"{path}.color")
    alpha = _require_number(data.get("alpha"), f"{path}.alpha", minimum=0.0)
    if alpha > 1.0:
        _invalid("Layer alpha must be <= 1.", f"{path}.alpha")

    raw_shapes = _require_list(data.get("shapes"), f"{path}.shapes", nonempty=True)
    shapes: list[ShapeSpec] = []
    shape_ids: set[str] = set()
    for index, raw_shape in enumerate(raw_shapes):
        shape = _parse_shape(raw_shape, f"{path}.shapes[{index}]", index)
        if shape.id in shape_ids:
            _invalid("Duplicate shape id within layer.", f"{path}.shapes[{index}].id")
        shape_ids.add(shape.id)
        shapes.append(shape)

    return LayerSpec(
        id=layer_id,
        color=color,
        alpha=alpha,
        shapes=tuple(shapes),
        order=order,
    )


def load_scene(path: str | Path) -> SceneSpec:
    scene_path = Path(path)
    try:
        with scene_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        raise InputValidationError(
            "Input JSON file does not exist.",
            field="input",
            details={"path": str(scene_path)},
        ) from None
    except OSError as exc:
        raise InputValidationError(
            "Input JSON file could not be read.",
            field="input",
            details={"path": str(scene_path), "reason": str(exc)},
        ) from None
    except json.JSONDecodeError as exc:
        raise InputValidationError(
            "Input file is not valid JSON.",
            field="input",
            details={"line": exc.lineno, "column": exc.colno, "reason": exc.msg},
        ) from None

    root = _require_mapping(raw, "$")
    if "version" in root:
        version = _require_int(root["version"], "$.version", minimum=1)
        if version != 1:
            _invalid("Unsupported scene version; expected 1.", "$.version")

    name = root.get("name", scene_path.stem)
    name = _require_string(name, "$.name")
    raw_layers = _require_list(root.get("layers"), "$.layers", nonempty=True)
    layers: list[LayerSpec] = []
    layer_ids: set[str] = set()
    for index, raw_layer in enumerate(raw_layers):
        layer = _parse_layer(raw_layer, f"$.layers[{index}]", index)
        if layer.id in layer_ids:
            _invalid("Duplicate layer id.", f"$.layers[{index}].id")
        layer_ids.add(layer.id)
        layers.append(layer)

    background: tuple[int, int, int, int] = (0, 0, 0, 0)
    if "background" in root:
        raw_background = _require_list(root["background"], "$.background")
        if len(raw_background) != 4:
            _invalid("Background must be [r, g, b, a].", "$.background")
        values: list[int] = []
        for index, value in enumerate(raw_background):
            channel = _require_int(value, f"$.background[{index}]", minimum=0)
            if channel > 255:
                _invalid("Background channel must be <= 255.", f"$.background[{index}]")
            values.append(channel)
        background = values[0], values[1], values[2], values[3]

    normal_strength = 1.0
    if "normal_strength" in root:
        normal_strength = _require_number(
            root["normal_strength"], "$.normal_strength", minimum=0.0
        )
        if normal_strength <= 0.0:
            _invalid("normal_strength must be > 0.", "$.normal_strength")

    return SceneSpec(
        name=name,
        layers=tuple(layers),
        background=background,
        normal_strength=normal_strength,
        source_path=scene_path,
    )


def validate_runtime_options(
    *,
    width: Any,
    height: Any,
    lod: Any,
    grid_size: Any,
) -> tuple[int, int, str, int]:
    if isinstance(width, bool):
        _invalid("Canvas width must be an integer.", "width")
    try:
        width_int = int(width)
    except (TypeError, ValueError):
        _invalid("Canvas width must be an integer.", "width")
    if str(width_int) != str(width).strip() and not isinstance(width, int):
        _invalid("Canvas width must be an integer.", "width")
    if width_int < 1 or width_int > MAX_CANVAS_DIMENSION:
        _invalid(
            f"Canvas width must be between 1 and {MAX_CANVAS_DIMENSION}.",
            "width",
            actual=width_int,
        )

    if isinstance(height, bool):
        _invalid("Canvas height must be an integer.", "height")
    try:
        height_int = int(height)
    except (TypeError, ValueError):
        _invalid("Canvas height must be an integer.", "height")
    if str(height_int) != str(height).strip() and not isinstance(height, int):
        _invalid("Canvas height must be an integer.", "height")
    if height_int < 1 or height_int > MAX_CANVAS_DIMENSION:
        _invalid(
            f"Canvas height must be between 1 and {MAX_CANVAS_DIMENSION}.",
            "height",
            actual=height_int,
        )
    if width_int * height_int > MAX_CANVAS_PIXELS:
        _invalid(
            f"Canvas area must be <= {MAX_CANVAS_PIXELS} pixels.",
            "canvas",
            width=width_int,
            height=height_int,
        )

    if lod not in VALID_LODS:
        _invalid("LOD must be one of high, medium, low.", "lod", actual=lod)

    if isinstance(grid_size, bool):
        _invalid("Grid size must be an integer.", "grid_size")
    try:
        grid_int = int(grid_size)
    except (TypeError, ValueError):
        _invalid("Grid size must be an integer.", "grid_size")
    if str(grid_int) != str(grid_size).strip() and not isinstance(grid_size, int):
        _invalid("Grid size must be an integer.", "grid_size")
    if grid_int < 1 or grid_int > MAX_GRID_SIZE:
        _invalid(
            f"Grid size must be between 1 and {MAX_GRID_SIZE}.",
            "grid_size",
            actual=grid_int,
        )

    return width_int, height_int, lod, grid_int
