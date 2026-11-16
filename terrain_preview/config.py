"""JSON configuration loading and strict validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import config_error, config_parse_error


@dataclass(frozen=True)
class NoiseConfig:
    frequency: float
    octaves: int
    persistence: float
    lacunarity: float


@dataclass(frozen=True)
class TerrainConfig:
    rows: int
    cols: int
    seed: int
    world_width: float
    world_depth: float
    height_scale: float
    tile_cells: int
    noise: NoiseConfig
    height_patches: tuple["HeightPatch", ...]


@dataclass(frozen=True)
class HeightPatch:
    center: tuple[float, float]
    radius: float
    delta: float


@dataclass(frozen=True)
class CameraConfig:
    azimuth_deg: float
    elevation_deg: float
    distance: float
    fov_y_deg: float
    near: float
    far: float
    target_fraction: tuple[float, float, float]


@dataclass(frozen=True)
class MaterialLayer:
    name: str
    color_srgb: tuple[float, float, float]
    weight: float


@dataclass(frozen=True)
class MaterialConfig:
    elevation_thresholds: tuple[float, float, float]
    elevation_transition: float
    slope_start: float
    slope_end: float
    slope_rock_strength: float
    layers: tuple[MaterialLayer, MaterialLayer, MaterialLayer, MaterialLayer]


@dataclass(frozen=True)
class AmbientConfig:
    sky_color_srgb: tuple[float, float, float]
    ground_color_srgb: tuple[float, float, float]
    intensity: float


@dataclass(frozen=True)
class ShadowConfig:
    enabled: bool
    resolution: int
    depth_bias: float
    normal_offset: float
    pcf_samples: int
    filter_radius: float


@dataclass(frozen=True)
class LightConfig:
    azimuth_deg: float
    elevation_deg: float
    color_srgb: tuple[float, float, float]
    intensity: float
    ambient: AmbientConfig
    shadow: ShadowConfig


@dataclass(frozen=True)
class RenderConfig:
    width: int
    height: int
    tile_size: int
    background_srgb: tuple[float, float, float]
    exposure: float
    gamma: float
    write_debug_images: bool


@dataclass(frozen=True)
class OutputConfig:
    image: Path
    normal_image: Path | None
    shadow_image: Path | None
    terrain_file: Path
    tile_report: Path
    cache_dir: Path


@dataclass(frozen=True)
class LimitsConfig:
    max_memory_mb: float | None
    max_total_samples: int | None
    max_terrain_cells: int


@dataclass(frozen=True)
class AppConfig:
    config_path: Path
    terrain: TerrainConfig
    camera: CameraConfig
    materials: MaterialConfig
    light: LightConfig
    render: RenderConfig
    output: OutputConfig
    limits: LimitsConfig


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(
    section: dict[str, Any],
    key: str,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    integer: bool = False,
) -> float | int:
    dotted = f"{path}.{key}"
    if key not in section:
        raise config_error(f"missing required value '{dotted}'")
    value = section[key]
    if not _is_number(value):
        raise config_error(f"'{dotted}' must be a number")
    if integer:
        if isinstance(value, float) and not value.is_integer():
            raise config_error(f"'{dotted}' must be an integer")
        value = int(value)
    else:
        value = float(value)
    if not math.isfinite(float(value)):
        raise config_error(f"'{dotted}' must be finite")
    if minimum is not None and float(value) < minimum:
        raise config_error(f"'{dotted}' must be >= {minimum}")
    if maximum is not None and float(value) > maximum:
        raise config_error(f"'{dotted}' must be <= {maximum}")
    return value


def _object(section: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    if key not in section:
        raise config_error(f"missing required object '{path}.{key}'")
    value = section[key]
    if not isinstance(value, dict):
        raise config_error(f"'{path}.{key}' must be an object")
    return value


def _array(
    section: dict[str, Any],
    key: str,
    path: str,
    *,
    length: int | None = None,
    numeric: bool = False,
) -> list[Any]:
    if key not in section:
        raise config_error(f"missing required array '{path}.{key}'")
    value = section[key]
    if not isinstance(value, list):
        raise config_error(f"'{path}.{key}' must be an array")
    if length is not None and len(value) != length:
        raise config_error(f"'{path}.{key}' must contain exactly {length} values")
    if numeric and not all(_is_number(item) for item in value):
        raise config_error(f"'{path}.{key}' must contain only numbers")
    return value


def _color(
    section: dict[str, Any], key: str, path: str
) -> tuple[float, float, float]:
    values = _array(section, key, path, length=3, numeric=True)
    color = tuple(float(item) for item in values)
    if any(not math.isfinite(item) or item < 0.0 or item > 1.0 for item in color):
        raise config_error(f"'{path}.{key}' values must be in [0, 1]")
    return color  # type: ignore[return-value]


def _bool(section: dict[str, Any], key: str, path: str) -> bool:
    if key not in section:
        raise config_error(f"missing required value '{path}.{key}'")
    value = section[key]
    if not isinstance(value, bool):
        raise config_error(f"'{path}.{key}' must be true or false")
    return value


def _optional_path(
    section: dict[str, Any], key: str, path: str, base: Path
) -> Path | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise config_error(f"'{path}.{key}' must be a non-empty path string")
    return (base / value).resolve()


def _required_path(
    section: dict[str, Any], key: str, path: str, base: Path
) -> Path:
    result = _optional_path(section, key, path, base)
    if result is None:
        raise config_error(f"missing required path '{path}.{key}'")
    return result


def _parse_material_layers(
    raw: list[Any], path: str
) -> tuple[MaterialLayer, MaterialLayer, MaterialLayer, MaterialLayer]:
    if len(raw) != 4:
        raise config_error(f"'{path}.layers' must contain exactly four materials")
    parsed: list[MaterialLayer] = []
    for index, item in enumerate(raw):
        item_path = f"{path}.layers[{index}]"
        if not isinstance(item, dict):
            raise config_error(f"'{item_path}' must be an object")
        name = item.get("name")
        if name not in ("sand", "grass", "rock", "snow"):
            raise config_error(
                f"'{item_path}.name' must be one of sand, grass, rock, snow"
            )
        color = _color(item, "color", item_path)
        weight = float(_number(item, "weight", item_path, minimum=0.0))
        parsed.append(MaterialLayer(name=name, color_srgb=color, weight=weight))
    names = [layer.name for layer in parsed]
    if names != ["sand", "grass", "rock", "snow"]:
        raise config_error(
            f"'{path}.layers' must be ordered as sand, grass, rock, snow"
        )
    if sum(layer.weight for layer in parsed) <= 0.0:
        raise config_error(f"'{path}.layers' weights must not all be zero")
    return tuple(parsed)  # type: ignore[return-value]


def _parse_height_patches(
    terrain_obj: dict[str, Any],
) -> tuple[HeightPatch, ...]:
    value = terrain_obj.get("height_patches", [])
    if not isinstance(value, list):
        raise config_error("'terrain.height_patches' must be an array")
    if len(value) > 256:
        raise config_error("'terrain.height_patches' cannot contain more than 256 patches")
    patches: list[HeightPatch] = []
    for index, item in enumerate(value):
        path = f"terrain.height_patches[{index}]"
        if not isinstance(item, dict):
            raise config_error(f"'{path}' must be an object")
        center_values = _array(item, "center", path, length=2, numeric=True)
        center = (float(center_values[0]), float(center_values[1]))
        if any(value < 0.0 or value > 1.0 for value in center):
            raise config_error(f"'{path}.center' values must be in [0, 1]")
        radius = float(_number(item, "radius", path, minimum=1e-4, maximum=2.0))
        delta = float(_number(item, "delta", path, minimum=-1.0, maximum=1.0))
        patches.append(HeightPatch(center=center, radius=radius, delta=delta))
    return tuple(patches)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise config_parse_error(f"configuration file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise config_parse_error(
            f"invalid JSON in {config_path}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from None
    except OSError as exc:
        raise config_parse_error(f"cannot read configuration file {config_path}: {exc}") from None

    if not isinstance(raw, dict):
        raise config_error("configuration root must be a JSON object")

    terrain_obj = _object(raw, "terrain", "config")
    noise_obj = _object(terrain_obj, "noise", "terrain")
    rows = int(_number(terrain_obj, "rows", "terrain", minimum=2, maximum=4096, integer=True))
    cols = int(_number(terrain_obj, "cols", "terrain", minimum=2, maximum=4096, integer=True))
    seed = int(_number(terrain_obj, "seed", "terrain", minimum=0, maximum=2**63 - 1, integer=True))
    world_width = float(_number(terrain_obj, "world_width", "terrain", minimum=1e-6))
    world_depth = float(_number(terrain_obj, "world_depth", "terrain", minimum=1e-6))
    height_scale = float(_number(terrain_obj, "height_scale", "terrain", minimum=1e-6))
    tile_cells = int(_number(terrain_obj, "tile_cells", "terrain", minimum=1, maximum=512, integer=True))
    noise = NoiseConfig(
        frequency=float(_number(noise_obj, "frequency", "terrain.noise", minimum=1e-6, maximum=256.0)),
        octaves=int(_number(noise_obj, "octaves", "terrain.noise", minimum=1, maximum=12, integer=True)),
        persistence=float(
            _number(noise_obj, "persistence", "terrain.noise", minimum=1e-6, maximum=1.0)
        ),
        lacunarity=float(
            _number(noise_obj, "lacunarity", "terrain.noise", minimum=1.0, maximum=8.0)
        ),
    )
    height_patches = _parse_height_patches(terrain_obj)

    camera_obj = _object(raw, "camera", "config")
    azimuth = float(_number(camera_obj, "azimuth_deg", "camera", minimum=-100000.0, maximum=100000.0))
    elevation = float(_number(camera_obj, "elevation_deg", "camera", minimum=-89.0, maximum=89.0))
    distance = float(_number(camera_obj, "distance", "camera", minimum=1e-6))
    fov = float(_number(camera_obj, "fov_y_deg", "camera", minimum=1.0, maximum=179.0))
    near = float(_number(camera_obj, "near", "camera", minimum=1e-6))
    far = float(_number(camera_obj, "far", "camera", minimum=1e-6))
    if far <= near:
        raise config_error("'camera.far' must be greater than 'camera.near'")
    target_values = _array(camera_obj, "target", "camera", length=3, numeric=True)
    target = tuple(float(item) for item in target_values)
    if any(item < 0.0 or item > 1.0 for item in target):
        raise config_error("'camera.target' values must be normalized fractions in [0, 1]")
    camera = CameraConfig(
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        distance=distance,
        fov_y_deg=fov,
        near=near,
        far=far,
        target_fraction=target,  # type: ignore[arg-type]
    )

    material_obj = _object(raw, "materials", "config")
    thresholds_raw = _array(
        material_obj, "elevation_thresholds", "materials", length=3, numeric=True
    )
    thresholds = tuple(float(item) for item in thresholds_raw)
    if not all(0.0 <= item <= 1.0 for item in thresholds):
        raise config_error("'materials.elevation_thresholds' values must be in [0, 1]")
    if not (thresholds[0] < thresholds[1] < thresholds[2]):
        raise config_error(
            "'materials.elevation_thresholds' must be strictly increasing "
            "(sand/grass, grass/rock, rock/snow)"
        )
    transition = float(
        _number(material_obj, "elevation_transition", "materials", minimum=1e-6, maximum=0.5)
    )
    slope_start = float(_number(material_obj, "slope_start", "materials", minimum=0.0, maximum=1.0))
    slope_end = float(_number(material_obj, "slope_end", "materials", minimum=0.0, maximum=1.0))
    if slope_start >= slope_end:
        raise config_error("'materials.slope_start' must be less than 'materials.slope_end'")
    slope_strength = float(
        _number(material_obj, "slope_rock_strength", "materials", minimum=0.0, maximum=1.0)
    )
    layers_raw = _array(material_obj, "layers", "materials")
    layers = _parse_material_layers(layers_raw, "materials")
    materials = MaterialConfig(
        elevation_thresholds=thresholds,  # type: ignore[arg-type]
        elevation_transition=transition,
        slope_start=slope_start,
        slope_end=slope_end,
        slope_rock_strength=slope_strength,
        layers=layers,
    )

    light_obj = _object(raw, "light", "config")
    direction_obj = _object(light_obj, "direction", "light")
    light_azimuth = float(
        _number(direction_obj, "azimuth_deg", "light.direction", minimum=-100000.0, maximum=100000.0)
    )
    light_elevation = float(
        _number(direction_obj, "elevation_deg", "light.direction", minimum=0.1, maximum=89.9)
    )
    light_color = _color(light_obj, "color", "light")
    light_intensity = float(_number(light_obj, "intensity", "light", minimum=0.0))
    ambient_obj = _object(light_obj, "ambient", "light")
    ambient = AmbientConfig(
        sky_color_srgb=_color(ambient_obj, "sky_color", "light.ambient"),
        ground_color_srgb=_color(ambient_obj, "ground_color", "light.ambient"),
        intensity=float(_number(ambient_obj, "intensity", "light.ambient", minimum=0.0)),
    )
    shadow_obj = _object(light_obj, "shadow", "light")
    shadow_enabled = _bool(shadow_obj, "enabled", "light.shadow")
    resolution = int(
        _number(shadow_obj, "resolution", "light.shadow", minimum=64, maximum=4096, integer=True)
    )
    if resolution & (resolution - 1):
        raise config_error("'light.shadow.resolution' must be a power of two")
    shadow = ShadowConfig(
        enabled=shadow_enabled,
        resolution=resolution,
        depth_bias=float(_number(shadow_obj, "depth_bias", "light.shadow", minimum=0.0)),
        normal_offset=float(_number(shadow_obj, "normal_offset", "light.shadow", minimum=0.0)),
        pcf_samples=int(
            _number(shadow_obj, "pcf_samples", "light.shadow", minimum=1, maximum=256, integer=True)
        ),
        filter_radius=float(
            _number(shadow_obj, "filter_radius", "light.shadow", minimum=0.0, maximum=32.0)
        ),
    )
    light = LightConfig(
        azimuth_deg=light_azimuth,
        elevation_deg=light_elevation,
        color_srgb=light_color,
        intensity=light_intensity,
        ambient=ambient,
        shadow=shadow,
    )

    render_obj = _object(raw, "render", "config")
    width = int(_number(render_obj, "width", "render", minimum=1, maximum=8192, integer=True))
    height = int(_number(render_obj, "height", "render", minimum=1, maximum=8192, integer=True))
    tile_size = int(_number(render_obj, "tile_size", "render", minimum=8, maximum=512, integer=True))
    render = RenderConfig(
        width=width,
        height=height,
        tile_size=tile_size,
        background_srgb=_color(render_obj, "background_color", "render"),
        exposure=float(_number(render_obj, "exposure", "render", minimum=1e-6)),
        gamma=float(_number(render_obj, "gamma", "render", minimum=0.1, maximum=10.0)),
        write_debug_images=_bool(render_obj, "write_debug_images", "render"),
    )

    output_obj = _object(raw, "output", "config")
    base_dir = config_path.parent
    image = _required_path(output_obj, "image", "output", base_dir)
    normal = _optional_path(output_obj, "normal_image", "output", base_dir)
    shadow_image = _optional_path(output_obj, "shadow_image", "output", base_dir)
    terrain_file = _required_path(output_obj, "terrain_file", "output", base_dir)
    tile_report = _required_path(output_obj, "tile_report", "output", base_dir)
    cache_dir = _required_path(output_obj, "cache_dir", "output", base_dir)
    if render.write_debug_images:
        if normal is None:
            raise config_error(
                "'output.normal_image' is required when render.write_debug_images is true"
            )
        if shadow_image is None:
            raise config_error(
                "'output.shadow_image' is required when render.write_debug_images is true"
            )
    output = OutputConfig(
        image=image,
        normal_image=normal,
        shadow_image=shadow_image,
        terrain_file=terrain_file,
        tile_report=tile_report,
        cache_dir=cache_dir,
    )

    limits_obj = _object(raw, "limits", "config")
    max_terrain_cells = int(
        _number(
            limits_obj,
            "max_terrain_cells",
            "limits",
            minimum=16,
            maximum=100_000_000,
            integer=True,
        )
    )
    if rows * cols > max_terrain_cells:
        raise config_error(
            f"terrain size {rows}x{cols} exceeds limits.max_terrain_cells={max_terrain_cells}"
        )
    max_memory_raw = limits_obj.get("max_memory_mb")
    max_samples_raw = limits_obj.get("max_total_samples")
    max_memory: float | None = None
    max_samples: int | None = None
    if max_memory_raw is not None:
        max_memory = float(
            _number(limits_obj, "max_memory_mb", "limits", minimum=4.0, maximum=65536.0)
        )
    if max_samples_raw is not None:
        max_samples = int(
            _number(limits_obj, "max_total_samples", "limits", minimum=1, maximum=2**62, integer=True)
        )
    limits = LimitsConfig(
        max_memory_mb=max_memory,
        max_total_samples=max_samples,
        max_terrain_cells=max_terrain_cells,
    )

    resolved_paths = [
        image,
        terrain_file,
        tile_report,
        cache_dir,
    ]
    if normal is not None:
        resolved_paths.append(normal)
    if shadow_image is not None:
        resolved_paths.append(shadow_image)
    if len({str(path).casefold() for path in resolved_paths}) != len(resolved_paths):
        raise config_error("output paths must be distinct")

    return AppConfig(
        config_path=config_path,
        terrain=TerrainConfig(
            rows=rows,
            cols=cols,
            seed=seed,
            world_width=world_width,
            world_depth=world_depth,
            height_scale=height_scale,
            tile_cells=tile_cells,
            noise=noise,
            height_patches=height_patches,
        ),
        camera=camera,
        materials=materials,
        light=light,
        render=render,
        output=output,
        limits=limits,
    )

