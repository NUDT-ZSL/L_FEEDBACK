from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .errors import UsageError
from .geometry import make_view_camera, prepare_screen_triangles
from .imageio import downsample, pixel_result_hash
from .presets import QualityPreset, get_preset, validate_block_size
from .raster import render_blocks, touched_blocks
from .scene import Scene
from .shading import empty_shadow_statistics, shade_gbuffer
from .shadows import ShadowMap, build_shadow_maps


@dataclass
class LayerBuffer:
    material_id: int
    color: np.ndarray
    depth: np.ndarray


@dataclass
class RenderResult:
    frame: np.ndarray
    layers: Dict[int, LayerBuffer]
    preset: QualityPreset
    block_size: int
    no_tiling: bool
    render_width: int
    render_height: int
    material_id: Optional[int]
    background: np.ndarray
    shadow_statistics: Dict[str, Dict[str, int]]
    timings_ms: Dict[str, float]
    light_shadow_resolutions: Dict[str, int]

    @property
    def pixel_hash(self) -> str:
        return pixel_result_hash(self.frame)

    @property
    def width(self) -> int:
        return int(self.frame.shape[1])

    @property
    def height(self) -> int:
        return int(self.frame.shape[0])


def compose_layers(
    layers: Dict[int, LayerBuffer],
    background: np.ndarray,
) -> np.ndarray:
    if not layers:
        raise ValueError("at least one layer is required")
    first = next(iter(layers.values()))
    height, width = first.depth.shape
    result = np.zeros((height, width, 3), dtype=np.float32)
    best_depth = np.full((height, width), np.inf, dtype=np.float32)
    covered_any = np.zeros((height, width), dtype=bool)
    for material_id in sorted(layers):
        layer = layers[material_id]
        if layer.depth.shape != (height, width) or layer.color.shape != (height, width, 3):
            raise ValueError("all layers must have identical dimensions")
        covered = np.isfinite(layer.depth)
        update = covered & (layer.depth < best_depth)
        if np.any(update):
            result[update] = layer.color[update]
            best_depth[update] = layer.depth[update]
            covered_any[update] = True
    if np.any(~covered_any):
        result[~covered_any] = np.asarray(background, dtype=np.float32)
    return result


def render_scene(
    scene: Scene,
    block_size: int = 4,
    preset_name: str = "medium",
    material_id: Optional[int] = None,
    no_tiling: bool = False,
) -> RenderResult:
    total_started = time.perf_counter()
    requested_block_size = validate_block_size(block_size)
    preset = get_preset(preset_name)
    if material_id is not None and material_id not in scene.materials:
        raise UsageError("unknown material ID %d" % material_id)

    render_width = scene.width * preset.supersample
    render_height = scene.height * preset.supersample
    if render_width > 16384 or render_height > 16384:
        raise UsageError(
            "supersampled render is too large (%dx%d)" % (render_width, render_height)
        )
    effective_block_size = (
        max(render_width, render_height) if no_tiling else requested_block_size
    )
    camera = make_view_camera(scene.camera, render_width, render_height)

    timings: Dict[str, float] = {}
    stage_started = time.perf_counter()
    shadow_maps, shadow_timings = build_shadow_maps(scene, preset.shadow_resolution)
    timings["shadow_maps"] = (time.perf_counter() - stage_started) * 1000.0
    for name, duration in shadow_timings.items():
        timings["shadow_map_" + name] = duration

    light_shadow_resolutions: Dict[str, int] = {}
    for light in scene.lights:
        if light.cast_shadow:
            light_shadow_resolutions[str(light.id)] = (
                light.shadow_resolution or preset.shadow_resolution
            )

    selected_materials = [material_id] if material_id is not None else list(scene.material_order)
    layers: Dict[int, LayerBuffer] = {}
    statistics = empty_shadow_statistics(scene)
    stage_started = time.perf_counter()
    geometry_ms = 0.0
    raster_ms = 0.0
    for current_material in selected_materials:
        prepare_started = time.perf_counter()
        triangles = prepare_screen_triangles(scene, camera, [current_material])
        geometry_ms += (time.perf_counter() - prepare_started) * 1000.0

        def shade_tile(gbuffer: Dict[str, np.ndarray]) -> np.ndarray:
            return shade_gbuffer(
                gbuffer,
                scene,
                camera,
                scene.materials,
                shadow_maps,
                preset.pcf_samples,
                statistics,
                scene.materials[current_material],
            )

        material_started = time.perf_counter()
        if triangles:
            blocks = touched_blocks(
                triangles, render_width, render_height, effective_block_size
            )
            color, depth = render_blocks(
                triangles,
                render_width,
                render_height,
                effective_block_size,
                camera.far,
                shade_tile,
                background=np.zeros(3, dtype=np.float64),
                blocks=blocks,
            )
        else:
            color = np.zeros((render_height, render_width, 3), dtype=np.float64)
            depth = np.full((render_height, render_width), np.inf, dtype=np.float64)
        raster_ms += (time.perf_counter() - material_started) * 1000.0
        layers[current_material] = LayerBuffer(
            material_id=current_material,
            color=np.ascontiguousarray(color, dtype=np.float32),
            depth=np.ascontiguousarray(depth, dtype=np.float32),
        )
    timings["geometry_prepare"] = geometry_ms
    timings["raster_shade"] = raster_ms

    stage_started = time.perf_counter()
    if material_id is not None:
        rendered = layers[material_id].color
        background = np.zeros(3, dtype=np.float64)
    else:
        rendered = compose_layers(layers, scene.background)
        background = scene.background
    timings["layer_composite"] = (time.perf_counter() - stage_started) * 1000.0

    stage_started = time.perf_counter()
    frame = downsample(rendered, preset.supersample)
    timings["resolve_supersample"] = (time.perf_counter() - stage_started) * 1000.0
    timings["total"] = (time.perf_counter() - total_started) * 1000.0

    return RenderResult(
        frame=np.ascontiguousarray(frame, dtype=np.float32),
        layers=layers,
        preset=preset,
        block_size=effective_block_size,
        no_tiling=no_tiling,
        render_width=render_width,
        render_height=render_height,
        material_id=material_id,
        background=np.asarray(background, dtype=np.float64),
        shadow_statistics=statistics,
        timings_ms=timings,
        light_shadow_resolutions=light_shadow_resolutions,
    )
