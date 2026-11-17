from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import tempfile
import time
from typing import Sequence

import numpy as np

from .cache import RenderCache
from .config import (
    Camera,
    DirectionalLight,
    Material,
    PointLight,
    RenderSettings,
    default_materials,
    default_point_lights,
)
from .errors import ValidationError
from .geometry import TerrainGeometry, affected_triangle_indices, build_geometry, changed_world_bounds
from .manifest import array_digest, build_manifest, digest_json, structural_config
from .math3d import camera_matrices, normalize, to_homogeneous
from .pngio import encode_gray16, encode_gray8, encode_rgb8
from .raster import rasterize_selected_tiles, tile_grid
from .shade import quantize_u16, quantize_u8, tone_map
from .shadows import (
    build_cascades,
    rasterize_cascade_depth,
    update_cascade_depths,
)


@dataclass
class RenderResult:
    color_linear: np.ndarray
    depth_linear: np.ndarray
    visibility: np.ndarray
    color_u8: np.ndarray
    depth_u16: np.ndarray
    shadow_u8: np.ndarray
    manifest: dict
    warnings: dict
    stats: dict
    cache: RenderCache

    def save(self, output_dir: str, cache_path: str | None = None) -> dict:
        output_dir = os.path.abspath(output_dir)
        parent = os.path.dirname(output_dir) or "."
        os.makedirs(parent, exist_ok=True)
        temporary = tempfile.mkdtemp(prefix=".terrain-output-", dir=parent)
        names = ("color.png", "depth.png", "shadow_visibility.png", "manifest.json")
        targets = {name: os.path.join(output_dir, name) for name in names}
        start = time.perf_counter()
        try:
            payloads = {
                "color.png": encode_rgb8(self.color_u8),
                "depth.png": encode_gray16(self.depth_u16),
                "shadow_visibility.png": encode_gray8(self.shadow_u8),
            }
            for name, payload in payloads.items():
                with open(os.path.join(temporary, name), "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            io_ms = (time.perf_counter() - start) * 1000.0
            manifest = json.loads(json.dumps(self.manifest, ensure_ascii=False, allow_nan=False))
            render_total_ms = float(manifest["timings_ms"].get("total_ms", 0.0)) - float(
                manifest["timings_ms"].get("io_ms", 0.0)
            )
            manifest["timings_ms"]["io_ms"] = round(io_ms, 3)
            manifest["timings_ms"]["total_ms"] = round(render_total_ms + io_ms, 3)
            manifest_path = os.path.join(temporary, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.makedirs(output_dir, exist_ok=True)
            for name in names:
                os.replace(os.path.join(temporary, name), targets[name])
            self.manifest = manifest
        finally:
            try:
                for name in names:
                    candidate = os.path.join(temporary, name)
                    if os.path.exists(candidate):
                        os.unlink(candidate)
                os.rmdir(temporary)
            except OSError:
                pass
        if cache_path:
            self.cache.save(cache_path)
        return self.manifest


def _normalize_inputs(
    height: np.ndarray,
    weight_maps: np.ndarray,
    camera: Camera,
    directional_light: DirectionalLight,
    point_lights: Sequence[PointLight] | None,
    materials: Sequence[Material] | None,
    settings: RenderSettings,
    unit_scale: float,
    height_scale: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    Camera,
    DirectionalLight,
    tuple[PointLight, ...],
    tuple[Material, ...],
    RenderSettings,
]:
    height = np.asarray(height, dtype=np.float64)
    weights = np.asarray(weight_maps, dtype=np.float64)
    if height.ndim != 2 or min(height.shape) < 2:
        raise ValidationError("高度场必须是至少 2x2 的二维数组")
    if not np.all(np.isfinite(height)):
        raise ValidationError("高度场包含 NaN 或无穷值")
    if weights.ndim != 3 or weights.shape[1:] != height.shape:
        raise ValidationError(
            f"权重图尺寸不匹配: 期望 (*, {height.shape[0]}, {height.shape[1]}), 实际 {weights.shape}"
        )
    if not np.all(np.isfinite(weights)):
        raise ValidationError("权重图包含 NaN 或无穷值")
    if not math.isfinite(unit_scale) or unit_scale <= 0.0:
        raise ValidationError("unit_scale 必须为正有限数")
    if not math.isfinite(height_scale) or height_scale <= 0.0:
        raise ValidationError("height_scale 必须为正有限数")
    settings = settings.normalized()
    camera = camera.normalized(settings.width, settings.height)
    light = directional_light.normalized()
    if point_lights is None:
        points = default_point_lights()
    else:
        points = tuple(light.normalized(index) for index, light in enumerate(point_lights))
    if len(points) < 2:
        raise ValidationError("至少需要两个点光源")
    if materials is None:
        material_tuple = default_materials()
    else:
        material_tuple = tuple(material.normalized(index) for index, material in enumerate(materials))
    if len(material_tuple) != weights.shape[0]:
        raise ValidationError(
            f"材质图层数 ({len(material_tuple)}) 与权重图通道数 ({weights.shape[0]}) 不一致"
        )
    return height, weights, camera, light, points, material_tuple, settings


def validate_dirty_rect(rect: tuple[int, int, int, int] | None, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if rect is None:
        return None
    if len(rect) != 4:
        raise ValidationError("脏区必须为 x0 y0 x1 y1 四个整数")
    for value in rect:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValidationError("脏区坐标必须是整数")
    values = tuple(int(value) for value in rect)
    x0, y0, x1, y1 = values
    rows, columns = shape
    if not (0 <= x0 < x1 <= columns and 0 <= y0 < y1 <= rows):
        raise ValidationError(
            f"脏区越界: ({x0},{y0},{x1},{y1})，高度场尺寸为 {columns}x{rows}，要求 x0<x1、y0<y1"
        )
    return values


def _diff_bounding_rect(old: np.ndarray, new: np.ndarray) -> tuple[int, int, int, int] | None:
    if old.shape != new.shape:
        return None
    changed = old != new
    if changed.ndim == 3:
        changed = np.any(changed, axis=0)
    if not np.any(changed):
        return None
    rows = np.flatnonzero(np.any(changed, axis=1))
    columns = np.flatnonzero(np.any(changed, axis=0))
    return int(columns[0]), int(rows[0]), int(columns[-1] + 1), int(rows[-1] + 1)


def _rect_covers(rect: tuple[int, int, int, int], diff: tuple[int, int, int, int]) -> bool:
    return rect[0] <= diff[0] and rect[1] <= diff[1] and rect[2] >= diff[2] and rect[3] >= diff[3]


def _project_bounds_to_tiles(
    points: np.ndarray,
    camera: Camera,
    settings: RenderSettings,
) -> set[int]:
    columns, rows = tile_grid(settings.width, settings.height, settings.tile_size)
    total = columns * rows
    _, _, view_projection = camera_matrices(camera, settings.width, settings.height)
    clip = to_homogeneous(np.asarray(points, dtype=np.float64)) @ view_projection.T
    w = clip[:, 3]
    if np.any(~np.isfinite(clip)) or np.any(w <= 1e-7):
        return set(range(total))
    ndc = clip[:, :3] / w[:, None]
    if np.any(np.abs(ndc[:, :2]) > 1e8):
        return set(range(total))
    x = (ndc[:, 0] * 0.5 + 0.5) * settings.width
    y = (0.5 - ndc[:, 1] * 0.5) * settings.height
    min_x = max(0.0, float(np.min(x)))
    max_x = min(float(settings.width), float(np.max(x)))
    min_y = max(0.0, float(np.min(y)))
    max_y = min(float(settings.height), float(np.max(y)))
    if max_x <= 0.0 or max_y <= 0.0 or min_x >= settings.width or min_y >= settings.height:
        return set()
    tx0 = max(0, int(math.floor(min_x / settings.tile_size)) - 1)
    tx1 = min(columns - 1, int(math.floor(max_x / settings.tile_size)) + 1)
    ty0 = max(0, int(math.floor(min_y / settings.tile_size)) - 1)
    ty1 = min(rows - 1, int(math.floor(max_y / settings.tile_size)) + 1)
    return {ty * columns + tx for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)}


def _dirty_primary_tiles(
    world_min: np.ndarray,
    world_max: np.ndarray,
    scene_min: np.ndarray,
    scene_max: np.ndarray,
    camera: Camera,
    directional_light: DirectionalLight,
    settings: RenderSettings,
) -> set[int]:
    points = [
        (x, y, z)
        for x in (world_min[0], world_max[0])
        for y in (world_min[1], world_max[1])
        for z in (world_min[2], world_max[2])
    ]
    extent = float(np.linalg.norm(scene_max - scene_min))
    direction = normalize(np.asarray(directional_light.direction, dtype=np.float64))
    for x, y, z in list(points):
        shifted = np.asarray((x, y, z)) + direction * extent
        shifted = np.clip(shifted, scene_min, scene_max)
        points.append(tuple(shifted))
    return _project_bounds_to_tiles(np.asarray(points, dtype=np.float64), camera, settings)


def _quantize_outputs(
    color_linear: np.ndarray,
    depth_linear: np.ndarray,
    visibility: np.ndarray,
    camera: Camera,
    settings: RenderSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mapped = tone_map(color_linear, settings.tone.exposure, settings.tone.gamma)
    color_u8 = quantize_u8(mapped)
    depth_normalized = np.clip((depth_linear - camera.near) / (camera.far - camera.near), 0.0, 1.0)
    depth_normalized[~np.isfinite(depth_linear)] = 1.0
    depth_u16 = quantize_u16(depth_normalized)
    shadow_u8 = quantize_u8(np.where(np.isfinite(visibility), visibility, 0.0))
    return color_u8, depth_u16, shadow_u8


def _make_cache(
    height: np.ndarray,
    source_weights: np.ndarray,
    color: np.ndarray,
    depth: np.ndarray,
    visibility: np.ndarray,
    cascades,
    structural_hash: str,
    height_hash: str,
    weights_hash: str,
    structural: dict,
    warning_codes: list[str],
) -> RenderCache:
    shadow_maps = np.stack([cascade.depth_map for cascade in cascades], axis=0)
    return RenderCache(
        height=height.copy(),
        source_weights=source_weights.copy(),
        color_linear=color.copy(),
        depth_linear=depth.copy(),
        visibility=visibility.copy(),
        shadow_maps=shadow_maps.copy(),
        structural_hash=structural_hash,
        height_hash=height_hash,
        weights_hash=weights_hash,
        structural_config=structural,
        cascade_splits=[cascade.split_far for cascade in cascades],
        warning_codes=sorted(set(warning_codes)),
    )


def render_scene(
    height: np.ndarray,
    weight_maps: np.ndarray,
    camera: Camera | None = None,
    directional_light: DirectionalLight | None = None,
    point_lights: Sequence[PointLight] | None = None,
    materials: Sequence[Material] | None = None,
    settings: RenderSettings | None = None,
    unit_scale: float = 1.0,
    height_scale: float = 1.0,
    cache: RenderCache | None = None,
    dirty_rect: tuple[int, int, int, int] | None = None,
    input_metadata: dict | None = None,
) -> RenderResult:
    start_total = time.perf_counter()
    camera = Camera() if camera is None else camera
    directional_light = DirectionalLight() if directional_light is None else directional_light
    settings = RenderSettings() if settings is None else settings
    (
        height,
        source_weights,
        camera,
        directional_light,
        point_lights,
        materials,
        settings,
    ) = _normalize_inputs(
        height,
        weight_maps,
        camera,
        directional_light,
        point_lights,
        materials,
        settings,
        unit_scale,
        height_scale,
    )
    requested_dirty = validate_dirty_rect(dirty_rect, height.shape)
    structural = structural_config(camera, directional_light, point_lights, materials, settings)
    structural["geometry_scales"] = {
        "unit_scale": float(unit_scale),
        "height_scale": float(height_scale),
    }
    structural_hash = digest_json(structural)
    height_hash = array_digest(height)
    weights_hash = array_digest(source_weights)
    geometry_start = time.perf_counter()
    geometry = build_geometry(height, source_weights, unit_scale, height_scale)
    geometry_ms = (time.perf_counter() - geometry_start) * 1000.0
    warning_codes = set(geometry.warnings.codes)
    warning_details = list(geometry.warnings.details)
    incremental: dict = {
        "mode": "full",
        "full_rebuild": True,
        "reason": "initial_render",
        "requested_dirty_rect": list(requested_dirty) if requested_dirty else None,
        "effective_dirty_rect": list(requested_dirty) if requested_dirty else None,
        "cache_used": cache is not None,
        "unit_scale": float(unit_scale),
        "height_scale": float(height_scale),
        "total_tiles": 0,
        "rendered_tiles": 0,
        "shadow_tiles_updated": 0,
    }
    timings = {"geometry_ms": round(geometry_ms, 3)}
    full_reason = "initial_render"
    use_incremental = False
    effective_dirty = requested_dirty
    if cache is not None:
        if cache.structural_hash != structural_hash:
            full_reason = "structural_signature_changed"
        elif cache.height.shape != height.shape or cache.source_weights.shape != source_weights.shape:
            full_reason = "input_shape_changed"
        else:
            height_diff = _diff_bounding_rect(cache.height, height)
            weight_diff = _diff_bounding_rect(cache.source_weights, source_weights)
            diffs = [rect for rect in (height_diff, weight_diff) if rect is not None]
            if not diffs:
                if effective_dirty is None:
                    effective_dirty = (0, 0, 1, 1)
                    use_incremental = True
                    incremental["reason"] = "no_visible_input_change"
                else:
                    use_incremental = True
                    incremental["reason"] = "dirty_rect_without_data_change"
            else:
                actual = (
                    min(rect[0] for rect in diffs),
                    min(rect[1] for rect in diffs),
                    max(rect[2] for rect in diffs),
                    max(rect[3] for rect in diffs),
                )
                if effective_dirty is None:
                    effective_dirty = actual
                    use_incremental = True
                    incremental["reason"] = "data_diff_inside_cache"
                elif _rect_covers(effective_dirty, actual):
                    use_incremental = True
                    incremental["reason"] = "valid_dirty_rect"
                else:
                    full_reason = "dirty_rect_incomplete"
                    warning_codes.add("DIRTY_RECT_INCOMPLETE")
                    warning_details.append(
                        {
                            "code": "DIRTY_RECT_INCOMPLETE",
                            "message": "脏区未覆盖全部输入变化，已安全回退为整体重建",
                            "requested_dirty_rect": list(effective_dirty),
                            "actual_changed_rect": list(actual),
                        }
                    )
    if use_incremental:
        old_geometry = build_geometry(cache.height, cache.source_weights, unit_scale, height_scale)
        affected = affected_triangle_indices(old_geometry, geometry, effective_dirty)
        if affected.size == 0:
            selected_tiles: set[int] = set()
            shadow_updates = [None] * settings.cascades.count
            color = cache.color_linear.copy()
            depth = cache.depth_linear.copy()
            visibility = cache.visibility.copy()
            cascades = build_cascades(camera, directional_light, settings.width, settings.height, settings.cascades)
            for index, cascade in enumerate(cascades):
                cascade.depth_map = cache.shadow_maps[index].copy()
            shadow_ms = 0.0
            raster_ms = 0.0
            raster_stats = {
                "fragments": 0,
                "rendered_triangles": 0,
                "projected_triangles": 0,
                "total_tiles": tile_grid(settings.width, settings.height, settings.tile_size)[0]
                * tile_grid(settings.width, settings.height, settings.tile_size)[1],
                "rendered_tiles": 0,
                "projection": {},
            }
        else:
            world_min, world_max = changed_world_bounds(old_geometry, geometry, affected)
            shadow_start = time.perf_counter()
            cascades = build_cascades(camera, directional_light, settings.width, settings.height, settings.cascades)
            for index, cascade in enumerate(cascades):
                cascade.depth_map = cache.shadow_maps[index].copy()
            shadow_updates = update_cascade_depths(cascades, geometry, world_min, world_max)
            shadow_ms = (time.perf_counter() - shadow_start) * 1000.0
            scene_min = np.minimum(old_geometry.bounds[0], geometry.bounds[0])
            scene_max = np.maximum(old_geometry.bounds[1], geometry.bounds[1])
            selected_tiles = _dirty_primary_tiles(
                world_min,
                world_max,
                scene_min,
                scene_max,
                camera,
                directional_light,
                settings,
            )
            color = cache.color_linear.copy()
            depth = cache.depth_linear.copy()
            visibility = cache.visibility.copy()
            raster_start = time.perf_counter()
            raster_stats = rasterize_selected_tiles(
                geometry,
                camera,
                directional_light,
                point_lights,
                materials,
                settings,
                cascades,
                color,
                depth,
                visibility,
                selected_tiles,
            )
            raster_ms = (time.perf_counter() - raster_start) * 1000.0
        incremental = {
            "mode": "incremental",
            "full_rebuild": False,
            "reason": incremental["reason"],
            "requested_dirty_rect": list(requested_dirty) if requested_dirty else None,
            "effective_dirty_rect": list(effective_dirty) if effective_dirty else None,
            "cache_used": True,
            "unit_scale": float(unit_scale),
            "height_scale": float(height_scale),
            "total_tiles": raster_stats["total_tiles"],
            "rendered_tiles": raster_stats["rendered_tiles"],
            "rendered_triangles": raster_stats["rendered_triangles"],
            "shadow_tiles_updated": int(
                sum(
                    update["dirty_texels"]
                    for update in shadow_updates
                    if update is not None
                )
            ),
            "projection": raster_stats["projection"],
        }
        timings["shadows_ms"] = round(shadow_ms, 3)
        timings["raster_ms"] = round(raster_ms, 3)
    else:
        incremental["reason"] = full_reason
        shadow_start = time.perf_counter()
        cascades = build_cascades(camera, directional_light, settings.width, settings.height, settings.cascades)
        shadow_updates = []
        for cascade in cascades:
            shadow_updates.append(rasterize_cascade_depth(cascade, geometry))
        shadow_ms = (time.perf_counter() - shadow_start) * 1000.0
        color = np.empty((settings.height, settings.width, 3), dtype=np.float32)
        depth = np.empty((settings.height, settings.width), dtype=np.float32)
        visibility = np.empty((settings.height, settings.width), dtype=np.float32)
        columns, rows = tile_grid(settings.width, settings.height, settings.tile_size)
        all_tiles = set(range(columns * rows))
        raster_start = time.perf_counter()
        raster_stats = rasterize_selected_tiles(
            geometry,
            camera,
            directional_light,
            point_lights,
            materials,
            settings,
            cascades,
            color,
            depth,
            visibility,
            all_tiles,
        )
        raster_ms = (time.perf_counter() - raster_start) * 1000.0
        timings["shadows_ms"] = round(shadow_ms, 3)
        timings["raster_ms"] = round(raster_ms, 3)
        incremental.update(
            {
                "mode": "full",
                "full_rebuild": True,
                "total_tiles": raster_stats["total_tiles"],
                "rendered_tiles": raster_stats["rendered_tiles"],
                "rendered_triangles": raster_stats["rendered_triangles"],
                "shadow_tiles_updated": int(
                    sum(update.get("writes", 0) for update in shadow_updates if update is not None)
                ),
                "projection": raster_stats["projection"],
                "cache_used": cache is not None,
            }
        )
    composite_start = time.perf_counter()
    color_u8, depth_u16, shadow_u8 = _quantize_outputs(color, depth, visibility, camera, settings)
    composite_ms = (time.perf_counter() - composite_start) * 1000.0
    timings["composite_ms"] = round(composite_ms, 3)
    total_render_ms = (time.perf_counter() - start_total) * 1000.0
    timings["total_ms"] = round(total_render_ms, 3)
    output_files = {
        "color": "color.png",
        "depth": "depth.png",
        "shadow_visibility": "shadow_visibility.png",
    }
    warnings = {"codes": sorted(warning_codes), "details": warning_details}
    manifest = build_manifest(
        settings=settings,
        camera=camera,
        directional_light=directional_light,
        point_lights=point_lights,
        materials=materials,
        height=height,
        source_weights=source_weights,
        height_hash=height_hash,
        weights_hash=weights_hash,
        structural_hash=structural_hash,
        cascades=cascades,
        warnings=warnings,
        timings=timings,
        incremental=incremental,
        output_files=output_files,
        input_metadata=input_metadata,
    )
    stats = {
        "incremental": incremental,
        "raster": raster_stats,
        "timings_ms": timings,
        "warning_codes": sorted(warning_codes),
    }
    cache_object = _make_cache(
        height,
        source_weights,
        color,
        depth,
        visibility,
        cascades,
        structural_hash,
        height_hash,
        weights_hash,
        structural,
        sorted(warning_codes),
    )
    return RenderResult(
        color_linear=color,
        depth_linear=depth,
        visibility=visibility,
        color_u8=color_u8,
        depth_u16=depth_u16,
        shadow_u8=shadow_u8,
        manifest=manifest,
        warnings=warnings,
        stats=stats,
        cache=cache_object,
    )


def incremental_render(
    height: np.ndarray,
    weight_maps: np.ndarray,
    cache: RenderCache,
    dirty_rect: tuple[int, int, int, int],
    camera: Camera | None = None,
    directional_light: DirectionalLight | None = None,
    point_lights: Sequence[PointLight] | None = None,
    materials: Sequence[Material] | None = None,
    settings: RenderSettings | None = None,
    unit_scale: float = 1.0,
    height_scale: float = 1.0,
    input_metadata: dict | None = None,
) -> RenderResult:
    return render_scene(
        height,
        weight_maps,
        camera=camera,
        directional_light=directional_light,
        point_lights=point_lights,
        materials=materials,
        settings=settings,
        unit_scale=unit_scale,
        height_scale=height_scale,
        cache=cache,
        dirty_rect=dirty_rect,
        input_metadata=input_metadata,
    )
