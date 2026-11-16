"""Top-level incremental terrain preview pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cache import (
    MANIFEST_FORMAT,
    cleanup_cache,
    load_cached_tile,
    load_manifest,
    load_shadow_map,
    save_cached_tile,
    save_manifest,
    save_shadow_map,
    screen_tile_shape,
    stable_hash,
    write_json_atomic,
)
from .config import AppConfig
from .errors import cache_error, output_error, render_error, resource_error
from .light import (
    LightingResources,
    build_lighting_resources,
    fit_light_camera,
    pcf_visibility,
    render_shadow_map,
)
from .material import material_albedo
from .png import StreamingPNG
from .raster import allocate_gbuffer, rasterize_tile_camera
from .scene import (
    OrbitCamera,
    VisibilityResult,
    build_world_tiles,
    extract_tile_mesh,
    setup_camera,
)
from .shading import shade_fragments
from .terrain import TerrainData, build_terrain


_GBUFFER_BYTES_PER_PIXEL = 41


@dataclass(frozen=True)
class RenderStats:
    world_tile_count: int
    visible_world_tile_count: int
    culled_frustum: int
    culled_backface: int
    screen_tile_count: int
    redrawn_tiles: int
    reused_tiles: int
    shadow_generated: bool
    shadow_reused: bool
    shadow_check_tiles: int
    shadow_check_samples: int
    shadow_extra_dirty: int
    memory_chunked: bool
    batch_columns: int
    estimated_samples: int
    actual_samples: int
    terrain_sha256: str


def _screen_key(row: int, col: int) -> str:
    return f"r{row}c{col}"


def _screen_counts(config: AppConfig) -> tuple[int, int]:
    return (
        math.ceil(config.render.height / config.render.tile_size),
        math.ceil(config.render.width / config.render.tile_size),
    )


def _screen_keys(config: AppConfig) -> list[tuple[int, int]]:
    tile_rows, tile_cols = _screen_counts(config)
    return [(row, col) for row in range(tile_rows) for col in range(tile_cols)]


def _geometry_key(config: AppConfig) -> str:
    terrain = config.terrain
    return stable_hash(
        {
            "rows": terrain.rows,
            "cols": terrain.cols,
            "world_width": terrain.world_width,
            "world_depth": terrain.world_depth,
            "height_scale": terrain.height_scale,
            "tile_cells": terrain.tile_cells,
        }
    )


def _camera_key(config: AppConfig, geometry_key: str) -> str:
    camera = config.camera
    return stable_hash(
        {
            "geometry": geometry_key,
            "azimuth_deg": camera.azimuth_deg,
            "elevation_deg": camera.elevation_deg,
            "distance": camera.distance,
            "fov_y_deg": camera.fov_y_deg,
            "near": camera.near,
            "far": camera.far,
            "target_fraction": camera.target_fraction,
            "width": config.render.width,
            "height": config.render.height,
            "tile_size": config.render.tile_size,
        }
    )


def _lighting_key(config: AppConfig) -> str:
    light = config.light
    return stable_hash(
        {
            "direction": [light.azimuth_deg, light.elevation_deg],
            "color": light.color_srgb,
            "intensity": light.intensity,
            "ambient": {
                "sky": light.ambient.sky_color_srgb,
                "ground": light.ambient.ground_color_srgb,
                "intensity": light.ambient.intensity,
            },
            "shadow": {
                "enabled": light.shadow.enabled,
                "resolution": light.shadow.resolution,
                "depth_bias": light.shadow.depth_bias,
                "normal_offset": light.shadow.normal_offset,
                "pcf_samples": light.shadow.pcf_samples,
                "filter_radius": light.shadow.filter_radius,
            },
            "exposure": config.render.exposure,
            "gamma": config.render.gamma,
            "background": config.render.background_srgb,
        }
    )


def _shadow_key(config: AppConfig, geometry_key: str, terrain: TerrainData) -> str:
    light = config.light
    return stable_hash(
        {
            "enabled": light.shadow.enabled,
            "direction": [light.azimuth_deg, light.elevation_deg],
            "resolution": light.shadow.resolution,
            "depth_bias": light.shadow.depth_bias,
            "normal_offset": light.shadow.normal_offset,
            "pcf_samples": light.shadow.pcf_samples,
            "filter_radius": light.shadow.filter_radius,
            "geometry": geometry_key,
            "terrain_sha256": terrain.sha256,
        }
    )


def _hash_parts(parts: list[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(struct.pack("<Q", len(part)))
        digest.update(part)
    return digest.hexdigest()


def _height_version(config: AppConfig, terrain: TerrainData, tile: Any) -> str:
    values = np.ascontiguousarray(
        terrain.heights[
            tile.vertex_row0 : tile.vertex_row1 + 1,
            tile.vertex_col0 : tile.vertex_col1 + 1,
        ],
        dtype="<f4",
    )
    return _hash_parts(
        [
            struct.pack("<II", values.shape[0], values.shape[1]),
            values.tobytes(order="C"),
        ]
    )


def _material_version(config: AppConfig, terrain: TerrainData, tile: Any) -> str:
    heights = terrain.heights[
        tile.vertex_row0 : tile.vertex_row1 + 1,
        tile.vertex_col0 : tile.vertex_col1 + 1,
    ]
    normals = terrain.normals[
        tile.vertex_row0 : tile.vertex_row1 + 1,
        tile.vertex_col0 : tile.vertex_col1 + 1,
    ]
    altitude_vertices = heights.reshape(-1)
    normal_y_vertices = normals[..., 1].reshape(-1)
    altitude_centers = 0.25 * (
        heights[:-1, :-1] + heights[1:, :-1] + heights[:-1, 1:] + heights[1:, 1:]
    )
    normal_y_centers = 0.25 * (
        normals[:-1, :-1, 1]
        + normals[1:, :-1, 1]
        + normals[:-1, 1:, 1]
        + normals[1:, 1:, 1]
    )
    altitude = np.concatenate(
        (altitude_vertices, altitude_centers.reshape(-1))
    ).astype(np.float64)
    normal_y = np.concatenate(
        (normal_y_vertices, normal_y_centers.reshape(-1))
    ).astype(np.float64)
    weights, albedo = material_albedo(altitude, normal_y, config.materials)
    return _hash_parts(
        [
            weights.astype("<f4").tobytes(order="C"),
            albedo.astype("<f4").tobytes(order="C"),
        ]
    )


def _world_metadata(
    config: AppConfig, terrain: TerrainData, visibility: VisibilityResult
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for tile in visibility.all_tiles:
        metadata[tile.tile_id] = {
            "height_version": _height_version(config, terrain, tile),
            "material_version": _material_version(config, terrain, tile),
            "screen_bbox": list(tile.screen_bbox),
            "world_bbox": tile.bbox_dict(),
        }
    return metadata


def _bbox_screen_keys(
    bbox: list[int] | tuple[int, int, int, int], config: AppConfig
) -> set[tuple[int, int]]:
    x0, y0, x1, y1 = (int(value) for value in bbox)
    if x1 < x0 or y1 < y0:
        return set()
    tile_size = config.render.tile_size
    row0 = max(0, y0 // tile_size)
    row1 = min(config.render.height - 1, y1) // tile_size
    col0 = max(0, x0 // tile_size)
    col1 = min(config.render.width - 1, x1) // tile_size
    return {
        (row, col)
        for row in range(row0, row1 + 1)
        for col in range(col0, col1 + 1)
    }


def _validate_cached_entries(
    config: AppConfig, cache_dir: Any, manifest: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    entries_raw = manifest.get("screen_tiles")
    if not isinstance(entries_raw, dict):
        return {}
    tile_rows, tile_cols = _screen_counts(config)
    valid: dict[str, dict[str, Any]] = {}
    for row in range(tile_rows):
        for col in range(tile_cols):
            key = _screen_key(row, col)
            entry = entries_raw.get(key)
            if not isinstance(entry, dict):
                continue
            shape = screen_tile_shape(
                row, col, config.render.width, config.render.height, config.render.tile_size
            )
            if load_cached_tile(cache_dir, entry, shape) is None:
                continue
            valid[key] = entry
    return valid


def _compute_dirty_tiles(
    config: AppConfig,
    old_manifest: dict[str, Any] | None,
    valid_entries: dict[str, dict[str, Any]],
    current_metadata: dict[str, dict[str, Any]],
    render_key: str,
    camera_key: str,
    lighting_key: str,
    shadow_key: str,
    force_full: bool,
) -> tuple[set[tuple[int, int]], str]:
    all_keys = set(_screen_keys(config))
    if force_full:
        return all_keys, "forced full redraw"
    if old_manifest is None:
        return all_keys, "cache miss"
    if (
        old_manifest.get("render_key") != render_key
        or old_manifest.get("camera_key") != camera_key
        or old_manifest.get("width") != config.render.width
        or old_manifest.get("height") != config.render.height
        or old_manifest.get("tile_size") != config.render.tile_size
    ):
        return all_keys, "camera/viewport changed"
    if old_manifest.get("lighting_key") != lighting_key:
        return all_keys, "lighting parameters changed"
    dirty: set[tuple[int, int]] = set()
    for key in all_keys:
        screen_key = _screen_key(*key)
        if screen_key not in valid_entries:
            dirty.add(key)

    old_world = old_manifest.get("world_tiles")
    if not isinstance(old_world, dict):
        return all_keys, "cache world metadata missing"
    changed: set[str] = set()
    for tile_id in set(old_world) | set(current_metadata):
        old_meta = old_world.get(tile_id)
        current_meta = current_metadata.get(tile_id)
        if not isinstance(old_meta, dict) or current_meta is None:
            changed.add(tile_id)
            continue
        if (
            old_meta.get("height_version") != current_meta["height_version"]
            or old_meta.get("material_version") != current_meta["material_version"]
        ):
            changed.add(tile_id)

    for tile_id in changed:
        old_meta = old_world.get(tile_id)
        current_meta = current_metadata.get(tile_id)
        for metadata in (old_meta, current_meta):
            if isinstance(metadata, dict) and isinstance(metadata.get("screen_bbox"), list):
                dirty.update(_bbox_screen_keys(metadata["screen_bbox"], config))

    for screen_key, entry in valid_entries.items():
        world_ids = entry.get("world_tiles")
        if isinstance(world_ids, list) and changed.intersection(world_ids):
            row_text, col_text = screen_key[1:].split("c", 1)
            dirty.add((int(row_text), int(col_text)))
        if (
            entry.get("lighting_version") != lighting_key
        ):
            row_text, col_text = screen_key[1:].split("c", 1)
            dirty.add((int(row_text), int(col_text)))
    return dirty, "incremental signatures compared"


def _shadow_sensitive_tiles(
    config: AppConfig,
    valid_entries: dict[str, dict[str, Any]],
    directly_dirty: set[tuple[int, int]],
    lighting: LightingResources,
) -> tuple[set[tuple[int, int]], int, int]:
    """Find otherwise-static tiles whose cached PCF mask changes under a new shadow map."""

    extra: set[tuple[int, int]] = set()
    checked_samples = 0
    checked_tiles = 0
    tile_size = config.render.tile_size
    for row, col in _screen_keys(config):
        key = _screen_key(row, col)
        if (row, col) in directly_dirty or key not in valid_entries:
            continue
        shape = screen_tile_shape(
            row, col, config.render.width, config.render.height, tile_size
        )
        cached = load_cached_tile(
            config.output.cache_dir,
            valid_entries[key],
            shape,
            include_geometry=True,
        )
        if cached is None:
            extra.add((row, col))
            continue
        covered = cached["covered"]
        if not np.any(covered):
            continue
        positions = cached["positions"][covered].astype(np.float64)
        normals = cached["normals"][covered].astype(np.float64)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(lengths, 1e-15)
        visibility = pcf_visibility(positions, normals, config, lighting)
        new_shadow = np.clip(np.rint(visibility * 255.0), 0, 255).astype(np.uint8)
        checked_tiles += 1
        checked_samples += positions.shape[0] * config.light.shadow.pcf_samples
        if not np.array_equal(new_shadow, cached["shadow"][covered]):
            extra.add((row, col))
    return extra, checked_samples, checked_tiles


def _sample_estimate(
    config: AppConfig, dirty: set[tuple[int, int]]
) -> int:
    tile_size = config.render.tile_size
    pixels = 0
    for row, col in dirty:
        height, width = screen_tile_shape(
            row, col, config.render.width, config.render.height, tile_size
        )
        pixels += height * width
    samples_per_fragment = 1
    if config.light.shadow.enabled:
        samples_per_fragment += config.light.shadow.pcf_samples
    return pixels * samples_per_fragment


def _memory_layout(
    config: AppConfig, terrain: TerrainData, dirty: set[tuple[int, int]], shadow_bytes: int
) -> tuple[bool, int, float]:
    base_bytes = int(terrain.heights.nbytes + terrain.normals.nbytes + shadow_bytes)
    full_bytes = base_bytes + (
        config.render.width * config.render.height * _GBUFFER_BYTES_PER_PIXEL
    )
    max_bytes = (
        int(config.limits.max_memory_mb * 1024 * 1024)
        if config.limits.max_memory_mb is not None
        else None
    )
    tile_pixels = config.render.tile_size * config.render.tile_size
    if max_bytes is None:
        return False, max(1, math.ceil(config.render.width / config.render.tile_size)), full_bytes / (1024 * 1024)

    chunked = full_bytes > max_bytes
    available = max_bytes - base_bytes
    minimum_batch_bytes = tile_pixels * _GBUFFER_BYTES_PER_PIXEL
    if available < minimum_batch_bytes:
        raise resource_error(
            "limits.max_memory_mb is too small for terrain arrays, the shadow map, "
            "and one screen tile; reduce shadow resolution or increase the limit"
        )
    max_pixels = available // _GBUFFER_BYTES_PER_PIXEL
    max_columns = max_pixels // max(config.render.tile_size, 1)
    max_columns = max(1, int(max_columns // config.render.tile_size))
    max_columns = min(
        max_columns, math.ceil(config.render.width / config.render.tile_size)
    )
    return chunked, max_columns, full_bytes / (1024 * 1024)


def _batch_dirty_tiles(
    config: AppConfig, dirty: set[tuple[int, int]], max_columns: int
) -> list[list[tuple[int, int]]]:
    rows: dict[int, list[int]] = {}
    for row, col in dirty:
        rows.setdefault(row, []).append(col)
    batches: list[list[tuple[int, int]]] = []
    for row in sorted(rows):
        columns = sorted(rows[row])
        start = 0
        while start < len(columns):
            end = start
            while (
                end + 1 < len(columns)
                and columns[end + 1] == columns[end] + 1
                and end - start + 1 < max_columns
            ):
                end += 1
            if end - start + 1 > max_columns:
                end = start + max_columns - 1
            batches.append([(row, col) for col in columns[start : end + 1]])
            start = end + 1
    return batches


def _visible_tile_index(visibility: VisibilityResult) -> dict[str, int]:
    return {tile.tile_id: index for index, tile in enumerate(visibility.all_tiles)}


def _render_dirty_batches(
    config: AppConfig,
    terrain: TerrainData,
    visibility: VisibilityResult,
    camera: OrbitCamera,
    lighting: LightingResources,
    dirty: set[tuple[int, int]],
    max_columns: int,
    current_metadata: dict[str, dict[str, Any]],
    lighting_key: str,
    shadow_key: str,
) -> tuple[dict[str, dict[str, Any]], int]:
    tile_size = config.render.tile_size
    background = np.clip(
        np.rint(np.asarray(config.render.background_srgb) * 255.0), 0, 255
    ).astype(np.uint8)
    tile_id_by_index = _visible_tile_index(visibility)
    tile_index_by_id = _visible_tile_index(visibility)
    tile_entries: dict[str, dict[str, Any]] = {}
    shaded_fragments = 0

    for batch in _batch_dirty_tiles(config, dirty, max_columns):
        batch_row = batch[0][0]
        first_col = batch[0][1]
        last_col = batch[-1][1]
        batch_x0 = first_col * tile_size
        batch_x1 = min(config.render.width, (last_col + 1) * tile_size)
        batch_y0 = batch_row * tile_size
        batch_y1 = min(config.render.height, (batch_row + 1) * tile_size)
        batch_width = batch_x1 - batch_x0
        batch_height = batch_y1 - batch_y0
        if batch_width <= 0 or batch_height <= 0:
            continue

        gbuffer = allocate_gbuffer(batch_width, batch_height)
        dirty_mask = np.zeros((batch_height, batch_width), dtype=bool)
        for row, col in batch:
            x0 = col * tile_size - batch_x0
            y0 = row * tile_size - batch_y0
            height, width = screen_tile_shape(
                row, col, config.render.width, config.render.height, tile_size
            )
            dirty_mask[y0 : y0 + height, x0 : x0 + width] = True

        for tile in visibility.visible:
            bx0, by0, bx1, by1 = tile.screen_bbox
            ix0 = max(bx0, batch_x0) - batch_x0
            ix1 = min(bx1, batch_x1 - 1) - batch_x0
            iy0 = max(by0, batch_y0) - batch_y0
            iy1 = min(by1, batch_y1 - 1) - batch_y0
            if ix0 > ix1 or iy0 > iy1:
                continue
            if not dirty_mask[iy0 : iy1 + 1, ix0 : ix1 + 1].any():
                continue
            mesh = extract_tile_mesh(config, terrain, tile, with_normals=True)
            rasterize_tile_camera(
                mesh,
                camera.view_projection,
                gbuffer,
                tile_index_by_id[tile.tile_id],
                image_width=config.render.width,
                image_height=config.render.height,
                x_offset=batch_x0,
                y_offset=batch_y0,
            )

        rgb_buffer = np.empty((batch_height, batch_width, 3), dtype=np.uint8)
        rgb_buffer[...] = background
        normal_buffer = np.zeros((batch_height, batch_width, 3), dtype=np.uint8)
        shadow_buffer = np.full((batch_height, batch_width), 255, dtype=np.uint8)
        shade_mask = gbuffer.covered & dirty_mask
        shaded_count = int(np.count_nonzero(shade_mask))
        if shaded_count:
            positions = gbuffer.positions[shade_mask].astype(np.float64)
            normals = gbuffer.normals[shade_mask].astype(np.float64)
            rgb, normal_debug, shadow_debug = shade_fragments(
                positions, normals, config, lighting
            )
            rgb_buffer[shade_mask] = rgb
            normal_buffer[shade_mask] = normal_debug
            shadow_buffer[shade_mask] = shadow_debug
            shaded_fragments += shaded_count

        for row, col in batch:
            local_x0 = col * tile_size - batch_x0
            local_y0 = row * tile_size - batch_y0
            tile_height, tile_width = screen_tile_shape(
                row, col, config.render.width, config.render.height, tile_size
            )
            rgb = np.ascontiguousarray(
                rgb_buffer[
                    local_y0 : local_y0 + tile_height,
                    local_x0 : local_x0 + tile_width,
                ]
            )
            normal = np.ascontiguousarray(
                normal_buffer[
                    local_y0 : local_y0 + tile_height,
                    local_x0 : local_x0 + tile_width,
                ]
            )
            shadow = np.ascontiguousarray(
                shadow_buffer[
                    local_y0 : local_y0 + tile_height,
                    local_x0 : local_x0 + tile_width,
                ]
            )
            covered = np.ascontiguousarray(
                gbuffer.covered[
                    local_y0 : local_y0 + tile_height,
                    local_x0 : local_x0 + tile_width,
                ]
            )
            cached_positions = np.ascontiguousarray(
                gbuffer.positions[
                    local_y0 : local_y0 + tile_height,
                    local_x0 : local_x0 + tile_width,
                ]
            )
            cached_normals = np.ascontiguousarray(
                gbuffer.normals[
                    local_y0 : local_y0 + tile_height,
                    local_x0 : local_x0 + tile_width,
                ]
            )
            tile_indices = gbuffer.tile_indices[
                local_y0 : local_y0 + tile_height,
                local_x0 : local_x0 + tile_width,
            ][covered]
            world_ids = sorted(
                {
                    tile_id_by_index.get(int(index))
                    for index in np.unique(tile_indices)
                    if int(index) >= 0
                }
                - {None}
            )
            relative = save_cached_tile(
                config.output.cache_dir,
                rgb,
                normal,
                shadow,
                covered,
                cached_positions,
                cached_normals,
            )
            screen_key = _screen_key(row, col)
            tile_entries[screen_key] = {
                "file": relative,
                "world_tiles": world_ids,
                "height_versions": {
                    tile_id: current_metadata[tile_id]["height_version"]
                    for tile_id in world_ids
                },
                "material_versions": {
                    tile_id: current_metadata[tile_id]["material_version"]
                    for tile_id in world_ids
                },
                "lighting_version": lighting_key,
                "shadow_version": shadow_key,
                "covered_pixels": int(np.count_nonzero(covered)),
            }

    return tile_entries, shaded_fragments


def _write_preview_images(
    config: AppConfig, entries: dict[str, dict[str, Any]]
) -> None:
    tile_size = config.render.tile_size
    tile_rows, tile_cols = _screen_counts(config)
    debug = config.render.write_debug_images
    with ExitStack() as stack:
        rgb_writer = stack.enter_context(
            StreamingPNG(config.output.image, config.render.width, config.render.height)
        )
        normal_writer = None
        shadow_writer = None
        if debug:
            assert config.output.normal_image is not None
            assert config.output.shadow_image is not None
            normal_writer = stack.enter_context(
                StreamingPNG(
                    config.output.normal_image,
                    config.render.width,
                    config.render.height,
                )
            )
            shadow_writer = stack.enter_context(
                StreamingPNG(
                    config.output.shadow_image,
                    config.render.width,
                    config.render.height,
                    grayscale=True,
                )
            )

        for row in range(tile_rows):
            expected_height = min(
                tile_size, config.render.height - row * tile_size
            )
            rgb_parts: list[np.ndarray] = []
            normal_parts: list[np.ndarray] = []
            shadow_parts: list[np.ndarray] = []
            for col in range(tile_cols):
                key = _screen_key(row, col)
                entry = entries.get(key)
                if entry is None:
                    raise cache_error(f"missing cache entry for screen tile {key}")
                shape = screen_tile_shape(
                    row, col, config.render.width, config.render.height, tile_size
                )
                cached = load_cached_tile(config.output.cache_dir, entry, shape)
                if cached is None:
                    raise cache_error(f"cached data for screen tile {key} is missing or invalid")
                rgb_parts.append(cached["rgb"])
                if debug:
                    normal_parts.append(cached["normal"])
                    shadow_parts.append(cached["shadow"])
            rgb_band = np.concatenate(rgb_parts, axis=1)[:, : config.render.width]
            if rgb_band.shape[0] != expected_height:
                raise cache_error("cached screen tile row has inconsistent height")
            if debug:
                normal_band = np.concatenate(normal_parts, axis=1)[
                    :, : config.render.width
                ]
                shadow_band = np.concatenate(shadow_parts, axis=1)[
                    :, : config.render.width
                ]
            for index in range(rgb_band.shape[0]):
                rgb_writer.write_row(rgb_band[index])
                if debug and normal_writer is not None and shadow_writer is not None:
                    normal_writer.write_row(normal_band[index])
                    shadow_writer.write_row(shadow_band[index])


def _tile_report(
    config: AppConfig,
    terrain: TerrainData,
    camera: OrbitCamera,
    visibility: VisibilityResult,
    stats: RenderStats,
    dirty: set[tuple[int, int]],
    current_metadata: dict[str, dict[str, Any]],
    chunk_reason: str,
    full_memory_mb: float,
) -> dict[str, Any]:
    visible_ids = {tile.tile_id for tile in visibility.visible}
    world_tiles = []
    for tile in visibility.all_tiles:
        metadata = current_metadata[tile.tile_id]
        world_tiles.append(
            {
                "id": tile.tile_id,
                "row": tile.row,
                "col": tile.col,
                "visible": tile.tile_id in visible_ids,
                "world_bbox": metadata["world_bbox"],
                "screen_bbox": metadata["screen_bbox"],
                "height_version": metadata["height_version"],
                "material_version": metadata["material_version"],
            }
        )
    return {
        "terrain": {
            "file": str(terrain.path),
            "rows": config.terrain.rows,
            "cols": config.terrain.cols,
            "sha256": terrain.sha256,
        },
        "camera": {
            "eye": camera.eye.tolist(),
            "target": camera.target.tolist(),
            "fov_y_deg": config.camera.fov_y_deg,
            "near": config.camera.near,
            "far": config.camera.far,
        },
        "visible_world_tiles": sorted(visible_ids),
        "world_tiles": world_tiles,
        "tile_culling": {
            "total": stats.world_tile_count,
            "visible": stats.visible_world_tile_count,
            "culled_frustum": stats.culled_frustum,
            "culled_backface": stats.culled_backface,
        },
        "screen_tiles": {
            "total": stats.screen_tile_count,
            "redrawn_tiles": stats.redrawn_tiles,
            "reused_tiles": stats.reused_tiles,
            "dirty": [_screen_key(*key) for key in sorted(dirty)],
            "decision": chunk_reason,
        },
        "lighting": {
            "shadow_generated": stats.shadow_generated,
            "shadow_reused": stats.shadow_reused,
            "shadow_check_tiles": stats.shadow_check_tiles,
            "shadow_check_samples": stats.shadow_check_samples,
            "shadow_extra_dirty_tiles": stats.shadow_extra_dirty,
            "shadow_resolution": (
                config.light.shadow.resolution if config.light.shadow.enabled else None
            ),
            "pcf_samples": config.light.shadow.pcf_samples,
            "estimated_samples": stats.estimated_samples,
            "actual_samples": stats.actual_samples,
        },
        "memory": {
            "limit_mb": config.limits.max_memory_mb,
            "estimated_full_gbuffer_mb": full_memory_mb,
            "chunked": stats.memory_chunked,
            "batch_columns": stats.batch_columns,
        },
        "outputs": {
            "image": str(config.output.image),
            "normal_image": (
                str(config.output.normal_image)
                if config.output.normal_image is not None
                else None
            ),
            "shadow_image": (
                str(config.output.shadow_image)
                if config.output.shadow_image is not None
                else None
            ),
            "cache_dir": str(config.output.cache_dir),
        },
    }


def render(config: AppConfig, *, force_full: bool = False) -> RenderStats:
    terrain = build_terrain(config.terrain, config.output.terrain_file)
    print(f"terrain_sha256={terrain.sha256}")

    camera = setup_camera(config)
    visibility = build_world_tiles(config, terrain, camera)
    visible_ids = [tile.tile_id for tile in visibility.visible]
    print("visible_world_tiles=" + json.dumps(visible_ids, separators=(",", ":")))
    print(
        "tile_culling: "
        f"total={len(visibility.all_tiles)} visible={len(visibility.visible)} "
        f"culled_frustum={visibility.culled_frustum} "
        f"culled_backface={visibility.culled_backface}"
    )
    if not visibility.visible:
        print("warning: no world tiles survived culling; output will contain background only")

    geometry_key = _geometry_key(config)
    camera_key = _camera_key(config, geometry_key)
    render_key = stable_hash(
        {
            "camera_key": camera_key,
            "geometry_key": geometry_key,
            "width": config.render.width,
            "height": config.render.height,
            "tile_size": config.render.tile_size,
        }
    )
    lighting_key = _lighting_key(config)
    shadow_key = _shadow_key(config, geometry_key, terrain)
    current_metadata = _world_metadata(config, terrain, visibility)

    old_manifest, cache_warning = load_manifest(config.output.cache_dir)
    if cache_warning:
        print(f"warning: {cache_warning}")
    if force_full:
        old_manifest = None
    valid_entries: dict[str, dict[str, Any]] = {}
    if old_manifest is not None:
        valid_entries = _validate_cached_entries(
            config, config.output.cache_dir, old_manifest
        )
    dirty, cache_decision = _compute_dirty_tiles(
        config,
        old_manifest,
        valid_entries,
        current_metadata,
        render_key,
        camera_key,
        lighting_key,
        shadow_key,
        force_full,
    )
    print(f"cache_decision={cache_decision}")

    shadow_generated = False
    shadow_reused = False
    shadow_relative = old_manifest.get("shadow_file") if old_manifest else None
    shadow_depth: np.ndarray | None = None
    shadow_content_changed = bool(
        config.light.shadow.enabled
        and old_manifest is not None
        and old_manifest.get("lighting_key") == lighting_key
        and old_manifest.get("shadow_key") != shadow_key
    )
    need_shadow = bool(
        config.light.shadow.enabled and (dirty or shadow_content_changed)
    )
    if need_shadow:
        if (
            old_manifest is not None
            and old_manifest.get("shadow_key") == shadow_key
            and shadow_relative is not None
        ):
            shadow_depth = load_shadow_map(
                config.output.cache_dir,
                old_manifest,
                config.light.shadow.resolution,
            )
            shadow_reused = shadow_depth is not None
        if shadow_depth is None:
            fitted = fit_light_camera(config, terrain)
            shadow_depth = render_shadow_map(
                config, terrain, visibility.all_tiles, fitted
            )
            shadow_relative = save_shadow_map(
                config.output.cache_dir, shadow_depth, shadow_key
            )
            shadow_generated = True
    elif config.light.shadow.enabled:
        if (
            old_manifest is not None
            and old_manifest.get("shadow_key") == shadow_key
            and shadow_relative is not None
        ):
            shadow_reused = True

    lighting: LightingResources | None = None
    if need_shadow:
        assert shadow_depth is not None
        lighting = build_lighting_resources(
            config, terrain, visibility.all_tiles, shadow_depth
        )
    shadow_check_tiles = 0
    shadow_check_samples = 0
    shadow_extra_dirty = 0
    if shadow_content_changed and lighting is not None:
        extra_dirty, shadow_check_samples, shadow_check_tiles = (
            _shadow_sensitive_tiles(config, valid_entries, dirty, lighting)
        )
        dirty |= extra_dirty
        shadow_extra_dirty = len(extra_dirty)
        print(
            "shadow_invalidation: "
            f"checked_tiles={shadow_check_tiles} "
            f"comparison_samples={shadow_check_samples} "
            f"extra_dirty_tiles={shadow_extra_dirty}"
        )

    estimated_samples = _sample_estimate(config, dirty) + shadow_check_samples
    if (
        config.limits.max_total_samples is not None
        and estimated_samples > config.limits.max_total_samples
    ):
        raise resource_error(
            f"estimated {estimated_samples} samples exceeds "
            f"limits.max_total_samples={config.limits.max_total_samples}; "
            "reduce PCF samples, resolution, or dirty tile count"
        )

    shadow_bytes = shadow_depth.nbytes if shadow_depth is not None else 0
    if config.light.shadow.enabled and shadow_bytes == 0:
        shadow_bytes = config.light.shadow.resolution**2 * 4
    memory_chunked, batch_columns, full_memory_mb = _memory_layout(
        config, terrain, dirty, shadow_bytes
    )
    print(
        "memory: "
        f"limit_mb={config.limits.max_memory_mb} "
        f"estimated_full_gbuffer_mb={full_memory_mb:.2f} "
        f"chunked={memory_chunked} batch_columns={batch_columns}"
    )
    if memory_chunked:
        print(
            "memory_notice: full-frame G-buffer exceeds limit; "
            "rendering dirty screen tiles in column chunks"
        )

    if dirty and lighting is None:
        lighting = build_lighting_resources(
            config, terrain, visibility.all_tiles, shadow_depth
        )
    if dirty:
        assert lighting is not None
        rendered_entries, shaded_fragments = _render_dirty_batches(
            config,
            terrain,
            visibility,
            camera,
            lighting,
            dirty,
            batch_columns,
            current_metadata,
            lighting_key,
            shadow_key,
        )
    else:
        rendered_entries = {}
        shaded_fragments = 0

    entries: dict[str, dict[str, Any]] = {}
    valid_keys = set(valid_entries)
    dirty_screen_keys = {_screen_key(*key) for key in dirty}
    for row, col in _screen_keys(config):
        key = _screen_key(row, col)
        if key in rendered_entries:
            entries[key] = rendered_entries[key]
        elif key in valid_keys and key not in dirty_screen_keys:
            reused_entry = dict(valid_entries[key])
            reused_entry["shadow_version"] = shadow_key
            entries[key] = reused_entry
        else:
            raise render_error(f"screen tile {key} was neither rendered nor reused")

    actual_samples = shaded_fragments
    if config.light.shadow.enabled:
        actual_samples += shaded_fragments * config.light.shadow.pcf_samples
    actual_samples += shadow_check_samples

    manifest = {
        "format": MANIFEST_FORMAT,
        "render_key": render_key,
        "camera_key": camera_key,
        "geometry_key": geometry_key,
        "lighting_key": lighting_key,
        "shadow_key": shadow_key,
        "width": config.render.width,
        "height": config.render.height,
        "tile_size": config.render.tile_size,
        "world_tiles": current_metadata,
        "screen_tiles": entries,
        "shadow_file": shadow_relative if config.light.shadow.enabled else None,
        "terrain_sha256": terrain.sha256,
    }

    try:
        _write_preview_images(config, entries)
    except (OSError, ValueError) as exc:
        raise output_error(f"cannot write preview images: {exc}") from None

    stats = RenderStats(
        world_tile_count=len(visibility.all_tiles),
        visible_world_tile_count=len(visibility.visible),
        culled_frustum=visibility.culled_frustum,
        culled_backface=visibility.culled_backface,
        screen_tile_count=len(_screen_keys(config)),
        redrawn_tiles=len(dirty),
        reused_tiles=len(_screen_keys(config)) - len(dirty),
        shadow_generated=shadow_generated,
        shadow_reused=shadow_reused,
        shadow_check_tiles=shadow_check_tiles,
        shadow_check_samples=shadow_check_samples,
        shadow_extra_dirty=shadow_extra_dirty,
        memory_chunked=memory_chunked,
        batch_columns=batch_columns,
        estimated_samples=estimated_samples,
        actual_samples=actual_samples,
        terrain_sha256=terrain.sha256,
    )
    report = _tile_report(
        config,
        terrain,
        camera,
        visibility,
        stats,
        dirty,
        current_metadata,
        cache_decision,
        full_memory_mb,
    )
    write_json_atomic(config.output.tile_report, report)
    save_manifest(config.output.cache_dir, manifest)
    cleanup_cache(config.output.cache_dir, manifest)

    print(
        "screen_tiles: "
        f"total={stats.screen_tile_count} "
        f"redrawn_tiles={stats.redrawn_tiles} "
        f"reused_tiles={stats.reused_tiles}"
    )
    print(
        "shadow_map: "
        f"enabled={config.light.shadow.enabled} "
        f"resolution={config.light.shadow.resolution} "
        f"generated={shadow_generated} reused={shadow_reused}"
    )
    if shadow_content_changed:
        print(
            "shadow_invalidation: "
            f"checked_tiles={shadow_check_tiles} "
            f"extra_dirty_tiles={shadow_extra_dirty}"
        )
    print(
        "samples: "
        f"estimated={stats.estimated_samples} actual={stats.actual_samples}"
    )
    print(f"cache_dir={config.output.cache_dir}")
    print(f"output_image={config.output.image}")
    if config.render.write_debug_images:
        print(f"normal_image={config.output.normal_image}")
        print(f"shadow_image={config.output.shadow_image}")
    print(f"tile_report={config.output.tile_report}")
    return stats

