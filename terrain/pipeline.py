"""End-to-end pipeline orchestration and deterministic contract output."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import gl_backend, software_backend
from .config import PipelineConfig, write_json
from .errors import E_BAD_FRUSTUM, E_NO_GL, PipelineError
from .heightfield import Heightfield
from .lod import (
    CullCounters,
    HeightBoundsCache,
    enforce_restricted_quadtree,
    recount_leaves,
    select_leaves,
)
from .mesh import TileMesh, build_tile_mesh_safe
from .normals import gamma_encode, shade_vertex_colors, tile_normals
from .quadtree import TileKey


@dataclass
class PipelineResult:
    image: np.ndarray
    contract: dict
    backend_used: str
    fallback: str | None


def _vertex_colors(
    mesh: TileMesh,
    heightfield: Heightfield,
    config: PipelineConfig,
) -> np.ndarray:
    normals_grid = tile_normals(mesh.key, heightfield, config)
    base_normals = normals_grid.reshape(
        normals_grid.shape[0] * normals_grid.shape[1], 3
    )
    base_linear = shade_vertex_colors(base_normals, config.light)
    if mesh.vertices.shape[0] == base_linear.shape[0]:
        return base_linear
    extra_count = mesh.vertices.shape[0] - base_linear.shape[0]
    extra_normals = np.zeros((extra_count, 3), dtype=np.float64)
    if extra_count:
        extra_vertices = mesh.vertices[base_linear.shape[0]:]
        x0, z0, x1, z1 = mesh.key.bounds(
            config.root_min, config.root_size
        )
        spacing = (x1 - x0) / (
            (config.tile_resolution - 1) * 2
        )
        vx = extra_vertices[:, 0]
        vz = extra_vertices[:, 2]
        height_left = heightfield.sample_array(vx - spacing, vz)
        height_right = heightfield.sample_array(vx + spacing, vz)
        height_down = heightfield.sample_array(vx, vz - spacing)
        height_up = heightfield.sample_array(vx, vz + spacing)
        extra_normals[:, 0] = -(height_right - height_left) / (2.0 * spacing)
        extra_normals[:, 1] = 1.0
        extra_normals[:, 2] = -(height_up - height_down) / (2.0 * spacing)
        extra_normals /= np.linalg.norm(extra_normals, axis=1, keepdims=True)
    extra_linear = shade_vertex_colors(extra_normals, config.light)
    return np.concatenate([base_linear, extra_linear], axis=0)


def _build_contract(
    config: PipelineConfig,
    leaves: frozenset[TileKey],
    counters: CullCounters,
    meshes: list[TileMesh],
    backend_used: str,
    fallback: str | None,
    elapsed_seconds: float,
) -> dict:
    visible_by_level: dict[int, int] = {}
    for level, count in enumerate(recount_leaves(leaves, config.max_depth)):
        visible_by_level[level] = count
    total_patched_edges = sum(len(mesh.stitched_edges) for mesh in meshes)
    rollback_edges = sum(
        1 for mesh in meshes if mesh.stitched_edges == ("ROLLBACK",)
    )
    return {
        "backend": str(backend_used),
        "backend_fallback": str(fallback) if fallback is not None else None,
        "config": {
            "seed": int(config.seed),
            "root_min": [float(v) for v in config.root_min],
            "root_size": float(config.root_size),
            "max_depth": int(config.max_depth),
            "tile_resolution": int(config.tile_resolution),
            "sse_threshold": float(config.sse_threshold),
            "sse_relative_tolerance": float(config.sse_relative_tolerance),
            "enable_stitching": bool(config.enable_stitching),
        },
        "culled_cumulative": int(counters.total_culled),
        "elapsed_seconds": float(round(elapsed_seconds, 6)),
        "image_height": int(config.image_height),
        "image_width": int(config.image_width),
        "leaf_tile_count": int(len(leaves)),
        "patched_edge_count": int(total_patched_edges),
        "rollback_edge_count": int(rollback_edges),
        "tested_per_level": [int(v) for v in counters.tested_per_level],
        "total_triangle_count": int(
            sum(mesh.triangle_count for mesh in meshes)
        ),
        "visible_per_level": {
            str(level): int(visible_by_level.get(level, 0))
            for level in range(config.max_depth + 1)
        },
        "culled_per_level": {
            str(level): int(counters.culled_per_level[level])
            for level in range(config.max_depth + 1)
        },
    }


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    config.validate()
    start = time.perf_counter()
    heightfield = Heightfield(
        config.seed, config.height_scale, config.noise_octaves
    )
    bounds_cache = HeightBoundsCache(config, heightfield)
    leaves, counters = select_leaves(config, bounds_cache)
    leaves = enforce_restricted_quadtree(leaves, config.max_depth)

    ordered_leaves = sorted(
        leaves, key=lambda tile: (tile.level, tile.iz, tile.ix)
    )
    meshes = [
        build_tile_mesh_safe(
            key,
            leaves,
            heightfield,
            config,
            bounds_cache.grid(key),
        )
        for key in ordered_leaves
    ]
    colors_linear = [
        _vertex_colors(mesh, heightfield, config) for mesh in meshes
    ]

    fallback = None
    if config.backend == "moderngl":
        try:
            linear_image = gl_backend.render_scene(meshes, colors_linear, config)
            backend_used = gl_backend.BACKEND_NAME
        except PipelineError as exc:
            if exc.code != E_NO_GL:
                raise
            fallback = E_NO_GL
            linear_image = software_backend.render_scene(
                meshes, colors_linear, config
            )
            backend_used = software_backend.BACKEND_NAME
    elif config.backend == "software":
        linear_image = software_backend.render_scene(meshes, colors_linear, config)
        backend_used = software_backend.BACKEND_NAME
    else:
        raise PipelineError(E_BAD_FRUSTUM, f"unknown backend {config.backend!r}")

    image = gamma_encode(linear_image.astype(np.float64) / 255.0, config.light.gamma)
    elapsed = time.perf_counter() - start
    contract = _build_contract(
        config, leaves, counters, meshes, backend_used, fallback, elapsed
    )
    return PipelineResult(image=image, contract=contract, backend_used=backend_used, fallback=fallback)


def write_outputs(result: PipelineResult, config: PipelineConfig) -> None:
    from PIL import Image

    png_path = Path(config.output_png)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.image, mode="L").save(png_path, format="PNG")
    write_json(config.output_contract, result.contract)
