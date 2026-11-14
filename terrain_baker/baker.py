"""End-to-end baking orchestration with atomic all-or-nothing output."""

from __future__ import annotations

import os
import time
from typing import Callable

import numpy as np

from . import preview as preview_mod
from .atomicio import AtomicStage
from .checksums import sha256_file
from .consistency import check_level
from .errors import IOFailure, ParameterError, PreviewUnavailable
from .heightfield import (
    HeightfieldSpec,
    build_heightfield,
    open_heightfield,
    validate_spec,
)
from .lod import build_lod_chain, lod_resolution
from .memory import MemoryBudget
from .metadata import write_manifest
from .normal import bake_normal_map, edge_mask, normal_length_error
from .pngutil import write_srgb_png
from .verifier import verify_output

DEFAULT_MEMORY_LIMIT = 64 * 1024 * 1024
DEFAULT_TILE = 512


def _rel(*parts: str) -> str:
    return "/".join(part.replace("\\", "/").strip("/") for part in parts)


def bake(
    spec: HeightfieldSpec,
    output_dir: str,
    levels: list[int],
    *,
    tile: int = DEFAULT_TILE,
    memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT,
    preview_width: int = 512,
    preview_height: int = 512,
    consistency_samples: int = 2048,
    inject_failure: Callable[[str], None] | None = None,
    skip_preview: bool = False,
) -> dict:
    """Run every baking stage and atomically publish the output set."""
    validate_spec(spec)
    for level in levels:
        lod_resolution(spec.resolution, int(level))
    if tile < 1:
        raise ParameterError("tile size must be positive")
    if memory_limit_bytes <= 0:
        raise ParameterError("memory limit must be positive")

    budget = MemoryBudget(memory_limit_bytes)
    started = time.perf_counter()
    stage = AtomicStage(output_dir, inject_failure=inject_failure)
    previous_manifest_path = os.path.join(output_dir, "metadata.json")
    previous_files: set[str] = set()
    if os.path.exists(previous_manifest_path):
        import json

        with open(previous_manifest_path, "r", encoding="utf-8") as handle:
            previous_files = set(json.load(handle).get("files", []))

    with stage:
        height_rel = _rel("data", "heightfield.dat")
        normal_rel = _rel("textures", "normal.png")
        mesh_rels = {
            int(level): _rel("meshes", f"lod{int(level)}.bin") for level in levels
        }
        preview_diffuse_rel = _rel("preview", "diffuse.png")
        preview_normal_rel = _rel("preview", "normal.png")
        metadata_rel = "metadata.json"

        height_path = stage.path(height_rel)
        grid = build_heightfield(height_path, spec, budget)
        stage.register(height_rel)

        base_spacing_world = spec.world_size / (spec.resolution - 1)
        normal_image, normal_tile_stats = bake_normal_map(
            grid, budget, tile=tile, spacing_world=base_spacing_world
        )
        normal_path = stage.path(normal_rel)
        write_srgb_png(normal_path, normal_image)
        stage.register(normal_rel)

        mesh_dir = os.path.join(stage.staging_dir, "meshes")
        lod_infos, temp_grids = build_lod_chain(
            base_grid=grid,
            levels=[int(level) for level in levels],
            world_size=spec.world_size,
            budget=budget,
            temp_dir=stage.staging_dir,
            mesh_dir=mesh_dir,
            relative_mesh_dir="meshes",
        )
        for level in levels:
            stage.register(mesh_rels[int(level)])

        rng = np.random.default_rng(1234567 + spec.seed)
        consistency: list[dict] = []
        coarse_grid = grid
        current_level = 0
        for info in lod_infos:
            while current_level < info.level:
                coarse_grid = open_heightfield(
                    temp_grids[current_level],
                    lod_resolution(spec.resolution, current_level + 1),
                )
                current_level += 1
            spacing_world = (
                spec.world_size
                * info.spacing
                / (info.resolution - 1)
            )
            report = check_level(
                height=np.asarray(coarse_grid),
                spacing_world=spacing_world,
                normal_image=normal_image,
                sample_count=min(consistency_samples, info.resolution ** 2),
                rng=rng,
            )
            report["level"] = info.level
            consistency.append(report)

        preview_info: dict = {"rendered": False, "available": not skip_preview,
                              "skipped": bool(skip_preview)}
        if not skip_preview:
            coarsest = lod_infos[-1]
            try:
                preview_info = preview_mod.render_previews(
                    binary_path=os.path.join(
                        stage.staging_dir, coarsest.binary_path
                    ),
                    vertex_count=coarsest.vertex_count,
                    index_count=coarsest.index_count,
                    diffuse_png=stage.path(preview_diffuse_rel),
                    normal_png=stage.path(preview_normal_rel),
                    width=preview_width,
                    height=preview_height,
                )
                preview_info["available"] = True
                stage.register(preview_diffuse_rel)
                stage.register(preview_normal_rel)
            except PreviewUnavailable as exc:
                preview_info = {
                    "rendered": False,
                    "available": False,
                    "error": str(exc),
                    "diagnostics": preview_mod.diagnostics(),
                    "exit_code": 6,
                }

        tile_stats = list(normal_tile_stats)
        for info in lod_infos:
            tile_stats.extend(info.tile_stats)

        core_files = [height_rel, normal_rel] + [
            mesh_rels[int(level)] for level in sorted(mesh_rels)
        ]
        if preview_info.get("rendered"):
            core_files += [preview_diffuse_rel, preview_normal_rel]

        normal_quality = normal_length_error(normal_image)
        normal_quality["edge_pixel_count"] = int(edge_mask(spec.resolution).sum())

        # Hash core outputs first; the manifest hash is computed over files only.
        per_file = {rel: sha256_file(stage.path(rel)) for rel in core_files}
        from .checksums import overall_sha256

        manifest = {
            "schema_version": 1,
            "inputs": spec.to_dict(),
            "tile_size": int(tile),
            "memory": {"limit_bytes": int(memory_limit_bytes), **budget.snapshot()},
            "heightfield": {
                "path": height_rel,
                "dtype": "float32",
                "endianness": "little",
            },
            "normal_map": {
                "path": normal_rel,
                "format": "png",
                "color_space": "sRGB",
                "encoding": "rgb8",
                "tangent_space": {"x": "u", "y": "v_down", "z": "viewer"},
                "quality": normal_quality,
            },
            "lod_levels": [info.to_dict() for info in lod_infos],
            "consistency_checks": consistency,
            "tile_stats": [stat.to_dict() for stat in tile_stats],
            "preview": preview_info,
            "files": core_files,
            "sha256": {
                "per_file": per_file,
                "overall": overall_sha256(per_file),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        metadata_path = stage.path(metadata_rel)
        write_manifest(metadata_path, manifest)
        stage.register(metadata_rel)

        if inject_failure is not None:
            inject_failure("__commit__")
        stage.commit(previous_manifest=previous_files)

    for temp_path in temp_grids:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    verification = verify_output(output_dir)
    if not verification["ok"]:
        raise IOFailure("post-bake verification failed: " + "; ".join(verification["errors"]))
    return {"manifest": manifest, "verification": verification}
