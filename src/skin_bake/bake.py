"""End-to-end bake orchestration and traceable report generation."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import ATLAS_FORMAT, __version__
from .basis import build_bind_basis
from .errors import output_io, tolerance
from .packing import PackResult, plan_and_pack
from .projection import (
    frame_count,
    project_to_bind_basis,
    resample_projection,
    sample_input_curves,
)
from .scene import Scene, load_scene
from .skinning import animation_frame_times
from .texture import write_texture


@dataclass
class BakeResult:
    output_dir: Path
    report: dict[str, Any]
    atlas_map: dict[str, Any]


def _rgba_block(
    packed: PackResult,
    sampled_curves: np.ndarray,
    padding: int,
) -> np.ndarray:
    atlas = np.zeros(
        (packed.atlas_height, packed.atlas_width, 2), dtype=np.float64
    )
    capacity_by_vertex = {
        block.vertex_index: block.payload_width * block.payload_height
        for block in packed.blocks
    }
    for block in packed.blocks:
        payload_w = block.payload_width
        payload_h = block.payload_height
        capacity = capacity_by_vertex[block.vertex_index]
        if packed.sample_count > capacity:
            raise RuntimeError(f"block for vertex {block.vertex_index} has insufficient texels")
        payload = np.zeros((capacity, 2), dtype=np.float64)
        payload[: packed.sample_count] = sampled_curves[: packed.sample_count, block.vertex_index, :]
        payload = payload.reshape(payload_h, payload_w, 2)
        px, py = block.payload_offset
        atlas[py : py + payload_h, px : px + payload_w, :] = payload

        left = px
        right = px + payload_w
        bottom = py
        top = py + payload_h
        for offset in range(1, padding + 1):
            atlas[bottom - offset, left:right, :] = payload[0]
            atlas[top + offset - 1, left:right, :] = payload[-1]
            atlas[bottom:top, left - offset, :] = payload[:, 0]
            atlas[bottom:top, right + offset - 1, :] = payload[:, -1]
            atlas[bottom - offset, left - offset, :] = payload[0, 0]
            atlas[bottom - offset, right + offset - 1, :] = payload[0, -1]
            atlas[top + offset - 1, left - offset, :] = payload[-1, 0]
            atlas[top + offset - 1, right + offset - 1, :] = payload[-1, -1]
    return atlas


def _atomic_write_output(
    output_dir: Path,
    scene: Scene,
    report: dict[str, Any],
    atlas_map: dict[str, Any],
    pixels: np.ndarray,
    started: float,
) -> None:
    parent = output_dir.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=parent))
    except OSError as exc:
        raise output_io(f"cannot create output staging directory: {exc}") from exc
    try:
        texture_path = staging / "displacement.rg16f"
        header, texture_file_hash = write_texture(texture_path, pixels)
        report["texture"]["fileSha256"] = texture_file_hash
        report["texture"]["dataSha256"] = header["dataSha256"]
        atlas_map["textureSha256"] = texture_file_hash
        atlas_map["textureDataSha256"] = header["dataSha256"]
        mapping_path = staging / "atlas_map.json"
        mapping_path.write_text(
            json.dumps(atlas_map, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        scene_copy = staging / "scene.json"
        scene_copy.write_bytes(scene.raw_bytes)
        report["bakeDurationSeconds"] = time.perf_counter() - started
        report_path = staging / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

        backup: Path | None = None
        if output_dir.exists():
            backup = output_dir.with_name(
                f".{output_dir.name}.old-{os.getpid()}-{time.time_ns()}"
            )
            os.replace(output_dir, backup)
        try:
            os.replace(staging, output_dir)
        except Exception:
            if backup is not None and backup.exists() and not output_dir.exists():
                os.replace(backup, output_dir)
            raise
        if backup is not None and backup.exists():
            try:
                if backup.is_dir():
                    shutil.rmtree(backup)
                else:
                    backup.unlink()
            except OSError:
                pass
        return
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise output_io(f"cannot finalize output directory '{output_dir}': {exc}") from exc
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def bake_scene(scene_path: str | Path, output_dir: str | Path) -> BakeResult:
    started = time.perf_counter()
    scene = load_scene(scene_path)
    output = Path(output_dir)

    basis = build_bind_basis(scene.normals, scene.tangents)
    frame_times = animation_frame_times(scene)
    input_vectors = sample_input_curves(scene, frame_times)
    projection = project_to_bind_basis(scene, basis, frame_times, input_vectors)

    if projection.overall_max_relative_error > scene.bake.error_tolerance:
        failing = [
            vertex
            for vertex, value in projection.max_relative_error_by_vertex.items()
            if value > scene.bake.error_tolerance
        ]
        raise tolerance(
            "projection error "
            f"{projection.overall_max_relative_error:.9g} exceeds tolerance "
            f"{scene.bake.error_tolerance:.9g} at vertices {failing[:16]}"
        )

    pack = plan_and_pack(
        curves=projection.curves,
        duration=scene.duration,
        loop=scene.loop_mode == "loop",
        texture_sample_rate=scene.bake.texture_sample_rate,
        atlas_width=scene.bake.atlas_width,
        atlas_height=scene.bake.atlas_height,
        min_block_size=scene.bake.min_block_size,
        padding=scene.bake.padding,
    )
    sampled_curves = resample_projection(scene, projection, pack.sample_count)
    atlas = _rgba_block(pack, sampled_curves, pack.padding)
    quantized = atlas.astype(np.float16).astype(np.float64)
    encoding_error = quantized - atlas
    max_encoding_abs_error = float(np.max(np.abs(encoding_error))) if atlas.size else 0.0

    mapping_blocks = []
    for block in pack.blocks:
        px, py = block.payload_offset
        mapping_blocks.append(
            {
                "vertex": block.vertex_index,
                "blockOrigin": [block.x, block.y],
                "pixelOffset": [px, py],
                "blockSize": [block.payload_width, block.payload_height],
                "paddedSize": [block.padded_width, block.padded_height],
                "sampleCount": pack.sample_count,
                "sampleRate": pack.actual_sample_rate,
                "curveBounds": {
                    "min": list(block.curve_min),
                    "max": list(block.curve_max),
                },
            }
        )
    atlas_map: dict[str, Any] = {
        "format": ATLAS_FORMAT,
        "version": "1.0.0",
        "atlasSize": [pack.atlas_width, pack.atlas_height],
        "origin": "bottom-left",
        "rowOrder": "bottom-to-top",
        "sampleOrder": "row-major-within-block",
        "sampleRate": pack.actual_sample_rate,
        "sampleCount": pack.sample_count,
        "duration": scene.duration,
        "loopMode": scene.loop_mode,
        "texture": "displacement.rg16f",
        "sceneContentHash": scene.content_hash,
        "packing": {
            "algorithm": "maxrects-bssf",
            "baseSampleRate": pack.base_sample_rate,
            "baseSampleCount": pack.base_sample_count,
            "downsampleFactor": pack.downsample_factor,
            "attempts": pack.attempts,
            "padding": pack.padding,
        },
        "blocks": mapping_blocks,
    }

    elapsed = time.perf_counter() - started
    report: dict[str, Any] = {
        "format": "skin-bake-report",
        "version": "1.0.0",
        "toolVersion": __version__,
        "status": "ok",
        "inputFile": str(scene.source_path),
        "inputFileSha256": scene.input_file_sha256,
        "sceneContentHash": scene.content_hash,
        "bakeDurationSeconds": elapsed,
        "actualSampleRate": pack.actual_sample_rate,
        "baseSampleRate": pack.base_sample_rate,
        "atlasSize": [pack.atlas_width, pack.atlas_height],
        "sampleCount": pack.sample_count,
        "downsampleFactor": pack.downsample_factor,
        "packingAttempts": pack.attempts,
        "frameCount": frame_count(scene),
        "fps": scene.fps,
        "duration": scene.duration,
        "loopMode": scene.loop_mode,
        "fallbackVertices": list(basis.fallback_vertices),
        "fallbackReasons": {
            str(vertex): reason for vertex, reason in sorted(basis.fallback_reasons.items())
        },
        "maxProjectionError": projection.overall_max_relative_error,
        "errorTolerance": scene.bake.error_tolerance,
        "maxProjectionErrorByVertex": [
            {
                "vertex": vertex,
                "relative": projection.max_relative_error_by_vertex[vertex],
            }
            for vertex in range(scene.vertex_count)
        ],
        "precision": {
            "workingType": "float64",
            "accumulation": "kahan-compensated" if scene.displacement_mode == "delta" else "direct",
            "maxAccumulationCorrection": projection.max_accumulation_correction,
            "loopClosureError": projection.loop_closure_error,
            "maxFloat16EncodingAbsError": max_encoding_abs_error,
        },
        "texture": {
            "file": "displacement.rg16f",
            "format": "RG16F",
        },
        "mappingFile": "atlas_map.json",
    }
    _atomic_write_output(output, scene, report, atlas_map, atlas, started)
    return BakeResult(output, report, atlas_map)


def load_atlas_map(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise output_io(f"cannot load atlas map '{source}': {exc}") from exc
    if data.get("format") != ATLAS_FORMAT:
        raise output_io("atlas map format is invalid")
    return data

