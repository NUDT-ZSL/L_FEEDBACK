"""End-to-end deterministic material baking pipeline."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .assets import LoadedAssets, load_assets
from .color import display_transform
from .errors import MaterialError
from .pngio import StreamingPNGWriter, write_png
from .preview import render_previews
from .scene import Scene
from .shading import shade_chunk
from .storage import CheckpointManager, NpyStreamWriter
from .tangent import TangentField, build_tangent_field
from .util import (
    RUN_SCHEMA,
    atomic_write_json,
    canonical_json_bytes,
    ensure_output_name,
    sha256_bytes,
    sha256_file,
)
from .validation import validate_normals


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _output_names(scene: Scene) -> dict[str, str]:
    kinds = (
        "baked_color",
        "baked_normal",
        "material_scalar",
        "lighting_diffuse",
        "lighting_specular",
        "lighting_indirect",
        "preview_main",
        "preview_normal",
        "preview_single_light",
        "validation_mask",
    )
    pattern = scene.output["filename_pattern"]
    names: dict[str, str] = {}
    for kind in kinds:
        name = pattern.replace("{kind}", kind)
        ensure_output_name(name)
        if not name.lower().endswith(".png"):
            raise MaterialError(
                "E009",
                "output.filename_pattern must produce a .png filename",
            )
        names[kind] = name
    if len(set(names.values())) != len(names):
        raise MaterialError("E009", "output filename pattern creates duplicate names")
    return names


def _run_fingerprint(
    scene: Scene,
    assets: LoadedAssets,
    *,
    chunk_rows: int,
    bit_depth: int,
    preview_resolution: tuple[int, int],
    filename_pattern: str,
) -> str:
    asset_hashes: dict[str, Any] = {}
    for label, info in assets.manifest["textures"].items():
        if "sha256" in info:
            asset_hashes[label] = info["sha256"]
        else:
            asset_hashes[label] = {
                "source": info.get("source", "default"),
                "value": info.get("value"),
            }
    payload = {
        "tool_version": __version__,
        "scene": scene.raw,
        "asset_hashes": asset_hashes,
        "resolution": assets.manifest["resolution"],
        "chunk_rows": chunk_rows,
        "bit_depth": bit_depth,
        "preview_resolution": preview_resolution,
        "filename_pattern": filename_pattern,
        "lights": scene.lights,
        "surface": scene.surface,
        "baking": scene.baking,
        "validation": scene.validation,
        "environment": scene.environment,
        "camera": scene.camera,
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _bake_chunks(
    scene: Scene,
    assets: LoadedAssets,
    tangent: TangentField,
    checkpoints: CheckpointManager,
    *,
    fail_after_chunks: int | None,
) -> int:
    newly_completed = 0
    skipped = 0
    strength = float(scene.baking["normal_strength"])
    for index in range(checkpoints.total_chunks):
        start, end = checkpoints.rows_for(index)
        if checkpoints.is_complete(index):
            skipped += 1
            continue
        object_normals = tangent.object_normals_rows(
            assets.normal_tangent, start, end, strength
        )
        components = shade_chunk(
            scene,
            assets.base_color[start:end],
            assets.roughness[start:end],
            assets.metallic[start:end],
            object_normals,
        )
        checkpoints.save(index, components.packed())
        newly_completed += 1
        _progress(
            f"[bake] chunk {index + 1}/{checkpoints.total_chunks} "
            f"rows={start}:{end} state=committed"
        )
        if fail_after_chunks is not None and newly_completed >= fail_after_chunks:
            from .errors import SimulatedInterrupt

            raise SimulatedInterrupt(newly_completed, checkpoints.total_chunks)
    _progress(
        f"[bake] complete chunks={checkpoints.total_chunks} "
        f"new={newly_completed} resumed={skipped}"
    )
    return newly_completed


def _move_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def _assemble_outputs(
    scene: Scene,
    assets: LoadedAssets,
    tangent: TangentField,
    checkpoints: CheckpointManager,
    out_dir: Path,
    names: dict[str, str],
    bit_depth: int,
) -> tuple[Path, Path, Path, list[dict[str, Any]]]:
    height, width = assets.manifest["resolution"][1], assets.manifest["resolution"][0]
    assembly_dir = checkpoints.work_dir / "assembly"
    if assembly_dir.exists():
        shutil.rmtree(assembly_dir)
    assembly_dir.mkdir(parents=True)

    temp_components = assembly_dir / "lighting_components.npy"
    temp_baked_linear = assembly_dir / "baked_linear.npy"
    temp_single_light = assembly_dir / "single_light_linear.npy"
    components_writer = NpyStreamWriter(temp_components, (height, width, 9))
    baked_writer = NpyStreamWriter(temp_baked_linear, (height, width, 3))
    single_writer = NpyStreamWriter(temp_single_light, (height, width, 3))

    png_kinds = {
        "baked_color": 3,
        "baked_normal": 3,
        "material_scalar": 3,
        "lighting_diffuse": 3,
        "lighting_specular": 3,
        "lighting_indirect": 3,
    }
    png_writers = {
        kind: StreamingPNGWriter(
            assembly_dir / names[kind],
            width,
            height,
            channels,
            bit_depth,
        )
        for kind, channels in png_kinds.items()
    }
    exposure = float(scene.baking["exposure"])
    strength = float(scene.baking["normal_strength"])
    first_light = [scene.lights[0]] if scene.lights else []
    for index, (start, end), packed in checkpoints.iter_chunks():
        if packed.shape != (end - start, width, 9):
            raise MaterialError(
                "E010", f"checkpoint chunk {index} has inconsistent channels"
            )
        diffuse = packed[:, :, 0:3]
        specular = packed[:, :, 3:6]
        indirect = packed[:, :, 6:9]
        total = (
            diffuse.astype(np.float64)
            + specular.astype(np.float64)
            + indirect.astype(np.float64)
        ).astype(np.float32)
        object_normals = tangent.object_normals_rows(
            assets.normal_tangent, start, end, strength
        )
        components_writer.append(packed)
        baked_writer.append(total)
        single_components = shade_chunk(
            scene,
            assets.base_color[start:end],
            assets.roughness[start:end],
            assets.metallic[start:end],
            object_normals,
            lights=first_light,
            include_environment=False,
        )
        single_writer.append(single_components.total)
        png_writers["baked_color"].append_rows(
            display_transform(total, exposure)
        )
        png_writers["baked_normal"].append_rows(
            np.clip(0.5 + 0.5 * object_normals, 0.0, 1.0)
        )
        scalar = np.concatenate(
            [
                assets.roughness[start:end],
                assets.metallic[start:end],
                np.ones((end - start, width, 1), dtype=np.float32),
            ],
            axis=2,
        )
        png_writers["material_scalar"].append_rows(scalar)
        png_writers["lighting_diffuse"].append_rows(
            display_transform(diffuse, exposure)
        )
        png_writers["lighting_specular"].append_rows(
            display_transform(specular, exposure)
        )
        png_writers["lighting_indirect"].append_rows(
            display_transform(indirect, exposure)
        )

    component_hash = components_writer.finalize()
    baked_hash = baked_writer.finalize()
    single_hash = single_writer.finalize()
    png_hashes = {
        kind: writer.finalize() for kind, writer in png_writers.items()
    }
    output_records: list[dict[str, Any]] = []
    for kind, channels in png_kinds.items():
        destination = out_dir / names[kind]
        _move_atomic(assembly_dir / names[kind], destination)
        output_records.append(
            {
                "name": names[kind],
                "kind": kind,
                "role": "baked_texture",
                "width": width,
                "height": height,
                "channels": channels,
                "bit_depth": bit_depth,
                "sha256": png_hashes[kind],
            }
        )
    destination_components = out_dir / "lighting_components.npy"
    destination_baked = out_dir / "baked_linear.npy"
    destination_single = out_dir / "single_light_linear.npy"
    _move_atomic(temp_components, destination_components)
    _move_atomic(temp_baked_linear, destination_baked)
    _move_atomic(temp_single_light, destination_single)
    output_records.extend(
        [
            {
                "name": destination_components.name,
                "kind": "lighting_components",
                "role": "float_cache",
                "shape": [height, width, 9],
                "dtype": "float32",
                "channels": ["diffuse_rgb", "specular_rgb", "indirect_rgb"],
                "sha256": component_hash,
            },
            {
                "name": destination_baked.name,
                "kind": "baked_linear",
                "role": "float_cache",
                "shape": [height, width, 3],
                "dtype": "float32",
                "sha256": baked_hash,
            },
            {
                "name": destination_single.name,
                "kind": "single_light_linear",
                "role": "float_cache",
                "shape": [height, width, 3],
                "dtype": "float32",
                "sha256": single_hash,
            },
        ]
    )
    return destination_baked, destination_single, destination_components, output_records


def run_pipeline(
    scene_path: Path,
    out_dir: Path,
    *,
    chunk_rows_override: int | None = None,
    preview_resolution_override: tuple[int, int] | None = None,
    bit_depth_override: int | None = None,
    name_pattern_override: str | None = None,
    seed_override: int | None = None,
    force: bool = False,
    fail_after_chunks: int | None = None,
) -> int:
    from .scene import load_scene

    scene = load_scene(scene_path)
    if chunk_rows_override is not None:
        scene.baking["chunk_rows"] = int(chunk_rows_override)
    if preview_resolution_override is not None:
        scene.output["preview_resolution"] = list(preview_resolution_override)
    if bit_depth_override is not None:
        if bit_depth_override not in {8, 16}:
            raise MaterialError("E011", "--bit-depth must be 8 or 16")
        scene.output["bit_depth"] = int(bit_depth_override)
    if name_pattern_override is not None:
        if name_pattern_override.count("{kind}") != 1:
            raise MaterialError(
                "E011", "--name-pattern must contain exactly one {kind}"
            )
        scene.output["filename_pattern"] = name_pattern_override
    if seed_override is not None:
        if seed_override < 0:
            raise MaterialError("E011", "--seed must be non-negative")
        scene.validation["seed"] = int(seed_override)

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(".*.tmp-*"):
        if stale.is_file():
            stale.unlink()
    names = _output_names(scene)
    assets = load_assets(scene)
    atomic_write_json(out_dir / "asset_manifest.json", assets.manifest)
    tangent = build_tangent_field(
        scene,
        (
            int(assets.manifest["resolution"][1]),
            int(assets.manifest["resolution"][0]),
        ),
    )
    chunk_rows = int(scene.baking["chunk_rows"])
    bit_depth = int(scene.output["bit_depth"])
    preview_resolution = (
        int(scene.output["preview_resolution"][0]),
        int(scene.output["preview_resolution"][1]),
    )
    fingerprint = _run_fingerprint(
        scene,
        assets,
        chunk_rows=chunk_rows,
        bit_depth=bit_depth,
        preview_resolution=preview_resolution,
        filename_pattern=scene.output["filename_pattern"],
    )
    checkpoints = CheckpointManager(
        out_dir,
        fingerprint,
        int(assets.manifest["resolution"][0]),
        int(assets.manifest["resolution"][1]),
        chunk_rows,
        force=force,
    )
    if checkpoints.resumed_chunks:
        _progress(
            f"[resume] {checkpoints.resumed_chunks}/{checkpoints.total_chunks} "
            "checkpoint chunks already complete"
        )
    _bake_chunks(
        scene,
        assets,
        tangent,
        checkpoints,
        fail_after_chunks=fail_after_chunks,
    )
    _progress("[assemble] streaming final textures and component cache")
    baked_path, single_path, _, output_records = _assemble_outputs(
        scene,
        assets,
        tangent,
        checkpoints,
        out_dir,
        names,
        bit_depth,
    )

    strength = float(scene.baking["normal_strength"])
    object_normals = tangent.object_normals(assets.normal_tangent, strength)
    baked_linear = np.load(baked_path, mmap_mode="r")
    single_light_linear = np.load(single_path, mmap_mode="r")
    preview_records = render_previews(
        scene,
        out_dir,
        baked_linear=baked_linear,
        object_normals=object_normals,
        single_light_linear=single_light_linear,
        bit_depth=bit_depth,
    )
    output_records.extend(preview_records)

    _progress("[validate] normal-space and UV seam consistency")
    validation = validate_normals(scene, assets, tangent, object_normals)
    validation.report["tangent_basis"] = tangent.info
    mask_hash = write_png(
        out_dir / names["validation_mask"], validation.mask, bit_depth
    )
    output_records.append(
        {
            "name": names["validation_mask"],
            "kind": "validation_mask",
            "role": "localization_mask",
            "width": int(validation.mask.shape[1]),
            "height": int(validation.mask.shape[0]),
            "channels": 3,
            "bit_depth": bit_depth,
            "sha256": mask_hash,
        }
    )
    atomic_write_json(out_dir / "validation_report.json", validation.report)
    report_hash = sha256_file(out_dir / "validation_report.json")
    asset_hash = sha256_file(out_dir / "asset_manifest.json")

    all_records = sorted(output_records, key=lambda item: item["name"])
    all_records.extend(
        [
            {
                "name": "asset_manifest.json",
                "kind": "asset_manifest",
                "role": "manifest",
                "sha256": asset_hash,
            },
            {
                "name": "validation_report.json",
                "kind": "validation_report",
                "role": "report",
                "sha256": report_hash,
            },
        ]
    )
    manifest = {
        "schema": RUN_SCHEMA,
        "tool_version": __version__,
        "status": validation.report["status"],
        "scene": {
            "name": scene.path.name,
            "sha256": sha256_file(scene.path),
        },
        "input_resolution": assets.manifest["resolution"],
        "preview_resolution": list(preview_resolution),
        "bit_depth": bit_depth,
        "chunk_rows": chunk_rows,
        "chunk_count": checkpoints.total_chunks,
        "run_fingerprint": fingerprint,
        "determinism": {
            "sampling_seed": int(scene.validation["seed"]),
            "stratified_sampling": bool(scene.validation["stratified"]),
            "light_order": "content-sorted before accumulation",
            "png_encoder": "filter0-zlib9-no-metadata",
            "array_format": "npy-v1-little-endian-float32",
            "output_timestamp": False,
        },
        "assets": {
            "manifest": "asset_manifest.json",
            "textures": assets.manifest["textures"],
        },
        "tangent_basis": tangent.info,
        "outputs": sorted(all_records, key=lambda item: item["name"]),
        "summary": {
            "normal_anomaly_pixels": validation.report["normal_space"][
                "pixels_with_anomaly"
            ],
            "seam_anomaly_pairs": validation.report["seams"][
                "anomalous_pairs"
            ],
        },
    }
    atomic_write_json(out_dir / "manifest.json", manifest)
    _progress(
        f"[done] status={validation.report['status']} out={out_dir} "
        f"chunks={checkpoints.total_chunks}"
    )
    return 0
