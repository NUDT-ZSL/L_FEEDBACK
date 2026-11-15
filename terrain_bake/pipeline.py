"""Transactional terrain bake pipeline and manifest contract."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .errors import (
    E_GEOMETRY_INVALID,
    E_INTERNAL_ERROR,
    E_INVALID_ARGUMENT,
    E_INVALID_HEIGHTFIELD,
    E_IO_ERROR,
    ERROR_EXIT_CODES,
    BakeError,
)
from .generator import (
    TerrainSamples,
    generate_samples,
    samples_from_external,
)
from .mesh import (
    TerrainMesh,
    build_base_mesh,
    lod_error_thresholds,
    lod_target_ratios,
    simplify_mesh,
    topology_summary,
    write_obj,
)
from .normals import (
    bake_material_preview,
    bake_normal_map,
    save_material_png,
    save_normal_png,
)
from .render import render_preview, save_preview_png
from .seam import build_transition_mesh


SCHEMA_VERSION = "1.0"
MANIFEST_NAME = "bake_manifest.json"
Reporter = Callable[[str], None]


@dataclass(frozen=True)
class BakeConfig:
    size: int = 65
    seed: int = 20261116
    lod_count: int = 3
    normal_resolution: int = 512
    preview_width: int = 800
    preview_height: int = 600
    output_dir: Path = Path("artifacts/terrain")
    max_relative_error: float = 0.03
    weight_tolerance: float = 1e-6
    preview_backend: str = "auto"


@dataclass
class BakeResult:
    manifest: dict[str, Any]
    manifest_path: Path
    exit_code: int


def _now_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def _emit(reporter: Reporter | None, message: str) -> None:
    if reporter is not None:
        reporter(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _base_manifest(config: BakeConfig) -> dict[str, Any]:
    try:
        output_name = str(Path(config.output_dir))
    except Exception:
        output_name = str(config.output_dir)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "error": None,
        "error_message": None,
        "error_details": {},
        "seed": int(config.seed),
        "request": {
            "size": [int(config.size), int(config.size)],
            "seed": int(config.seed),
            "lod_count": int(config.lod_count),
            "normal_resolution": int(config.normal_resolution),
            "preview_resolution": [
                int(config.preview_width),
                int(config.preview_height),
            ],
            "max_relative_error": float(config.max_relative_error),
            "weight_tolerance": float(config.weight_tolerance),
            "lod_target_ratios": [],
            "lod_error_thresholds": [],
            "preview_backend": str(config.preview_backend),
            "output_dir": output_name,
        },
        "source": {
            "kind": "procedural",
            "heightfield_path": None,
            "material_weights_path": None,
            "heightfield_sha256": None,
            "material_weights_sha256": None,
            "height_min": None,
            "height_max": None,
            "weight_sum_max_error": None,
        },
        "backend": {
            "requested": str(config.preview_backend),
            "attempted": [],
            "used": None,
            "fallback_reason": None,
        },
        "timings_ms": {},
        "lods": [],
        "normal_map": {
            "path": None,
            "metadata_path": None,
            "resolution": None,
            "pixel_step": None,
            "height_scale": None,
            "tangent_axis": "+X",
            "bitangent_axis": "-Z",
            "normal_axis": "+Y",
            "sha256": None,
            "metadata_sha256": None,
        },
        "material_preview": {
            "path": None,
            "resolution": None,
            "sha256": None,
        },
        "preview": {
            "path": None,
            "backend": None,
            "resolution": None,
            "sha256": None,
        },
        "seam": {
            "pairs": [],
            "all_manifold": None,
            "t_junction_count": 0,
        },
        "checksums": {},
        "artifacts": [],
        "warnings": [],
    }


def _validate_config(config: BakeConfig, output: Path) -> None:
    if config.size < 2:
        raise BakeError(
            E_INVALID_HEIGHTFIELD,
            "heightfield size must be at least 2",
            {"size": int(config.size)},
        )
    if not isinstance(config.seed, int) or config.seed < 0 or config.seed > 0xFFFFFFFFFFFFFFFF:
        raise BakeError(
            E_INVALID_ARGUMENT,
            "seed must be an unsigned 64-bit integer",
            {"seed": config.seed},
        )
    if config.lod_count < 3:
        raise BakeError(
            E_INVALID_ARGUMENT,
            "at least three LOD levels are required",
            {"lod_count": int(config.lod_count)},
        )
    if config.normal_resolution < 2:
        raise BakeError(
            E_INVALID_ARGUMENT,
            "normal map resolution must be at least 2",
            {"normal_resolution": int(config.normal_resolution)},
        )
    if config.preview_width < 16 or config.preview_height < 16:
        raise BakeError(
            E_INVALID_ARGUMENT,
            "preview dimensions must both be at least 16",
        )
    if not math.isfinite(config.max_relative_error) or config.max_relative_error <= 0:
        raise BakeError(
            E_INVALID_ARGUMENT,
            "maximum relative simplification error must be positive and finite",
        )
    if not math.isfinite(config.weight_tolerance) or config.weight_tolerance <= 0:
        raise BakeError(
            E_INVALID_ARGUMENT,
            "weight tolerance must be positive and finite",
        )
    if config.preview_backend not in {"auto", "cpu", "opengl"}:
        raise BakeError(
            E_INVALID_ARGUMENT,
            "preview backend must be auto, cpu, or opengl",
            {"preview_backend": config.preview_backend},
        )
    if output == output.parent:
        raise BakeError(
            E_INVALID_ARGUMENT,
            "output directory must not be a filesystem root",
        )
    if output.exists() and not output.is_dir():
        raise BakeError(
            E_INVALID_ARGUMENT,
            "output path exists and is not a directory",
            {"output": str(output)},
        )
    if output.is_symlink():
        raise BakeError(
            E_INVALID_ARGUMENT,
            "output directory must not be a symbolic link",
            {"output": str(output)},
        )


def _register_artifact(
    path: Path,
    stage_dir: Path,
    manifest: dict[str, Any],
) -> str:
    relative = path.relative_to(stage_dir).as_posix()
    sha256 = _sha256_file(path)
    manifest["artifacts"].append(relative)
    manifest["checksums"][relative] = sha256
    return sha256


def _write_manifest(stage_dir: Path, manifest: dict[str, Any]) -> Path:
    path = stage_dir / MANIFEST_NAME
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _replace_directory(stage_dir: Path, output_dir: Path) -> None:
    parent = output_dir.parent
    backup: Path | None = None
    if output_dir.exists():
        backup = parent / f".{output_dir.name}.backup-{uuid.uuid4().hex}"
        output_dir.rename(backup)
    try:
        stage_dir.rename(output_dir)
    except Exception:
        if backup is not None and backup.exists() and not output_dir.exists():
            backup.rename(output_dir)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def _finish_failure(
    config: BakeConfig,
    output_dir: Path,
    stage_dir: Path,
    manifest: dict[str, Any],
    error: BakeError,
) -> BakeResult:
    shutil.rmtree(stage_dir, ignore_errors=True)
    manifest["status"] = "error"
    manifest["error"] = error.code
    manifest["error_message"] = error.message
    manifest["error_details"] = error.details
    manifest["lods"] = []
    manifest["normal_map"] = {
        "path": None,
        "metadata_path": None,
        "resolution": None,
        "pixel_step": None,
        "height_scale": None,
        "tangent_axis": "+X",
        "bitangent_axis": "-Z",
        "normal_axis": "+Y",
        "sha256": None,
        "metadata_sha256": None,
    }
    manifest["material_preview"] = {
        "path": None,
        "resolution": None,
        "sha256": None,
    }
    manifest["preview"] = {
        "path": None,
        "backend": None,
        "resolution": None,
        "sha256": None,
    }
    manifest["seam"] = {
        "pairs": [],
        "all_manifold": None,
        "t_junction_count": 0,
    }
    manifest["checksums"] = {}
    manifest["artifacts"] = []
    failure_stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.failure-",
            dir=str(output_dir.parent),
        )
    )
    try:
        manifest_path = _write_manifest(failure_stage, manifest)
        _replace_directory(failure_stage, output_dir)
        return BakeResult(
            manifest=manifest,
            manifest_path=output_dir / manifest_path.name,
            exit_code=ERROR_EXIT_CODES.get(error.code, 8),
        )
    except Exception as exc:
        shutil.rmtree(failure_stage, ignore_errors=True)
        raise BakeError(
            E_IO_ERROR,
            "failed to write atomic failure manifest",
            {"reason": str(exc)},
        ) from exc


def _source_summary(
    samples: TerrainSamples,
    heightfield_provided: bool,
    weights_provided: bool,
) -> dict[str, Any]:
    if heightfield_provided and weights_provided:
        kind = "provided_heightfield_and_weights"
    elif heightfield_provided:
        kind = "provided_heightfield"
    elif weights_provided:
        kind = "procedural_height_with_provided_weights"
    else:
        kind = "procedural"
    return {
        "kind": kind,
        "heightfield_path": None,
        "material_weights_path": None,
        "heightfield_sha256": samples.heightfield_sha256,
        "material_weights_sha256": samples.weights_sha256,
        "height_min": float(samples.heightfield.min()),
        "height_max": float(samples.heightfield.max()),
        "weight_sum_max_error": float(
            np.max(np.abs(np.sum(samples.weights, axis=2) - 1.0))
        ),
    }


def _lod_manifest_entry(
    level: int,
    mesh: TerrainMesh,
    path: Path,
    relative_error: float,
    error_threshold: float,
    target_vertices: int,
    reached_target: bool,
    collapse_count: int,
    sha256: str,
) -> dict[str, Any]:
    topology = topology_summary(mesh, level=level)
    return {
        "level": int(level),
        "path": path.name,
        "vertex_count": mesh.vertex_count,
        "face_count": mesh.face_count,
        "simplification_error": float(relative_error),
        "relative_simplification_error": float(relative_error),
        "error_threshold": float(error_threshold),
        "target_vertices": int(target_vertices),
        "reached_target": bool(reached_target),
        "collapse_count": int(collapse_count),
        "topology": topology,
        "sha256": sha256,
    }


def _write_json_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def bake(
    config: BakeConfig,
    heightfield: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    reporter: Reporter | None = None,
) -> BakeResult:
    """Run the complete bake and atomically publish exactly one output tree."""

    output_dir = Path(config.output_dir).expanduser().absolute()
    manifest = _base_manifest(config)
    total_start = time.perf_counter()
    stage_dir: Path | None = None
    try:
        if output_dir == output_dir.parent:
            raise BakeError(
                E_INVALID_ARGUMENT,
                "output directory must not be a filesystem root",
            )
        if output_dir.exists() and not output_dir.is_dir():
            raise BakeError(
                E_INVALID_ARGUMENT,
                "output path exists and is not a directory",
                {"output": str(output_dir)},
            )
        if output_dir.is_symlink():
            raise BakeError(
                E_INVALID_ARGUMENT,
                "output directory must not be a symbolic link",
                {"output": str(output_dir)},
            )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        stage_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.stage-",
                dir=str(output_dir.parent),
            )
        )
        _validate_config(config, output_dir)
        if output_dir.exists() and not os.access(output_dir, os.W_OK):
            raise BakeError(E_IO_ERROR, "output directory is not writable")
    except BakeError as error:
        manifest["timings_ms"]["total"] = _now_ms(total_start)
        _emit(reporter, f"[error] {error.code}: {error.message}")
        if stage_dir is not None:
            return _finish_failure(
                config, output_dir, stage_dir, manifest, error
            )
        raise
    except Exception as error:
        manifest["timings_ms"]["total"] = _now_ms(total_start)
        _emit(reporter, f"[error] {E_INTERNAL_ERROR}: {error}")
        raise

    try:
        stage_start = time.perf_counter()
        _emit(reporter, "[stage] validate heightfield and material weights")
        _validate_config(config, output_dir)
        if heightfield is None:
            samples = generate_samples(
                config.size,
                config.size,
                config.seed,
                config.weight_tolerance,
            )
        else:
            samples = samples_from_external(
                heightfield,
                weights,
                config.seed,
                config.weight_tolerance,
            )
        rows, cols = samples.heightfield.shape
        manifest["request"]["size"] = [int(rows), int(cols)]
        manifest["source"] = _source_summary(
            samples,
            heightfield_provided=heightfield is not None,
            weights_provided=weights is not None,
        )
        heightfield_path = stage_dir / "heightfield.npy"
        weights_path = stage_dir / "material_weights.npy"
        np.save(heightfield_path, samples.heightfield, allow_pickle=False)
        np.save(weights_path, samples.weights, allow_pickle=False)
        _register_artifact(heightfield_path, stage_dir, manifest)
        _register_artifact(weights_path, stage_dir, manifest)
        manifest["source"]["heightfield_path"] = heightfield_path.name
        manifest["source"]["material_weights_path"] = weights_path.name
        manifest["timings_ms"]["samples"] = _now_ms(stage_start)
        _emit(
            reporter,
            f"[samples] {cols}x{rows} seed={config.seed} "
            f"sha256={samples.heightfield_sha256[:12]} "
            f"({manifest['timings_ms']['samples']:.3f} ms)",
        )

        targets = lod_target_ratios(config.lod_count)
        thresholds = lod_error_thresholds(
            config.lod_count, config.max_relative_error
        )
        manifest["request"]["lod_target_ratios"] = [float(value) for value in targets]
        manifest["request"]["lod_error_thresholds"] = [
            float(value) for value in thresholds
        ]

        stage_start = time.perf_counter()
        base_mesh = build_base_mesh(samples.heightfield, samples.weights)
        topology_summary(base_mesh, level=0)
        meshes: list[TerrainMesh] = [base_mesh]
        lod_entries: list[dict[str, Any]] = []
        for level in range(config.lod_count):
            if level == 0:
                mesh = base_mesh
                path = stage_dir / "lod_0.obj"
                write_obj(mesh, path, f"terrain LOD 0 seed={config.seed}")
                entry = _lod_manifest_entry(
                    level,
                    mesh,
                    path,
                    relative_error=0.0,
                    error_threshold=0.0,
                    target_vertices=base_mesh.vertex_count,
                    reached_target=True,
                    collapse_count=0,
                    sha256=_register_artifact(path, stage_dir, manifest),
                )
                lod_entries.append(entry)
                continue

            target = max(8, int(round(base_mesh.vertex_count * targets[level - 1])))
            result = simplify_mesh(
                meshes[-1],
                target,
                thresholds[level - 1],
                f"LOD {level}",
            )
            if result.mesh.vertex_count >= meshes[-1].vertex_count:
                raise BakeError(
                    E_GEOMETRY_INVALID,
                    f"LOD {level} did not reduce mesh complexity",
                )
            if result.relative_error > config.max_relative_error + 1e-12:
                raise BakeError(
                    E_GEOMETRY_INVALID,
                    f"LOD {level} exceeded the simplification error cap",
                    {
                        "relative_error": result.relative_error,
                        "maximum": config.max_relative_error,
                    },
                )
            mesh = result.mesh
            meshes.append(mesh)
            path = stage_dir / f"lod_{level}.obj"
            write_obj(
                mesh,
                path,
                f"terrain LOD {level} seed={config.seed} "
                f"relative_error={result.relative_error:.12g}",
            )
            entry = _lod_manifest_entry(
                level,
                mesh,
                path,
                relative_error=result.relative_error,
                error_threshold=result.error_threshold,
                target_vertices=result.target_vertices,
                reached_target=result.reached_target,
                collapse_count=result.collapse_count,
                sha256=_register_artifact(path, stage_dir, manifest),
            )
            lod_entries.append(entry)
        manifest["lods"] = lod_entries
        manifest["timings_ms"]["lod"] = _now_ms(stage_start)
        _emit(
            reporter,
            "[lod] "
            + ", ".join(
                f"L{entry['level']}={entry['vertex_count']}v/{entry['face_count']}f"
                for entry in lod_entries
            )
            + f" ({manifest['timings_ms']['lod']:.3f} ms)",
        )

        stage_start = time.perf_counter()
        seam_pairs: list[dict[str, Any]] = []
        for level in range(1, config.lod_count):
            seam = build_transition_mesh(
                meshes[level - 1],
                meshes[level],
                level - 1,
                level,
            )
            path = stage_dir / f"transition_lod_{level - 1}_{level}.obj"
            write_obj(
                seam.mesh,
                path,
                f"LOD transition {level - 1}->{level} seed={config.seed}",
            )
            sha256 = _register_artifact(path, stage_dir, manifest)
            seam_pairs.append(
                {
                    "high_level": seam.high_level,
                    "low_level": seam.low_level,
                    "path": path.name,
                    "high_boundary_vertices": seam.high_boundary_vertices,
                    "low_boundary_vertices": seam.low_boundary_vertices,
                    "stitch_triangles": seam.stitch_triangles,
                    "t_junction_count": seam.t_junction_count,
                    "high_vertices_reused": seam.high_vertices_reused,
                    "combined_topology": seam.combined_topology,
                    "manifold": True,
                    "sha256": sha256,
                }
            )
        manifest["seam"] = {
            "pairs": seam_pairs,
            "all_manifold": True,
            "t_junction_count": int(
                sum(pair["t_junction_count"] for pair in seam_pairs)
            ),
        }
        manifest["timings_ms"]["seam"] = _now_ms(stage_start)
        _emit(
            reporter,
            "[seam] "
            + ", ".join(
                f"L{p['high_level']}->L{p['low_level']} "
                f"{p['stitch_triangles']} tris"
                for p in seam_pairs
            )
            + f" | T-junctions=0 ({manifest['timings_ms']['seam']:.3f} ms)",
        )

        stage_start = time.perf_counter()
        normal_map = bake_normal_map(
            samples.heightfield, config.normal_resolution
        )
        normal_path = stage_dir / "normal_map.png"
        save_normal_png(normal_map, normal_path, config.seed)
        normal_sha = _register_artifact(normal_path, stage_dir, manifest)
        normal_meta_path = stage_dir / "normal_map.json"
        _write_json_metadata(
            normal_meta_path,
            {
                "seed": config.seed,
                "resolution": [
                    int(config.normal_resolution),
                    int(config.normal_resolution),
                ],
                "pixel_step": float(normal_map.pixel_step),
                "height_scale": float(normal_map.height_scale),
                "coordinate_system": "X=columns,Z=rows,Y=up",
                "tangent_axis": "+X",
                "bitangent_axis": "-Z",
                "normal_axis": "+Y",
                "encoding": "linear_rgb_0_1",
                "derivative": (
                    "central finite difference, one output pixel, "
                    "clamped at borders"
                ),
                "sha256": normal_sha,
            },
        )
        normal_meta_sha = _register_artifact(normal_meta_path, stage_dir, manifest)
        manifest["normal_map"] = {
            "path": normal_path.name,
            "metadata_path": normal_meta_path.name,
            "resolution": [
                int(config.normal_resolution),
                int(config.normal_resolution),
            ],
            "pixel_step": float(normal_map.pixel_step),
            "height_scale": float(normal_map.height_scale),
            "tangent_axis": "+X",
            "bitangent_axis": "-Z",
            "normal_axis": "+Y",
            "sha256": normal_sha,
            "metadata_sha256": normal_meta_sha,
        }
        manifest["timings_ms"]["normals"] = _now_ms(stage_start)
        _emit(
            reporter,
            f"[normal] {config.normal_resolution}x{config.normal_resolution} "
            f"step={normal_map.pixel_step:.8f} "
            f"({manifest['timings_ms']['normals']:.3f} ms)",
        )

        stage_start = time.perf_counter()
        material_resolution = max(
            128, min(1024, int(config.normal_resolution))
        )
        material_rgb = bake_material_preview(
            samples.heightfield,
            samples.weights,
            material_resolution,
        )
        material_path = stage_dir / "material_preview.png"
        save_material_png(
            material_rgb,
            material_path,
            config.seed,
            material_resolution,
        )
        material_sha = _register_artifact(material_path, stage_dir, manifest)
        manifest["material_preview"] = {
            "path": material_path.name,
            "resolution": [material_resolution, material_resolution],
            "sha256": material_sha,
        }
        manifest["timings_ms"]["materials"] = _now_ms(stage_start)
        _emit(
            reporter,
            f"[materials] {material_resolution}x{material_resolution} "
            f"({manifest['timings_ms']['materials']:.3f} ms)",
        )

        stage_start = time.perf_counter()
        preview = render_preview(
            meshes[-1],
            normal_map.rgb,
            config.preview_width,
            config.preview_height,
            backend=config.preview_backend,
        )
        preview_path = stage_dir / "preview.png"
        save_preview_png(preview, preview_path, config.seed)
        preview_sha = _register_artifact(preview_path, stage_dir, manifest)
        manifest["backend"] = {
            "requested": str(config.preview_backend),
            "attempted": list(preview.attempted_backends),
            "used": preview.backend,
            "fallback_reason": preview.fallback_reason,
        }
        manifest["preview"] = {
            "path": preview_path.name,
            "backend": preview.backend,
            "resolution": [preview.width, preview.height],
            "sha256": preview_sha,
        }
        manifest["timings_ms"]["preview"] = _now_ms(stage_start)
        if preview.fallback_reason:
            manifest["warnings"].append(
                "OpenGL unavailable; preview used the explicit CPU rasterizer: "
                + preview.fallback_reason
            )
            _emit(
                reporter,
                f"[preview] OpenGL unavailable, CPU fallback: "
                f"{preview.fallback_reason[:120]}",
            )
        _emit(
            reporter,
            f"[preview] backend={preview.backend} "
            f"{preview.width}x{preview.height} "
            f"({manifest['timings_ms']['preview']:.3f} ms)",
        )

        output_start = time.perf_counter()
        manifest["status"] = "ok"
        manifest["error"] = None
        manifest["error_message"] = None
        manifest["error_details"] = {}
        manifest["timings_ms"]["output"] = _now_ms(output_start)
        manifest["timings_ms"]["total"] = _now_ms(total_start)
        manifest_path = _write_manifest(stage_dir, manifest)
        _replace_directory(stage_dir, output_dir)
        final_manifest_path = output_dir / manifest_path.name
        _emit(
            reporter,
            f"[result] status=ok lods={config.lod_count} "
            f"artifacts={len(manifest['artifacts'])} "
            f"total={manifest['timings_ms']['total']:.3f} ms -> {output_dir}",
        )
        return BakeResult(
            manifest=manifest,
            manifest_path=final_manifest_path,
            exit_code=0,
        )
    except BakeError as error:
        manifest["timings_ms"]["total"] = _now_ms(total_start)
        _emit(reporter, f"[error] {error.code}: {error.message}")
        assert stage_dir is not None
        return _finish_failure(
            config,
            output_dir,
            stage_dir,
            manifest,
            error,
        )
    except Exception as error:
        manifest["timings_ms"]["total"] = _now_ms(total_start)
        wrapped = BakeError(
            E_INTERNAL_ERROR,
            "unexpected pipeline failure",
            {"type": error.__class__.__name__, "reason": str(error)},
        )
        _emit(reporter, f"[error] {wrapped.code}: {wrapped.message}: {error}")
        assert stage_dir is not None
        return _finish_failure(
            config,
            output_dir,
            stage_dir,
            manifest,
            wrapped,
        )
