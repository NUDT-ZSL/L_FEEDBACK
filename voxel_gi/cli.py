"""Command-line entry for baking and validating SH-9 caches."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .baker import bake_scene
from .cache import (
    ALGORITHM_VERSION,
    FORMAT_VERSION,
    _atomic_write,
    peek_cache,
    read_cache,
    validate_cache,
    write_cache,
)
from .errors import BusinessError, ErrorCode
from .scene import (
    ProbeLayout,
    Scene,
    build_probe_layout,
    compute_geometry_hash,
    compute_layout_hash,
    compute_lighting_hash,
    compute_scene_hash,
    load_scene,
)


def _report_path(cache_path: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return Path(explicit)
    return Path(str(cache_path) + ".report.json")


def _guard_distinct_paths(
    scene_path: Path,
    cache_path: Path,
    report_path: Path,
) -> None:
    scene_resolved = Path(scene_path).resolve()
    cache_resolved = Path(cache_path).resolve()
    report_resolved = Path(report_path).resolve()
    if scene_resolved == cache_resolved:
        raise BusinessError(
            ErrorCode.INVALID_SCHEMA,
            "Scene input and cache output must be different paths",
            {"path": str(scene_path)},
        )
    if report_resolved in (scene_resolved, cache_resolved):
        raise BusinessError(
            ErrorCode.INVALID_SCHEMA,
            "Report path must not overwrite the scene or cache",
            {"report": str(report_path)},
        )


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    data = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    try:
        _atomic_write(Path(path), data)
    except OSError as exc:
        raise BusinessError(
            ErrorCode.IO_ERROR,
            "Unable to write JSON report atomically",
            {"path": str(path), "reason": str(exc)},
        ) from exc


def _expected_hashes(scene: Scene, layout: ProbeLayout) -> Dict[str, str]:
    geometry = compute_geometry_hash(scene)
    layout_hash = compute_layout_hash(layout)
    lighting = compute_lighting_hash(scene)
    scene_hash = compute_scene_hash(
        scene,
        layout,
        geometry_hash=geometry,
        layout_hash=layout_hash,
        lighting_hash=lighting,
    )
    return {
        "geometry_hash": geometry,
        "layout_hash": layout_hash,
        "lighting_hash": lighting,
        "scene_hash": scene_hash,
    }


def _classify_existing_cache(
    cache_path: Path,
    expected: Dict[str, str],
) -> Dict[str, Any]:
    if not cache_path.exists():
        return {"status": "absent", "action": "created"}
    try:
        metadata = peek_cache(cache_path)
    except BusinessError as error:
        return {
            "status": "invalid",
            "action": "replaced_invalid_or_incompatible_cache",
            "reason_code": error.code,
            "reason": error.message,
        }
    if metadata.get("geometry_hash") != expected["geometry_hash"]:
        return {
            "status": "stale",
            "action": "rebaked_geometry_mismatch",
            "reason_code": ErrorCode.CACHE_GEOMETRY_MISMATCH,
        }
    if metadata.get("scene_hash") != expected["scene_hash"]:
        return {
            "status": "stale",
            "action": "rebaked_scene_mismatch",
            "reason_code": ErrorCode.CACHE_SCENE_MISMATCH,
        }
    metadata_layout = metadata.get("layout")
    if not isinstance(metadata_layout, dict):
        return {
            "status": "invalid",
            "action": "replaced_invalid_or_incompatible_cache",
            "reason_code": ErrorCode.CACHE_CORRUPT,
        }
    if metadata_layout.get("hash") != expected["layout_hash"]:
        return {
            "status": "stale",
            "action": "rebaked_layout_mismatch",
            "reason_code": ErrorCode.CACHE_LAYOUT_MISMATCH,
        }
    return {"status": "reusable", "action": "overwritten_with_deterministic_result"}


def _energy_report(energy: Dict[str, Any]) -> Dict[str, Any]:
    bound = float(energy.get("bound", 0.0))
    maximum = float(energy.get("max_probe_irradiance", 0.0))
    result = dict(energy)
    result["within_bound"] = bool(maximum <= bound + 1.0e-9)
    return result


def bake_command(
    scene_path: Path,
    cache_path: Path,
    report_path: Optional[Path],
    threads: Optional[int],
    command_start: float,
) -> Dict[str, Any]:
    _guard_distinct_paths(
        scene_path,
        cache_path,
        _report_path(cache_path, report_path),
    )
    scene = load_scene(scene_path)
    layout = build_probe_layout(scene)
    expected = _expected_hashes(scene, layout)
    existing = _classify_existing_cache(cache_path, expected)
    result = bake_scene(scene, layout=layout, threads=threads)
    written = write_cache(cache_path, scene, layout, result)
    elapsed = time.perf_counter() - command_start
    report = {
        "status": "ok",
        "mode": "bake",
        "algorithm_version": ALGORITHM_VERSION,
        "format_version": FORMAT_VERSION,
        "scene_id": scene.scene_id,
        "probe_count": result.probe_count,
        "mean_residual": result.mean_residual,
        "max_residual": result.max_residual,
        "total_elapsed_seconds": elapsed,
        "version_hash": written.file_sha256,
        "geometry_hash": result.geometry_hash,
        "scene_hash": result.scene_hash,
        "layout_hash": result.layout_hash,
        "lighting_hash": result.lighting_hash,
        "threads": result.thread_count,
        "energy": _energy_report(
            {
                "bound": result.energy_bound,
                "max_probe_irradiance": result.max_probe_irradiance,
                "max_channel_irradiance": result.max_channel_irradiance,
                "clamped_probes": result.clamped_probes,
                "albedo_upper_bound": scene.albedo_upper_bound,
                "measure": "sum_of_rgb_channels",
                "total_light_power": scene.total_light_power,
            }
        ),
        "voxelization": result.grid_info,
        "bake_seconds": {
            "voxelization": result.voxelization_seconds,
            "probes": result.probe_bake_seconds,
        },
        "cache": {
            "path": str(cache_path),
            "size_bytes": written.file_size,
            "payload_sha256": written.payload_sha256,
            "existing_cache": existing,
        },
    }
    _write_json(_report_path(cache_path, report_path), report)
    return report


def validate_command(
    scene_path: Path,
    cache_path: Path,
    report_path: Optional[Path],
    command_start: float,
) -> Dict[str, Any]:
    _guard_distinct_paths(
        scene_path,
        cache_path,
        _report_path(cache_path, report_path),
    )
    scene = load_scene(scene_path)
    layout = build_probe_layout(scene)
    cache = read_cache(cache_path)
    validate_cache(scene, layout, cache)
    elapsed = time.perf_counter() - command_start
    metadata = cache.metadata
    metadata_layout = metadata.get("layout")
    layout_hash = (
        metadata_layout.get("hash")
        if isinstance(metadata_layout, dict)
        else None
    )
    metadata_energy = metadata.get("energy")
    if not isinstance(metadata_energy, dict):
        metadata_energy = {}
    metadata_voxelization = metadata.get("voxelization")
    if not isinstance(metadata_voxelization, dict):
        metadata_voxelization = {}
    report = {
        "status": "ok",
        "mode": "validate",
        "rebaked": False,
        "algorithm_version": ALGORITHM_VERSION,
        "format_version": cache.format_version,
        "scene_id": scene.scene_id,
        "probe_count": cache.probe_count,
        "mean_residual": cache.mean_residual,
        "max_residual": cache.max_residual,
        "total_elapsed_seconds": elapsed,
        "version_hash": cache.file_sha256,
        "payload_sha256": cache.payload_sha256,
        "geometry_hash": metadata.get("geometry_hash"),
        "scene_hash": metadata.get("scene_hash"),
        "layout_hash": layout_hash,
        "lighting_hash": metadata.get("lighting_hash"),
        "cache_size_bytes": cache.file_size,
        "energy": _energy_report(metadata_energy),
        "voxelization": metadata_voxelization,
        "validation": {
            "version_checked": True,
            "geometry_hash_checked": True,
            "scene_hash_checked": True,
            "layout_hash_checked": True,
            "payload_checksum_checked": True,
        },
    }
    _write_json(_report_path(cache_path, report_path), report)
    return report


def _exit_code(error: BusinessError) -> int:
    if error.code.startswith("CACHE_"):
        return 3
    if error.code in (
        ErrorCode.INVALID_JSON,
        ErrorCode.INVALID_SCHEMA,
        ErrorCode.INVALID_MESH_NAN,
        ErrorCode.INVALID_MESH_DEGENERATE,
        ErrorCode.INVALID_MATERIAL,
        ErrorCode.INVALID_LIGHT,
        ErrorCode.INVALID_PROBE_SPACING,
        ErrorCode.INVALID_PROBE_BOUNDS,
        ErrorCode.INVALID_BAKE_CONFIG,
        ErrorCode.PROBE_INSIDE_OCCLUDER,
    ):
        return 2
    if error.code == ErrorCode.IO_ERROR:
        return 4
    return 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bake or validate a deterministic voxelized SH-9 irradiance cache."
    )
    parser.add_argument("scene", type=Path, help="Scene description JSON")
    parser.add_argument("cache", type=Path, help="SH cache output/input path")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate an existing cache without baking",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Override the scene's worker thread count during bake",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path (default: <cache>.report.json)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    command_start = time.perf_counter()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.validate:
            report = validate_command(
                args.scene,
                args.cache,
                args.report,
                command_start,
            )
        else:
            report = bake_command(
                args.scene,
                args.cache,
                args.report,
                args.threads,
                command_start,
            )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "mode": report["mode"],
                    "probe_count": report["probe_count"],
                    "mean_residual": report["mean_residual"],
                    "total_elapsed_seconds": report["total_elapsed_seconds"],
                    "version_hash": report["version_hash"],
                    "report": str(_report_path(args.cache, args.report)),
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
        )
        return 0
    except BusinessError as error:
        print(
            json.dumps(error.to_dict(), indent=2, sort_keys=True, ensure_ascii=True),
            file=sys.stderr,
        )
        return _exit_code(error)
    except Exception as error:
        internal = BusinessError(
            ErrorCode.INTERNAL_ERROR,
            "Unexpected internal failure",
            {"type": type(error).__name__, "reason": str(error)},
        )
        print(
            json.dumps(internal.to_dict(), indent=2, sort_keys=True, ensure_ascii=True),
            file=sys.stderr,
        )
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
