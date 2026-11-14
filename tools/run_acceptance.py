"""Run the complete local acceptance path and emit a machine-readable report."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from voxel_gi.baker import bake_scene
from voxel_gi.cache import _HEADER, read_cache, validate_cache, write_cache
from voxel_gi.errors import BusinessError, ErrorCode
from voxel_gi.sample_scene import generate_sample_scene, write_sample_scene
from voxel_gi.scene import build_probe_layout, load_scene, parse_scene
from voxel_gi.sh import IRRADIANCE_FROM_C00
from voxel_gi.voxel import VoxelGrid


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_cli(*arguments: object) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "voxel_gi.cli", *[str(item) for item in arguments]],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _error_code(function: Any) -> str:
    try:
        function()
    except BusinessError as error:
        return error.code
    return "NO_ERROR"


def run_acceptance() -> Dict[str, Any]:
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    scene_path = artifacts / "sample_scene.json"
    cache_path = artifacts / "sample_cache.vgic"
    bake_report_path = artifacts / "bake_report.json"
    validate_report_path = artifacts / "validate_report.json"
    verification_path = artifacts / "verification.json"

    write_sample_scene(scene_path)
    scene = load_scene(scene_path)
    layout = build_probe_layout(scene)
    grid = VoxelGrid.build(scene)
    serial_result = bake_scene(scene, layout, grid, threads=1)

    with tempfile.TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        repeated_scene_path = temporary_root / "sample_scene_repeat.json"
        write_sample_scene(repeated_scene_path)
        deterministic_scene_json = (
            repeated_scene_path.read_bytes() == scene_path.read_bytes()
        )
        serial_cache_path = temporary_root / "serial.vgic"
        write_cache(serial_cache_path, scene, layout, serial_result)

        completed_bake = _run_cli(
            scene_path,
            cache_path,
            "--threads",
            "4",
            "--report",
            bake_report_path,
        )
        if completed_bake.returncode != 0:
            raise RuntimeError(
                "CLI bake failed with code {}: {}".format(
                    completed_bake.returncode,
                    completed_bake.stderr,
                )
            )
        bake_report = json.loads(bake_report_path.read_text(encoding="utf-8"))
        cache = read_cache(cache_path)
        validate_cache(scene, layout, cache)

        cache_hash_before_validate = _sha256(cache_path)
        completed_validate = _run_cli(
            scene_path,
            cache_path,
            "--validate",
            "--report",
            validate_report_path,
        )
        if completed_validate.returncode != 0:
            raise RuntimeError(
                "CLI validate failed with code {}: {}".format(
                    completed_validate.returncode,
                    completed_validate.stderr,
                )
            )
        validate_report = json.loads(validate_report_path.read_text(encoding="utf-8"))
        cache_hash_after_validate = _sha256(cache_path)

        deterministic_parallel = (
            serial_cache_path.read_bytes() == cache_path.read_bytes()
        )
        serial_hash = _sha256(serial_cache_path)

        channel_irradiance = (
            cache.coefficients[:, 0, :].astype(np.float64) * IRRADIANCE_FROM_C00
        )
        total_irradiance = np.sum(channel_irradiance, axis=1)
        energy_bound = float(cache.metadata["energy"]["bound"])
        max_probe_irradiance = float(np.max(total_irradiance))
        energy_within_bound = bool(max_probe_irradiance <= energy_bound + 1.0e-9)
        mean_residual = float(
            np.mean(cache.residuals, dtype=np.float64)
        )

        changed_raw = copy.deepcopy(json.loads(scene_path.read_text(encoding="utf-8")))
        changed_raw["meshes"][0]["triangles"][0][0][0] += 0.01
        changed_scene = parse_scene(changed_raw)
        geometry_mismatch_code = _error_code(
            lambda: validate_cache(changed_scene, layout, cache)
        )

        modified_version_path = temporary_root / "bad_version.vgic"
        bad_version = bytearray(cache_path.read_bytes())
        _, version, _, _ = _HEADER.unpack_from(bad_version, 0)
        struct.pack_into("<I", bad_version, 8, int(version) + 1)
        modified_version_path.write_bytes(bytes(bad_version))
        version_error_code = _error_code(lambda: read_cache(modified_version_path))

        modified_checksum_path = temporary_root / "bad_checksum.vgic"
        bad_checksum = bytearray(cache_path.read_bytes())
        bad_checksum[-1] ^= 0xFF
        modified_checksum_path.write_bytes(bytes(bad_checksum))
        checksum_error_code = _error_code(lambda: read_cache(modified_checksum_path))

        invalid_paths: Dict[str, Path] = {}
        invalid_codes: Dict[str, str] = {}
        invalid_cases: Dict[str, tuple[Dict[str, Any], str]] = {}

        nan_case = copy.deepcopy(generate_sample_scene())
        nan_case["meshes"][0]["triangles"][0][0][0] = float("nan")
        invalid_cases["nan_vertex"] = (nan_case, ErrorCode.INVALID_MESH_NAN)

        degenerate_case = copy.deepcopy(generate_sample_scene())
        degenerate_case["meshes"][0]["triangles"][0][1] = list(
            degenerate_case["meshes"][0]["triangles"][0][0]
        )
        invalid_cases["degenerate_triangle"] = (
            degenerate_case,
            ErrorCode.INVALID_MESH_DEGENERATE,
        )

        light_case = copy.deepcopy(generate_sample_scene())
        light_case["lights"][0]["radiant_power"] = 0.0
        invalid_cases["light_out_of_range"] = (
            light_case,
            ErrorCode.INVALID_LIGHT,
        )

        spacing_case = copy.deepcopy(generate_sample_scene())
        spacing_case["probe_layout"]["spacing"] = 0.0
        invalid_cases["invalid_probe_spacing"] = (
            spacing_case,
            ErrorCode.INVALID_PROBE_SPACING,
        )

        for name, (raw, expected) in invalid_cases.items():
            path = temporary_root / (name + ".json")
            path.write_text(json.dumps(raw), encoding="utf-8")
            invalid_paths[name] = path
            invalid_codes[name] = _error_code(lambda path=path: load_scene(path))
            if invalid_codes[name] != expected:
                raise AssertionError(
                    "{} returned {}, expected {}".format(
                        name, invalid_codes[name], expected
                    )
                )

        atomic_cache_hash = _sha256(cache_path)
        atomic_invalid = _run_cli(invalid_paths["nan_vertex"], cache_path)
        atomic_after_hash = _sha256(cache_path)
        atomic_preserved = bool(
            atomic_invalid.returncode == 2 and atomic_after_hash == atomic_cache_hash
        )
        atomic_error_code = ""
        try:
            atomic_error_code = json.loads(atomic_invalid.stderr)["error"]["code"]
        except (KeyError, json.JSONDecodeError):
            atomic_error_code = "UNPARSEABLE"

    if not deterministic_parallel:
        raise AssertionError("Single-thread and four-thread caches differ")
    if not deterministic_scene_json:
        raise AssertionError("Repeated sample-scene generation differs")
    if not energy_within_bound:
        raise AssertionError("Energy conservation check failed")
    if geometry_mismatch_code != ErrorCode.CACHE_GEOMETRY_MISMATCH:
        raise AssertionError("Geometry mismatch was not rejected")
    if version_error_code != ErrorCode.CACHE_VERSION_UNSUPPORTED:
        raise AssertionError("Unsupported cache version was not rejected")
    if checksum_error_code != ErrorCode.CACHE_CHECKSUM_MISMATCH:
        raise AssertionError("Cache checksum mismatch was not rejected")
    if not atomic_preserved:
        raise AssertionError("Invalid input damaged the existing cache")
    for key in (
        "probe_count",
        "mean_residual",
        "total_elapsed_seconds",
        "version_hash",
    ):
        if key not in bake_report:
            raise AssertionError("Bake report missing {}".format(key))
    if validate_report.get("mode") != "validate":
        raise AssertionError("Validate report has the wrong mode")
    if cache_hash_after_validate != cache_hash_before_validate:
        raise AssertionError("--validate modified the cache")

    report = {
        "status": "pass",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": sys.platform,
        },
        "sample_bake": {
            "scene_path": str(scene_path),
            "cache_path": str(cache_path),
            "bake_report_path": str(bake_report_path),
            "probe_count": bake_report["probe_count"],
            "mean_residual": bake_report["mean_residual"],
            "max_residual": bake_report["max_residual"],
            "total_elapsed_seconds": bake_report["total_elapsed_seconds"],
            "version_hash": bake_report["version_hash"],
            "threads": bake_report["threads"],
            "voxelization": bake_report["voxelization"],
            "geometry_hash": bake_report["geometry_hash"],
            "scene_hash": bake_report["scene_hash"],
            "layout_hash": bake_report["layout_hash"],
        },
        "checks": {
            "deterministic_scene_json": deterministic_scene_json,
            "single_vs_multithread_bit_identical": {
                "passed": deterministic_parallel,
                "serial_cache_sha256": serial_hash,
                "parallel_cache_sha256": _sha256(cache_path),
                "max_abs_coefficient_difference": 0.0,
                "serial_probe_seconds": serial_result.probe_bake_seconds,
                "parallel_probe_seconds": bake_report["bake_seconds"]["probes"],
            },
            "cache_roundtrip_and_validate": {
                "passed": True,
                "file_sha256": cache.file_sha256,
                "payload_sha256": cache.payload_sha256,
                "probe_count": cache.probe_count,
                "coefficient_dtype": str(cache.coefficients.dtype),
                "coefficient_shape": list(cache.coefficients.shape),
                "residual_dtype": str(cache.residuals.dtype),
                "mean_residual_from_cache": mean_residual,
                "validate_mode": validate_report["mode"],
                "cache_unchanged_by_validate": cache_hash_after_validate
                == cache_hash_before_validate,
            },
            "energy_conservation": {
                "passed": energy_within_bound,
                "measure": "sum_of_rgb_c00_integrals",
                "energy_bound": energy_bound,
                "max_probe_irradiance": max_probe_irradiance,
                "clamped_probes": cache.metadata["energy"]["clamped_probes"],
            },
            "cache_version_rejection": {
                "passed": version_error_code == ErrorCode.CACHE_VERSION_UNSUPPORTED,
                "error_code": version_error_code,
            },
            "cache_checksum_rejection": {
                "passed": checksum_error_code == ErrorCode.CACHE_CHECKSUM_MISMATCH,
                "error_code": checksum_error_code,
            },
            "geometry_hash_mismatch_rejection": {
                "passed": geometry_mismatch_code
                == ErrorCode.CACHE_GEOMETRY_MISMATCH,
                "error_code": geometry_mismatch_code,
            },
            "structured_input_errors": {
                "passed": True,
                "codes": invalid_codes,
            },
            "atomic_cache_preservation": {
                "passed": atomic_preserved,
                "invalid_input_error_code": atomic_error_code,
                "cache_sha256_before": atomic_cache_hash,
                "cache_sha256_after": atomic_after_hash,
            },
        },
    }
    verification_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    try:
        report = run_acceptance()
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "probe_count": report["sample_bake"]["probe_count"],
                    "mean_residual": report["sample_bake"]["mean_residual"],
                    "version_hash": report["sample_bake"]["version_hash"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        artifacts = ROOT / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "fail",
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
        (artifacts / "verification.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
