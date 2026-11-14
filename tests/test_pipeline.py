"""End-to-end acceptance tests for the voxelized SH-9 baker."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from voxel_gi.baker import bake_scene
from voxel_gi.cache import (
    _HEADER,
    read_cache,
    validate_cache,
    write_cache,
)
from voxel_gi.errors import BusinessError, ErrorCode
from voxel_gi.sample_scene import generate_sample_scene, write_sample_scene
from voxel_gi.scene import (
    SCHEMA_VERSION,
    build_probe_layout,
    compute_geometry_hash,
    load_scene,
    parse_scene,
)
from voxel_gi.sh import sh_orthonormality_error
from voxel_gi.sampler import hemisphere_directions
from voxel_gi.voxel import VoxelGrid

ROOT = Path(__file__).resolve().parents[1]


def _reduced_scene_raw() -> Dict[str, Any]:
    raw = copy.deepcopy(generate_sample_scene())
    raw["bake"].update(
        {
            "voxel_resolution": 20,
            "ray_count": 32,
            "threads": 4,
        }
    )
    raw["probe_layout"]["spacing"] = 2.0
    for light in raw["lights"]:
        light["samples"] = 1
    return raw


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _run_cli(*arguments: object) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "voxel_gi.cli", *[str(item) for item in arguments]],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


class PipelineTests(unittest.TestCase):
    def test_sample_scene_generation_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.json"
            second = root / "second.json"
            write_sample_scene(first)
            write_sample_scene(second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            scene = load_scene(first)
            self.assertEqual(scene.schema_version, SCHEMA_VERSION)
            self.assertEqual(len(scene.meshes), 7)
            self.assertEqual(len(scene.lights), 2)
            layout = build_probe_layout(scene)
            self.assertEqual(layout.counts, (4, 2, 2))
            self.assertEqual(layout.probe_count, 16)

    def test_sh_basis_and_hemisphere_uniformity(self) -> None:
        diagonal_error, off_diagonal_error = sh_orthonormality_error(4096)
        self.assertLess(diagonal_error, 1.0e-5)
        self.assertLess(off_diagonal_error, 1.0e-3)
        directions = hemisphere_directions(
            4096, 20261115, 7, np.array([0.0, 1.0, 0.0])
        )
        self.assertAlmostEqual(float(np.mean(directions[:, 1])), 0.5, places=4)
        mean_vector = np.mean(directions, axis=0)
        self.assertAlmostEqual(float(mean_vector[0]), 0.0, places=3)
        self.assertAlmostEqual(float(mean_vector[2]), 0.0, places=3)
        self.assertAlmostEqual(float(mean_vector[1]), 0.5, places=3)
        self.assertTrue(np.all(directions[:, 1] >= 0.0))

    def test_voxelization_spatial_hash_and_ray_hits(self) -> None:
        scene = parse_scene(_reduced_scene_raw())
        grid = VoxelGrid.build(scene)
        self.assertGreater(grid.occupied_voxel_count, 0)
        self.assertGreater(grid.spatial_hash.bucket_count, 0)
        self.assertGreater(grid.spatial_hash.entry_count, 0)
        hit = grid.trace_primary(
            np.array([7.0, 2.0, 5.0]),
            np.array([0.0, -1.0, 0.0]),
        )
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertAlmostEqual(float(hit.point[1]), 0.0, places=6)
        self.assertAlmostEqual(abs(float(hit.normal[1])), 1.0, places=6)
        self.assertTrue(
            grid.is_occluded(
                np.array([4.0, 3.0, 0.6]),
                np.array([4.0, 3.0, -2.0]),
            )
        )

    def test_single_and_multithread_results_are_bit_identical(self) -> None:
        scene = parse_scene(_reduced_scene_raw())
        layout = build_probe_layout(scene)
        grid = VoxelGrid.build(scene)
        serial = bake_scene(scene, layout, grid, threads=1)
        parallel = bake_scene(scene, layout, grid, threads=4)
        self.assertTrue(np.array_equal(serial.coefficients, parallel.coefficients))
        self.assertTrue(np.array_equal(serial.residuals, parallel.residuals))
        self.assertEqual(serial.mean_residual, parallel.mean_residual)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "serial.vgic"
            second = root / "parallel.vgic"
            write_cache(first, scene, layout, serial)
            write_cache(second, scene, layout, parallel)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_cache_roundtrip_and_validate_mode_does_not_modify_cache(self) -> None:
        scene = parse_scene(_reduced_scene_raw())
        layout = build_probe_layout(scene)
        result = bake_scene(scene, layout, threads=4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene_path = root / "scene.json"
            cache_path = root / "cache.vgic"
            validate_report = root / "validate.json"
            _write_json(scene_path, _reduced_scene_raw())
            written = write_cache(cache_path, scene, layout, result)
            cache = read_cache(cache_path)
            self.assertEqual(cache.file_sha256, written.file_sha256)
            self.assertEqual(cache.payload_sha256, written.payload_sha256)
            self.assertTrue(
                np.array_equal(cache.coefficients, result.coefficients)
            )
            self.assertTrue(np.array_equal(cache.residuals, result.residuals))
            validate_cache(scene, layout, cache)
            before = hashlib.sha256(cache_path.read_bytes()).hexdigest()
            completed = _run_cli(
                scene_path,
                cache_path,
                "--validate",
                "--report",
                validate_report,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            after = hashlib.sha256(cache_path.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            report = json.loads(validate_report.read_text(encoding="utf-8"))
            self.assertEqual(report["mode"], "validate")
            self.assertFalse(report["rebaked"])
            self.assertEqual(report["probe_count"], layout.probe_count)
            self.assertEqual(report["version_hash"], before)

    def test_geometry_hash_mismatch_is_rejected(self) -> None:
        scene = parse_scene(_reduced_scene_raw())
        layout = build_probe_layout(scene)
        result = bake_scene(scene, layout, threads=4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path = root / "cache.vgic"
            write_cache(cache_path, scene, layout, result)
            changed_raw = copy.deepcopy(_reduced_scene_raw())
            changed_raw["meshes"][0]["triangles"][0][0][0] += 0.01
            changed_scene = parse_scene(changed_raw)
            self.assertNotEqual(
                compute_geometry_hash(scene),
                compute_geometry_hash(changed_scene),
            )
            cache = read_cache(cache_path)
            with self.assertRaises(BusinessError) as context:
                validate_cache(changed_scene, layout, cache)
            self.assertEqual(context.exception.code, ErrorCode.CACHE_GEOMETRY_MISMATCH)

    def test_cli_geometry_mismatch_rejects_then_rebakes_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_path = root / "original.json"
            changed_path = root / "changed.json"
            cache_path = root / "cache.vgic"
            validate_report = root / "rejected.json"
            rebake_report = root / "rebaked.json"
            _write_json(original_path, _reduced_scene_raw())
            first = _run_cli(original_path, cache_path, "--threads", "4")
            self.assertEqual(first.returncode, 0, first.stderr)
            original_hash = hashlib.sha256(cache_path.read_bytes()).hexdigest()

            changed_raw = copy.deepcopy(_reduced_scene_raw())
            changed_raw["meshes"][0]["triangles"][0][0][0] += 0.01
            _write_json(changed_path, changed_raw)
            rejected = _run_cli(
                changed_path,
                cache_path,
                "--validate",
                "--report",
                validate_report,
            )
            self.assertEqual(rejected.returncode, 3, rejected.stderr)
            error = json.loads(rejected.stderr)
            self.assertEqual(
                error["error"]["code"], ErrorCode.CACHE_GEOMETRY_MISMATCH
            )
            self.assertEqual(
                hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                original_hash,
            )

            rebaked = _run_cli(
                changed_path,
                cache_path,
                "--threads",
                "4",
                "--report",
                rebake_report,
            )
            self.assertEqual(rebaked.returncode, 0, rebaked.stderr)
            report = json.loads(rebake_report.read_text(encoding="utf-8"))
            self.assertEqual(
                report["cache"]["existing_cache"]["reason_code"],
                ErrorCode.CACHE_GEOMETRY_MISMATCH,
            )
            self.assertEqual(
                report["cache"]["existing_cache"]["action"],
                "rebaked_geometry_mismatch",
            )
            self.assertNotEqual(
                hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                original_hash,
            )

    def test_unsupported_version_and_checksum_mismatch(self) -> None:
        scene = parse_scene(_reduced_scene_raw())
        layout = build_probe_layout(scene)
        result = bake_scene(scene, layout, threads=4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path = root / "cache.vgic"
            write_cache(cache_path, scene, layout, result)
            original = cache_path.read_bytes()
            magic, version, metadata_length, payload_length = _HEADER.unpack_from(
                original, 0
            )
            self.assertEqual(magic, b"VXGI9SH1")
            corrupted_version = bytearray(original)
            struct.pack_into("<I", corrupted_version, 8, version + 1)
            cache_path.write_bytes(bytes(corrupted_version))
            with self.assertRaises(BusinessError) as context:
                read_cache(cache_path)
            self.assertEqual(
                context.exception.code, ErrorCode.CACHE_VERSION_UNSUPPORTED
            )
            checksum_corrupt = bytearray(original)
            checksum_corrupt[-1] ^= 0xFF
            cache_path.write_bytes(bytes(checksum_corrupt))
            with self.assertRaises(BusinessError) as context:
                read_cache(cache_path)
            self.assertEqual(
                context.exception.code, ErrorCode.CACHE_CHECKSUM_MISMATCH
            )

    def test_energy_guard_clamps_every_probe(self) -> None:
        raw = _reduced_scene_raw()
        raw["bake"]["ambient_radiance"] = [100.0, 100.0, 100.0]
        scene = parse_scene(raw)
        result = bake_scene(scene, build_probe_layout(scene), threads=4)
        self.assertGreater(result.clamped_probes, 0)
        self.assertTrue(result.within_energy_bound)
        self.assertLessEqual(
            result.max_probe_irradiance,
            result.energy_bound + 1.0e-9,
        )
        self.assertGreater(
            result.max_probe_irradiance,
            result.energy_bound * 0.999,
        )

    def test_cli_bake_report_and_repeat_binary_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene_path = root / "scene.json"
            first_cache = root / "first.vgic"
            second_cache = root / "second.vgic"
            first_report = root / "first_report.json"
            second_report = root / "second_report.json"
            _write_json(scene_path, _reduced_scene_raw())
            first = _run_cli(
                scene_path,
                first_cache,
                "--threads",
                "4",
                "--report",
                first_report,
            )
            second = _run_cli(
                scene_path,
                second_cache,
                "--threads",
                "1",
                "--report",
                second_report,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(first_cache.read_bytes(), second_cache.read_bytes())
            report = json.loads(first_report.read_text(encoding="utf-8"))
            for key in (
                "probe_count",
                "mean_residual",
                "total_elapsed_seconds",
                "version_hash",
            ):
                self.assertIn(key, report)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["mode"], "bake")
            self.assertGreater(report["probe_count"], 0)
            self.assertGreaterEqual(report["mean_residual"], 0.0)
            self.assertGreaterEqual(report["total_elapsed_seconds"], 0.0)
            self.assertEqual(len(report["version_hash"]), 64)

    def test_structured_input_errors_do_not_damage_existing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid_scene_path = root / "scene.json"
            invalid_scene_path = root / "invalid.json"
            cache_path = root / "cache.vgic"
            _write_json(valid_scene_path, _reduced_scene_raw())
            valid = _run_cli(valid_scene_path, cache_path, "--threads", "4")
            self.assertEqual(valid.returncode, 0, valid.stderr)
            original_cache = cache_path.read_bytes()

            cases = []
            nan_case = copy.deepcopy(_reduced_scene_raw())
            nan_case["meshes"][0]["triangles"][0][0][0] = float("nan")
            cases.append((nan_case, ErrorCode.INVALID_MESH_NAN))
            degenerate_case = copy.deepcopy(_reduced_scene_raw())
            degenerate_case["meshes"][0]["triangles"][0][1] = list(
                degenerate_case["meshes"][0]["triangles"][0][0]
            )
            cases.append((degenerate_case, ErrorCode.INVALID_MESH_DEGENERATE))
            light_case = copy.deepcopy(_reduced_scene_raw())
            light_case["lights"][0]["radiant_power"] = 0.0
            cases.append((light_case, ErrorCode.INVALID_LIGHT))
            spacing_case = copy.deepcopy(_reduced_scene_raw())
            spacing_case["probe_layout"]["spacing"] = 0.0
            cases.append((spacing_case, ErrorCode.INVALID_PROBE_SPACING))

            for index, (raw, expected_code) in enumerate(cases):
                with self.subTest(code=expected_code):
                    _write_json(invalid_scene_path, raw)
                    completed = _run_cli(invalid_scene_path, cache_path)
                    self.assertEqual(completed.returncode, 2, completed.stderr)
                    error = json.loads(completed.stderr)
                    self.assertEqual(error["status"], "error")
                    self.assertEqual(error["error"]["code"], expected_code)
                    self.assertEqual(cache_path.read_bytes(), original_cache)


if __name__ == "__main__":
    unittest.main()
