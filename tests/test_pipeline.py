from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from terrain_bake import BakeConfig, bake
from terrain_bake.generator import generate_material_weights, generate_samples
from terrain_bake.mesh import (
    build_base_mesh,
    lod_error_thresholds,
    lod_target_ratios,
    simplify_mesh,
)
from terrain_bake.normals import bake_normal_map
from terrain_bake.render import render_preview
from terrain_bake.seam import build_transition_mesh


TOP_LEVEL_SCHEMA = {
    "schema_version",
    "status",
    "error",
    "error_message",
    "error_details",
    "seed",
    "request",
    "source",
    "backend",
    "timings_ms",
    "lods",
    "normal_map",
    "material_preview",
    "preview",
    "seam",
    "checksums",
    "artifacts",
    "warnings",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class GeneratorTests(unittest.TestCase):
    def test_fixed_seed_is_bitwise_deterministic(self) -> None:
        first = generate_samples(17, 19, 424242, 1e-9)
        second = generate_samples(17, 19, 424242, 1e-9)
        self.assertTrue(np.array_equal(first.heightfield, second.heightfield))
        self.assertTrue(np.array_equal(first.weights, second.weights))
        self.assertEqual(first.heightfield_sha256, second.heightfield_sha256)
        self.assertEqual(first.weights_sha256, second.weights_sha256)
        self.assertLessEqual(
            float(np.max(np.abs(np.sum(first.weights, axis=2) - 1.0))),
            1e-9,
        )

    def test_flat_heightfield_produces_valid_material_weights(self) -> None:
        flat = np.full((9, 9), 0.4, dtype=np.float64)
        weights = generate_material_weights(flat, 1)
        self.assertEqual(weights.shape, (9, 9, 3))
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertLessEqual(
            float(np.max(np.abs(np.sum(weights, axis=2) - 1.0))),
            1e-9,
        )


class NormalMapTests(unittest.TestCase):
    def test_flat_normal_encodes_tangent_space_up(self) -> None:
        flat = np.full((8, 8), 0.5, dtype=np.float64)
        result = bake_normal_map(flat, 32)
        np.testing.assert_array_equal(
            result.rgb[1:-1, 1:-1],
            np.broadcast_to(np.array([128, 128, 255], dtype=np.uint8), (30, 30, 3)),
        )
        self.assertAlmostEqual(result.pixel_step, 1.0 / 31.0)

    def test_gradient_axes_and_resolution_scaled_step(self) -> None:
        x_ramp = np.tile(np.linspace(0.2, 0.8, 16), (16, 1))
        z_ramp = np.tile(np.linspace(0.2, 0.8, 16)[:, None], (1, 16))
        x_normal = bake_normal_map(x_ramp, 16)
        z_normal = bake_normal_map(z_ramp, 16)
        center = (8, 8)
        self.assertLess(int(x_normal.rgb[center][0]), 120)
        self.assertGreater(int(z_normal.rgb[center][1]), 136)
        self.assertAlmostEqual(x_normal.pixel_step, 1.0 / 15.0)
        self.assertAlmostEqual(
            bake_normal_map(x_ramp, 32).pixel_step,
            1.0 / 31.0,
        )

    def test_auto_preview_records_cpu_fallback(self) -> None:
        samples = generate_samples(9, 9, 5, 1e-9)
        mesh = build_base_mesh(samples.heightfield, samples.weights)
        normal_map = bake_normal_map(samples.heightfield, 16)
        with mock.patch(
            "terrain_bake.render.render_preview_opengl",
            side_effect=RuntimeError("forced missing context"),
        ):
            result = render_preview(
                mesh,
                normal_map.rgb,
                64,
                48,
                backend="auto",
            )
        self.assertEqual(result.backend, "cpu")
        self.assertEqual(result.attempted_backends, ["opengl", "cpu"])
        self.assertIn("forced missing context", result.fallback_reason or "")
        self.assertEqual(result.image.shape, (48, 64, 3))


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = generate_samples(33, 33, 20261116, 1e-9)
        cls.base = build_base_mesh(cls.samples.heightfield, cls.samples.weights)
        cls.meshes = [cls.base]
        cls.errors = lod_error_thresholds(3, 0.03)
        cls.targets = lod_target_ratios(3)
        for level, (ratio, threshold) in enumerate(
            zip(cls.targets, cls.errors), start=1
        ):
            result = simplify_mesh(
                cls.meshes[-1],
                max(8, round(cls.base.vertex_count * ratio)),
                threshold,
                f"LOD {level}",
            )
            cls.meshes.append(result.mesh)

    def test_lods_are_progressively_smaller_and_error_bounded(self) -> None:
        for previous, current, threshold in zip(
            self.meshes, self.meshes[1:], self.errors
        ):
            self.assertLess(current.vertex_count, previous.vertex_count)
            self.assertLess(current.face_count, previous.face_count)
            self.assertGreater(threshold, 0.0)
        self.assertLessEqual(self.errors[0], self.errors[-1])

    def test_transition_reuses_high_vertices_and_has_no_t_junctions(self) -> None:
        for level in range(1, len(self.meshes)):
            seam = build_transition_mesh(
                self.meshes[level - 1],
                self.meshes[level],
                level - 1,
                level,
            )
            self.assertTrue(seam.high_vertices_reused)
            self.assertEqual(seam.t_junction_count, 0)
            self.assertTrue(seam.combined_topology["manifold"])
            self.assertEqual(seam.combined_topology["boundary_loops"], 1)
            self.assertGreater(seam.stitch_triangles, 0)


class PipelineTests(unittest.TestCase):
    def small_config(self, output: Path, **overrides: object) -> BakeConfig:
        values = {
            "size": 33,
            "seed": 777,
            "lod_count": 3,
            "normal_resolution": 64,
            "preview_width": 160,
            "preview_height": 120,
            "output_dir": output,
            "preview_backend": "cpu",
        }
        values.update(overrides)
        return BakeConfig(**values)

    def test_success_manifest_and_artifact_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "terrain"
            result = bake(self.small_config(output))
            self.assertEqual(result.exit_code, 0)
            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(set(manifest), TOP_LEVEL_SCHEMA)
            self.assertEqual(manifest["status"], "ok")
            self.assertIsNone(manifest["error"])
            self.assertEqual(manifest["seed"], 777)
            self.assertEqual(len(manifest["lods"]), 3)
            self.assertTrue(manifest["seam"]["all_manifold"])
            self.assertEqual(manifest["seam"]["t_junction_count"], 0)
            self.assertTrue(manifest["normal_map"]["path"].endswith(".png"))
            self.assertTrue(manifest["preview"]["path"].endswith(".png"))
            self.assertTrue(manifest["material_preview"]["path"].endswith(".png"))

            previous_vertices = None
            for lod in manifest["lods"]:
                path = output / lod["path"]
                self.assertTrue(path.exists())
                self.assertEqual(lod["sha256"], sha256_file(path))
                self.assertTrue(lod["topology"]["manifold"])
                self.assertLessEqual(
                    lod["simplification_error"],
                    lod["error_threshold"] + 1e-12,
                )
                if previous_vertices is not None:
                    self.assertLess(lod["vertex_count"], previous_vertices)
                previous_vertices = lod["vertex_count"]

            for relative, checksum in manifest["checksums"].items():
                path = output / relative
                self.assertTrue(path.is_file(), relative)
                self.assertEqual(checksum, sha256_file(path), relative)
            self.assertEqual(
                set(manifest["checksums"]),
                set(manifest["artifacts"]),
            )

    def test_invalid_inputs_leave_only_failure_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "terrain"
            result = bake(self.small_config(output, size=1))
            self.assertEqual(result.exit_code, 3)
            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(set(manifest), TOP_LEVEL_SCHEMA)
            self.assertEqual(manifest["error"], "E_INVALID_HEIGHTFIELD")
            self.assertEqual(manifest["artifacts"], [])
            self.assertEqual(sorted(path.name for path in output.iterdir()), [
                "bake_manifest.json"
            ])

            nan_height = np.zeros((9, 9), dtype=np.float64)
            nan_height[2, 3] = np.nan
            result = bake(
                self.small_config(output, size=9),
                heightfield=nan_height,
            )
            self.assertEqual(result.exit_code, 3)
            self.assertEqual(result.manifest["error"], "E_INVALID_HEIGHTFIELD")
            self.assertEqual(
                [path.name for path in output.iterdir()],
                ["bake_manifest.json"],
            )

            bad_weights = np.full((9, 9, 3), 0.5, dtype=np.float64)
            result = bake(
                self.small_config(output, size=9),
                heightfield=np.zeros((9, 9), dtype=np.float64),
                weights=bad_weights,
            )
            self.assertEqual(result.exit_code, 4)
            self.assertEqual(result.manifest["error"], "E_WEIGHT_MISMATCH")
            self.assertEqual(
                [path.name for path in output.iterdir()],
                ["bake_manifest.json"],
            )

    def test_geometry_error_code_and_no_partial_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "terrain"
            result = bake(
                self.small_config(
                    output,
                    size=17,
                    max_relative_error=1e-12,
                )
            )
            self.assertEqual(result.exit_code, 5)
            self.assertEqual(result.manifest["error"], "E_GEOMETRY_INVALID")
            self.assertEqual(result.manifest["artifacts"], [])
            self.assertEqual(
                [path.name for path in output.iterdir()],
                ["bake_manifest.json"],
            )

    def test_cli_uses_business_exit_code_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "cli-terrain"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "terrain_bake",
                    "--size",
                    "1",
                    "--output",
                    str(output),
                    "--preview-backend",
                    "cpu",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 3)
            self.assertIn("E_INVALID_HEIGHTFIELD", completed.stderr)
            self.assertNotIn("Traceback", completed.stderr)
            manifest = json.loads((output / "bake_manifest.json").read_text())
            self.assertEqual(manifest["error"], "E_INVALID_HEIGHTFIELD")

    def test_cli_external_height_and_weight_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            height = np.linspace(0.0, 1.0, 9 * 9).reshape(9, 9)
            bad_height = height.copy()
            bad_height[4, 4] = np.inf
            height_path = root / "height.npy"
            bad_height_path = root / "bad_height.npy"
            weights_path = root / "bad_weights.npy"
            np.save(height_path, height)
            np.save(bad_height_path, bad_height)
            np.save(weights_path, np.full((9, 9, 3), 0.4))
            command = [
                sys.executable,
                "-m",
                "terrain_bake",
                "--size",
                "9",
                "--output",
                str(root / "output"),
                "--normal-resolution",
                "32",
                "--preview-width",
                "64",
                "--preview-height",
                "64",
                "--preview-backend",
                "cpu",
            ]
            completed = subprocess.run(
                command + ["--heightfield", str(bad_height_path)],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 3)
            self.assertIn("E_INVALID_HEIGHTFIELD", completed.stderr)
            output = root / "output"
            self.assertEqual(
                [path.name for path in output.iterdir()],
                ["bake_manifest.json"],
            )

            completed = subprocess.run(
                command
                + ["--heightfield", str(height_path), "--weights", str(weights_path)],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 4)
            self.assertIn("E_WEIGHT_MISMATCH", completed.stderr)
            self.assertEqual(
                [path.name for path in output.iterdir()],
                ["bake_manifest.json"],
            )


if __name__ == "__main__":
    unittest.main()
