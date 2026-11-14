"""Deterministic and engineering-constraint tests for the software CSM pipeline."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from csm_pipeline.camera_path import load_path, save_path
from csm_pipeline.csm import build_cascades, cascade_splits
from csm_pipeline.depth_archive import (
    read_depth_archive,
    write_depth_archive,
)
from csm_pipeline.errors import (
    ERR_ARCHIVE_HEADER,
    ERR_ARCHIVE_TRUNCATED,
    ERR_CONFIG,
    ERR_PATH_MISSING,
    PipelineError,
)
from csm_pipeline.geometry import box, plane, sphere
from csm_pipeline.matrices import look_at, orthographic, perspective
from csm_pipeline.metrics import PSNR_IDENTICAL_DB, psnr
from csm_pipeline.pipeline import run_pipeline, validate_config
from csm_pipeline.rasterizer import rasterize
from csm_pipeline.scene import generate_scene, scene_fingerprint
from csm_pipeline.cli import main


SEED = 20261115


def base_config(path: str, output_dir: str) -> dict:
    return {
        "seed": SEED,
        "camera_path": path,
        "cascade_count": 3,
        "shadow_resolution": 64,
        "width": 128,
        "height": 96,
        "camera_near": 0.1,
        "camera_far": 40.0,
        "frame_time": 1.0,
        "pcf_kernel": 3,
        "blend_fraction": 0.15,
        "normal_bias_scale": 0.02,
        "depth_bias": 2.0e-4,
    }


class SceneDeterminismTest(unittest.TestCase):
    def test_bit_identical_scene(self):
        first = generate_scene(SEED)
        second = generate_scene(SEED)
        self.assertEqual(
            scene_fingerprint(first), scene_fingerprint(second)
        )
        self.assertTrue(
            np.array_equal(
                first.merged_mesh.positions.view(np.uint8),
                second.merged_mesh.positions.view(np.uint8),
            )
        )
        self.assertTrue(
            np.allclose(first.light_direction, second.light_direction)
        )

    def test_different_seed_changes_scene(self):
        self.assertNotEqual(
            scene_fingerprint(generate_scene(1)),
            scene_fingerprint(generate_scene(2)),
        )

    def test_mesh_winding_and_outward_normals(self):
        cube = box("c", np.zeros(3), np.ones(3))
        self.assertEqual(cube.triangle_count, 12)
        ground = plane("g", np.zeros(3), 4.0, 4.0, 2)
        self.assertEqual(ground.triangle_count, 8)
        ball = sphere("s", np.zeros(3), 1.0, 12, 8)
        self.assertGreater(ball.triangle_count, 0)


class MatrixAndCascadeTest(unittest.TestCase):
    def test_perspective_depth_maps_zero_to_one(self):
        proj = perspective(np.pi / 2, 1.0, 1.0, 10.0)
        near_clip = proj @ np.array((0, 0, -1, 1.0))
        far_clip = proj @ np.array((0, 0, -10, 1.0))
        near_ndc = near_clip[2] / near_clip[3]
        far_ndc = far_clip[2] / far_clip[3]
        self.assertAlmostEqual(near_ndc, 0.0, places=12)
        self.assertAlmostEqual(far_ndc, 1.0, places=12)

    def test_orthographic_depth_maps_zero_to_one(self):
        proj = orthographic(-1, 1, -1, 1, 2.0, 8.0)
        near_ndc = (proj @ np.array((0, 0, -2, 1.0)))[2]
        far_ndc = (proj @ np.array((0, 0, -8, 1.0)))[2]
        self.assertAlmostEqual(near_ndc, 0.0, places=12)
        self.assertAlmostEqual(far_ndc, 1.0, places=12)

    def test_cascade_splits_monotonic(self):
        splits = cascade_splits(0.1, 40.0, 4, lam=0.5)
        self.assertEqual(splits.shape, (5,))
        self.assertTrue(np.all(np.diff(splits) > 0))
        self.assertEqual(splits[0], 0.1)
        self.assertEqual(splits[-1], 40.0)

    def test_snapping_moves_by_whole_texels(self):
        scene = generate_scene(SEED)
        mesh = scene.merged_mesh
        bounds = mesh.positions.reshape(-1, 3)
        offsets = np.array(
            [
                (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1),
            ],
            float,
        )
        box_bounds = bounds.min(0) + offsets * (bounds.max(0) - bounds.min(0))
        cascades = build_cascades(
            eye=np.array((9.0, 5.5, 9.0)),
            target=np.array((0.0, 1.0, 0.0)),
            up=np.array((0.0, 1.0, 0.0)),
            fovy_rad=np.radians(55.0),
            aspect=4 / 3,
            camera_near=0.1,
            camera_far=40.0,
            light_direction=scene.light_direction,
            light_up=scene.light_up,
            cascade_count=3,
            shadow_resolution=128,
            scene_bounds=box_bounds,
        )
        for cascade in cascades:
            raw = cascade.view_proj
            snapped = cascade.snapped_view_proj
            # Rotation/scale rows stay identical; only translation changes.
            np.testing.assert_allclose(raw[:2, :3], snapped[:2, :3])
            left, right, bottom, top = cascade.raw_bounds
            s_left, s_right, s_bottom, s_top = cascade.snapped_bounds
            texel_x = (right - left) / 128
            texel_y = (top - bottom) / 128
            # Snapped frustum edges must lie on the integer-texel grid whose
            # origin is the light-view origin; raw edges generally do not.
            self.assertAlmostEqual(
                s_left / texel_x, round(s_left / texel_x), places=6
            )
            self.assertAlmostEqual(
                s_bottom / texel_y, round(s_bottom / texel_y), places=6
            )
            # Frustum width/height and depth must be preserved by snapping.
            self.assertAlmostEqual(s_right - s_left, right - left, places=9)
            self.assertAlmostEqual(s_top - s_bottom, top - bottom, places=9)


class RasterizerTest(unittest.TestCase):
    def test_depth_buffer_range_and_coverage(self):
        scene = generate_scene(SEED)
        mesh = scene.merged_mesh
        view_proj = perspective(np.radians(55.0), 4 / 3, 0.1, 40.0) @ look_at(
            np.array((9.0, 5.5, 9.0)),
            np.array((0.0, 1.0, 0.0)),
            np.array((0.0, 1.0, 0.0)),
        )
        result = rasterize(mesh, view_proj, 96, 72)
        self.assertTrue(np.isfinite(result["depth"]).all())
        hit = result["depth"][result["hit"]]
        self.assertTrue((hit >= 0.0).all() and (hit <= 1.0).all())
        self.assertGreater(result["hit"].mean(), 0.2)


class MetricsTest(unittest.TestCase):
    def test_psnr_identical_uses_defined_constant(self):
        image = np.random.default_rng(1).random((8, 8)).astype(np.float32)
        self.assertEqual(psnr(image, image), PSNR_IDENTICAL_DB)

    def test_psnr_decreases_with_error(self):
        image = np.zeros((16, 16), dtype=np.float32)
        noisy = image + 0.1
        self.assertLess(psnr(image, noisy), PSNR_IDENTICAL_DB)

    def test_pcf_respects_map_bounds(self):
        from csm_pipeline.shadows import _sample_shadow_map

        depth = np.full((4, 4), 0.5, dtype=np.float32)
        ndc = np.array([[[-0.99, -0.99], [0.99, 0.99]]], dtype=np.float64)
        ref = np.full((1, 2), 0.4, dtype=np.float32)
        valid = np.ones((1, 2), dtype=bool)
        factor = _sample_shadow_map(depth, ndc, ref, 1, valid)
        self.assertTrue(np.isfinite(factor).all())
        invalid = np.zeros((1, 2), dtype=bool)
        outside = _sample_shadow_map(depth, ndc, ref, 1, invalid)
        np.testing.assert_array_equal(outside, np.ones((1, 2), np.float32))


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "depth.bin")
        self.stack = (
            np.random.default_rng(7).random((3, 16, 12)).astype(np.float32)
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_bit_identical(self):
        write_depth_archive(self.path, self.stack)
        loaded = read_depth_archive(self.path)
        np.testing.assert_array_equal(loaded, self.stack)

    def test_truncated_payload_reports_fixed_code(self):
        write_depth_archive(self.path, self.stack)
        with open(self.path, "r+b") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.truncate(size - 7)
        with self.assertRaises(PipelineError) as ctx:
            read_depth_archive(self.path)
        self.assertEqual(ctx.exception.code, ERR_ARCHIVE_TRUNCATED)

    def test_bad_header_reports_fixed_code(self):
        write_depth_archive(self.path, self.stack)
        with open(self.path, "r+b") as handle:
            handle.write(b"XXXX")
        with self.assertRaises(PipelineError) as ctx:
            read_depth_archive(self.path)
        self.assertEqual(ctx.exception.code, ERR_ARCHIVE_HEADER)

    def test_missing_archive(self):
        with self.assertRaises(PipelineError) as ctx:
            read_depth_archive(os.path.join(self.tmp.name, "nope.bin"))
        self.assertEqual(ctx.exception.code, "E_ARCHIVE_MISSING")


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.camera_path = os.path.join(self.tmp.name, "path.json")
        scene = generate_scene(SEED)
        save_path(self.camera_path, scene.keyframes)
        self.output_a = os.path.join(self.tmp.name, "a")
        self.output_b = os.path.join(self.tmp.name, "b")
        self.config = base_config(self.camera_path, self.output_a)

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_run_reproducible_and_consistent(self):
        report_a = run_pipeline(dict(self.config), self.output_a)
        config_b = dict(self.config)
        config_b["seed"] = SEED
        report_b = run_pipeline(config_b, self.output_b)
        self.assertEqual(report_a["checksums"], report_b["checksums"])
        self.assertEqual(
            report_a["scene"]["fingerprint_sha256"],
            report_b["scene"]["fingerprint_sha256"],
        )
        self.assertTrue(
            report_a["selection_consistency"]["cascade_index_identical"]
        )
        self.assertTrue(report_a["selection_consistency"]["blend_identical"])
        for cascade in report_a["cascades"]:
            self.assertGreaterEqual(cascade["stats"]["shadow_coverage"], 0.0)
            self.assertGreater(cascade["texel_world_size"], 0.0)
        # Artifacts referenced by the report must exist.
        for name in (
            "shadow_factor_raw.npy",
            "shadow_factor_snapped.npy",
            "shadow_diff.npy",
            "shadow_diff.pgm",
            "shadow_depth_snapped.bin",
            "report.json",
        ):
            self.assertTrue(os.path.exists(os.path.join(self.output_a, name)))
        # Archive written by pipeline must reload and match report checksum.
        archive = read_depth_archive(
            os.path.join(self.output_a, "shadow_depth_snapped.bin")
        )
        from csm_pipeline.metrics import array_checksum

        for index, cascade in enumerate(report_a["cascades"]):
            self.assertEqual(
                cascade["snapped_depth_checksum"],
                array_checksum(archive[index]),
            )

    def test_cli_writes_json_error_for_missing_path(self):
        missing = os.path.join(self.tmp.name, "missing.json")
        out = os.path.join(self.tmp.name, "cli_err")
        argv = [
            "run", "--seed", str(SEED), "--camera-path", missing,
            "--cascades", "2", "--shadow-resolution", "32",
            "--width", "64", "--height", "48", "--output-dir", out,
        ]
        code = main(argv)
        self.assertNotEqual(code, 0)
        with open(os.path.join(out, "report.json"), encoding="utf-8") as h:
            doc = json.load(h)
        self.assertEqual(doc["status"], "error")
        self.assertEqual(doc["error_code"], ERR_PATH_MISSING)

    def test_invalid_config_does_not_crash(self):
        config = dict(self.config)
        config["cascade_count"] = 0
        with self.assertRaises(PipelineError) as ctx:
            validate_config(config)
        self.assertEqual(ctx.exception.code, ERR_CONFIG)

    def test_malformed_path_file(self):
        bad = os.path.join(self.tmp.name, "bad.json")
        with open(bad, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        with self.assertRaises(PipelineError) as ctx:
            load_path(bad)
        self.assertEqual(ctx.exception.code, "E_PATH_FORMAT")


class CascadeBlendTest(unittest.TestCase):
    def test_blend_fraction_zero_disables_fading(self):
        from csm_pipeline.shadows import _blend_weights
        from csm_pipeline.csm import cascade_splits

        class _Stub:
            def __init__(self, index, near, far):
                self.index = index
                self.near = near
                self.far = far

        splits = cascade_splits(1.0, 10.0, 3)
        cascades = [
            _Stub(i, float(splits[i]), float(splits[i + 1])) for i in range(3)
        ]
        depth = np.linspace(1.0, 10.0, 50, dtype=np.float32)
        index = np.full(depth.shape, 2, dtype=np.int32)
        for cascade in cascades:
            index[(index == 2) & (depth <= cascade.far)] = cascade.index
        zero_blend = _blend_weights(depth, index, cascades, 0.0)
        self.assertTrue(np.all(zero_blend == 0.0))
        active_blend = _blend_weights(depth, index, cascades, 0.2)
        # Pixels inside a fade band must partially transition.
        self.assertTrue(np.any((active_blend > 0.0) & (active_blend < 1.0)))
        # The last cascade never fades toward a non-existent next cascade.
        self.assertTrue(np.all(active_blend[index == 2][depth[index == 2] >= splits[-1]] == 0.0))

    def test_pcf_kernel_must_be_odd_positive(self):
        config = base_config("", "")
        config["pcf_kernel"] = 2
        with self.assertRaises(PipelineError):
            validate_config(config)

    def test_reversed_near_far_rejected(self):
        config = base_config("", "")
        config["camera_near"], config["camera_far"] = 50.0, 1.0
        with self.assertRaises(PipelineError) as ctx:
            validate_config(config)
        self.assertEqual(ctx.exception.code, ERR_CONFIG)

    def test_invalid_depth_map_is_rejected(self):
        from csm_pipeline.rasterizer import validate_depth_map
        from csm_pipeline.errors import ERR_INVALID_DEPTH

        bad = np.full((8, 8), np.nan, dtype=np.float32)
        hit = np.ones((8, 8), dtype=bool)
        with self.assertRaises(PipelineError) as ctx:
            validate_depth_map(bad, hit)
        self.assertEqual(ctx.exception.code, ERR_INVALID_DEPTH)

        out_of_range = np.full((8, 8), 1.5, dtype=np.float32)
        with self.assertRaises(PipelineError) as ctx:
            validate_depth_map(out_of_range, hit)
        self.assertEqual(ctx.exception.code, ERR_INVALID_DEPTH)


if __name__ == "__main__":
    unittest.main(verbosity=2)

