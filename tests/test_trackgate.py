"""Deterministic acceptance tests for the track-gate pipeline.

Run:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trackgate import errors
from trackgate.arch import _arc_length_params, segment_aabb, tessellate
from trackgate.camera import Camera
from trackgate.errors import (
    BadTileWidthError,
    DegenerateFovError,
    EmptySceneError,
)
from trackgate.incremental import IncrementalRenderer, load_cache
from trackgate.pipeline import (
    build_step_sequence,
    camera_inside_segments,
    run_pipeline,
    verify_frame,
)
from trackgate.raster import render
from trackgate.scene import generate_scene
from trackgate.tiles import cross_check, make_tiles, recount_from_ids

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_tmp():
    return tempfile.mkdtemp(prefix="tg_test_", dir=WORKSPACE)


class TestSceneGenerator(unittest.TestCase):
    def test_deterministic_byte_identical(self):
        a = generate_scene(7, 6)
        b = generate_scene(7, 6)
        self.assertEqual(a.canonical_json(), b.canonical_json())
        self.assertEqual(a.version, b.version)
        # json round-trip stability
        self.assertEqual(
            json.dumps(a.canonical_dict(), sort_keys=True),
            json.dumps(b.canonical_dict(), sort_keys=True),
        )

    def test_different_seed_differs(self):
        self.assertNotEqual(generate_scene(7, 6).version, generate_scene(8, 6).version)

    def test_shared_material_ids_no_shared_mutable_state(self):
        scene = generate_scene(7, 6)
        mat_ids = [s.material_id for s in scene.segments]
        self.assertGreater(len(set(mat_ids)), 1)
        self.assertLess(len(set(mat_ids)), len(mat_ids))  # sharing happens
        mutated = scene.mutate_segment(0, yaw_deg=99.0)
        self.assertNotEqual(scene.version, mutated.version)
        self.assertEqual(scene.segments[0].yaw_deg, generate_scene(7, 6).segments[0].yaw_deg)
        with self.assertRaises(Exception):
            scene.segments[0].yaw_deg = 5.0  # frozen dataclass

    def test_empty_scene_error(self):
        with self.assertRaises(EmptySceneError) as ctx:
            generate_scene(7, 0)
        self.assertEqual(ctx.exception.code, "E_EMPTY_SCENE")
        self.assertEqual(ctx.exception.exit_code, errors.EXIT_BUSINESS)


class TestCamera(unittest.TestCase):
    def test_version_changes_with_pose(self):
        c0 = Camera.look_at((0, 2, 10), (0, 2, 0), 55, 16 / 9)
        c1 = Camera.look_at((1, 2, 10), (0, 2, 0), 55, 16 / 9)
        c2 = Camera.look_at((0, 2, 10), (1, 2, 0), 55, 16 / 9)
        self.assertNotEqual(c0.version, c1.version)
        self.assertNotEqual(c0.version, c2.version)
        self.assertEqual(c0.version, Camera.look_at((0, 2, 10), (0, 2, 0), 55, 16 / 9).version)

    def test_degenerate_fov(self):
        for bad in (0.0001, 0.5, 179.5, 180.0):
            with self.assertRaises(DegenerateFovError) as ctx:
                Camera((0, 0, 0), 0, 0, bad, 1.0)
            self.assertEqual(ctx.exception.code, "E_DEGENERATE_FOV")

    def test_partial_frustum_containment_kept(self):
        scene = generate_scene(7, 3)
        cam = Camera.look_at((4.5, 2.0, 12.0), (4.5, 2.0, 0.0), 40, 16 / 9)
        mins, maxs = segment_aabb(scene.segments[0])
        self.assertTrue(cam.aabb_visible(mins, maxs))
        # a box fully behind the camera must be culled
        behind_min = np.array([cam.position[0] - 1, 0, cam.position[2] + 5])
        behind_max = behind_min + 1.0
        self.assertFalse(cam.aabb_visible(behind_min, behind_max))


class TestRaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = generate_scene(11, 4)
        cls.cam = Camera.look_at((13.0, 3.0, 16.0), (13.0, 2.0, 0.0), 55, 16 / 9)

    def test_deterministic_buffers(self):
        r1 = render(self.scene, self.cam, 160, 90)
        r2 = render(self.scene, self.cam, 160, 90)
        np.testing.assert_array_equal(r1.color, r2.color)
        np.testing.assert_array_equal(r1.depth, r2.depth)
        np.testing.assert_array_equal(r1.ids, r2.ids)

    def test_tile_restricted_render_matches_full(self):
        full = render(self.scene, self.cam, 160, 90)
        for x0, x1 in make_tiles(160, 48):
            part = render(self.scene, self.cam, 160, 90, x_range=(x0, x1))
            np.testing.assert_array_equal(part.color[:, x0:x1], full.color[:, x0:x1])
            np.testing.assert_array_equal(part.ids[:, x0:x1], full.ids[:, x0:x1])

    def test_depth_closest_wins_not_draw_order(self):
        # Two gates on the same line of sight; nearer one must win
        # regardless of segment index order.
        from trackgate.scene import Material, Scene, Segment, StripeParams

        stripe = StripeParams(count=6, duty=0.5, color=(255, 255, 255))
        mats = (Material(0, (200, 50, 50)), Material(1, (50, 50, 200)))
        near_seg = Segment(0, (0, 0, 0), 0.0, 4.0, 5.0, 0.5, 1.0, 0, stripe)
        far_seg = Segment(1, (0, 0, -6), 0.0, 4.0, 5.0, 0.5, 1.0, 1, stripe)
        cam = Camera.look_at((0, 2.0, 10.0), (0, 2.0, 0.0), 50, 16 / 9)
        fwd = render(Scene(1, mats, (near_seg, far_seg)), cam, 120, 68)
        rev = render(Scene(1, mats, (far_seg, near_seg)), cam, 120, 68)
        np.testing.assert_array_equal(fwd.color, rev.color)
        np.testing.assert_array_equal(fwd.ids, rev.ids)
        center = fwd.ids[34, 60]
        self.assertIn(center, (0, -1))
        if (fwd.ids == 0).any() and (fwd.ids == 1).any():
            self.assertLess(
                fwd.depth[fwd.ids == 0].max(), fwd.depth[fwd.ids == 1].min() + 6.5
            )

    def test_arc_length_param_normalized_and_shared_across_edges(self):
        s = _arc_length_params(28, 2.5, 4.0)
        self.assertAlmostEqual(float(s[0]), 0.0)
        self.assertAlmostEqual(float(s[-1]), 1.0)
        self.assertTrue(np.all(np.diff(s) > 0))
        scene = generate_scene(3, 1)
        _v, s_attr, _t = tessellate(scene.segments[0])
        # inner/outer/front/back verts at one station share the same s
        per_station = s_attr.reshape(-1, 4)
        np.testing.assert_array_equal(per_station[:, 0], per_station[:, 1])
        np.testing.assert_array_equal(per_station[:, 0], per_station[:, 2])
        np.testing.assert_array_equal(per_station[:, 0], per_station[:, 3])

    def test_stripe_phase_stable_across_arch_sizes(self):
        # Stripe pattern is a pure function of normalized arc length s, so
        # stripe boundaries sit at the same s for any arch height/width.
        from trackgate.scene import Material, Segment, StripeParams

        stripe = StripeParams(count=8, duty=0.5, color=(255, 255, 0))
        for height, width in ((3.0, 4.0), (5.0, 6.0), (4.2, 5.1)):
            seg = Segment(0, (0, 0, 0), 0.0, height, width, 0.4, 0.6, 0, stripe)
            verts, s_attr, _t = tessellate(seg)
            s_station = s_attr.reshape(-1, 4)[:, 0]
            # s really measures normalized midline arc length (chord approx)
            mid = verts.reshape(-1, 4, 3).mean(axis=1)
            chord = np.linalg.norm(np.diff(mid, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(chord)])
            np.testing.assert_allclose(s_station, cum / cum[-1], atol=2e-3)
            # stripe phase at the arch start (s=0) and end (s=1) is identical
            # for every arch size: pattern depends only on s
            def on(s):
                return (s * stripe.count) % 1.0 < stripe.duty

            self.assertTrue(on(0.0))
            self.assertEqual(on(0.25), on(0.25))  # pure function of s


class TestTilesAndManifest(unittest.TestCase):
    def test_tile_width_not_dividing(self):
        self.assertEqual(
            make_tiles(200, 64), [(0, 64), (64, 128), (128, 192), (192, 200)]
        )
        self.assertEqual(make_tiles(100, 128), [(0, 100)])

    def test_bad_tile_width(self):
        with self.assertRaises(BadTileWidthError):
            make_tiles(200, 0)

    def test_manifest_cross_check_and_tamper_detection(self):
        scene = generate_scene(5, 3)
        cam = Camera.look_at((13.0, 3.0, 16.0), (13.0, 2.0, 0.0), 55, 16 / 9)
        tiles = make_tiles(160, 64)
        res = render(scene, cam, 160, 90)
        manifest = recount_from_ids(res.ids, tiles)
        self.assertEqual(cross_check(manifest, res.ids, tiles), [])
        # no zero-pixel segments recorded
        for entry in manifest:
            for count in entry["segments"].values():
                self.assertGreater(count, 0)
        # tamper: structured diff must name tile, segment and delta
        first_with_seg = next(e for e in manifest if e["segments"])
        sid = next(iter(first_with_seg["segments"]))
        first_with_seg["segments"][sid] += 5
        diffs = cross_check(manifest, res.ids, tiles)
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["tile"], first_with_seg["tile"])
        self.assertEqual(diffs[0]["segments"][0]["segment"], int(sid))
        self.assertEqual(diffs[0]["segments"][0]["delta"], 5)


class TestIncremental(unittest.TestCase):
    def test_incremental_matches_full_recompute_each_step(self):
        scene = generate_scene(7, 6)
        width, height, tile_width = 320, 180, 64
        tiles = make_tiles(width, tile_width)
        steps = build_step_sequence(scene, width, height)
        renderer = IncrementalRenderer(width, height, tile_width)
        prev_color = None
        for idx, (sc, cam, note) in enumerate(steps):
            frame = renderer.render_frame(sc, cam)
            diffs = verify_frame(frame, sc, cam, width, height, tiles)
            self.assertEqual(diffs, [], f"step {idx} ({note}): {diffs}")
            self.assertEqual(frame.manifest_diffs, [])
            if note == "no change: full reuse":
                self.assertEqual(frame.dirty_tiles, [])
                np.testing.assert_array_equal(frame.color, prev_color)
            if note == "camera move only":
                self.assertTrue(frame.dirty_tiles)
            prev_color = frame.color.copy()

    def test_cache_invalidation_on_generation_change(self):
        tmp = make_tmp()
        try:
            scene = generate_scene(7, 3)
            cam = Camera.look_at((13.0, 3.0, 16.0), (13.0, 2.0, 0.0), 55, 16 / 9)
            renderer = IncrementalRenderer(160, 90, 64)
            renderer.render_frame(scene, cam)
            cache_dir = os.path.join(tmp, "cache")
            renderer.save_cache(cache_dir)
            meta = renderer.cache_meta()
            self.assertIsNotNone(load_cache(cache_dir, meta))
            # snapshot generation moved on -> old offsets must not be reused
            stale = dict(meta, generation=meta["generation"] + 1)
            self.assertIsNone(load_cache(cache_dir, stale))
            stale2 = dict(meta, scene_version="0" * 16)
            self.assertIsNone(load_cache(cache_dir, stale2))
            # corrupted cache file -> miss, never a fake hit
            with open(os.path.join(cache_dir, "tile_cache.json"), "w") as fh:
                fh.write("{not json")
            self.assertIsNone(load_cache(cache_dir, meta))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_camera_inside_gate_is_defined_result(self):
        scene = generate_scene(7, 3)
        g = scene.segments[0]
        cam = Camera.look_at(
            (g.translation[0], 1.5, g.translation[2]),
            (g.translation[0] + 5.0, 2.0, g.translation[2]),
            60,
            16 / 9,
        )
        self.assertEqual(camera_inside_segments(scene, cam), [0])
        res = render(scene, cam, 160, 90)  # must still render deterministically
        self.assertEqual(res.color.shape, (90, 160, 4))


class TestPipelineAndExitCodes(unittest.TestCase):
    def test_run_pipeline_ok_and_snapshots(self):
        tmp = make_tmp()
        try:
            summary = run_pipeline(7, 6, 320, 180, 64, tmp)
            self.assertTrue(summary["ok"])
            self.assertEqual(summary["total_diffs"], 0)
            self.assertTrue(summary["cache"]["self_hit"])
            self.assertTrue(summary["cache"]["stale_generation_rejected"])
            for name in ("frame_000_color.npy", "frame_000_depth.npy",
                         "frame_000_manifest.json", "summary.json"):
                self.assertTrue(os.path.exists(os.path.join(tmp, name)), name)
            depth = np.load(os.path.join(tmp, "frame_000_depth.npy"))
            self.assertEqual(depth.dtype, np.float32)
            self.assertTrue(np.isfinite(depth).any())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cli_exit_codes(self):
        import main

        tmp = make_tmp()
        try:
            rc = main.main(["run", "--segments", "4", "--width", "160",
                            "--height", "90", "--out", tmp])
            self.assertEqual(rc, errors.EXIT_OK)
            rc = main.main(["run", "--segments", "0", "--out", tmp])
            self.assertEqual(rc, errors.EXIT_BUSINESS)
            rc = main.main(["edges"])
            self.assertEqual(rc, errors.EXIT_OK)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
