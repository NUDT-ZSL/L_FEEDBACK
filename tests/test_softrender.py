"""Unit + integration tests for the softrender package."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from softrender import assets, geometry, raster  # noqa: E402
from softrender.errors import AssetError, ParameterError  # noqa: E402
from softrender.obj_loader import load_obj  # noqa: E402
from softrender.scene import load_material  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestObjLoader(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="objtest_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write(self, name, text):
        path = os.path.join(self.dir, name)
        with open(path, "w") as fh:
            fh.write(text)
        return path

    def test_negative_indices(self):
        path = self._write("neg.obj", """
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
vn 0 0 1
f -3/-3/-1 -2/-2/-1 -1/-1/-1
""")
        obj = load_obj(path)
        self.assertEqual(obj["pos"].shape, (1, 3, 3))
        np.testing.assert_allclose(obj["pos"][0, 1], [1, 0, 0])
        np.testing.assert_allclose(obj["uv"][0, 2], [0, 1])

    def test_positive_and_polygon_fan(self):
        path = self._write("quad.obj", """
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
f 1/1/1 2/2/1 3/3/1 4/4/1
""")
        obj = load_obj(path)
        self.assertEqual(obj["pos"].shape, (2, 3, 3))  # quad -> 2 triangles

    def test_index_out_of_range(self):
        path = self._write("bad.obj", "v 0 0 0\nf 1 2 3\n")
        with self.assertRaises(AssetError):
            load_obj(path)

    def test_zero_index_rejected(self):
        path = self._write("zero.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n")
        with self.assertRaises(AssetError):
            load_obj(path)

    def test_missing_file(self):
        with self.assertRaises(AssetError):
            load_obj(os.path.join(self.dir, "nope.obj"))


class TestMaterial(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="mattest_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _mat(self, data):
        path = os.path.join(self.dir, "m.json")
        with open(path, "w") as fh:
            json.dump(data, fh)
        return path

    def test_valid(self):
        m = load_material(self._mat({"albedo": [0.5, 0.5, 0.5], "roughness": 0.4}))
        self.assertAlmostEqual(m.roughness, 0.4)
        self.assertIsNone(m.normal_map)

    def test_missing_field(self):
        with self.assertRaises(AssetError):
            load_material(self._mat({"albedo": [0.5, 0.5, 0.5]}))

    def test_bad_albedo(self):
        with self.assertRaises(AssetError):
            load_material(self._mat({"albedo": [0.5, 2.0, 0.5], "roughness": 0.4}))
        with self.assertRaises(AssetError):
            load_material(self._mat({"albedo": [0.5, 0.5], "roughness": 0.4}))

    def test_missing_normal_map_file(self):
        with self.assertRaises(AssetError):
            load_material(self._mat({"albedo": [0.5, 0.5, 0.5], "roughness": 0.4,
                                     "normal_map": "does_not_exist.png"}))


class TestParameters(unittest.TestCase):
    def test_threads_range(self):
        with self.assertRaises(ParameterError):
            raster.validate_threads(0)
        with self.assertRaises(ParameterError):
            raster.validate_threads((os.cpu_count() or 1) + 1)
        self.assertEqual(raster.validate_threads(1), 1)


class TestRasterRules(unittest.TestCase):
    def _quad_stris(self, seed=0, grid=4, size=64):
        """A grid of quads (2 triangles each) covering the whole frame."""
        rng = np.random.default_rng(seed)
        sx, sy = [], []
        step = size * 256 // grid
        for gy in range(grid):
            for gx in range(grid):
                x0, y0 = gx * step, gy * step
                x1, y1 = (gx + 1) * step, (gy + 1) * step
                # Random interior jitter on shared vertices keeps edges shared.
                sx.append([x0, x1, x1]); sy.append([y0, y0, y1])
                sx.append([x0, x1, x0]); sy.append([y0, y1, y1])
        n = len(sx)
        return {
            "sx": np.array(sx, dtype=np.int64),
            "sy": np.array(sy, dtype=np.int64),
            "z": np.zeros((n, 3)),
            "w": np.ones((n, 3)),
            "world": np.zeros((n, 3, 3)),
            "nrm": np.zeros((n, 3, 3)),
            "uv": np.zeros((n, 3, 2)),
            "mat": np.zeros(n, dtype=np.int32),
        }

    def test_top_left_rule_no_cracks_no_overlap(self):
        stris = self._quad_stris()
        counts = raster.coverage_count(stris, 64, 64)
        self.assertEqual(int(counts.min()), 1, "crack: uncovered pixel inside quad grid")
        self.assertEqual(int(counts.max()), 1, "double-covered pixel on shared edge")

    def test_threads_identical_results(self):
        stris = self._quad_stris()
        _, d1 = raster.rasterize(stris, 64, 64, nthreads=1)
        _, d2 = raster.rasterize(stris, 64, 64, nthreads=4)
        np.testing.assert_array_equal(d1, d2)


class TestClipping(unittest.TestCase):
    def test_near_clip_produces_positive_w(self):
        # Triangle crossing the near plane of the main camera.
        scene_dir = tempfile.mkdtemp(prefix="cliptest_")
        try:
            scene_path = assets.generate_assets(scene_dir, 7)
            from softrender.scene import load_scene
            from softrender.pipeline import _load_world
            scene = load_scene(scene_path)
            world, nrm, uv, mat, _ = _load_world(scene)
            bmin = world.reshape(-1, 3).min(0)
            bmax = world.reshape(-1, 3).max(0)
            rig = geometry.main_camera(bmin, bmax, 64, 64)
            stris = geometry.assemble_triangles(world, nrm, uv, mat, rig, 64, 64)
            self.assertGreater(stris["sx"].shape[0], 0)
            self.assertTrue(np.all(stris["w"] > 0), "w <= 0 vertex after clipping")
            self.assertTrue(np.all(np.isfinite(stris["z"])))
        finally:
            shutil.rmtree(scene_dir, ignore_errors=True)


class TestEndToEnd(unittest.TestCase):
    def test_determinism_and_thread_invariance(self):
        workdir = tempfile.mkdtemp(prefix="e2e_")
        try:
            scene_path = assets.generate_assets(os.path.join(workdir, "assets"), 123)
            from softrender import pipeline, tonemap
            hashes = []
            for threads in (1, 2, 1):
                hdr, _ = pipeline.render_scene(scene_path, 64, 64, threads, 64)
                rgb8 = tonemap.hdr_to_rgb8(hdr)
                hashes.append(rgb8.tobytes())
            self.assertEqual(hashes[0], hashes[1], "multithreaded != single-threaded")
            self.assertEqual(hashes[0], hashes[2], "run-to-run mismatch")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def test_cli_error_paths(self):
        workdir = tempfile.mkdtemp(prefix="clierr_")
        try:
            out = os.path.join(workdir, "out.png")
            env = dict(os.environ)
            # Invalid shadow resolution (not a power of two).
            r = subprocess.run(
                [sys.executable, "-m", "softrender", "run", "--seed", "1",
                 "--output", out, "--assets-dir", os.path.join(workdir, "a"),
                 "--width", "64", "--height", "64", "--shadow-res", "100"],
                capture_output=True, text=True, cwd=ROOT, env=env)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("error:", r.stderr)
            self.assertFalse(os.path.exists(out), "partial image must not be written")
            # Invalid thread count.
            r = subprocess.run(
                [sys.executable, "-m", "softrender", "run", "--seed", "1",
                 "--output", out, "--assets-dir", os.path.join(workdir, "a"),
                 "--width", "64", "--height", "64", "--shadow-res", "64",
                 "--threads", "0"],
                capture_output=True, text=True, cwd=ROOT, env=env)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("error:", r.stderr)
            self.assertFalse(os.path.exists(out))
            # Missing material file referenced by the scene.
            scene_dir = os.path.join(workdir, "assets")
            scene_path = assets.generate_assets(scene_dir, 5)
            os.unlink(os.path.join(scene_dir, "mech_a.json"))
            r = subprocess.run(
                [sys.executable, "-m", "softrender", "render",
                 "--scene", scene_path, "--output", out,
                 "--width", "64", "--height", "64", "--shadow-res", "64"],
                capture_output=True, text=True, cwd=ROOT, env=env)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("error:", r.stderr)
            self.assertFalse(os.path.exists(out))
            # Invalid material JSON (missing roughness).
            assets.generate_assets(scene_dir, 5)
            with open(os.path.join(scene_dir, "mech_a.json"), "w") as fh:
                json.dump({"albedo": [0.5, 0.5, 0.5]}, fh)
            r = subprocess.run(
                [sys.executable, "-m", "softrender", "render",
                 "--scene", scene_path, "--output", out,
                 "--width", "64", "--height", "64", "--shadow-res", "64"],
                capture_output=True, text=True, cwd=ROOT, env=env)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("error:", r.stderr)
            self.assertFalse(os.path.exists(out))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
