"""Acceptance tests for the volumetric fog renderer (stdlib unittest)."""

import io
import json
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from volfog.cli import main
from volfog.config import RenderConfig
from volfog.density import generate_density_field, load_density_field
from volfog.errors import RenderError
from volfog.pipeline import render_to_files
from volfog.renderer import render_frame


def fast_config(**overrides) -> RenderConfig:
    config = RenderConfig(
        width=48, height=48, voxel_res=40,
        step_size=0.04, max_samples=80, light_samples=4,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class DensityFieldTests(unittest.TestCase):
    def test_deterministic_and_normalized(self):
        a = generate_density_field(40, seed=7)
        b = generate_density_field(40, seed=7)
        c = generate_density_field(40, seed=8)
        self.assertEqual(a.shape, (40, 40, 40))
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))
        self.assertGreaterEqual(a.min(), 0.0)
        self.assertLessEqual(a.max(), 1.0)

    def test_external_npy_normalized_and_validated(self):
        raw = np.linspace(2.0, 10.0, 27).reshape(3, 3, 3).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "density.npy")
            np.save(path, raw)
            loaded = load_density_field(path)
        self.assertAlmostEqual(loaded.min(), 0.0, places=7)
        self.assertAlmostEqual(loaded.max(), 1.0, places=7)

        bad = np.ones((4, 4, 4))
        bad[0, 0, 0] = np.nan
        with self.assertRaises(RenderError) as ctx:
            from volfog.config import validate_density_array
            validate_density_array(bad, "density_file")
        self.assertEqual(ctx.exception.param, "density_file")


class RenderConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.density = generate_density_field(40, seed=7)

    def test_serial_parallel_pixel_identical(self):
        serial = render_frame(fast_config(parallel=False), self.density)
        parallel = render_frame(fast_config(parallel=True, threads=2), self.density)
        names = ("color", "transmittance", "depth")
        for name, s, p in zip(names, serial, parallel):
            self.assertEqual(s.shape, p.shape)
            self.assertTrue(
                np.all(np.abs(s - p) <= 1e-6),
                f"{name} mismatch max={np.abs(s - p).max()}",
            )

    def test_modes_share_transmittance_and_depth(self):
        single = render_frame(fast_config(mode="single"), self.density)
        multiple = render_frame(fast_config(mode="multiple"), self.density)
        np.testing.assert_allclose(single[1], multiple[1], atol=1e-12)
        np.testing.assert_allclose(single[2], multiple[2], atol=1e-12)
        self.assertGreater(np.abs(multiple[0] - single[0]).max(), 1e-6)
        self.assertTrue(np.all(multiple[0] >= single[0] - 1e-12))

    def test_transmittance_and_depth_rules(self):
        config = fast_config(extinction=8.0)
        _, trans, depth = render_frame(config, self.density)
        self.assertTrue(np.all((trans >= 0.0) & (trans <= 1.0)))
        self.assertGreater(int(np.sum(trans < config.depth_threshold)), 0)
        hit = trans < config.depth_threshold
        if np.any(hit):
            self.assertTrue(np.all(depth[hit] < config.max_distance))
        self.assertTrue(np.all(depth[~hit] == config.max_distance))
        self.assertTrue(np.all(depth >= 0.0))


class PipelineAndErrorTests(unittest.TestCase):
    def test_output_files_and_formats(self):
        config = fast_config(output_dir=tempfile.mkdtemp(), output_prefix="t")
        meta = render_to_files(config)
        with Image.open(meta["files"]["color_png"]) as color, \
                Image.open(meta["files"]["transmittance_png"]) as trans, \
                Image.open(meta["files"]["depth_png"]) as depth:
            self.assertEqual(color.mode, "RGB")
            self.assertEqual(trans.mode, "L")
            self.assertIn(depth.mode, ("I;16", "I"))
            depth_array = np.asarray(depth)
            self.assertEqual(depth_array.dtype, np.uint16)
            self.assertAlmostEqual(depth_array.max() / 65535.0, 1.0, places=4)
        self.assertTrue(all(os.path.exists(p) for p in meta["files"].values() if p.endswith(".npy")))

    def test_cli_error_payloads(self):
        cases = [
            (["--focal-length", "-1"], "focal_length"),
            (["--focal-length", "nan"], "focal_length"),
            (["--step-size", "inf"], "step_size"),
            (["--width", "0"], "width"),
            (["--voxel-res", "-3"], "voxel_res"),
            (["--light-color", "1", "1", "nan"], "light_color"),
        ]
        for argv, param in cases:
            with self.subTest(argv=argv):
                err = io.StringIO()
                old_err = sys.stderr
                sys.stderr = err
                try:
                    code = main(argv)
                finally:
                    sys.stderr = old_err
                self.assertEqual(code, 1)
                lines = [line for line in err.getvalue().strip().splitlines() if line]
                self.assertEqual(len(lines), 1)
                payload = json.loads(lines[0])
                self.assertEqual(payload["error_code"], "E_INVALID_INPUT")
                self.assertEqual(payload["param"], param)

    def test_missing_density_file(self):
        config = fast_config(density_file="/nonexistent/field.npy")
        with self.assertRaises(RenderError) as ctx:
            render_to_files(config)
        self.assertEqual(ctx.exception.param, "density_file")


if __name__ == "__main__":
    unittest.main()
