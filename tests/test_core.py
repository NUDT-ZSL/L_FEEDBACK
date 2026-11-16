from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from paint_review.environment import HdrEnvironment
from paint_review.export import save_png_deterministic
from paint_review.geometry import hood_mesh
from paint_review.grains import build_grain_texture
from paint_review.params import OrbitCamera, PaintParams, preset_params, render_hash
from paint_review.renderer import PaintRenderer, RenderCancelled, ggx_ndf


class ParameterTests(unittest.TestCase):
    def test_material_and_render_hashes_are_canonical(self) -> None:
        params = preset_params("金属漆")
        same = PaintParams(**params.canonical_dict())  # type: ignore[arg-type]
        self.assertEqual(params.material_hash(), same.material_hash())
        changed = PaintParams(**{**params.canonical_dict(), "roughness": 0.31})
        self.assertNotEqual(params.material_hash(), changed.material_hash())
        camera = OrbitCamera()
        self.assertEqual(
            render_hash(params, camera, 1920, 1080),
            render_hash(params, camera, 1920, 1080),
        )
        self.assertNotEqual(
            render_hash(params, camera, 1920, 1080),
            render_hash(changed, camera, 1920, 1080),
        )


class ProceduralResourceTests(unittest.TestCase):
    def test_hood_mesh_minimum_subdivision_and_upward_normals(self) -> None:
        mesh = hood_mesh(96, 96)
        self.assertEqual(mesh.subdivisions_u, 96)
        self.assertEqual(mesh.subdivisions_v, 96)
        self.assertEqual(mesh.triangles.shape[0], 96 * 96 * 2)
        self.assertGreater(float(np.mean(mesh.normals[:, 1])), 0.9)

    def test_environment_prefilter_is_multilevel_and_deterministic(self) -> None:
        first = HdrEnvironment.build(256, 128)
        second = HdrEnvironment.build(256, 128)
        self.assertEqual(first.level_count, 8)
        self.assertGreaterEqual(first.level_count, 4)
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.level_dimensions[0], (256, 128))
        for a, b in zip(first.levels, second.levels):
            self.assertTrue(np.array_equal(a, b))
        first.clear()
        self.assertEqual(first.level_count, 0)

    def test_grain_texture_is_large_and_deterministic(self) -> None:
        first = build_grain_texture(0.4, 211, "metallic")
        second = build_grain_texture(0.4, 211, "metallic")
        self.assertGreaterEqual(first.width, 512)
        self.assertGreaterEqual(first.height, 512)
        self.assertTrue(np.array_equal(first.normals, second.normals))
        self.assertTrue(np.array_equal(first.active, second.active))
        self.assertEqual(first.digest, second.digest)
        active = first.active
        lengths = np.linalg.norm(first.normals[active], axis=-1)
        np.testing.assert_allclose(lengths, 1.0, atol=2e-6)


class RenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.params = preset_params("珠光漆")
        cls.camera = OrbitCamera()

    def test_independent_renderers_are_pixel_identical(self) -> None:
        first_renderer = PaintRenderer(96, 96, 256, 128)
        second_renderer = PaintRenderer(96, 96, 256, 128)
        try:
            first, first_stats = first_renderer.render(
                self.params,
                self.camera,
                320,
                180,
                threading.Event(),
            )
            second, second_stats = second_renderer.render(
                self.params,
                self.camera,
                320,
                180,
                threading.Event(),
            )
        finally:
            first_renderer.close()
            second_renderer.close()
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first_stats["render_hash"], second_stats["render_hash"])
        self.assertEqual(first_stats["env_digest"], second_stats["env_digest"])
        self.assertEqual(first_stats["grain_digest"], second_stats["grain_digest"])

    def test_png_round_trip_is_pixel_identical(self) -> None:
        renderer = PaintRenderer(96, 96, 256, 128)
        try:
            image, _ = renderer.render(
                preset_params("金属漆"),
                self.camera,
                256,
                144,
                threading.Event(),
            )
        finally:
            renderer.close()
        with tempfile.TemporaryDirectory() as temporary:
            first_path = Path(temporary) / "first.png"
            second_path = Path(temporary) / "second.png"
            save_png_deterministic(image, first_path)
            save_png_deterministic(image, second_path)
            first_loaded = np.asarray(Image.open(first_path).convert("RGB"))
            second_loaded = np.asarray(Image.open(second_path).convert("RGB"))
        self.assertTrue(np.array_equal(first_loaded, second_loaded))
        self.assertTrue(np.array_equal(image, first_loaded))

    def test_render_cancels_mid_rasterization_and_releases(self) -> None:
        renderer = PaintRenderer(96, 96, 256, 128)
        cancel = threading.Event()
        observed = []

        def on_progress(value: int, _message: str) -> None:
            observed.append(value)
            if value >= 60:
                cancel.set()

        with self.assertRaises(RenderCancelled):
            renderer.render(
                preset_params("金属漆"),
                self.camera,
                640,
                360,
                cancel,
                on_progress,
            )
        self.assertGreaterEqual(len(observed), 2)
        renderer.close()
        self.assertTrue(renderer.closed)
        self.assertEqual(renderer.environment.level_count, 0)

    def test_clearcoat_ggx_is_finite(self) -> None:
        n_dot_h = np.linspace(0.0, 1.0, 64)
        values = ggx_ndf(n_dot_h, 0.055)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(values >= 0.0))
        self.assertGreater(float(values[-1]), float(values[0]))


if __name__ == "__main__":
    unittest.main()

