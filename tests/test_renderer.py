from __future__ import annotations

import math
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from lrt.materials import calibration_report
from lrt.renderer import Renderer
from lrt.scene import create_default_scene, parse_scene


def make_scene(
    width=32,
    height=32,
    seed=158,
    spp=1,
    aa=1,
    shadow_samples=4,
    max_depth=5,
):
    description = create_default_scene(seed=seed, width=width, height=height)
    description["render"].update(
        {
            "samples_per_pixel": spp,
            "aa_samples": aa,
            "shadow_samples": shadow_samples,
            "max_depth": max_depth,
        }
    )
    return parse_scene(description)


class RendererTests(unittest.TestCase):
    def test_render_is_deterministic_and_counts_special_rays(self) -> None:
        scene = make_scene(width=20, height=18, aa=2, shadow_samples=3)
        first = Renderer(scene).render()
        second = Renderer(scene).render()
        self.assertTrue(np.array_equal(first.rgb, second.rgb))
        self.assertTrue(np.array_equal(first.linear_rgb, second.linear_rgb))
        self.assertEqual(first.stats, second.stats)
        self.assertGreater(first.stats["shadow_rays"], 0)
        self.assertGreater(first.stats["reflection_rays"], 0)
        self.assertGreater(first.stats["refraction_rays"], 0)
        self.assertGreaterEqual(first.stats["depth_clip_count"], 0)
        self.assertGreater(first.stats["total_rays"], first.stats["total_primary_rays"])
        self.assertEqual(first.rgb.shape, (18, 20, 3))
        self.assertEqual(first.rgb.dtype, np.uint8)
        self.assertLessEqual(first.stats["max_layer_reflectance"], 1.0)
        self.assertEqual(first.stats["nan_pixels"], 0)

    def test_png_and_stats_output(self) -> None:
        scene = make_scene(width=12, height=10)
        result = Renderer(scene).render()
        with tempfile.TemporaryDirectory() as temporary:
            png = os.path.join(temporary, "image.png")
            stats = os.path.join(temporary, "stats.json")
            result.save_png(png)
            result.save_stats(stats)
            with Image.open(png) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (12, 10))
                loaded = np.asarray(image)
            self.assertTrue(np.array_equal(loaded, result.rgb))
            self.assertTrue(os.path.getsize(stats) > 100)

    def test_calibration_is_in_range(self) -> None:
        scene = make_scene()
        report = calibration_report(scene.materials.values())
        self.assertTrue(report["all_in_range"])
        self.assertGreaterEqual(report["minimum_reflectance"], 0.0)
        self.assertLessEqual(report["maximum_reflectance"], 1.0)

    def test_nan_sample_isolated_to_magenta_pixel(self) -> None:
        class InjectNaN(Renderer):
            def _trace_sample(self, origin, direction, pixel_id, sub_id, path_id):
                if pixel_id == 0:
                    return (math.nan, math.nan, math.nan), 0.0, 0.0
                return super()._trace_sample(origin, direction, pixel_id, sub_id, path_id)

        result = InjectNaN(make_scene(width=8, height=8)).render()
        self.assertEqual(result.stats["nan_pixels"], 1)
        self.assertGreaterEqual(result.stats["nan_samples"], 1)
        self.assertEqual(tuple(result.rgb[0, 0]), (255, 0, 255))
        non_magenta = np.any(result.rgb.reshape(-1, 3) != np.array([255, 0, 255]), axis=1)
        self.assertTrue(np.any(non_magenta))

    def test_shadow_noise_decreases_with_samples(self) -> None:
        low = Renderer(make_scene(width=64, height=64, spp=2, shadow_samples=1)).render()
        high = Renderer(make_scene(width=64, height=64, spp=2, shadow_samples=16)).render()
        self.assertLess(high.stats["shadow_noise"], low.stats["shadow_noise"])
        self.assertGreater(
            high.stats["average_shadow_rays_per_pixel"],
            low.stats["average_shadow_rays_per_pixel"],
        )

    def test_aa4_reduces_edge_gradient(self) -> None:
        aa1 = Renderer(make_scene(width=48, height=48, spp=1, aa=1, shadow_samples=2)).render()
        aa4 = Renderer(make_scene(width=48, height=48, spp=1, aa=4, shadow_samples=2)).render()
        self.assertLess(aa4.stats["edge_gradient"], 0.8 * aa1.stats["edge_gradient"])

    def test_same_seed_reproduces_every_pixel(self) -> None:
        first = Renderer(make_scene(width=16, height=16, seed=991)).render()
        second = Renderer(make_scene(width=16, height=16, seed=991)).render()
        self.assertTrue(np.array_equal(first.rgb, second.rgb))
        self.assertEqual(first.stats, second.stats)


if __name__ == "__main__":
    unittest.main()

