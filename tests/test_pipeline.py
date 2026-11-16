from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from deferred_validation.gbuffer import GBUFFER_NAMES, layout_manifest
from deferred_validation.gl_context import EGLContext
from deferred_validation.renderer import DeferredRenderer
from deferred_validation.report import write_outputs
from deferred_validation.scene import load_scene, refresh_checksums
from tests.helpers import write_test_scene


class PipelineTests(unittest.TestCase):
    def test_multipass_render_report_and_ssaa_reset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene_path = write_test_scene(root / "scene.json", ssaa=1)
            scene = load_scene(scene_path)
            with EGLContext() as context:
                self.assertIsNotNone(context.info)
                self.assertTrue(context.info.version.startswith("4.5"))
                renderer = DeferredRenderer(context.info)
                try:
                    first = renderer.render(scene)
                    first_generation = renderer.resource_generation
                    report = write_outputs(scene, first, root / "out1")
                    self.assertEqual(report["validation"]["status"], "pass")
                    self.assertEqual(report["gbuffer_layout"], layout_manifest())
                    self.assertEqual(report["gbuffer_layout"]["draw_buffer_order"], list(GBUFFER_NAMES))
                    self.assertEqual(first.final_rgb.shape, (96, 128, 3))
                    self.assertEqual(first.internal_width, 128)
                    self.assertGreater(float(np.mean(first.coverage)), 0.05)
                    self.assertLess(
                        report["consistency"]["direct_plus_indirect_vs_composite"]["max_abs"],
                        1e-4,
                    )
                    self.assertEqual(
                        report["consistency"]["reconstructed_tone_map_vs_final"]["fraction_over_1_percent"],
                        0.0,
                    )
                    self.assertTrue((root / "out1" / "final.png").is_file())
                    self.assertTrue((root / "out1" / "g_normal.png").is_file())
                    self.assertTrue((root / "out1" / "g_metallic.png").is_file())
                    self.assertTrue((root / "out1" / "report.json").is_file())

                    scene.rendering.ssaa = 2
                    scene.rendering.tone_mapper = "reinhard"
                    refresh_checksums(scene)
                    second = renderer.render(scene)
                    self.assertGreater(renderer.resource_generation, first_generation)
                    self.assertEqual(second.internal_width, 256)
                    self.assertEqual(second.final_rgb.shape, (96, 128, 3))
                    self.assertEqual(second.gbuffer_target["albedo"].shape, first.gbuffer_target["albedo"].shape)
                    self.assertNotEqual(
                        second.gbuffer_internal["albedo"].shape,
                        first.gbuffer_internal["albedo"].shape,
                    )
                finally:
                    renderer.close()

    def test_report_contains_pixel_statistics_and_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene_path = write_test_scene(root / "scene.json")
            scene = load_scene(scene_path)
            with EGLContext() as context:
                renderer = DeferredRenderer(context.info)
                try:
                    result = renderer.render(scene)
                finally:
                    renderer.close()
            report = write_outputs(scene, result, root / "out")
            self.assertIn("direct_linear", report["statistics"])
            self.assertIn("final_srgb_u8", report["statistics"])
            for name in ("mean", "variance", "min", "max"):
                self.assertIn(name, report["statistics"]["direct_linear"]["channels"]["r"])
            self.assertEqual(set(report["scene"]["checksums"]), {"geometry", "materials", "lighting", "render_settings", "scene"})
            self.assertEqual(len(report["artifacts"]), 9)
            json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
