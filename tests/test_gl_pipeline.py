from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import glfw
import numpy as np
from OpenGL.GL import GL_NO_ERROR, glGetError, glGetString, GL_VERSION

from proc_gi.config import DiagnosticLog, RenderConfig
from proc_gi.renderer import Renderer
from proc_gi.scene_generator import generate_scene


class OpenGLPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not glfw.init():
            raise unittest.SkipTest("GLFW cannot initialize in this environment")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        cls.window = glfw.create_window(320, 240, "proc_gi-test", None, None)
        if not cls.window:
            glfw.terminate()
            raise unittest.SkipTest("OpenGL 3.3 core context unavailable")
        glfw.make_context_current(cls.window)

    @classmethod
    def tearDownClass(cls) -> None:
        glfw.destroy_window(cls.window)
        glfw.terminate()

    def test_render_incremental_shadow_and_postprocess(self) -> None:
        config, mesh = generate_scene()
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            config = replace(
                config,
                render=RenderConfig(
                    width=320,
                    height=240,
                    exposure=config.render.exposure,
                    screenshot_path=str(temporary_path / "color.png"),
                    debug_screenshot_path=str(temporary_path / "linear.png"),
                    warning_path=str(temporary_path / "warnings.json"),
                ),
            )
            diagnostics = DiagnosticLog()
            renderer = Renderer(config, mesh, diagnostics, width=320, height=240)
            try:
                initial = renderer.render()
                self.assertEqual(initial.cluster_count, 16)
                self.assertEqual(initial.cube_builds, 16)
                self.assertEqual(initial.material_draw_calls, 5)
                hdr = renderer.read_hdr()
                self.assertTrue(np.isfinite(hdr).all())
                self.assertGreater(float(hdr.max()), 0.0)

                light = config.lights[0]
                renderer.update_light(
                    light.id,
                    intensity=light.intensity + 9.0,
                    position=[light.position[0] + 0.1, light.position[1], light.position[2]],
                )
                incremental = renderer.render()
                incremental_hdr = renderer.read_hdr()
                self.assertFalse(incremental.full_rebuild)
                self.assertGreater(incremental.cube_reused, 0)
                self.assertLess(incremental.cube_builds, initial.cube_builds)
                self.assertGreater(incremental.delta_entries, 0)
                self.assertGreater(incremental.subtract_entries, 0)

                renderer.render(rebuild=True)
                full_hdr = renderer.read_hdr()
                self.assertLess(float(np.abs(incremental_hdr - full_hdr).max()), 0.08)

                color_path = renderer.save_screenshot(tonemap=True)
                linear_path = renderer.save_screenshot(tonemap=False)
                self.assertTrue(color_path.exists())
                self.assertTrue(linear_path.exists())
                self.assertGreater(color_path.stat().st_size, 1000)
                self.assertNotEqual(color_path.read_bytes(), linear_path.read_bytes())
                self.assertEqual(glGetError(), GL_NO_ERROR)
            finally:
                renderer.close()

    def test_context_is_core_33_or_newer(self) -> None:
        version = glGetString(GL_VERSION).decode("ascii")
        major = int(version.split(".", 1)[0])
        self.assertGreaterEqual(major, 3)


if __name__ == "__main__":
    unittest.main()

