import os
import tempfile
import unittest
from unittest import mock

from terrain_baker import preview as preview_mod
from terrain_baker.baker import bake
from terrain_baker.cli import main
from terrain_baker.errors import EXIT_PREVIEW_UNAVAILABLE
from terrain_baker.heightfield import HeightfieldSpec
from terrain_baker.verifier import verify_output


class PreviewFailureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, "pv")
        self.spec = HeightfieldSpec(seed=4, resolution=257, height_scale=8.0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_backend_failure_keeps_core_artifacts_and_reports_diagnostic(self):
        saved = list(preview_mod._DIAGNOSTICS)
        preview_mod._DIAGNOSTICS.clear()
        try:
            with mock.patch.object(preview_mod, "_load_first", return_value=None):
                result = bake(
                    self.spec, self.out, levels=[0, 1], tile=128,
                    preview_width=64, preview_height=64,
                )
        finally:
            preview_mod._DIAGNOSTICS[:] = saved

        manifest = result["manifest"]
        self.assertFalse(manifest["preview"]["available"])
        self.assertFalse(manifest["preview"]["rendered"])
        self.assertEqual(manifest["preview"]["exit_code"], EXIT_PREVIEW_UNAVAILABLE)
        self.assertTrue(manifest["preview"]["diagnostics"])
        # Core artifacts still present and verify cleanly.
        self.assertTrue(verify_output(self.out)["ok"])
        self.assertTrue(os.path.exists(os.path.join(self.out, "textures", "normal.png")))
        self.assertFalse(os.path.exists(os.path.join(self.out, "preview", "diffuse.png")))

    def test_cli_returns_preview_exit_code(self):
        with mock.patch.object(preview_mod, "_load_first", return_value=None):
            code = main([
                "--seed", "4", "--resolution", "257", "--height-scale", "8",
                "--levels", "0", "--output",
                os.path.join(self.tmp.name, "cli"), "--tile", "128",
            ])
        self.assertEqual(code, EXIT_PREVIEW_UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
