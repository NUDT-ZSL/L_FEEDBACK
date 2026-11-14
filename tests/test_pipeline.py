from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from bake_tool.pipeline import BakeConfig, run_bake


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_NAME = "runtime_profile.json"


class PipelineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls._temporary.name)
        cls.gl_dir = cls.base / "gl"
        cls.dx_dir = cls.base / "dx"
        cls.repeat_dir = cls.base / "repeat"
        cls.gl = run_bake(
            BakeConfig(resolution=48, normal_convention="opengl", dilation_radius=2),
            cls.gl_dir,
        )
        cls.dx = run_bake(
            BakeConfig(resolution=48, normal_convention="directx", dilation_radius=2),
            cls.dx_dir,
        )
        cls.repeat = run_bake(
            BakeConfig(resolution=48, normal_convention="opengl", dilation_radius=2),
            cls.repeat_dir,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_required_outputs_and_statistics(self) -> None:
        required = {
            "normal.png",
            "albedo.png",
            "roughness.png",
            "metallic.png",
            "tangent.png",
            "bitangent.png",
            "coverage_mask.png",
            "coverage_mask_dilated.png",
            "triangle_index.png",
            "dilation_difference.png",
            "normal_consistency_mask.png",
            "normal_consistency_alternate_mask.png",
            "preview.png",
            "bake_report.json",
        }
        self.assertTrue(required.issubset(set(self.gl.texture_bytes)))
        report = self.gl.report
        for channel in ("normal", "albedo", "roughness", "metallic"):
            self.assertIn(channel, report["channels"])
            self.assertIn("mean", report["channels"][channel])
            self.assertIn("variance", report["channels"][channel])
        self.assertGreater(report["rasterization"]["coverage_pixels"], 0)
        self.assertEqual(report["rasterization"]["culled_backface_triangles"], 0)
        self.assertEqual(report["raycast"]["fallback_pixels"], 0)
        self.assertLessEqual(report["raycast"]["fallback_ratio"], 0.02)
        self.assertEqual(report["dilation"]["valid_pixels_overwritten"], 0)
        self.assertGreater(report["dilation"]["dilated_pixels"], 0)
        self.assertEqual(report["normal_consistency"]["mismatch_pixels"], 0)
        self.assertIn("total_elapsed_ms", report["timing"])
        self.assertEqual(report["parameters"]["resolution"], 48)

        with Image.open(self.gl_dir / "preview.png") as preview:
            self.assertEqual(preview.size, (48, 48))
            self.assertEqual(preview.mode, "RGB")
        with Image.open(self.gl_dir / "triangle_index.png") as triangle_map:
            values = np.asarray(triangle_map)
            coverage = np.asarray(Image.open(self.gl_dir / "coverage_mask.png")) > 0
            self.assertEqual(int(np.count_nonzero(values)), int(np.count_nonzero(coverage)))
            self.assertEqual(int(np.count_nonzero(values[coverage] == 0)), 0)

    def test_directx_green_channel_flip_and_selected_consistency(self) -> None:
        gl_normal = np.asarray(Image.open(self.gl_dir / "normal.png"))
        dx_normal = np.asarray(Image.open(self.dx_dir / "normal.png"))
        coverage = np.asarray(Image.open(self.gl_dir / "coverage_mask.png")) > 0
        self.assertTrue(np.array_equal(gl_normal[..., 0][coverage], dx_normal[..., 0][coverage]))
        self.assertTrue(np.array_equal(gl_normal[..., 2][coverage], dx_normal[..., 2][coverage]))
        self.assertTrue(
            np.all(
                np.abs(
                    gl_normal[..., 1][coverage].astype(int)
                    + dx_normal[..., 1][coverage].astype(int)
                    - 255
                )
                <= 1
            )
        )
        self.assertEqual(self.gl.report["normal_consistency"]["mismatch_pixels"], 0)
        self.assertEqual(self.dx.report["normal_consistency"]["mismatch_pixels"], 0)
        self.assertGreater(
            self.gl.report["normal_consistency"]["alternate_mismatch_pixels"], 0
        )
        self.assertGreater(
            self.dx.report["normal_consistency"]["alternate_mismatch_pixels"], 0
        )
        self.assertEqual(
            self.gl.report["normal_consistency"]["selected_convention"], "opengl"
        )
        self.assertEqual(
            self.dx.report["normal_consistency"]["selected_convention"], "directx"
        )

    def test_byte_determinism(self) -> None:
        left = {
            name: payload
            for name, payload in self.gl.texture_bytes.items()
            if name != RUNTIME_NAME
        }
        right = {
            name: payload
            for name, payload in self.repeat.texture_bytes.items()
            if name != RUNTIME_NAME
        }
        self.assertEqual(sorted(left), sorted(right))
        for name in sorted(left):
            self.assertEqual(left[name], right[name], msg=name)
        self.assertEqual(
            json.loads(left["bake_report.json"])["outputs"]["texture_sha256"],
            json.loads(right["bake_report.json"])["outputs"]["texture_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(left["normal.png"]).hexdigest(),
            self.gl.report["outputs"]["texture_sha256"]["normal.png"],
        )
        runtime = json.loads((self.gl_dir / RUNTIME_NAME).read_text(encoding="utf-8"))
        self.assertGreaterEqual(runtime["actual_elapsed_ms"], 0.0)


class ErrorPathTests(unittest.TestCase):
    def _run_error(self, expected_code: str, extra_args: list[str]) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bake_tool",
                    "--out",
                    str(output),
                    "--resolution",
                    "32",
                    *extra_args,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2, msg=completed.stderr)
            report = json.loads((output / "error_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["business_code"], expected_code)
            self.assertIsInstance(report["business_code"], str)
            self.assertIsInstance(report["exit_code"], int)
            self.assertTrue(report["context"])
            self.assertTrue(report["recovery_suggestions"])
            self.assertFalse(list(output.glob("*.png")))
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                ["error_report.json"],
            )

    def test_uv_out_of_range(self) -> None:
        self._run_error("E_UV_OUT_OF_RANGE", ["--scenario", "uv_out_of_range"])

    def test_missing_high_mesh(self) -> None:
        self._run_error("E_HIGH_MESH_MISSING", ["--scenario", "missing_high"])

    def test_resolution_limit(self) -> None:
        self._run_error(
            "E_RESOLUTION_LIMIT",
            ["--resolution", "33", "--max-resolution", "32"],
        )

    def test_miss_rate_limit(self) -> None:
        self._run_error(
            "E_MISS_RATE_EXCEEDED",
            ["--scenario", "miss_rate", "--miss-threshold", "0.01"],
        )


if __name__ == "__main__":
    unittest.main()
