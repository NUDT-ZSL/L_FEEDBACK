from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from soft_renderer.imageio import read_pfm
from soft_renderer.sample_scene import write_indoor_scene

from tests.helpers import run_cli, temporary_files


class CliTests(unittest.TestCase):
    def test_render_layers_composite_png_report_and_no_temp_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            scene_path = directory / "scene.json"
            write_indoor_scene(str(scene_path), 32, 24)
            whole_path = directory / "whole.pfm"
            layers_dir = directory / "layers"
            png_path = directory / "preview.png"
            report_path = directory / "report.json"
            completed = run_cli(
                [
                    "render",
                    str(scene_path),
                    str(whole_path),
                    "--preset",
                    "low",
                    "--block-size",
                    "4",
                    "--layers-dir",
                    str(layers_dir),
                    "--png",
                    str(png_path),
                    "--report",
                    str(report_path),
                ]
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(whole_path.is_file())
            self.assertTrue(png_path.is_file())
            self.assertTrue((layers_dir / "layers.json").is_file())
            self.assertTrue((layers_dir / "material_0.pfm").is_file())
            composite_path = directory / "composite.pfm"
            composite_report = directory / "composite.json"
            completed = run_cli(
                [
                    "composite",
                    "--layers-dir",
                    str(layers_dir),
                    "-o",
                    str(composite_path),
                    "--report",
                    str(composite_report),
                ]
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                hashlib.sha256(whole_path.read_bytes()).digest(),
                hashlib.sha256(composite_path.read_bytes()).digest(),
            )
            whole_report = json.loads(report_path.read_text(encoding="utf-8"))
            composite_data = json.loads(composite_report.read_text(encoding="utf-8"))
            self.assertEqual(
                whole_report["pixel_result_sha256"],
                composite_data["pixel_result_sha256"],
            )
            self.assertEqual(whole_report["parameters"]["block_size"], 4)
            for key in ("shadow_maps", "raster_shade", "layer_composite", "total"):
                self.assertIn(key, whole_report["timings_ms"])
            self.assertEqual(temporary_files(directory), [])

    def test_report_hashes_are_stable_and_presets_change_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            scene_path = directory / "scene.json"
            write_indoor_scene(str(scene_path), 24, 18)
            reports = []
            for index, preset in enumerate(("low", "low", "high")):
                output = directory / ("out_%d.pfm" % index)
                report = directory / ("report_%d.json" % index)
                arguments = ["render", str(scene_path), str(output)]
                if index == 1:
                    arguments = [
                        "render",
                        "--scene",
                        str(scene_path),
                        "-o",
                        str(output),
                        "--no-tiling",
                    ]
                arguments.extend(
                    [
                        "--preset",
                        preset,
                        "--block-size",
                        "4",
                        "--report",
                        str(report),
                    ]
                )
                completed = run_cli(arguments)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                reports.append(json.loads(report.read_text(encoding="utf-8")))
            self.assertEqual(
                reports[0]["pixel_result_sha256"], reports[1]["pixel_result_sha256"]
            )
            self.assertEqual(reports[0]["parameters"]["pcf_samples"], 1)
            self.assertEqual(reports[2]["parameters"]["pcf_samples"], 9)
            self.assertEqual(reports[0]["parameters"]["supersample"], 1)
            self.assertEqual(reports[2]["parameters"]["supersample"], 3)
            self.assertGreater(
                reports[2]["parameters"]["shadow_resolution"],
                reports[0]["parameters"]["shadow_resolution"],
            )
            frame = read_pfm(str(directory / "out_0.pfm"))
            self.assertEqual(frame.shape, (18, 24, 3))

    def test_failures_are_nonzero_specific_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            scene_path = directory / "scene.json"
            write_indoor_scene(str(scene_path), 20, 15)

            bad_json = directory / "bad.json"
            bad_json.write_text('{"materials": [}', encoding="utf-8")
            completed = run_cli(
                ["render", str(bad_json), str(directory / "bad.pfm"), "--preset", "low"]
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("invalid JSON", completed.stderr)
            self.assertFalse((directory / "bad.pfm").exists())

            completed = run_cli(
                [
                    "render",
                    str(scene_path),
                    str(directory / "bad_block.pfm"),
                    "--preset",
                    "low",
                    "--block-size",
                    "0",
                ]
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("unsupported block size", completed.stderr)
            self.assertFalse((directory / "bad_block.pfm").exists())

            blocker = directory / "not_a_directory"
            blocker.write_text("blocked", encoding="utf-8")
            valid_pfm = directory / "atomic.pfm"
            completed = run_cli(
                [
                    "render",
                    str(scene_path),
                    str(valid_pfm),
                    "--preset",
                    "low",
                    "--block-size",
                    "8",
                    "--png",
                    str(blocker / "preview.png"),
                ]
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("cannot create output directory", completed.stderr)
            self.assertFalse(valid_pfm.exists())
            self.assertEqual(temporary_files(directory), [])

            completed = run_cli(
                [
                    "render",
                    str(scene_path),
                    str(blocker / "out.pfm"),
                    "--preset",
                    "low",
                    "--block-size",
                    "8",
                ]
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("error:", completed.stderr)
            self.assertFalse((blocker / "out.pfm").exists())
            self.assertEqual(temporary_files(directory), [])


if __name__ == "__main__":
    unittest.main()
