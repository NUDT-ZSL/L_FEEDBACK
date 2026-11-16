from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from prism_tracer.errors import BusinessError
from prism_tracer.integrator import _dielectric_response
from prism_tracer.pipeline import RenderConfig, run_pipeline
from prism_tracer.scene import (
    build_scene,
    scene_definition,
    validate_scene_data,
)


ROOT = Path(__file__).resolve().parents[1]


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ValidationTests(unittest.TestCase):
    def test_missing_scene_field_is_business_error(self) -> None:
        data = scene_definition("prism")
        data.pop("camera")
        with self.assertRaises(BusinessError) as raised:
            validate_scene_data(data)
        self.assertEqual(raised.exception.code, "E_SCENE_FIELDS")
        self.assertEqual(raised.exception.exit_code, 11)

    def test_invalid_optical_table_is_business_error(self) -> None:
        data = scene_definition("prism")
        data["water"]["ior"]["values"][0] = -1.0
        with self.assertRaises(BusinessError) as raised:
            validate_scene_data(data)
        self.assertEqual(raised.exception.code, "E_SCENE_VALUE")

    def test_camera_inside_glass_is_business_error(self) -> None:
        scene = build_scene("prism")
        scene.camera_position[:] = [0.0, 0.8, 0.0]
        with self.assertRaises(BusinessError) as raised:
            scene.validate_camera()
        self.assertEqual(raised.exception.code, "E_CAMERA_MEDIUM")
        self.assertEqual(raised.exception.exit_code, 20)

    def test_camera_inside_water_is_business_error(self) -> None:
        scene = build_scene("prism")
        scene.camera_position[:] = [0.0, 0.1, 0.0]
        with self.assertRaises(BusinessError) as raised:
            scene.validate_camera()
        self.assertEqual(raised.exception.code, "E_CAMERA_MEDIUM")

    def test_invalid_cli_does_not_write_ppm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "out"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "render.py"),
                    "--res",
                    "0x10",
                    "--spp",
                    "1",
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("[E_RESOLUTION]", result.stderr)
            self.assertEqual(list(out.glob("*.ppm")) if out.exists() else [], [])

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "render.py"),
                    "--res",
                    "16x12",
                    "--spp",
                    "-4",
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("[E_SPP]", result.stderr)
            self.assertEqual(list(out.glob("*.ppm")) if out.exists() else [], [])


class SpectralAndFresnelTests(unittest.TestCase):
    def test_fixed_weights_form_equal_energy_white(self) -> None:
        scene = build_scene("prism")
        assert scene.spectral is not None
        np.testing.assert_allclose(
            scene.spectral.white_error(), np.ones(3), atol=5.0e-4
        )

    def test_fresnel_and_total_internal_reflection(self) -> None:
        scene = build_scene("prism")
        direction = np.array([[0.0, -1.0, 0.0]])
        normal = np.array([[0.0, 1.0, 0.0]])
        wavelength = np.array([550.0])

        _, medium, factor, tir = _dielectric_response(
            scene,
            direction,
            normal,
            np.array([0], dtype=np.int8),
            np.array([1], dtype=np.int8),
            wavelength,
            np.array([0.5]),
            radiance_mode=True,
        )
        self.assertEqual(int(medium[0]), 1)
        self.assertFalse(bool(tir[0]))
        self.assertAlmostEqual(float(factor[0]), (1.0 / 1.499) ** 2, places=4)

        angle = np.deg2rad(62.0)
        outgoing = np.array([[np.sin(angle), np.cos(angle), 0.0]])
        reflected, medium, factor, tir = _dielectric_response(
            scene,
            outgoing,
            normal,
            np.array([1], dtype=np.int8),
            np.array([0], dtype=np.int8),
            wavelength,
            np.array([0.01]),
            radiance_mode=True,
        )
        self.assertTrue(bool(tir[0]))
        self.assertEqual(int(medium[0]), 1)
        self.assertAlmostEqual(float(factor[0]), 1.0)
        self.assertLess(float(reflected[0, 1]), 0.0)


class PipelineTests(unittest.TestCase):
    config = RenderConfig(
        width=32,
        height=24,
        spp=4,
        seed=7,
        tile=8,
        max_depth=6,
        use_indirect=True,
        caustic_photons=10_000,
        caustic_map_resolution=64,
        exposure=1.1,
    )

    def test_resume_is_byte_identical(self) -> None:
        scene = build_scene("prism")
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            out_full = Path(a)
            out_resume = Path(b)
            report_full = run_pipeline(scene, self.config, out_full)
            image_name = report_full["outputs"]["image"]
            aux_name = report_full["outputs"]["auxiliary"]
            report_name = report_full["outputs"]["report"]

            with self.assertRaises(KeyboardInterrupt):
                run_pipeline(
                    scene,
                    self.config,
                    out_resume,
                    stop_after_tiles=2,
                )
            self.assertEqual(list(out_resume.glob("*.ppm")), [])
            self.assertTrue(list(out_resume.glob("*.partial-*.npz")))

            report_resume = run_pipeline(scene, self.config, out_resume)
            self.assertEqual(report_full["seed"], 7)
            self.assertEqual(report_full["samples_per_pixel"], 4)
            self.assertEqual(report_full["max_path_depth"], 6)
            self.assertIn("channel_stats", report_full)
            self.assertIn("variance", report_full["channel_stats"]["R"])
            self.assertEqual(
                _file_hash(out_full / image_name),
                _file_hash(out_resume / image_name),
            )
            self.assertEqual(
                _file_hash(out_full / aux_name),
                _file_hash(out_resume / aux_name),
            )
            self.assertEqual(
                _file_hash(out_full / report_name),
                _file_hash(out_resume / report_name),
            )
            self.assertEqual(report_resume["hashes"], report_full["hashes"])

    def test_tile_size_does_not_change_image_bytes(self) -> None:
        scene = build_scene("prism")
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = run_pipeline(scene, self.config, Path(a))
            changed = RenderConfig(
                width=self.config.width,
                height=self.config.height,
                spp=self.config.spp,
                seed=self.config.seed,
                tile=16,
                max_depth=self.config.max_depth,
                use_indirect=self.config.use_indirect,
                caustic_photons=self.config.caustic_photons,
                caustic_map_resolution=self.config.caustic_map_resolution,
                exposure=self.config.exposure,
            )
            second = run_pipeline(scene, changed, Path(b))
            self.assertEqual(
                _file_hash(Path(a) / first["outputs"]["image"]),
                _file_hash(Path(b) / second["outputs"]["image"]),
            )

    def test_no_indirect_reports_zero_indirect_energy(self) -> None:
        scene = build_scene("prism")
        direct_only = RenderConfig(
            width=24,
            height=18,
            spp=3,
            seed=11,
            tile=8,
            max_depth=5,
            use_indirect=False,
            caustic_photons=5_000,
            caustic_map_resolution=48,
            exposure=1.1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            report = run_pipeline(scene, direct_only, Path(temporary))
        self.assertFalse(report["indirect_enabled"])
        for channel in ("R", "G", "B"):
            self.assertEqual(report["channel_stats"][channel]["indirect_mean"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

