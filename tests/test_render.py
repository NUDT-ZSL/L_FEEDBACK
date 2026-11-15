from __future__ import annotations

import colorsys
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from matrender.color import luminance
from matrender.renderer import (
    RenderOptions,
    make_render_plan,
    render,
)
from matrender.scene import load_scene


ROOT = Path(__file__).resolve().parents[1]


def _options(
    *,
    samples: int,
    threads: int,
    width: int,
    height: int,
    tile_size: int = 16,
    ssao: bool | None = None,
) -> RenderOptions:
    return RenderOptions(
        samples=samples,
        threads=threads,
        tile_size=tile_size,
        max_memory_bytes=1024 * 1024 * 1024,
        ssao=ssao,
        width=width,
        height=height,
    )


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grid_scene = load_scene(ROOT / "scenes" / "material_grid.json")
        cls.grid_results = {}
        for threads in (1, 2, 4):
            plan = make_render_plan(
                cls.grid_scene,
                _options(
                    samples=2, threads=threads, width=96, height=72
                ),
            )
            cls.grid_results[threads] = render(cls.grid_scene, plan)

    def test_thread_counts_are_bitwise_identical(self) -> None:
        reference = self.grid_results[1]
        for threads in (2, 4):
            candidate = self.grid_results[threads]
            np.testing.assert_array_equal(
                reference.linear_hdr, candidate.linear_hdr
            )
            self.assertEqual(reference.material_regions, candidate.material_regions)
            self.assertEqual(reference.region_stats, candidate.region_stats)

    def test_repeat_render_is_deterministic(self) -> None:
        repeat_plan = make_render_plan(
            self.grid_scene,
            _options(samples=2, threads=4, width=96, height=72),
        )
        repeat = render(self.grid_scene, repeat_plan)
        difference = np.max(
            np.abs(
                repeat.linear_hdr.astype(np.float64)
                - self.grid_results[1].linear_hdr.astype(np.float64)
            )
        )
        self.assertLess(float(difference), 1.0e-6)

    def test_ssao_linear_mean_absolute_difference(self) -> None:
        scene = load_scene(ROOT / "scenes" / "ssao_contrast.json")
        on_plan = make_render_plan(
            scene,
            _options(
                samples=1, threads=2, width=160, height=107, ssao=True
            ),
        )
        off_plan = make_render_plan(
            scene,
            _options(
                samples=1, threads=2, width=160, height=107, ssao=False
            ),
        )
        on_image = render(scene, on_plan).linear_hdr.astype(np.float64)
        off_image = render(scene, off_plan).linear_hdr.astype(np.float64)
        mean_absolute_difference = float(np.mean(np.abs(on_image - off_image)))
        self.assertGreater(mean_absolute_difference, 0.02)

    def test_material_report_matches_hdr_masks(self) -> None:
        result = self.grid_results[4]
        self.assertEqual(
            [entry["index"] for entry in result.material_regions],
            list(range(len(self.grid_scene.spheres))),
        )
        self.assertEqual(
            len({entry["name"] for entry in result.material_regions}),
            len(self.grid_scene.spheres),
        )
        for entry, mask in zip(
            result.material_regions, result.sphere_masks, strict=True
        ):
            values = result.linear_hdr[mask].astype(np.float64)
            self.assertGreater(values.shape[0], 0)
            average = float(np.mean(luminance(values)))
            maximum = float(np.max(luminance(values)))
            self.assertLess(
                abs(average - float(entry["average_luminance"]))
                / max(abs(average), 1.0e-12),
                0.01,
            )
            self.assertLess(
                abs(maximum - float(entry["max_luminance"]))
                / max(abs(maximum), 1.0e-12),
                0.01,
            )
            mean_rgb = np.mean(values, axis=0)
            normalized = mean_rgb / np.max(mean_rgb)
            hue = colorsys.rgb_to_hsv(*normalized.tolist())[0] * 360.0
            self.assertLess(
                abs(hue - float(entry["dominant_hue_deg"])), 1.0e-9
            )

    def test_resource_plan_degrades_and_records_decisions(self) -> None:
        scene = load_scene(ROOT / "scenes" / "ssao_contrast.json")
        plan = make_render_plan(
            scene,
            RenderOptions(
                samples=64,
                threads=2,
                tile_size=32,
                max_memory_bytes=1024 * 1024,
                width=160,
                height=107,
            ),
        )
        self.assertTrue(plan.degraded)
        self.assertLess(plan.samples, 64)
        self.assertGreater(plan.initial_estimated_bytes, plan.estimated_bytes)
        self.assertLessEqual(plan.estimated_bytes, plan.max_memory_bytes)
        self.assertEqual(plan.warnings[0]["code"], "E_RESOURCE_LIMIT")
        self.assertTrue(plan.degradation_events)

    def test_thread_limit_warning(self) -> None:
        plan = make_render_plan(
            self.grid_scene,
            RenderOptions(
                samples=1,
                threads=100000,
                tile_size=16,
                max_memory_bytes=1024 * 1024 * 1024,
                width=32,
                height=24,
            ),
        )
        self.assertEqual(plan.threads, plan.logical_cores)
        self.assertEqual(plan.warnings[0]["code"], "W_THREAD_LIMIT")

    def test_cli_thread_limit_warning_is_logged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "out"
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "matrender",
                    "render",
                    str(ROOT / "scenes" / "material_grid.json"),
                    "-o",
                    str(output_dir),
                    "--width",
                    "32",
                    "--height",
                    "24",
                    "--samples",
                    "1",
                    "--threads",
                    "100000",
                    "--tile-size",
                    "16",
                    "--quiet",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("W_THREAD_LIMIT", process.stderr)
            log = json.loads(
                (output_dir / "render_log.json").read_text(encoding="utf-8")
            )
            warning_codes = {warning["code"] for warning in log["warnings"]}
            self.assertIn("W_THREAD_LIMIT", warning_codes)
            self.assertEqual(
                log["threading"]["effective_threads"],
                log["threading"]["logical_cores"],
            )

    def test_cli_resource_fallback_outputs_are_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "out"
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "matrender",
                    "render",
                    str(ROOT / "scenes" / "material_grid.json"),
                    "-o",
                    str(output_dir),
                    "--width",
                    "96",
                    "--height",
                    "72",
                    "--samples",
                    "8",
                    "--threads",
                    "2",
                    "--tile-size",
                    "16",
                    "--max-memory-mb",
                    "0.5",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("E_RESOURCE_LIMIT", process.stderr)

            expected = {
                "linear_hdr.npy",
                "preview_aces_srgb.png",
                "render_log.json",
                "material_report.json",
                "sphere_masks.npz",
            }
            self.assertEqual(
                {path.name for path in output_dir.iterdir()}, expected
            )

            hdr = np.load(output_dir / "linear_hdr.npy", allow_pickle=False)
            log = json.loads(
                (output_dir / "render_log.json").read_text(encoding="utf-8")
            )
            report = json.loads(
                (output_dir / "material_report.json").read_text(encoding="utf-8")
            )
            masks = np.load(output_dir / "sphere_masks.npz", allow_pickle=False)
            with Image.open(output_dir / "preview_aces_srgb.png") as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (hdr.shape[1], hdr.shape[0]))

            self.assertTrue(log["resource"]["degraded"])
            self.assertEqual(log["linear_hdr"]["dtype"], "float32")
            self.assertEqual(
                log["linear_hdr"]["global"]["linear_scalar_max"],
                float(np.max(hdr)),
            )
            self.assertEqual(len(report["regions"]), 12)
            self.assertEqual(len(masks.files), 12)
            for entry in report["regions"]:
                mask = masks[f"sphere_{entry['index']:03d}"]
                values = hdr[mask].astype(np.float64)
                average = float(np.mean(luminance(values)))
                self.assertLess(
                    abs(average - entry["average_luminance"])
                    / max(abs(average), 1.0e-12),
                    0.01,
                )

