from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from deferred_validation.verify import compare_reports, load_reference_report


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*arguments: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-m", "deferred_validation", *arguments],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


class VerifyTests(unittest.TestCase):
    def test_cli_same_scene_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = _write_scene(root / "scene.json")
            rendered = run_cli("render", "--scene", str(scene), "-o", str(root / "baseline"))
            self.assertEqual(rendered.returncode, 0, rendered.stderr)
            verified = run_cli(
                "verify",
                "--scene",
                str(scene),
                "--report",
                str(root / "baseline" / "report.json"),
                "-o",
                str(root / "current"),
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            summary = json.loads(verified.stdout)
            self.assertEqual(summary["status"], "pass")
            self.assertIn(summary["classification"], {"identical", "floating_point_noise"})

    def test_cli_light_change_returns_three(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = _write_scene(root / "scene.json")
            altered = _write_scene(root / "altered.json", light_scale=1.25)
            rendered = run_cli("render", "--scene", str(scene), "-o", str(root / "baseline"))
            self.assertEqual(rendered.returncode, 0, rendered.stderr)
            verified = run_cli(
                "verify",
                "--scene",
                str(altered),
                "--report",
                str(root / "baseline" / "report.json"),
                "-o",
                str(root / "altered_out"),
                "--json-out",
                str(root / "summary.json"),
            )
            self.assertEqual(verified.returncode, 3, verified.returncode)
            self.assertIn("substantive differences", verified.stderr)
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["classification"], "substantive")
            self.assertTrue(
                any(reason["kind"] == "checksum_changed" for reason in summary["substantive_reasons"])
            )
            self.assertTrue(
                any(
                    reason["kind"] == "pixel_difference_exceeds_threshold"
                    for reason in summary["substantive_reasons"]
                )
            )

    def test_subthreshold_image_change_is_float_noise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = _write_scene(root / "scene.json")
            first = run_cli("render", "--scene", str(scene), "-o", str(root / "reference"))
            second = run_cli("render", "--scene", str(scene), "-o", str(root / "current"))
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            reference_report, reference_dir = load_reference_report(
                root / "reference" / "report.json"
            )
            current_report, current_dir = load_reference_report(root / "current" / "report.json")
            final_path = reference_dir / "final.png"
            with Image.open(final_path) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            pixels[0, 0, 0] = np.uint8(min(255, int(pixels[0, 0, 0]) + 1))
            Image.fromarray(pixels, mode="RGB").save(final_path, format="PNG", compress_level=9)
            reference_report["artifacts"]["final"]["sha256"] = hashlib.sha256(
                final_path.read_bytes()
            ).hexdigest()
            summary = compare_reports(
                reference_report,
                reference_dir,
                current_report,
                current_dir,
                threshold=0.01,
            )
            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["classification"], "floating_point_noise")
            self.assertLessEqual(summary["comparison"]["max_abs_pixel_difference"], 1.0 / 255.0)


def _write_scene(path: Path, *, light_scale: float = 1.0) -> Path:
    from tests.helpers import write_test_scene

    return write_test_scene(path, light_scale=light_scale)


if __name__ == "__main__":
    unittest.main()
