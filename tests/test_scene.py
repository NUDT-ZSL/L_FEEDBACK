from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from matrender.scene import SceneValidationError, load_scene, parse_scene


ROOT = Path(__file__).resolve().parents[1]


def _minimal_scene() -> dict:
    return {
        "scene_name": "test",
        "resolution": {"width": 32, "height": 24},
        "camera": {
            "position": [0, 0, 4],
            "target": [0, 0, 0],
            "up": [0, 1, 0],
            "fov_y_deg": 40,
        },
        "light": {"direction": [-0.5, -1, -0.5], "intensity": 1.0},
        "spheres": [
            {
                "name": "s",
                "center": [0, 0, 0],
                "radius": 1.0,
                "base_color": [0.5, 0.5, 0.5],
                "metallic": 0.0,
                "roughness": 0.5,
                "normal_perturbation": 0.0,
            }
        ],
    }


class SceneValidationTests(unittest.TestCase):
    def test_valid_scene(self) -> None:
        scene = parse_scene(_minimal_scene(), "test")
        self.assertEqual(scene.width, 32)
        self.assertEqual(len(scene.spheres), 1)

    def test_missing_required_field(self) -> None:
        payload = _minimal_scene()
        del payload["camera"]["fov_y_deg"]
        with self.assertRaisesRegex(SceneValidationError, "fov_y_deg"):
            parse_scene(payload, "test")

    def test_roughness_zero_rejected(self) -> None:
        payload = _minimal_scene()
        payload["spheres"][0]["roughness"] = 0.0
        with self.assertRaisesRegex(SceneValidationError, "roughness"):
            parse_scene(payload, "test")

    def test_metallic_out_of_range_rejected(self) -> None:
        payload = _minimal_scene()
        payload["spheres"][0]["metallic"] = 1.01
        with self.assertRaisesRegex(SceneValidationError, "metallic"):
            parse_scene(payload, "test")

    def test_malformed_json_is_business_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaisesRegex(SceneValidationError, "invalid JSON"):
                load_scene(path)

    def test_cli_invalid_scene_exits_zero_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "should_not_exist"
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "matrender",
                    "render",
                    str(ROOT / "scenes" / "invalid_roughness.json"),
                    "-o",
                    str(output_dir),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0)
            self.assertIn("E_INVALID_SCENE", process.stderr)
            self.assertFalse(output_dir.exists())

    def test_validate_cli(self) -> None:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "matrender",
                "validate",
                str(ROOT / "scenes" / "material_grid.json"),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(process.returncode, 0)
        payload = json.loads(process.stdout)
        self.assertEqual(payload["status"], "valid")
        self.assertEqual(payload["spheres"], 12)

