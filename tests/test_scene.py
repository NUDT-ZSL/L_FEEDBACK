from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from deferred_validation.cli import main
from deferred_validation.errors import ConfigError
from deferred_validation.scene import load_scene
from tests.helpers import write_test_scene


class SceneValidationTests(unittest.TestCase):
    def test_loads_hierarchy_materials_lights_and_normal_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scene_path = write_test_scene(Path(directory) / "scene.json")
            scene = load_scene(scene_path)
            self.assertEqual(len(scene.meshes), 5)
            self.assertGreaterEqual(len(scene.materials), 5)
            self.assertEqual(len(scene.lights), 3)
            self.assertIsNotNone(scene.materials["floor"].normal_map)
            self.assertEqual(scene.materials["floor"].normal_map.shape, (32, 32, 4))
            self.assertIn("geometry", scene.checksums)
            self.assertIn("lighting", scene.checksums)
            self.assertTrue(all(len(value) == 64 for value in scene.checksums.values()))

    def test_unknown_material_reports_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_test_scene(Path(directory) / "scene.json")
            data = json.loads(path.read_text(encoding="utf-8"))
            data["nodes"][0]["material"] = "missing_material"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, r"nodes\.floor_node\.material: unknown material"):
                load_scene(path)

    def test_invalid_roughness_reports_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_test_scene(Path(directory) / "scene.json")
            data = json.loads(path.read_text(encoding="utf-8"))
            data["materials"]["copper"]["roughness"] = 1.5
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, r"materials\.copper\.roughness: must be <= 1.0"):
                load_scene(path)

    def test_missing_camera_reports_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_test_scene(Path(directory) / "scene.json")
            data = json.loads(path.read_text(encoding="utf-8"))
            del data["camera"]
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, r"camera: required field is missing"):
                load_scene(path)

    def test_invalid_json_has_location(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text('{"camera": [}', encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, r"broken\.json:1:\d+: invalid JSON"):
                load_scene(path)

    def test_cli_missing_scene_returns_2_and_stderr(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main(["render", "--scene", "/definitely/missing/scene.json", "-o", "/tmp/out"])
        self.assertEqual(code, 2)
        self.assertIn("configuration error", stderr.getvalue())
        self.assertIn("does not exist", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
