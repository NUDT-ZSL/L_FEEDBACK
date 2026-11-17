from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from lrt.errors import SceneError
from lrt.geometry import AABB, Plane, Sphere, Triangle
from lrt.scene import create_default_scene, load_scene, parse_scene

ROOT = Path(__file__).resolve().parents[1]


class SceneParsingTests(unittest.TestCase):
    def test_default_scene_is_deterministic_and_has_all_primitives(self) -> None:
        first = create_default_scene(seed=158, width=64, height=48)
        second = create_default_scene(seed=158, width=64, height=48)
        self.assertEqual(first, second)
        scene = parse_scene(first)
        self.assertEqual((scene.width, scene.height, scene.seed), (64, 48, 158))
        self.assertTrue(any(isinstance(item, Sphere) for item in scene.geometry))
        self.assertTrue(any(isinstance(item, Plane) for item in scene.geometry))
        self.assertTrue(any(isinstance(item, AABB) for item in scene.geometry))
        self.assertTrue(any(isinstance(item, Triangle) for item in scene.geometry))
        self.assertEqual(len(scene.lights), 2)
        self.assertEqual(scene.description_hash, parse_scene(second).description_hash)
        self.assertNotEqual(
            scene.description_hash, parse_scene(create_default_scene(seed=159, width=64, height=48)).description_hash
        )

    def test_json_round_trip(self) -> None:
        description = create_default_scene(seed=22, width=12, height=10)
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "scene.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(description, handle)
            scene = load_scene(path)
            self.assertEqual(scene.seed, 22)
            self.assertEqual((scene.width, scene.height), (12, 10))

    def assert_error_code(self, mutator, expected_code: int) -> None:
        description = copy.deepcopy(create_default_scene(seed=158, width=8, height=8))
        mutator(description)
        with self.assertRaises(SceneError) as caught:
            parse_scene(description)
        self.assertEqual(caught.exception.code, expected_code)

    def test_parse_error_codes(self) -> None:
        def negative_radius(data):
            data["geometry"][2]["radius"] = -1.0

        def unknown_material(data):
            data["materials"]["mystery"] = {"type": "glass"}

        def empty_lights(data):
            data["lights"] = []

        def missing_field(data):
            del data["camera"]["look_at"]

        def bad_ior(data):
            data["materials"]["ceramic"]["coating"]["ior"] = 0.5

        def bad_thickness(data):
            data["materials"]["ceramic"]["coating"]["thickness"] = 1.0

        def bad_aa(data):
            data["render"]["aa_samples"] = 3

        self.assert_error_code(negative_radius, 4)
        self.assert_error_code(unknown_material, 5)
        self.assert_error_code(empty_lights, 6)
        self.assert_error_code(missing_field, 3)
        self.assert_error_code(bad_ior, 7)
        self.assert_error_code(bad_thickness, 7)
        self.assert_error_code(bad_aa, 8)

    def test_cli_integer_error_codes(self) -> None:
        cases = []

        data = create_default_scene(seed=158, width=8, height=8)
        data["geometry"][2]["radius"] = -2.0
        cases.append((data, 4))

        data = create_default_scene(seed=158, width=8, height=8)
        data["materials"]["bad"] = {"type": "unknown"}
        cases.append((data, 5))

        data = create_default_scene(seed=158, width=8, height=8)
        data["lights"] = []
        cases.append((data, 6))

        data = create_default_scene(seed=158, width=8, height=8)
        data["materials"]["ceramic"]["coating"]["ior"] = 9.0
        cases.append((data, 7))

        with tempfile.TemporaryDirectory() as temporary:
            for index, (description, expected) in enumerate(cases):
                scene_path = os.path.join(temporary, "scene_%d.json" % index)
                with open(scene_path, "w", encoding="utf-8") as handle:
                    json.dump(description, handle)
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "lrt",
                        "--scene",
                        scene_path,
                        "--output",
                        os.path.join(temporary, "out_%d.png" % index),
                        "--stats",
                        os.path.join(temporary, "stats_%d.json" % index),
                    ],
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, expected, completed.stderr)
                self.assertIn("ERROR %d:" % expected, completed.stderr)

            malformed_path = os.path.join(temporary, "malformed.json")
            with open(malformed_path, "w", encoding="utf-8") as handle:
                handle.write("{not json")
            completed = subprocess.run(
                [sys.executable, "-m", "lrt", "--scene", malformed_path],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2, completed.stderr)


if __name__ == "__main__":
    unittest.main()

