from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from spatial_audio_renderer.errors import ParameterError
from spatial_audio_renderer.scene import load_scene, parse_scene
from tests.helpers import base_scene, clone_scene, write_scene


class SceneValidationTests(unittest.TestCase):
    def test_source_timestamps_must_be_strictly_increasing(self) -> None:
        scene = clone_scene(base_scene())
        scene["sources"][0]["position_keys"][1]["time"] = 100
        scene["sources"][0]["position_keys"].insert(
            1, {"time": 100, "position": [-1.4, 0.0, 2.0]}
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = write_scene(Path(temporary), scene)
            with self.assertRaises(ParameterError) as context:
                load_scene(path)
            self.assertIn("strictly increasing", str(context.exception))

    def test_head_timestamps_must_be_strictly_increasing(self) -> None:
        scene = clone_scene(base_scene())
        scene["head"]["quaternion_keys"][1]["time"] = 0
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scene.json"
            import json

            path.write_text(json.dumps(scene), encoding="utf-8")
            with self.assertRaises(ParameterError) as context:
                load_scene(path)
            self.assertIn("head.quaternion_keys", str(context.exception))

    def test_zero_quaternion_is_rejected(self) -> None:
        scene = clone_scene(base_scene())
        scene["head"]["quaternion_keys"][0]["quaternion"] = [0, 0, 0, 0]
        with self.assertRaises(ParameterError):
            parse_scene(scene, "/tmp/scene.json")

    def test_quaternion_is_normalized_and_position_velocity_is_piecewise(self) -> None:
        scene = clone_scene(base_scene(duration_samples=1000))
        scene["head"]["quaternion_keys"][0]["quaternion"] = [2.0, 0.0, 0.0, 0.0]
        scene["head"]["quaternion_keys"][1]["time"] = 1000
        parsed = parse_scene(scene, "/tmp/scene.json")
        sampled = parsed.head.rotation.sample(np.array([0, 500, 1000]))
        norms = np.linalg.norm(sampled, axis=1)
        self.assertTrue(np.allclose(norms, 1.0))
        velocity = parsed.sources[0].position.velocity_per_sample(
            np.array([0, 500, 1000])
        )
        self.assertTrue(np.allclose(velocity, 0.0))

    def test_relative_source_paths_are_scene_relative(self) -> None:
        scene = clone_scene(base_scene())
        with tempfile.TemporaryDirectory() as temporary:
            path = write_scene(Path(temporary), scene)
            parsed = load_scene(path)
            self.assertEqual(
                parsed.sources[0].file_path,
                (Path(temporary) / "left.wav").resolve(),
            )

    def test_windows_style_relative_source_path(self) -> None:
        scene = clone_scene(base_scene())
        scene["sources"][0]["file"] = "assets\\left.wav"
        with tempfile.TemporaryDirectory() as temporary:
            path = write_scene(Path(temporary), scene)
            parsed = load_scene(path)
            self.assertEqual(
                parsed.sources[0].file_path,
                Path(temporary) / "assets" / "left.wav",
            )

    def test_occluder_bounds_are_validated(self) -> None:
        scene = clone_scene(base_scene())
        scene["occluders"][0]["min"] = [1.0, 1.0, 1.0]
        scene["occluders"][0]["max"] = [0.0, 2.0, 2.0]
        with self.assertRaises(ParameterError):
            parse_scene(scene, "/tmp/scene.json")


if __name__ == "__main__":
    unittest.main()
