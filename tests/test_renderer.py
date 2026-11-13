from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from frame_render.renderer import RayScene, linear_to_srgb8
from frame_render.spec import load_spec


class RendererTests(unittest.TestCase):
    def test_srgb_endpoints_and_clamping(self) -> None:
        linear = np.asarray(
            [[[-1.0, 0.0, 0.5], [1.0, 2.0, 0.0031308]]], dtype=np.float64
        )
        encoded = linear_to_srgb8(linear)
        self.assertEqual(encoded.dtype, np.uint8)
        self.assertEqual(int(encoded[0, 0, 0]), 0)
        self.assertEqual(int(encoded[0, 0, 1]), 0)
        self.assertEqual(int(encoded[0, 1, 0]), 255)
        self.assertEqual(int(encoded[0, 1, 1]), 255)

    def test_nearest_hit_wins_over_farther_sphere(self) -> None:
        document = {
            "schema_version": 1,
            "scene": {
                "spheres": [
                    {
                        "id": "near-red",
                        "center": [0.0, 0.0, 0.0],
                        "radius": 1.0,
                        "albedo": [0.9, 0.05, 0.05],
                        "specular": 0.1,
                        "shininess": 16.0,
                    },
                    {
                        "id": "far-blue",
                        "center": [0.0, 0.0, -2.5],
                        "radius": 1.0,
                        "albedo": [0.05, 0.05, 0.9],
                        "specular": 0.1,
                        "shininess": 16.0,
                    },
                ],
                "light": {
                    "position": [0.0, 3.0, 3.0],
                    "color": [1.0, 1.0, 1.0],
                    "intensity": 1.0,
                },
                "camera": {
                    "width": 9,
                    "height": 7,
                    "fov_y_deg": 35.0,
                    "position": [0.0, 0.0, 5.0],
                    "target": [0.0, 0.0, 0.0],
                    "up": [0.0, 1.0, 0.0],
                },
                "ambient": [0.08, 0.08, 0.08],
                "background": [0.0, 0.0, 0.0],
            },
            "animation": {
                "frame_start": 0,
                "frame_end": 0,
                "frames": [
                    {
                        "frame": 0,
                        "position": [0.0, 0.0, 5.0],
                        "target": [0.0, 0.0, 0.0],
                        "up": [0.0, 1.0, 0.0],
                        "seed": 7,
                    }
                ],
            },
            "sampling": {
                "samples_per_axis": 2,
                "jitter": False,
                "seed_algorithm": "pcg64-grid-jitter-v1",
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            scene_path = Path(temporary) / "scene.json"
            scene_path.write_text(json.dumps(document), encoding="utf-8")
            spec = load_spec(scene_path)
            scene = RayScene(spec)
            image = scene.render_frame(spec.frames[0])
            center = image[3, 4].astype(np.int16)
            self.assertGreater(center[0], center[2])

    def test_frame_rendering_is_repeatable(self) -> None:
        document = {
            "schema_version": 1,
            "scene": {
                "spheres": [
                    {
                        "id": "sphere",
                        "center": [0.0, 0.0, 0.0],
                        "radius": 1.0,
                        "albedo": [0.7, 0.25, 0.1],
                        "specular": 0.4,
                        "shininess": 40.0,
                    }
                ],
                "light": {
                    "position": [2.0, 3.0, 4.0],
                    "color": [1.0, 0.95, 0.9],
                    "intensity": 1.1,
                },
                "camera": {
                    "width": 22,
                    "height": 15,
                    "fov_y_deg": 45.0,
                    "position": [0.0, 0.0, 4.5],
                    "target": [0.0, 0.0, 0.0],
                    "up": [0.0, 1.0, 0.0],
                },
                "ambient": [0.03, 0.04, 0.06],
                "background": [0.01, 0.01, 0.02],
            },
            "animation": {
                "frame_start": 0,
                "frame_end": 1,
                "frames": [
                    {
                        "frame": 0,
                        "position": [0.0, 0.0, 4.5],
                        "target": [0.0, 0.0, 0.0],
                        "seed": 123,
                    },
                    {
                        "frame": 1,
                        "position": [0.2, 0.1, 4.5],
                        "target": [0.0, 0.0, 0.0],
                        "seed": 124,
                    },
                ],
            },
            "sampling": {
                "samples_per_axis": 4,
                "jitter": True,
                "seed_algorithm": "pcg64-grid-jitter-v1",
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            scene_path = Path(temporary) / "scene.json"
            scene_path.write_text(json.dumps(document), encoding="utf-8")
            spec = load_spec(scene_path)
            scene = RayScene(spec)
            first = scene.render_frame(spec.frames[0])
            second = scene.render_frame(spec.frames[0])
            np.testing.assert_array_equal(first, second)
            self.assertEqual(first.dtype, np.uint8)
            self.assertEqual(first.shape, (15, 22, 3))


if __name__ == "__main__":
    unittest.main()

