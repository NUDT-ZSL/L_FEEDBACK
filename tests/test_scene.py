from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from soft_renderer.errors import SceneError
from soft_renderer.mathutil import srgb_to_linear
from soft_renderer.scene import load_scene


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


class SceneParsingTests(unittest.TestCase):
    def test_area_weighted_normals_and_linear_colors(self) -> None:
        scene_data = {
            "version": 1,
            "color_space": "srgb",
            "image": {"width": 24, "height": 18},
            "camera": {
                "position": [0.0, 0.5, 5.0],
                "target": [0.0, 0.5, 0.0],
                "up": [0.0, 1.0, 0.0],
                "fov_y_deg": 50.0,
            },
            "materials": [
                {
                    "id": 7,
                    "name": "gray",
                    "base_color": "#808080",
                    "roughness": 0.4,
                    "metallic": 0.2,
                }
            ],
            "meshes": [
                {
                    "name": "weighted_normals",
                    "vertices": [
                        {"position": [0.0, 0.0, 0.0], "uv": [0.0, 0.0]},
                        {"position": [0.0, 0.0, 1.0], "uv": [1.0, 0.0]},
                        {"position": [2.0, 0.0, 0.0], "uv": [0.0, 1.0]},
                        {"position": [1.0, 0.0, 0.0], "uv": [0.5, 0.0]},
                        {"position": [1.0, 1.0, 0.0], "uv": [0.5, 1.0]},
                    ],
                    "triangles": [
                        {"indices": [0, 1, 2], "material": 7},
                        {"indices": [0, 3, 4], "material": 7},
                    ],
                }
            ],
            "lights": [
                {
                    "type": "point",
                    "position": [0.0, 2.0, 2.0],
                    "color": [0.5, 0.5, 0.5],
                    "intensity": 4.0,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            _write_json(path, scene_data)
            scene = load_scene(str(path))

        expected = np.array([0.0, 2.0, 1.0])
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(scene.normals[0], expected, rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(scene.normals[1], [0.0, 1.0, 0.0], atol=1.0e-12)
        np.testing.assert_allclose(scene.normals[3], [0.0, 0.0, 1.0], atol=1.0e-12)
        expected_gray = srgb_to_linear(np.array([128.0 / 255.0]))[0]
        np.testing.assert_allclose(
            scene.materials[7].base_color,
            np.full(3, expected_gray),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            scene.lights[0].color,
            np.full(3, srgb_to_linear(np.array([0.5]))[0]),
        )
        self.assertEqual(scene.triangles.shape, (2, 3))
        self.assertEqual(scene.triangle_materials.tolist(), [7, 7])
        self.assertEqual(scene.uvs.shape, (5, 2))

    def test_invalid_triangle_index_is_reported(self) -> None:
        data = {
            "image": {"width": 8, "height": 8},
            "camera": {
                "position": [0.0, 0.0, 2.0],
                "target": [0.0, 0.0, 0.0],
            },
            "materials": [
                {"id": 0, "base_color": [0.5, 0.5, 0.5], "roughness": 0.5, "metallic": 0.0}
            ],
            "meshes": [
                {
                    "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                    "triangles": [[0, 1, 9]],
                    "material": 0,
                }
            ],
            "lights": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            _write_json(path, data)
            with self.assertRaisesRegex(SceneError, "out of range"):
                load_scene(str(path))

    def test_linear_color_space_skips_srgb_transfer(self) -> None:
        data = {
            "color_space": "linear",
            "image": {"width": 8, "height": 8},
            "camera": {
                "position": [0.0, 0.0, 2.0],
                "target": [0.0, 0.0, 0.0],
            },
            "background": [0.25, 0.5, 1.0],
            "materials": [
                {"id": 0, "base_color": [0.1, 0.2, 0.3], "roughness": 0.5, "metallic": 0.0}
            ],
            "meshes": [
                {
                    "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                    "triangles": [[0, 1, 2]],
                    "material": 0,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linear.json"
            _write_json(path, data)
            scene = load_scene(str(path))
        np.testing.assert_array_equal(scene.background, [0.25, 0.5, 1.0])
        np.testing.assert_array_equal(scene.materials[0].base_color, [0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
