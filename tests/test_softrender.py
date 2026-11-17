from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from softrender.hiz import HierarchicalZ
from softrender.renderer import render_scene
from softrender.scene import parse_scene


def _sample_data() -> dict:
    return json.loads((ROOT / "samples" / "character_scene.json").read_text(encoding="utf-8"))


def _small_scene_data() -> dict:
    data = _sample_data()
    data["camera"]["width"] = 160
    data["camera"]["height"] = 160
    for light in data["lights"]:
        light["shadow"]["resolution"] = 64
    return data


class SoftRenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_scene(self, data: dict, name: str = "scene.json") -> Path:
        path = self.temp_path / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def _run_cli(self, scene: Path, output: Path, stats: Path | None = None) -> subprocess.CompletedProcess:
        command = [
            sys.executable,
            str(ROOT / "render.py"),
            "--scene",
            str(scene),
            "--output",
            str(output),
        ]
        if stats is not None:
            command.extend(["--stats", str(stats)])
        return subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )

    def test_full_sample_render_statistics(self) -> None:
        scene = parse_scene(_sample_data(), "inline")
        result = render_scene(scene)
        self.assertEqual(result.image.shape, (512, 512, 3))
        self.assertEqual(result.image.dtype, np.uint8)
        counts = result.stats["counts"]
        self.assertEqual(counts["total_triangles"], 7504)
        self.assertGreater(counts["hiz_culled_triangles"], 0)
        self.assertGreater(counts["shaded_pixels"], 0)
        self.assertGreater(counts["transparent_fragments"], 0)
        self.assertEqual(counts["transparent_draw_calls"], 1)
        self.assertEqual(counts["opaque_draw_calls"], 5)
        self.assertEqual(result.stats["hiz"]["levels"], 10)
        light_stats = result.stats["lights"]
        self.assertEqual(len(light_stats), 2)
        self.assertEqual(light_stats[0]["shadowMapResolution"], [512, 512])
        self.assertEqual(light_stats[1]["shadowMapResolution"], [256, 256])
        self.assertEqual(light_stats[1]["shadowMapFaces"], 6)
        self.assertEqual(light_stats[0]["pcfKernel"], [3, 3])

    def test_cli_is_byte_reproducible_and_writes_required_stats(self) -> None:
        scene_path = self._write_scene(_small_scene_data())
        first = self._run_cli(
            scene_path,
            self.temp_path / "first",
            self.temp_path / "first" / "logs" / "stats.json",
        )
        second = self._run_cli(
            scene_path,
            self.temp_path / "second",
            self.temp_path / "second" / "logs" / "stats.json",
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        first_png = (self.temp_path / "first" / "render.png").read_bytes()
        second_png = (self.temp_path / "second" / "render.png").read_bytes()
        self.assertEqual(first_png, second_png)
        first_stats_bytes = (self.temp_path / "first" / "logs" / "stats.json").read_bytes()
        second_stats_bytes = (self.temp_path / "second" / "logs" / "stats.json").read_bytes()
        self.assertEqual(first_stats_bytes, second_stats_bytes)
        stats = json.loads(first_stats_bytes)
        self.assertIn("total_triangles", stats["counts"])
        self.assertIn("culled_triangles", stats["counts"])
        self.assertIn("shaded_pixels", stats["counts"])
        self.assertEqual(len(stats["lights"]), 2)
        self.assertEqual(stats["output"]["format"], "PNG")
        self.assertEqual(len(stats["output"]["sha256"]), 64)

    def test_near_plane_crossing_is_polygon_clipped(self) -> None:
        data = _small_scene_data()
        data["camera"].update(
            {
                "position": [0.0, 0.0, 0.0],
                "target": [0.0, 0.0, -1.0],
                "up": [0.0, 1.0, 0.0],
                "near": 0.5,
                "far": 20.0,
                "cullMode": "none",
            }
        )
        data["mesh"].update(
            {
                "vertices": [
                    [-0.6, -0.5, -1.0],
                    [0.6, -0.5, -1.0],
                    [0.0, 0.6, 0.5],
                ],
                "normals": [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
                "uvs": [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
                "tangents": [
                    [1.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0],
                ],
                "primitives": [{"material": "armor", "indices": [0, 1, 2]}],
            }
        )
        result = render_scene(parse_scene(data, "inline"))
        counts = result.stats["counts"]
        self.assertEqual(counts["total_triangles"], 1)
        self.assertEqual(counts["near_clipped_original_triangles"], 1)
        self.assertGreaterEqual(counts["near_clipped_output_triangles"], 2)
        self.assertGreater(counts["shaded_pixels"], 0)

    def test_hiz_is_conservative_for_partial_rectangles(self) -> None:
        depth = np.full((8, 8), 0.2, dtype=np.float64)
        depth[0, 0] = 0.9
        hiz = HierarchicalZ(depth)
        self.assertTrue(hiz.fully_occluded(1, 1, 6, 6, 0.4))
        self.assertFalse(hiz.fully_occluded(0, 0, 0, 0, 0.4))
        self.assertFalse(hiz.fully_occluded(0, 0, 6, 6, 0.4))
        self.assertFalse(hiz.fully_occluded(1, 1, 6, 6, 0.2))

    def test_wireframe_and_normal_overlays_render(self) -> None:
        data = _small_scene_data()
        data["debug"].update({"wireframe": True, "normals": True})
        result = render_scene(parse_scene(data, "inline"))
        self.assertGreater(result.stats["counts"]["wireframe_pixels"], 0)
        self.assertGreater(result.stats["counts"]["normal_overlay_pixels"], 0)

    def test_invalid_scene_and_missing_resource_error_codes(self) -> None:
        data = _small_scene_data()
        del data["camera"]
        missing_field = self._write_scene(data, "missing.json")
        result = self._run_cli(missing_field, self.temp_path / "missing_out")
        self.assertNotEqual(result.returncode, 0)
        error = json.loads(result.stderr)
        self.assertEqual(error["error"]["code"], "E_SCHEMA")
        self.assertEqual(error["error"]["path"], "$.camera")

        invalid = _small_scene_data()
        invalid["camera"]["fovY"] = 0.0
        invalid_path = self._write_scene(invalid, "invalid.json")
        result = self._run_cli(invalid_path, self.temp_path / "invalid_out")
        self.assertNotEqual(result.returncode, 0)
        error = json.loads(result.stderr)
        self.assertEqual(error["error"]["code"], "E_VALUE")
        self.assertEqual(error["error"]["path"], "$.camera.fovY")

        missing_path = self.temp_path / "does_not_exist.json"
        result = self._run_cli(missing_path, self.temp_path / "no_input_out")
        self.assertNotEqual(result.returncode, 0)
        error = json.loads(result.stderr)
        self.assertEqual(error["error"]["code"], "E_INPUT_READ")

    def test_non_writable_output_path_returns_output_error(self) -> None:
        scene_path = self._write_scene(_small_scene_data())
        blocking_file = self.temp_path / "not_a_directory"
        blocking_file.write_text("occupied", encoding="utf-8")
        result = self._run_cli(scene_path, blocking_file)
        self.assertNotEqual(result.returncode, 0)
        error = json.loads(result.stderr)
        self.assertEqual(error["error"]["code"], "E_OUTPUT")


if __name__ == "__main__":
    unittest.main()
