from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from PIL import Image

from terrain_pbr.sample import build_sample_scene, write_sample


ROOT = Path(__file__).resolve().parents[1]


def run_cli(
    scene_path: Path,
    output_dir: Path,
    *,
    width: str | int = 128,
    height: str | int = 96,
    lod: str = "high",
    grid_size: str | int = 16,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "terrain_pbr",
            "--input",
            str(scene_path),
            "--output-dir",
            str(output_dir),
            "--width",
            str(width),
            "--height",
            str(height),
            "--lod",
            lod,
            "--grid-size",
            str(grid_size),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def rounded(points: list[tuple[float, float]]) -> list[list[float]]:
    return [[round(x, 6), round(y, 6)] for x, y in points]


def polygon_shape(
    shape_id: str,
    points: list[tuple[float, float]],
) -> dict[str, Any]:
    levels = rounded(points)
    return {
        "id": shape_id,
        "kind": "polygon",
        "geometry": {"high": levels, "medium": levels, "low": levels},
    }


def write_scene(path: Path, layers: list[dict[str, Any]]) -> Path:
    scene = {"version": 1, "name": path.stem, "layers": layers}
    path.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    return path


def read_metadata(output_dir: Path) -> dict[str, Any]:
    return json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory(prefix="terrain-pbr-test-")
        self.temp = Path(self._temp.name)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_sample_end_to_end_metadata_and_outputs(self) -> None:
        scene = write_sample(self.temp / "sample.json")
        output = self.temp / "out"
        proc = run_cli(scene, output)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(len(proc.stdout.strip().splitlines()), 1)
        summary = json.loads(proc.stdout)
        self.assertTrue(summary["ok"])
        self.assertGreaterEqual(summary["elapsed_seconds"], 0.0)
        self.assertEqual(summary["canvas"], {"width": 128, "height": 96})
        for name in ("albedo.png", "height.png", "normal.png", "metadata.json"):
            path = output / name
            self.assertTrue(path.is_file(), name)
            self.assertEqual(Path(summary["outputs"][name]), path.resolve())

        metadata = read_metadata(output)
        self.assertEqual(metadata["lod"], "high")
        self.assertEqual(metadata["layer_order"], [
            "terrain-ground",
            "river",
            "buildings",
            "road",
            "plaza",
        ])
        self.assertEqual(len(metadata["layers"]), 5)
        by_id = {layer["id"]: layer for layer in metadata["layers"]}
        self.assertTrue(by_id["plaza"]["self_intersecting"])
        self.assertFalse(by_id["river"]["self_intersecting"])
        for layer in metadata["layers"]:
            self.assertIn("bbox", layer)
            self.assertGreater(layer["nontransparent_pixels"], 0)
            self.assertIn("selected", layer["vertex_counts"])
            x0, y0, x1, y1 = layer["bbox"]
            self.assertGreaterEqual(x0, 0)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(x1, 128)
            self.assertLessEqual(y1, 96)
        self.assertEqual(by_id["road"]["bbox"][0], 0)
        self.assertEqual(by_id["road"]["bbox"][2], 128)

        with Image.open(output / "albedo.png") as image:
            self.assertEqual((image.mode, image.size), ("RGBA", (128, 96)))
        with Image.open(output / "height.png") as image:
            self.assertEqual((image.mode, image.size), ("L", (128, 96)))
        with Image.open(output / "normal.png") as image:
            self.assertEqual((image.mode, image.size), ("RGB", (128, 96)))

    def test_over_blend_and_fixed_supersample_coverage(self) -> None:
        scene = write_scene(
            self.temp / "over.json",
            [
                {
                    "id": "base",
                    "color": [255, 0, 0],
                    "alpha": 1.0,
                    "shapes": [
                        polygon_shape(
                            "base-rect",
                            [(2.25, 2.25), (13.75, 2.25), (13.75, 13.75), (2.25, 13.75)],
                        )
                    ],
                },
                {
                    "id": "overlay",
                    "color": [0, 0, 255],
                    "alpha": 0.5,
                    "shapes": [
                        polygon_shape(
                            "overlay-rect",
                            [(6.5, 6.5), (18, 6.5), (18, 18), (6.5, 18)],
                        )
                    ],
                },
            ],
        )
        output = self.temp / "out"
        proc = run_cli(scene, output, width=20, height=20, grid_size=4)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        albedo = Image.open(output / "albedo.png").convert("RGBA")
        self.assertEqual(albedo.getpixel((5, 5)), (255, 0, 0, 255))
        self.assertEqual(albedo.getpixel((2, 2)), (255, 0, 0, 143))
        self.assertEqual(albedo.getpixel((10, 10)), (128, 0, 128, 255))
        self.assertEqual(albedo.getpixel((0, 0)), (0, 0, 0, 0))

    def test_height_and_sobel_normal(self) -> None:
        scene = write_scene(
            self.temp / "height.json",
            [
                {
                    "id": "plateau",
                    "color": [40, 80, 120],
                    "alpha": 1.0,
                    "shapes": [
                        polygon_shape(
                            "plateau-rect",
                            [(2.25, 2.25), (13.75, 2.25), (13.75, 13.75), (2.25, 13.75)],
                        )
                    ],
                }
            ],
        )
        output = self.temp / "out"
        proc = run_cli(scene, output, width=20, height=20, grid_size=5)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        height = Image.open(output / "height.png").convert("L")
        normal = Image.open(output / "normal.png").convert("RGB")
        self.assertEqual(height.getpixel((5, 5)), 255)
        self.assertEqual(height.getpixel((2, 2)), 143)
        self.assertEqual(height.getpixel((0, 0)), 0)
        self.assertEqual(normal.getpixel((5, 5)), (128, 128, 255))
        self.assertEqual(normal.getpixel((0, 0)), (128, 128, 255))
        self.assertNotEqual(normal.getpixel((2, 2)), (128, 128, 255))

        decoded = [
            (channel / 255.0) * 2.0 - 1.0
            for channel in normal.getpixel((2, 2))
        ]
        length = math.sqrt(sum(value * value for value in decoded))
        self.assertAlmostEqual(length, 1.0, delta=0.02)

    def test_lod_switch_changes_geometry_and_records_vertex_counts(self) -> None:
        scene = write_sample(self.temp / "sample.json")
        outputs = {}
        metadata = {}
        for lod in ("high", "medium", "low"):
            output = self.temp / lod
            proc = run_cli(scene, output, lod=lod, grid_size=8)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            outputs[lod] = (output / "albedo.png").read_bytes()
            metadata[lod] = read_metadata(output)
            self.assertEqual(metadata[lod]["lod"], lod)
            for name in ("albedo.png", "height.png", "normal.png"):
                with Image.open(output / name) as image:
                    self.assertEqual(image.size, (128, 96))

        high_layers = {layer["id"]: layer for layer in metadata["high"]["layers"]}
        medium_layers = {layer["id"]: layer for layer in metadata["medium"]["layers"]}
        low_layers = {layer["id"]: layer for layer in metadata["low"]["layers"]}
        for layer_id in ("terrain-ground", "river", "road"):
            high_count = high_layers[layer_id]["vertex_counts"]["selected"]
            medium_count = medium_layers[layer_id]["vertex_counts"]["selected"]
            low_count = low_layers[layer_id]["vertex_counts"]["selected"]
            self.assertLessEqual(low_count, medium_count)
            self.assertLessEqual(medium_count, high_count)
            self.assertLess(low_count, high_count)
        self.assertNotEqual(outputs["high"], outputs["medium"])
        self.assertNotEqual(outputs["high"], outputs["low"])

    def test_grid_stats_and_pixel_determinism(self) -> None:
        scene = write_sample(self.temp / "sample.json")
        small = self.temp / "grid8"
        large = self.temp / "grid32"
        repeat = self.temp / "grid8-repeat"
        self.assertEqual(run_cli(scene, small, grid_size=8).returncode, 0)
        self.assertEqual(run_cli(scene, large, grid_size=32).returncode, 0)
        self.assertEqual(run_cli(scene, repeat, grid_size=8).returncode, 0)

        small_meta = read_metadata(small)
        large_meta = read_metadata(large)
        self.assertEqual(small_meta["peak_rasterized_pixels"], 64)
        self.assertEqual(large_meta["peak_rasterized_pixels"], 1024)
        self.assertLess(
            small_meta["peak_rasterized_pixels"],
            large_meta["peak_rasterized_pixels"],
        )
        self.assertNotEqual(
            small_meta["total_rasterized_pixels"],
            large_meta["total_rasterized_pixels"],
        )
        for name in ("albedo.png", "height.png", "normal.png"):
            self.assertEqual((small / name).read_bytes(), (large / name).read_bytes())
            self.assertEqual((small / name).read_bytes(), (repeat / name).read_bytes())

    def test_self_intersecting_polygon_uses_even_odd_and_does_not_crash(self) -> None:
        sample = build_sample_scene()
        star_layer = next(layer for layer in sample["layers"] if layer["id"] == "plaza")
        scene = write_scene(self.temp / "star.json", [star_layer])
        output = self.temp / "out"
        proc = run_cli(scene, output, grid_size=8)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        metadata = read_metadata(output)
        self.assertTrue(metadata["layers"][0]["self_intersecting"])
        self.assertTrue(
            metadata["layers"][0]["shapes"][0]["self_intersecting_by_lod"]["high"]
        )
        alpha = Image.open(output / "albedo.png").convert("RGBA").getchannel("A")
        self.assertEqual(alpha.getpixel((94, 50)), 0)
        self.assertGreater(alpha.getpixel((94, 34)), 0)

        repeat = self.temp / "repeat"
        self.assertEqual(run_cli(scene, repeat, grid_size=8).returncode, 0)
        self.assertEqual(
            (output / "albedo.png").read_bytes(),
            (repeat / "albedo.png").read_bytes(),
        )

    def test_polygon_enclosing_canvas_is_clipped_without_holes(self) -> None:
        scene = write_scene(
            self.temp / "enclosing.json",
            [
                {
                    "id": "enclosing",
                    "color": [12, 34, 56],
                    "alpha": 1.0,
                    "shapes": [
                        polygon_shape(
                            "outside",
                            [(-20, -20), (40, -20), (40, 40), (-20, 40)],
                        )
                    ],
                }
            ],
        )
        output = self.temp / "out"
        proc = run_cli(scene, output, width=20, height=20, grid_size=6)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        albedo = Image.open(output / "albedo.png").convert("RGBA")
        height = Image.open(output / "height.png").convert("L")
        normal = Image.open(output / "normal.png").convert("RGB")
        self.assertEqual(albedo.getpixel((0, 0)), (12, 34, 56, 255))
        self.assertEqual(albedo.getpixel((19, 19)), (12, 34, 56, 255))
        self.assertEqual(height.getpixel((10, 10)), 255)
        self.assertEqual(normal.getpixel((10, 10)), (128, 128, 255))

    def test_invalid_inputs_are_structured_and_leave_no_partial_outputs(self) -> None:
        valid_scene = build_sample_scene()
        missing_layers = dict(valid_scene)
        missing_layers.pop("layers")
        missing_alpha = json.loads(json.dumps(valid_scene))
        missing_alpha["layers"][0].pop("alpha")
        bad_coordinate = json.loads(json.dumps(valid_scene))
        bad_coordinate["layers"][0]["shapes"][0]["geometry"]["high"][0][0] = "left"

        cases = [
            ("missing_layers", missing_layers, 128, 96, "high", 16),
            ("missing_alpha", missing_alpha, 128, 96, "high", 16),
            ("bad_coordinate", bad_coordinate, 128, 96, "high", 16),
            ("bad_lod", valid_scene, 128, 96, "ultra", 16),
            ("zero_width", valid_scene, 0, 96, "high", 16),
            ("negative_height", valid_scene, 128, -1, "high", 16),
            ("zero_grid", valid_scene, 128, 96, "high", 0),
            ("text_width", valid_scene, "wide", 96, "high", 16),
        ]
        for name, scene_data, width, height, lod, grid in cases:
            with self.subTest(name=name):
                scene = self.temp / f"{name}.json"
                scene.write_text(json.dumps(scene_data), encoding="utf-8")
                output = self.temp / f"out-{name}"
                output.mkdir()
                sentinel = output / "keep.txt"
                sentinel.write_text("keep", encoding="utf-8")
                proc = run_cli(
                    scene,
                    output,
                    width=width,
                    height=height,
                    lod=lod,
                    grid_size=grid,
                )
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertEqual(proc.stdout, "")
                lines = proc.stderr.strip().splitlines()
                self.assertEqual(len(lines), 1)
                payload = json.loads(lines[0])
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["error"]["code"], "E_INVALID_INPUT")
                self.assertTrue(sentinel.is_file())
                for output_name in ("albedo.png", "height.png", "normal.png", "metadata.json"):
                    self.assertFalse((output / output_name).exists())

    def test_invalid_json_is_reported_as_invalid_input(self) -> None:
        scene = self.temp / "broken.json"
        scene.write_text("{not json", encoding="utf-8")
        proc = run_cli(scene, self.temp / "out")
        self.assertEqual(proc.returncode, 2)
        payload = json.loads(proc.stderr.strip())
        self.assertEqual(payload["error"]["code"], "E_INVALID_INPUT")
        self.assertFalse((self.temp / "out").exists())


if __name__ == "__main__":
    unittest.main()
