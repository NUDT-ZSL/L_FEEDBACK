from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from render_pipeline import (
    E_SNAPSHOT_CORRUPT,
    E_TILE_FAILED,
    PipelineOptions,
    SnapshotCorruptError,
    load_scene,
    load_snapshot,
    render_reference,
    render_scene,
    shadow_visibility,
)
from render_pipeline.math3d import dot, normalize
from render_pipeline.scene import canonical_json, digest


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "sample_scene.json"


def small_scene() -> dict:
    scene = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    scene = copy.deepcopy(scene)
    scene["frame"] = {"width": 64, "height": 48}
    scene["tile"] = {"width": 16, "height": 16}
    scene["lights"][0]["resolution"] = 64
    scene["lights"][0]["pcf_kernel"] = 5
    scene["lights"][1]["resolution"] = 32
    scene["shadow"]["directional_resolution"] = 64
    scene["shadow"]["point_resolution"] = 32
    scene["shadow"]["pcf_kernel"] = 5
    scene["memory_limit_bytes"] = 16 * 1024 * 1024
    return scene


def write_scene(directory: str | Path, scene: dict, name: str = "scene.json") -> Path:
    path = Path(directory) / name
    path.write_text(json.dumps(scene, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def tile_at(tiles, scene, x: int, y: int):
    origin_x = (x // scene.tile_width) * scene.tile_width
    origin_y = (y // scene.tile_height) * scene.tile_height
    for tile in tiles:
        if tile.x == origin_x and tile.y == origin_y:
            return tile
    raise AssertionError(f"no tile at {x}, {y}")


class PipelineAcceptanceTests(unittest.TestCase):
    def test_tiled_gbuffer_matches_whole_frame_exactly(self):
        scene_data = small_scene()
        scene = load_scene(scene_data)
        reference = render_reference(scene)
        tiled = render_scene(scene, PipelineOptions(workers=4, snapshot_dir=None))

        self.assertEqual(reference.frame, tiled.frame)
        self.assertEqual(tiled.report["status"], "ok")
        for screen_y in range(scene.height):
            for screen_x in range(scene.width):
                tile = tile_at(tiled.gbuffer_tiles, scene, screen_x, screen_y)
                local_x = screen_x - tile.x
                local_y = screen_y - tile.y
                tile_index = tile.pixel_index(local_x, local_y)
                whole_index = screen_y * scene.width + screen_x
                self.assertEqual(reference.gbuffer.depth[whole_index], tile.depth[tile_index])
                self.assertEqual(reference.gbuffer.material_ids[whole_index], tile.material_ids[tile_index])
                for component in range(3):
                    self.assertEqual(
                        reference.gbuffer.normal[whole_index * 3 + component],
                        tile.normal[tile_index * 3 + component],
                    )
                    self.assertEqual(
                        reference.gbuffer.albedo[whole_index * 3 + component],
                        tile.albedo[tile_index * 3 + component],
                    )

    def test_parallel_execution_is_deterministic(self):
        scene_data = small_scene()
        scene = load_scene(scene_data)
        serial = render_scene(scene, PipelineOptions(workers=1, snapshot_dir=None))
        parallel = render_scene(scene, PipelineOptions(workers=4, snapshot_dir=None))
        wide_parallel = render_scene(scene, PipelineOptions(workers=8, snapshot_dir=None))
        self.assertEqual(serial.ppm_bytes, parallel.ppm_bytes)
        self.assertEqual(serial.ppm_bytes, wide_parallel.ppm_bytes)
        self.assertEqual(
            hashlib.sha256(serial.ppm_bytes).hexdigest(),
            hashlib.sha256(parallel.ppm_bytes).hexdigest(),
        )
        self.assertTrue(serial.report["tiles"][0]["shadow_maps"])

    def test_shadow_quality_presets_pcf_and_cache_decision(self):
        scene_data = small_scene()
        scene = load_scene(scene_data)
        reference = render_reference(scene)
        sun = scene.lights[0]
        light_vector = normalize((-sun.direction[0], -sun.direction[1], -sun.direction[2]))
        floor_visibility = []
        for pixel_y in range(scene.height):
            for pixel_x in range(scene.width):
                pixel_index = pixel_y * scene.width + pixel_x
                if reference.gbuffer.depth[pixel_index] == float("inf"):
                    continue
                if reference.gbuffer.material_ids[pixel_index] != 0:
                    continue
                normal = normalize(
                    tuple(reference.gbuffer.normal[pixel_index * 3 : pixel_index * 3 + 3])
                )
                world_position = reference.gbuffer.world_at(scene, pixel_x, pixel_y)
                floor_visibility.append(
                    shadow_visibility(
                        scene,
                        sun,
                        reference.shadow_maps,
                        world_position,
                        normal,
                        dot(normal, light_vector),
                    )
                )
        self.assertLess(min(floor_visibility), 0.8)
        self.assertGreater(
            sum(1 for value in floor_visibility if value >= 0.999) / len(floor_visibility),
            0.8,
        )
        with tempfile.TemporaryDirectory() as snapshot_dir:
            high = render_scene(
                scene,
                PipelineOptions(workers=4, snapshot_dir=snapshot_dir, quality="high"),
            )
            self.assertEqual(high.report["shadow"]["effective"]["sun"]["resolution"], 64)
            self.assertEqual(high.report["shadow"]["effective"]["lamp"]["resolution"], 32)
            self.assertEqual(high.report["shadow"]["effective"]["sun"]["pcf_kernel"], 5)
            self.assertTrue(all(len(tile["shadow_maps"]) == 2 for tile in high.report["tiles"]))
            point_snapshot = load_snapshot(
                Path(snapshot_dir) / "tile-0016-0016.snapshot.json"
            )
            point_map = point_snapshot.shadow_maps["lamp"]
            self.assertEqual(len(point_map.faces), 6)
            self.assertEqual(point_map.resolution, 32)

            reused = render_scene(
                scene,
                PipelineOptions(workers=4, snapshot_dir=snapshot_dir, quality="high"),
            )
            self.assertEqual(reused.report["summary"]["reused_tiles"], reused.report["summary"]["total_tiles"])

            fast = render_scene(
                scene,
                PipelineOptions(workers=4, snapshot_dir=snapshot_dir, quality="fast"),
            )
            fresh_fast = render_scene(
                scene,
                PipelineOptions(workers=4, snapshot_dir=None, quality="fast"),
            )
            self.assertEqual(fast.ppm_bytes, fresh_fast.ppm_bytes)
            self.assertNotEqual(high.ppm_bytes, fast.ppm_bytes)
            self.assertEqual(fast.report["shadow"]["effective"]["sun"]["resolution"], 32)
            self.assertEqual(fast.report["shadow"]["effective"]["lamp"]["resolution"], 16)
            self.assertEqual(fast.report["shadow"]["effective"]["sun"]["pcf_kernel"], 1)
            self.assertEqual(fast.report["summary"]["reused_tiles"], 0)
            self.assertFalse(any(tile["shadow_cache_reused"] for tile in fast.report["tiles"]))
            self.assertIn("quality_changed", fast.report["incremental"]["global_reasons"])

            pcf_three = render_scene(
                scene,
                PipelineOptions(
                    workers=2,
                    snapshot_dir=None,
                    quality="high",
                    pcf_kernel=3,
                ),
            )
            pcf_five = render_scene(
                scene,
                PipelineOptions(
                    workers=2,
                    snapshot_dir=None,
                    quality="high",
                    pcf_kernel=5,
                ),
            )
            self.assertNotEqual(pcf_three.ppm_bytes, pcf_five.ppm_bytes)

    def test_snapshot_round_trip_and_corruption_codes(self):
        scene_data = small_scene()
        scene = load_scene(scene_data)
        with tempfile.TemporaryDirectory() as snapshot_dir:
            result = render_scene(
                scene,
                PipelineOptions(workers=2, snapshot_dir=snapshot_dir),
            )
            snapshot_file = Path(snapshot_dir) / "tile-0016-0016.snapshot.json"
            snapshot = load_snapshot(snapshot_file)
            self.assertEqual(snapshot.version, 1)
            self.assertTrue(snapshot.checksum.startswith("sha256:"))
            tile = tile_at(result.gbuffer_tiles, scene, 16, 16)
            self.assertEqual(snapshot.gbuffer.depth, tile.depth)
            self.assertEqual(snapshot.gbuffer.normal, tile.normal)
            self.assertEqual(snapshot.gbuffer.albedo, tile.albedo)
            self.assertEqual(snapshot.scene["summary"], scene.summary())

            raw = json.loads(snapshot_file.read_text(encoding="utf-8"))
            raw["scene"]["summary"] = "tampered"
            snapshot_file.write_text(canonical_json(raw) + "\n", encoding="utf-8")
            with self.assertRaises(SnapshotCorruptError) as checksum_context:
                load_snapshot(snapshot_file)
            self.assertEqual(checksum_context.exception.code, E_SNAPSHOT_CORRUPT)

            raw = json.loads(snapshot_file.read_text(encoding="utf-8"))
            raw.pop("checksum", None)
            raw["version"] = 999
            raw["checksum"] = "sha256:" + digest(raw)
            snapshot_file.write_text(canonical_json(raw) + "\n", encoding="utf-8")
            with self.assertRaises(SnapshotCorruptError) as version_context:
                load_snapshot(snapshot_file)
            self.assertEqual(version_context.exception.code, E_SNAPSHOT_CORRUPT)

            repaired = render_scene(
                scene,
                PipelineOptions(workers=2, snapshot_dir=snapshot_dir),
            )
            self.assertEqual(repaired.report["status"], "ok")
            self.assertEqual(repaired.report["summary"]["rendered_tiles"], 1)
            self.assertEqual(repaired.report["snapshot_errors"][0]["code"], E_SNAPSHOT_CORRUPT)

    def test_incremental_material_color_and_scene_version(self):
        base_data = small_scene()
        with tempfile.TemporaryDirectory() as snapshot_dir:
            first = render_scene(
                load_scene(base_data),
                PipelineOptions(workers=4, snapshot_dir=snapshot_dir),
            )
            self.assertEqual(first.report["summary"]["rendered_tiles"], first.report["summary"]["total_tiles"])
            self.assertEqual(first.report["summary"]["reused_tiles"], 0)

            material_changed = copy.deepcopy(base_data)
            material_changed["materials"][2]["albedo"] = [0.24, 0.17, 0.61]
            second = render_scene(
                load_scene(material_changed),
                PipelineOptions(workers=4, snapshot_dir=snapshot_dir),
            )
            fresh_second = render_scene(
                load_scene(material_changed),
                PipelineOptions(workers=4, snapshot_dir=None),
            )
            self.assertEqual(second.ppm_bytes, fresh_second.ppm_bytes)
            self.assertIn("red", second.report["incremental"]["changed_materials"])
            self.assertGreater(second.report["summary"]["reused_tiles"], 0)
            self.assertLess(
                second.report["summary"]["rendered_tiles"],
                second.report["summary"]["total_tiles"],
            )
            self.assertTrue(any(tile["shadow_cache_reused"] for tile in second.report["tiles"]))

            light_changed = copy.deepcopy(material_changed)
            light_changed["lights"][1]["color"] = [1.0, 0.33, 0.14]
            third = render_scene(
                load_scene(light_changed),
                PipelineOptions(workers=4, snapshot_dir=snapshot_dir),
            )
            fresh_third = render_scene(
                load_scene(light_changed),
                PipelineOptions(workers=4, snapshot_dir=None),
            )
            self.assertEqual(third.ppm_bytes, fresh_third.ppm_bytes)
            self.assertIn("lamp", third.report["incremental"]["changed_light_colors"])
            self.assertGreater(third.report["summary"]["reused_tiles"], 0)
            self.assertLess(
                third.report["summary"]["rendered_tiles"],
                third.report["summary"]["total_tiles"],
            )

            version_changed = copy.deepcopy(light_changed)
            version_changed["version"] = 2
            fourth = render_scene(
                load_scene(version_changed),
                PipelineOptions(workers=4, snapshot_dir=snapshot_dir),
            )
            self.assertEqual(fourth.report["summary"]["reused_tiles"], 0)
            self.assertIn("scene_version_changed", fourth.report["incremental"]["global_reasons"])

    def test_failed_tiles_are_isolated_and_never_snapshotted(self):
        invalid_scene = small_scene()
        invalid_scene["materials"][2]["albedo"] = [-0.4, 0.1, 0.1]
        with tempfile.TemporaryDirectory() as snapshot_dir:
            valid = render_scene(
                load_scene(small_scene()),
                PipelineOptions(workers=2, snapshot_dir=snapshot_dir),
            )
            self.assertEqual(valid.report["status"], "ok")
            stale = render_scene(
                load_scene(invalid_scene),
                PipelineOptions(workers=4, snapshot_dir=snapshot_dir),
            )
            self.assertEqual(stale.report["status"], "partial")
            for failed in stale.report["failed_tiles"]:
                self.assertFalse(
                    (Path(snapshot_dir) / f"{failed['id']}.snapshot.json").exists()
                )

        with tempfile.TemporaryDirectory() as snapshot_dir:
            result = render_scene(
                load_scene(invalid_scene),
                PipelineOptions(workers=4, snapshot_dir=snapshot_dir),
            )
            self.assertEqual(result.report["status"], "partial")
            self.assertGreater(result.report["summary"]["succeeded_tiles"], 0)
            self.assertGreater(result.report["summary"]["failed_tiles"], 0)
            self.assertEqual(len(result.ppm_bytes), 64 * 48 * 3 + len(f"P6\n64 48\n255\n"))
            for failed in result.report["failed_tiles"]:
                self.assertEqual(failed["error"]["code"], E_TILE_FAILED)
                self.assertFalse(
                    (Path(snapshot_dir) / f"{failed['id']}.snapshot.json").exists()
                )
            snapshot_count = len(list(Path(snapshot_dir).glob("*.snapshot.json")))
            self.assertEqual(snapshot_count, result.report["summary"]["succeeded_tiles"])

        with tempfile.TemporaryDirectory() as snapshot_dir:
            result = render_scene(
                load_scene(small_scene()),
                PipelineOptions(
                    workers=4,
                    snapshot_dir=snapshot_dir,
                    memory_limit_bytes=4096,
                ),
            )
            self.assertEqual(result.report["summary"]["failed_tiles"], result.report["summary"]["total_tiles"])
            self.assertTrue(all(tile["error"]["code"] == E_TILE_FAILED for tile in result.report["failed_tiles"]))
            self.assertEqual(list(Path(snapshot_dir).glob("*.snapshot.json")), [])

    def test_cli_entry_point(self):
        with tempfile.TemporaryDirectory() as directory:
            scene_path = write_scene(directory, small_scene())
            output_path = Path(directory) / "out.ppm"
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "render_pipeline",
                    "--scene",
                    str(scene_path),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertTrue(output_path.exists())
            report_path = output_path.with_name("out.report.json")
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertEqual(
                report["output_sha256"],
                hashlib.sha256(output_path.read_bytes()).hexdigest(),
            )
            snapshot_directory = output_path.with_name("out.snapshots")
            self.assertEqual(len(list(snapshot_directory.glob("*.snapshot.json"))), 12)


if __name__ == "__main__":
    unittest.main()
