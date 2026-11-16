from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from terrain_preview.config import load_config
from terrain_preview.light import LightingResources, fit_light_camera, pcf_visibility
from terrain_preview.material import material_albedo, material_weights
from terrain_preview.scene import build_world_tiles, setup_camera
from terrain_preview.terrain import (
    build_terrain,
    generate_heightfield,
    read_terrain_binary,
    terrain_payload,
    write_terrain_binary,
)
from tests.helpers import small_config, write_config


class TerrainDeterminismTests(unittest.TestCase):
    def test_generation_payload_and_file_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = write_config(root, small_config())
            config = load_config(config_path)
            first = generate_heightfield(config.terrain)
            second = generate_heightfield(config.terrain)
            self.assertEqual(first.dtype, np.float32)
            self.assertEqual(terrain_payload(first), terrain_payload(second))

            first_path = root / "first.bin"
            second_path = root / "second.bin"
            digest_first = write_terrain_binary(first_path, first, config.terrain.seed)
            digest_second = write_terrain_binary(second_path, second, config.terrain.seed)
            self.assertEqual(digest_first, digest_second)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())

            loaded, seed = read_terrain_binary(first_path)
            self.assertEqual(seed, config.terrain.seed)
            np.testing.assert_array_equal(first, loaded)

    def test_different_seed_changes_terrain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = small_config()
            path_a = write_config(root, raw, "a.json")
            changed = copy.deepcopy(raw)
            changed["terrain"]["seed"] += 1
            path_b = write_config(root, changed, "b.json")
            a = generate_heightfield(load_config(path_a).terrain)
            b = generate_heightfield(load_config(path_b).terrain)
            self.assertFalse(np.array_equal(a, b))

    def test_terrain_file_checksum_detects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_config(write_config(root, small_config()))
            terrain = build_terrain(config.terrain, root / "terrain.bin")
            data = bytearray(terrain.path.read_bytes())
            data[-1] ^= 0xFF
            terrain.path.write_bytes(data)
            with self.assertRaises(Exception) as raised:
                read_terrain_binary(terrain.path)
            self.assertIn("checksum", str(raised.exception))


class MaterialTests(unittest.TestCase):
    def test_flat_snow_and_steep_same_altitude_differ(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_config(write_config(root, small_config()))
            altitude = np.array([0.88, 0.88], dtype=np.float64)
            normal_y = np.array([1.0, 0.45], dtype=np.float64)
            weights, albedo = material_albedo(
                altitude, normal_y, config.materials
            )
            self.assertGreater(weights[0, 3], 0.9)
            self.assertGreater(weights[1, 2], 0.5)
            self.assertGreater(np.linalg.norm(albedo[0] - albedo[1]), 0.1)

    def test_weights_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_config(write_config(root, small_config()))
            altitude = np.linspace(0.0, 1.0, 101)
            normal_y = np.linspace(1.0, 0.1, 101)
            weights = material_weights(altitude, normal_y, config.materials)
            np.testing.assert_allclose(np.sum(weights, axis=1), 1.0, atol=1e-12)
            self.assertTrue(np.all(weights >= -1e-15))


class CullingTests(unittest.TestCase):
    def test_narrow_fov_culls_outer_world_tiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = small_config()
            raw["camera"]["fov_y_deg"] = 12.0
            raw["camera"]["distance"] = 300.0
            raw["terrain"]["tile_cells"] = 4
            config = load_config(write_config(root, raw))
            terrain = build_terrain(config.terrain, root / "terrain.bin")
            camera = setup_camera(config)
            visibility = build_world_tiles(config, terrain, camera)
            self.assertGreater(len(visibility.all_tiles), len(visibility.visible))
            self.assertGreater(visibility.culled_frustum, 0)
            self.assertTrue(all(tile.tile_id for tile in visibility.visible))


class ShadowBoundaryTests(unittest.TestCase):
    def test_points_outside_shadow_map_are_fully_lit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_config(write_config(root, small_config()))
            terrain = build_terrain(config.terrain, root / "terrain.bin")
            light_camera = fit_light_camera(config, terrain)
            depth = np.ones(
                (config.light.shadow.resolution, config.light.shadow.resolution),
                dtype=np.float32,
            )
            resources = LightingResources(
                light_camera=light_camera,
                depth_map=depth,
                pcf_offsets=np.zeros((1, 2), dtype=np.float64),
            )
            positions = np.array(
                [[-1.0e6, 0.0, -1.0e6], [1.0e6, 0.0, 1.0e6]]
            )
            normals = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
            visibility = pcf_visibility(positions, normals, config, resources)
            np.testing.assert_allclose(visibility, np.ones(2))


if __name__ == "__main__":
    unittest.main()

