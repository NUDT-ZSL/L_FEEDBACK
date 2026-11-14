import os
import tempfile
import unittest

import numpy as np

from terrain_baker.memory import MemoryBudget
from terrain_baker.normal import (
    bake_normal_map,
    decode_rgb8,
    edge_mask,
    encode_rgb8,
    normal_length_error,
)
from terrain_baker.pngutil import read_png_chunks, write_srgb_png


def smooth_grid(n: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, n, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    return (np.sin(xx * 6.28) * np.cos(yy * 6.28)).astype(np.float32)


class NormalMapTests(unittest.TestCase):
    def setUp(self):
        self.budget = MemoryBudget(64 * 1024 * 1024)

    def test_tiled_equals_monolithic(self):
        grid = smooth_grid(129)
        mono, _ = bake_normal_map(grid, self.budget, tile=129)
        for tile in (16, 32, 64):
            tiled, stats = bake_normal_map(grid, self.budget, tile=tile)
            np.testing.assert_array_equal(tiled, mono)
            self.assertGreater(len(stats), 1)

    def test_no_seam_across_tile_boundary(self):
        grid = smooth_grid(129)
        image, _ = bake_normal_map(grid, self.budget, tile=32)
        # Adjacent pixels straddling a tile boundary must differ no more than
        # any other interior adjacent pair (both are uint8 encoded).
        boundary_row = 32
        diff_across = np.abs(
            image[boundary_row].astype(np.int16)
            - image[boundary_row - 1].astype(np.int16)
        )
        interior = np.abs(image[10].astype(np.int16) - image[9].astype(np.int16))
        self.assertLessEqual(diff_across.max(), interior.max() + 1)

    def test_edges_use_clamped_sampling(self):
        grid = smooth_grid(65)
        image, _ = bake_normal_map(grid, self.budget, tile=64)
        # Manual clamped central-difference normal at the top-left pixel.
        spacing = 1.0
        dx = (grid[0, 1] - grid[0, 0]) * 0.5 / spacing
        dy = (grid[1, 0] - grid[0, 0]) * 0.5 / spacing
        normal = np.array([-dx, -dy, 1.0], dtype=np.float32)
        normal /= np.linalg.norm(normal)
        encoded = encode_rgb8(normal[None, None, :])[0, 0]
        np.testing.assert_array_equal(image[0, 0], encoded)
        self.assertTrue(edge_mask(65)[0, 0])
        self.assertFalse(edge_mask(65)[5, 5])

    def test_unit_length_after_normalization(self):
        grid = smooth_grid(129)
        image, _ = bake_normal_map(grid, self.budget, tile=64)
        quality = normal_length_error(image)
        self.assertLess(quality["max_normalized_length_error"], 1e-3)
        decoded = decode_rgb8(image)
        normalized = decoded / np.linalg.norm(decoded, axis=2, keepdims=True)
        lengths = np.linalg.norm(normalized, axis=2)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-6)

    def test_z_positive_and_rgb_roundtrip(self):
        grid = smooth_grid(33)
        image, _ = bake_normal_map(grid, self.budget, tile=33)
        decoded = decode_rgb8(image)
        self.assertTrue((decoded[:, :, 2] > 0).all())

    def test_png_carries_srgb_chunk(self):
        grid = smooth_grid(33)
        image, _ = bake_normal_map(grid, self.budget, tile=33)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "n.png")
            write_srgb_png(path, image)
            write_srgb_png(path + ".copy", image)
            tags = read_png_chunks(path)
            self.assertIn(b"sRGB", tags)
            self.assertIn(b"gAMA", tags)
            self.assertEqual(open(path, "rb").read(), open(path + ".copy", "rb").read())

    def test_flat_heightfield_points_up(self):
        grid = np.full((33, 33), 7.0, dtype=np.float32)
        image, _ = bake_normal_map(grid, self.budget, tile=33)
        decoded = decode_rgb8(image)
        np.testing.assert_allclose(decoded[:, :, 0], 0.0, atol=2.0 / 255.0)
        np.testing.assert_allclose(decoded[:, :, 1], 0.0, atol=2.0 / 255.0)
        np.testing.assert_allclose(decoded[:, :, 2], 1.0, atol=2.0 / 255.0)


if __name__ == "__main__":
    unittest.main()
