import os
import tempfile
import unittest

import numpy as np

from terrain_baker.lod import (
    INDEX_DTYPE,
    VERTEX_DTYPE,
    downsample_heightfield,
    lod_resolution,
    tangent_normals_from_height,
    write_lod_mesh,
)
from terrain_baker.memory import MemoryBudget


def paraboloid(n: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    return (xx * xx + yy * yy).astype(np.float32)


class LodResolutionTests(unittest.TestCase):
    def test_halving(self):
        self.assertEqual(lod_resolution(257, 0), 257)
        self.assertEqual(lod_resolution(257, 1), 129)
        self.assertEqual(lod_resolution(257, 2), 65)

    def test_invalid_level(self):
        with self.assertRaises(ValueError):
            lod_resolution(257, -1)


class DownsampleTests(unittest.TestCase):
    def setUp(self):
        self.budget = MemoryBudget(64 * 1024 * 1024)

    def test_dimensions_and_corners_preserved(self):
        n = 129
        grid = paraboloid(n)
        coarse = downsample_heightfield(grid, self.budget, block_rows=16)
        self.assertEqual(coarse.shape, (65, 65))
        self.assertEqual(coarse[0, 0], grid[0, 0])
        self.assertEqual(coarse[0, -1], grid[0, -1])
        self.assertEqual(coarse[-1, 0], grid[-1, 0])
        self.assertEqual(coarse[-1, -1], grid[-1, -1])

    def test_lowpass_beats_naive_decimation_on_band_limited_signal(self):
        n = 129
        axis = np.linspace(0, 4 * np.pi, n, dtype=np.float32)
        xx, yy = np.meshgrid(axis, axis)
        signal = np.sin(xx) * np.cos(yy).astype(np.float32)
        filtered = downsample_heightfield(signal, self.budget)
        naive = signal[::2, ::2]
        # Filtering must reduce high-frequency energy vs naive 2x decimation.
        energy_filtered = float(np.var(filtered[1:-1, 1:-1]))
        energy_naive = float(np.var(naive[1:-1, 1:-1]))
        self.assertLessEqual(energy_filtered, energy_naive + 1e-6)

    def test_blocked_matches_single_shot(self):
        grid = paraboloid(129)
        blocked = downsample_heightfield(grid, self.budget, block_rows=8)
        whole = downsample_heightfield(grid, self.budget, block_rows=64)
        np.testing.assert_allclose(blocked, whole, rtol=1e-6, atol=1e-6)

    def test_constant_field_invariant(self):
        grid = np.full((65, 65), 3.5, dtype=np.float32)
        coarse = downsample_heightfield(grid, self.budget)
        np.testing.assert_allclose(coarse, 3.5, atol=1e-6)


class MeshTests(unittest.TestCase):
    def setUp(self):
        self.budget = MemoryBudget(64 * 1024 * 1024)
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_mesh_layout_counts_and_uvs(self):
        n = 33
        grid = paraboloid(n)
        path = os.path.join(self.tmp.name, "lod0.bin")
        info = write_lod_mesh(
            path, grid, level=0, spacing_texels=1, world_size=100.0,
            budget=self.budget, relative_path="meshes/lod0.bin",
        )
        size = os.path.getsize(path)
        self.assertEqual(info.vertex_count, n * n)
        self.assertEqual(info.triangle_count, (n - 1) * (n - 1) * 2)
        self.assertEqual(
            size,
            info.index_offset + info.index_count * INDEX_DTYPE.itemsize,
        )
        vertices = np.fromfile(path, dtype=VERTEX_DTYPE, count=info.vertex_count)
        indices = np.fromfile(
            path, dtype=INDEX_DTYPE, count=info.index_count,
            offset=info.index_offset,
        )
        self.assertEqual(indices.max(), n * n - 1)
        self.assertEqual(indices.min(), 0)
        uv = vertices["uv"]
        self.assertAlmostEqual(uv.min(), 0.0, places=6)
        self.assertAlmostEqual(uv.max(), 1.0, places=6)
        # Corner UVs map to the full texture domain.
        self.assertEqual(tuple(uv[0]), (0.0, 0.0))
        self.assertEqual(tuple(uv[-1]), (1.0, 1.0))
        normals = vertices["normal"]
        np.testing.assert_allclose(
            np.linalg.norm(normals, axis=1), 1.0, atol=1e-6
        )

    def test_mesh_normals_match_recomputed(self):
        n = 33
        grid = paraboloid(n)
        spacing_world = 100.0 / (n - 1)
        path = os.path.join(self.tmp.name, "lod.bin")
        info = write_lod_mesh(
            path, grid, level=0, spacing_texels=1, world_size=100.0,
            budget=self.budget, relative_path="meshes/lod.bin",
        )
        vertices = np.fromfile(
            path, dtype=VERTEX_DTYPE, count=info.vertex_count
        )
        expected = tangent_normals_from_height(grid, spacing_world).reshape(-1, 3)
        np.testing.assert_allclose(vertices["normal"], expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
