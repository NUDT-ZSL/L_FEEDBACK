from __future__ import annotations

import unittest

import numpy as np

from bake_tool.dilation import dilate_maps
from bake_tool.mesh import Mesh
from bake_tool.rasterizer import rasterize_texture_space
from bake_tool.tbn import build_vertex_tbn


def _quad(normals_z: float = 1.0) -> Mesh:
    positions = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ]
    )
    normals = np.tile(np.asarray([[0.0, 0.0, normals_z]]), (4, 1))
    return Mesh(
        positions=positions,
        normals=normals,
        uvs=np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        triangles=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        albedo=np.tile(np.asarray([[0.8, 0.5, 0.2]]), (4, 1)),
        roughness=np.asarray([0.4, 0.4, 0.4, 0.4]),
        metallic=np.asarray([0.0, 0.0, 0.0, 0.0]),
        name="quad",
    )


class RasterizerTests(unittest.TestCase):
    def test_boundary_clipping_pixel_centers_and_single_owner(self) -> None:
        mesh = _quad()
        result = rasterize_texture_space(mesh, resolution=32)
        self.assertEqual(int(result.coverage_mask.sum()), 32 * 32)
        self.assertGreaterEqual(result.clipped_triangles, 1)
        self.assertEqual(result.rasterized_triangles, 2)
        self.assertTrue(np.all(result.triangle_index >= 0))
        self.assertEqual(np.unique(result.triangle_index).size, 2)
        barycentric_sum = result.barycentric.sum(axis=2)[result.coverage_mask]
        self.assertTrue(np.allclose(barycentric_sum, 1.0))

    def test_backface_culling(self) -> None:
        mesh = _quad(normals_z=-1.0)
        result = rasterize_texture_space(mesh, resolution=32)
        self.assertEqual(int(result.coverage_mask.sum()), 0)
        self.assertEqual(result.culled_triangles, 2)
        self.assertEqual(result.rasterized_triangles, 0)


class TBNTests(unittest.TestCase):
    def test_mirrored_uv_degenerate_uv_and_stable_fallback(self) -> None:
        positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        normals = np.tile(np.asarray([[0.0, 0.0, 1.0]]), (6, 1))
        uvs = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.25, 0.25],
                [0.25, 0.25],
                [0.25, 0.25],
            ]
        )
        mesh = Mesh(
            positions=positions,
            normals=normals,
            uvs=uvs,
            triangles=np.asarray([[0, 2, 1], [3, 4, 5]], dtype=np.int32),
            albedo=np.ones((6, 3)),
            roughness=np.ones(6) * 0.5,
            metallic=np.zeros(6),
        )
        result = build_vertex_tbn(mesh)
        self.assertEqual(result.mirrored_triangles, 1)
        self.assertEqual(result.degenerate_triangles, 1)
        self.assertEqual(result.fallback_vertices, 3)
        self.assertTrue(
            np.allclose(np.linalg.norm(result.tangents, axis=1), 1.0)
        )
        self.assertTrue(
            np.allclose(
                np.einsum("ij,ij->i", result.tangents, result.normals), 0.0
            )
        )
        self.assertTrue(
            np.allclose(
                result.bitangents,
                np.cross(result.normals, result.tangents),
            )
        )


class DilationTests(unittest.TestCase):
    def test_dilation_does_not_overwrite_other_island(self) -> None:
        coverage = np.zeros((5, 9), dtype=bool)
        coverage[2, 1] = True
        coverage[2, 7] = True
        maps = np.zeros((5, 9, 1), dtype=np.uint8)
        maps[2, 1, 0] = 10
        maps[2, 7, 0] = 30
        result = dilate_maps(maps, coverage, radius=4)

        self.assertEqual(int(result.coverage_after[2, 1]), 1)
        self.assertEqual(int(result.coverage_after[2, 7]), 1)
        self.assertEqual(int(result.maps[2, 1, 0]), 10)
        self.assertEqual(int(result.maps[2, 7, 0]), 30)
        self.assertGreater(result.dilated_pixels, 0)
        self.assertEqual(result.seam_pixels, 2)
        self.assertFalse(np.any(result.maps[~result.coverage_after]))

        untouched_valid = coverage & (result.maps[..., 0] != 10) & (
            result.maps[..., 0] != 30
        )
        self.assertFalse(np.any(untouched_valid))


if __name__ == "__main__":
    unittest.main()

