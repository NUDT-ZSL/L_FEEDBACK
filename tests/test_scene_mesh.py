from __future__ import annotations

import unittest

import numpy as np

from proc_gi.scene_generator import generate_scene, point_in_polygon


class SceneMeshTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config, self.mesh = generate_scene()

    def test_deterministic_generation(self) -> None:
        _, regenerated = generate_scene()
        self.assertEqual(self.mesh.checksum(), regenerated.checksum())
        self.assertEqual(self.mesh.triangle_count, 520)
        self.assertEqual(len(self.mesh.groups), 5)

    def test_material_groups_are_contiguous_and_batched(self) -> None:
        expected_offset = 0
        expected_vertex_offset = 0
        for group in self.mesh.groups:
            self.assertEqual(group.index_offset, expected_offset)
            self.assertEqual(group.vertex_offset, expected_vertex_offset)
            self.assertEqual(group.index_count % 3, 0)
            group_indices = self.mesh.indices[
                group.index_offset : group.index_offset + group.index_count
            ]
            group_materials = self.mesh.vertices[group_indices, 6]
            np.testing.assert_array_equal(group_materials, group.material_index)
            expected_offset += group.index_count
            expected_vertex_offset += group.vertex_count
        self.assertEqual(expected_offset, len(self.mesh.indices))
        self.assertEqual(expected_vertex_offset, len(self.mesh.vertices))

    def test_triangle_winding_matches_vertex_normals(self) -> None:
        triangles = self.mesh.triangles()
        vertex_normals = self.mesh.vertices[self.mesh.indices, 3:6].reshape(-1, 3, 3)
        geometric = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        dots = np.einsum("ij,ij->i", geometric, vertex_normals.mean(axis=1))
        self.assertTrue(np.all(dots > 0.0))
        self.assertFalse(np.any(np.linalg.norm(geometric, axis=1) < 1.0e-10))

    def test_all_scene_materials_are_used(self) -> None:
        used = {group.material_index for group in self.mesh.groups}
        self.assertEqual(used, {material.id for material in self.config.materials})

    def test_all_source_lights_are_inside_room_polygon(self) -> None:
        polygon = np.asarray(self.config.room["room_polygon"], dtype=np.float64)
        for light in self.config.lights:
            with self.subTest(light=light.id):
                self.assertTrue(point_in_polygon((light.position[0], light.position[2]), polygon))


if __name__ == "__main__":
    unittest.main()

