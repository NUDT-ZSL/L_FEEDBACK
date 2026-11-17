from __future__ import annotations

import math
import unittest

from lrt.geometry import AABB, Plane, Sphere, Triangle


class DummyMaterial:
    pass


class GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.material = DummyMaterial()

    def test_sphere_intersection(self) -> None:
        sphere = Sphere((0.0, 0.0, 0.0), 1.0, self.material)
        hit = sphere.intersect((0.0, 0.0, 3.0), (0.0, 0.0, -1.0), 1.0e-9, math.inf)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertAlmostEqual(hit.t, 2.0)
        self.assertEqual(hit.normal, (0.0, 0.0, 1.0))
        self.assertTrue(hit.front)

    def test_plane_intersection(self) -> None:
        plane = Plane((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), self.material)
        hit = plane.intersect((0.0, 2.0, 0.0), (0.0, -1.0, 0.0), 1.0e-9, math.inf)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertAlmostEqual(hit.t, 2.0)
        self.assertTrue(hit.front)

    def test_aabb_intersection(self) -> None:
        box = AABB((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), self.material)
        hit = box.intersect((0.0, 0.0, 3.0), (0.0, 0.0, -1.0), 1.0e-9, math.inf)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertAlmostEqual(hit.t, 2.0)
        self.assertEqual(hit.normal, (0.0, 0.0, 1.0))

    def test_aabb_ray_starting_inside_hits_exit(self) -> None:
        box = AABB((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), self.material)
        hit = box.intersect((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0e-9, math.inf)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertAlmostEqual(hit.t, 1.0)
        self.assertEqual(hit.normal, (1.0, 0.0, 0.0))
        self.assertFalse(hit.front)

    def test_triangle_intersection(self) -> None:
        triangle = Triangle(
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
            self.material,
        )
        hit = triangle.intersect((0.0, 0.0, 1.0), (0.0, 0.0, -1.0), 1.0e-9, math.inf)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertAlmostEqual(hit.t, 1.0)
        self.assertAlmostEqual(abs(hit.normal[2]), 1.0)


if __name__ == "__main__":
    unittest.main()

