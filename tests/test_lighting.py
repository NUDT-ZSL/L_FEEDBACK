from __future__ import annotations

import unittest

import numpy as np

from proc_gi.clustering import VirtualPointLight
from proc_gi.config import Material
from proc_gi.lighting import accumulate_reference, evaluate_brdf


class LightingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Material(
            id=1,
            name="test",
            type="BlinnPhong",
            albedo=(0.52, 0.4, 0.3),
            roughness=0.35,
            specular=0.4,
            shininess=64.0,
        )
        self.vpl = VirtualPointLight(
            id="vpl-test",
            position=(0.0, 2.0, 0.0),
            color=(1.0, 0.8, 0.6),
            intensity=10.0,
            direction=(0.0, -1.0, 0.0),
            direction_range_deg=0.0,
            is_spot=False,
            range=5.0,
            radius=1.0,
            member_ids=("light",),
            source_count=1,
            signature="sig",
        )
        self.point = np.asarray((0.1, 0.72, 0.2))
        self.normal = np.asarray((0.0, 1.0, 0.0))
        self.view = np.asarray((0.0, 1.0, 0.5))

    def test_energy_linearity_and_additive_overlap(self) -> None:
        half_a = VirtualPointLight(
            **{**self.vpl.__dict__, "id": "a", "intensity": 5.0, "signature": "a"}
        )
        half_b = VirtualPointLight(
            **{**self.vpl.__dict__, "id": "b", "intensity": 5.0, "signature": "b"}
        )
        single = accumulate_reference([self.vpl], self.base, self.point, self.normal, self.view)
        split = accumulate_reference([half_a, half_b], self.base, self.point, self.normal, self.view)
        additive = accumulate_reference([self.vpl, half_a], self.base, self.point, self.normal, self.view)
        np.testing.assert_allclose(single, split, atol=1.0e-10)
        self.assertTrue(np.all(additive > single))
        np.testing.assert_allclose(additive, single + split * 0.5, atol=1.0e-10)

    def test_material_models_produce_distinct_responses(self) -> None:
        lambertian = Material(
            id=0,
            name="lambert",
            type="Lambertian",
            albedo=self.base.albedo,
            roughness=0.8,
        )
        oren = Material(
            id=1,
            name="oren",
            type="OrenNayar",
            albedo=self.base.albedo,
            roughness=0.9,
        )
        light_dir = np.asarray((0.4, 0.7, 0.2))
        responses = [
            evaluate_brdf(lambertian, self.normal, self.view, light_dir),
            evaluate_brdf(oren, self.normal, self.view, light_dir),
            evaluate_brdf(self.base, self.normal, self.view, light_dir),
        ]
        self.assertFalse(np.allclose(responses[0], responses[1]))
        self.assertFalse(np.allclose(responses[0], responses[2]))


if __name__ == "__main__":
    unittest.main()

