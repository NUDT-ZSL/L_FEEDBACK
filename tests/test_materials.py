from __future__ import annotations

import math
import unittest

from lrt.materials import (
    BaseMaterial,
    Coating,
    Material,
    calibration_report,
    effective_layer_reflectance,
    fresnel_dielectric,
    refract,
)


class MaterialTests(unittest.TestCase):
    def test_fresnel_bounds(self) -> None:
        for cos_i in (0.001, 0.1, 0.5, 1.0):
            value = fresnel_dielectric(cos_i, 1.0, 1.5)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertLess(fresnel_dielectric(1.0, 1.0, 1.5), fresnel_dielectric(0.1, 1.0, 1.5))

    def test_energy_conserving_composition(self) -> None:
        material = Material(
            "metal",
            "layered",
            BaseMaterial("metal", (0.95, 0.72, 0.30), 0.05),
            Coating(1.52, 0.002, (1.0, 0.98, 0.95), (0.1, 0.2, 0.3)),
        )
        for angle in range(0, 90, 2):
            cos_i = max(1.0e-5, math.cos(math.radians(angle)))
            reflectance = effective_layer_reflectance(material, cos_i)
            self.assertTrue(all(0.0 <= channel <= 1.0 for channel in reflectance))

    def test_refraction_reverses_across_parallel_interface(self) -> None:
        incident = (-0.2, -0.95, -0.1)
        magnitude = math.sqrt(sum(channel * channel for channel in incident))
        incident = tuple(channel / magnitude for channel in incident)
        normal = (0.0, 1.0, 0.0)
        entered, inside = refract(incident, normal, 1.0, 1.5)
        self.assertTrue(entered)
        exited, outside = refract(inside, normal, 1.5, 1.0)
        self.assertTrue(exited)
        for expected, actual in zip(incident, outside):
            self.assertAlmostEqual(expected, actual, places=12)

    def test_calibration_report(self) -> None:
        material = Material(
            "diffuse",
            "layered",
            BaseMaterial("diffuse", (0.4, 0.5, 0.6)),
            Coating(1.4, 0.001),
        )
        report = calibration_report([material])
        self.assertTrue(report["all_in_range"])
        self.assertLessEqual(report["maximum_reflectance"], 1.0)
        self.assertEqual(report["warnings"], [])

    def test_calibration_warns_instead_of_clamping(self) -> None:
        material = Material(
            "invalid_energy",
            "layered",
            BaseMaterial("diffuse", (1.2, 0.5, 0.5)),
            Coating(1.5, 0.001),
        )
        report = calibration_report([material])
        self.assertFalse(report["all_in_range"])
        self.assertGreater(report["maximum_reflectance"], 1.0)
        self.assertTrue(report["warnings"])


if __name__ == "__main__":
    unittest.main()

