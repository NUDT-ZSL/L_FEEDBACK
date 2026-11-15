import math
import unittest

from cinema_master.layouts import (
    default_roles_for_count,
    mapping_matrix,
    resolve_output_layout,
)


class LayoutTests(unittest.TestCase):
    def test_output_layouts_and_masks(self):
        name, roles, mask = resolve_output_layout("5.1")
        self.assertEqual(name, "5.1")
        self.assertEqual(roles, ["FL", "FR", "FC", "LFE", "BL", "BR"])
        self.assertEqual(mask, 0x3F)

        name, roles, mask = resolve_output_layout("stereo")
        self.assertEqual(name, "stereo")
        self.assertEqual(roles, ["FL", "FR"])
        self.assertEqual(mask, 0x3)

    def test_bs775_downmix_coefficients(self):
        source = ["FL", "FR", "FC", "LFE", "BL", "BR"]
        matrix, details = mapping_matrix(source, ["FL", "FR"], "stereo")
        coefficient = 1.0 / math.sqrt(2.0)
        self.assertAlmostEqual(matrix[0, 0], 1.0)
        self.assertAlmostEqual(matrix[1, 1], 1.0)
        self.assertAlmostEqual(matrix[2, 0], coefficient)
        self.assertAlmostEqual(matrix[2, 1], coefficient)
        self.assertAlmostEqual(matrix[3, 0], 0.0)
        self.assertAlmostEqual(matrix[3, 1], 0.0)
        self.assertAlmostEqual(matrix[4, 0], coefficient)
        self.assertAlmostEqual(matrix[4, 1], 0.0)
        self.assertAlmostEqual(matrix[5, 1], coefficient)
        self.assertEqual(details["downmix_standard"], "ITU-R BS.775")

    def test_51_missing_channels_stay_silent(self):
        matrix, details = mapping_matrix(["FC"], ["FL", "FR", "FC", "LFE", "BL", "BR"], "5.1")
        self.assertAlmostEqual(matrix[0, 2], 1.0)
        self.assertEqual(float(matrix.sum()), 1.0)
        self.assertEqual(details["dropped_roles"], [])

    def test_implicit_pcm_layouts(self):
        self.assertEqual(default_roles_for_count(2)[0], ["FL", "FR"])
        self.assertEqual(
            default_roles_for_count(6)[0],
            ["FL", "FR", "FC", "LFE", "BL", "BR"],
        )


if __name__ == "__main__":
    unittest.main()

