import unittest

import numpy as np

from offline_audio_render.channels import adapt_block


class ChannelAdaptationTests(unittest.TestCase):
    def test_mono_spatial_expansion(self):
        block = np.ones((4, 1), dtype=np.float64)
        gains = np.array([0.6, 0.8])
        out, warning = adapt_block(block, 1, "stereo", gains)
        np.testing.assert_allclose(out, np.tile(gains, (4, 1)))
        self.assertIsNone(warning)

    def test_surround_to_stereo_downmix_warning(self):
        block = np.eye(6, dtype=np.float64)
        gains = np.ones(2)
        out, warning = adapt_block(block, 6, "stereo", gains)
        self.assertEqual(out.shape, (6, 2))
        self.assertEqual(warning, "W2004")
        self.assertAlmostEqual(out[0, 0], 1.0)
        self.assertAlmostEqual(out[1, 1], 1.0)

    def test_surround_to_surround_drops_lfe(self):
        block = np.ones((2, 6), dtype=np.float64)
        gains = np.ones(6)
        out, warning = adapt_block(block, 6, "5.1", gains)
        self.assertEqual(out.shape, (2, 6))
        np.testing.assert_array_equal(out[:, 3], 0.0)
        self.assertEqual(warning, "W2004")

    def test_stereo_to_surround_expansion(self):
        block = np.ones((3, 2), dtype=np.float64)
        gains = np.ones(6)
        out, warning = adapt_block(block, 2, "5.1", gains)
        self.assertEqual(out.shape, (3, 6))
        np.testing.assert_array_equal(out[:, 2:], 0.0)
        self.assertIsNone(warning)


if __name__ == "__main__":
    unittest.main()
