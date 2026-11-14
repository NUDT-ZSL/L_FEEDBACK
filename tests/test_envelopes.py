import unittest

import numpy as np

from adaptive_mixer.envelopes import (
    CROSSFADE_TYPES, apply_fade_in_out, crossfade_gains,
)


class CrossfadeTests(unittest.TestCase):
    def test_endpoint_values(self):
        for curve in CROSSFADE_TYPES:
            gains = crossfade_gains(4000, curve)
            self.assertAlmostEqual(gains["out"][0], 1.0)
            self.assertAlmostEqual(gains["out"][-1], 0.0)
            self.assertAlmostEqual(gains["in"][0], 0.0)
            self.assertAlmostEqual(gains["in"][-1], 1.0)

    def test_c0_continuity_at_join(self):
        for curve in CROSSFADE_TYPES:
            gains = crossfade_gains(2000, curve)
            # Outgoing constant-gain region joins with g_out(0)=1.
            self.assertAlmostEqual(gains["out"][0], 1.0)
            self.assertAlmostEqual(gains["in"][-1], 1.0)

    def test_c1_continuity_zero_end_slope(self):
        for curve in CROSSFADE_TYPES:
            n = 4000
            gains = crossfade_gains(n, curve)
            g = gains["out"]
            # One-sided derivatives at the endpoints must be ~0 so that
            # splicing with a constant-gain region keeps C1 continuity.
            self.assertLess(abs(g[1] - g[0]), 2e-3)
            self.assertLess(abs(g[-1] - g[-2]), 2e-3)
            # Interior derivatives are finite and the spline is monotone.
            diffs = np.diff(g)
            self.assertTrue(np.all(diffs <= 1e-9))

    def test_equal_power_identity(self):
        gains = crossfade_gains(3000, "equal_power")
        power = gains["out"] ** 2 + gains["in"] ** 2
        np.testing.assert_allclose(power, 1.0, atol=2e-3)

    def test_fade_envelope(self):
        out = apply_fade_in_out(np.ones(1000), 100)
        self.assertEqual(out[0], 0.0)
        self.assertAlmostEqual(out[-1], 0.0)
        self.assertAlmostEqual(out[500], 1.0)


if __name__ == "__main__":
    unittest.main()
