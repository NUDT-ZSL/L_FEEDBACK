import math
import unittest

import numpy as np

from audiomixer.dsp import (
    db_to_gain,
    equal_power_gain_curve,
    equal_power_pan,
    fade_curve,
    linear_gain_curve,
)
from audiomixer.generator import render_sine


class PanLawTests(unittest.TestCase):
    def test_constant_power(self):
        for pan in (-1.0, -0.5, 0.0, 0.25, 1.0):
            left, right = equal_power_pan(pan)
            self.assertAlmostEqual(left * left + right * right, 1.0, places=12)

    def test_endpoints_and_center(self):
        left_far, right_far = equal_power_pan(-1.0)
        self.assertAlmostEqual(left_far, 1.0, places=12)
        self.assertAlmostEqual(right_far, 0.0, places=12)
        left_far, right_far = equal_power_pan(1.0)
        self.assertAlmostEqual(left_far, 0.0, places=12)
        self.assertAlmostEqual(right_far, 1.0, places=12)
        left, right = equal_power_pan(0.0)
        self.assertAlmostEqual(left, math.cos(math.pi / 4), places=12)
        self.assertAlmostEqual(right, math.sin(math.pi / 4), places=12)


class FadeCurveTests(unittest.TestCase):
    def test_linear_endpoints(self):
        curve = linear_gain_curve(5, fade_in=True)
        np.testing.assert_allclose(curve, [0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(
            linear_gain_curve(5, fade_in=False), [1.0, 0.75, 0.5, 0.25, 0.0]
        )

    def test_equal_power_endpoints_and_power(self):
        curve = equal_power_gain_curve(100, fade_in=True)
        self.assertAlmostEqual(curve[0], 0.0, places=12)
        self.assertAlmostEqual(curve[-1], 1.0, places=12)
        # Crossfade partner curves must sum in power to one at every sample.
        fade_in = fade_curve(64, "equal_power", True)
        fade_out = fade_curve(64, "equal_power", False)
        np.testing.assert_allclose(fade_in**2 + fade_out**2, 1.0, atol=1e-12)


class GeneratorTests(unittest.TestCase):
    def test_deterministic_sample_identical(self):
        first = render_sine(48000, 1.0, 440.0, 0.5)
        second = render_sine(48000, 1.0, 440.0, 0.5)
        np.testing.assert_array_equal(first, second)

    def test_amplitude_and_length(self):
        sine = render_sine(44100, 0.5, 1000.0, 0.25, channels=2)
        self.assertEqual(sine.shape, (22050, 2))
        self.assertLessEqual(np.max(np.abs(sine)), 0.25 + 1e-12)
        np.testing.assert_array_equal(sine[:, 0], sine[:, 1])

    def test_starts_at_zero_phase(self):
        sine = render_sine(48000, 0.01, 440.0, 1.0)
        self.assertEqual(sine[0, 0], 0.0)

    def test_db_to_gain(self):
        self.assertAlmostEqual(db_to_gain(6.0), 1.9952623149688795, places=10)
        self.assertEqual(db_to_gain(0.0), 1.0)


if __name__ == "__main__":
    unittest.main()
