import unittest

import numpy as np

from offline_audio_render.curves import (
    Curve,
    OnePoleLowPass,
    db_to_gain,
    lowpass_alpha,
)


class CurveFilterTests(unittest.TestCase):
    def test_piecewise_curve_endpoint_clamp_and_interpolation(self):
        curve = Curve(
            (
                (0.0, -12.0),
                (0.5, -6.0),
                (1.0, 0.0),
            )
        )
        values = curve.evaluate(np.array([-1.0, 0.25, 0.5, 1.5]))
        np.testing.assert_allclose(values, [-12.0, -9.0, -6.0, 0.0])

    def test_db_gain(self):
        np.testing.assert_allclose(db_to_gain(0.0), 1.0)
        np.testing.assert_allclose(db_to_gain(-6.0), 0.501187, rtol=1e-5)

    def test_lowpass_alpha_range(self):
        alpha = lowpass_alpha(np.array([1.0, 24000.0]), 48000)
        self.assertTrue(np.all(alpha > 0.0))
        self.assertTrue(np.all(alpha < 1.0))

    def test_filter_matches_direct_recurrence_and_keeps_state(self):
        sample_rate = 1000
        samples = np.random.default_rng(3).normal(size=1000)
        cutoff = np.linspace(50.0, 400.0, 1000)
        alpha = lowpass_alpha(cutoff, sample_rate)

        filter_state = OnePoleLowPass()
        output = np.concatenate(
            [
                filter_state.process(samples[:333], alpha[:333]),
                filter_state.process(samples[333:666], alpha[333:666]),
                filter_state.process(samples[666:], alpha[666:]),
            ]
        )

        reference = np.zeros(1000)
        state = 0.0
        for index in range(1000):
            state = alpha[index] * samples[index] + (1 - alpha[index]) * state
            reference[index] = state
        np.testing.assert_allclose(output, reference, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
