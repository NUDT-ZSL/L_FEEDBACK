import unittest

import numpy as np

from adaptive_mixer.limiter import (
    TRUE_PEAK_CEILING_DBTP, limit, true_peak, true_peak_db,
)


class LimiterTests(unittest.TestCase):
    def test_quiet_signal_passes_through(self):
        rng = np.random.default_rng(3)
        signal = rng.standard_normal(48000) * 0.1
        result = limit(signal)
        self.assertFalse(result.limited)
        self.assertEqual(result.gain_reduction_db, 0.0)
        np.testing.assert_allclose(result.output, signal, atol=1e-6)

    def test_hot_signal_respects_ceiling(self):
        rng = np.random.default_rng(4)
        signal = rng.standard_normal(48000 * 2)
        signal *= 1.5 / np.max(np.abs(signal))
        result = limit(signal)
        self.assertTrue(result.limited)
        self.assertLessEqual(result.true_peak_db,
                             TRUE_PEAK_CEILING_DBTP + 1e-6)
        self.assertLess(true_peak(result.output), 10.0 ** (-1.0 / 20.0))

    def test_inter_sample_peak_detection(self):
        # A 6 kHz full-scale square (period 8 @48k) has Gibbs inter-sample
        # overshoot: the 4x measurement must exceed the 0 dBFS sample peak.
        period = 8
        square = np.where((np.arange(4096) % period) < period // 2, 1.0, -1.0)
        self.assertGreater(true_peak(square), 1.0)
        self.assertGreaterEqual(true_peak_db(square), 0.5)

    def test_length_preserved(self):
        signal = np.sin(np.linspace(0, 40 * np.pi, 12345)) * 2.0
        result = limit(signal)
        self.assertEqual(len(result.output), len(signal))


if __name__ == "__main__":
    unittest.main()
