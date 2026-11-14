from __future__ import annotations

import math
import unittest

import numpy as np

from cinemix.config import LimiterSpec
from cinemix.dsp import true_peak
from cinemix.limiter import apply_limiter


class LimiterTests(unittest.TestCase):
    def test_limiter_prevents_true_peak_overshoot(self) -> None:
        sample_rate = 48000
        rng = np.random.default_rng(12345)
        time = np.arange(sample_rate * 2) / sample_rate
        tonal = (
            np.sin(2.0 * math.pi * 1000.0 * time)
            + 0.4 * np.sin(2.0 * math.pi * 7300.0 * time)
            + 0.2 * np.sin(2.0 * math.pi * 11300.0 * time)
        )
        noise = rng.standard_normal(time.size)
        monophonic = tonal + noise * 0.15
        monophonic *= 1.1 / float(np.max(np.abs(monophonic)))
        stereo = np.stack((monophonic, monophonic * 0.93), axis=1)
        spec = LimiterSpec(
            attack_ms=4.0,
            release_ms=100.0,
            ratio=20.0,
            lookahead_ms=6.0,
            compression_threshold_dbtp=None,
            safety_margin_db=0.25,
        )
        limited = apply_limiter(stereo, sample_rate, -1.0, spec)
        measured = true_peak(limited.samples, sample_rate)
        self.assertLessEqual(measured.dbtp, -1.0 + 1e-9)
        self.assertGreater(limited.max_reduction_db, 1.0)

    def test_quiet_signal_is_not_reduced(self) -> None:
        sample_rate = 48000
        time = np.arange(sample_rate) / sample_rate
        quiet = (np.sin(2.0 * math.pi * 440.0 * time) * 0.1)[:, None]
        spec = LimiterSpec(5.0, 120.0, 20.0, 7.0, None, 0.25)
        limited = apply_limiter(quiet, sample_rate, -1.0, spec)
        self.assertEqual(limited.max_reduction_db, 0.0)
        np.testing.assert_allclose(limited.samples, quiet)


if __name__ == "__main__":
    unittest.main()
