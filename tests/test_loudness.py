import unittest

import numpy as np

from adaptive_mixer.loudness import (
    integrated_loudness, k_weight, k_weighting_ir, lufs_to_linear_gain,
    short_term_loudness,
)


def sine(freq, seconds, sr=48000, amp=0.5):
    t = np.arange(int(seconds * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


class LoudnessTests(unittest.TestCase):
    def test_k_weighting_gain_at_1khz(self):
        ir = k_weighting_ir()
        out = k_weight(sine(1000.0, 2.0), ir)
        # Canonical BS.1770 K-weighting gain at 1 kHz is +0.691 dB.
        ratio = np.sqrt(np.mean(out ** 2) /
                        np.mean(sine(1000.0, 2.0) ** 2))
        self.assertAlmostEqual(20 * np.log10(ratio), 0.691, delta=0.05)

    def test_absolute_calibration_1khz(self):
        # -24 dBFS 1 kHz sine: RMS -27.02 dBFS, K gain +0.691 dB,
        # BS.1770 offset -0.691 dB -> -27.02 LUFS on the gated scale.
        tone = sine(1000.0, 4.0, amp=10.0 ** (-24.0 / 20.0))
        lufs = integrated_loudness(tone)
        rms_db = 20 * np.log10(np.sqrt(np.mean(tone ** 2)))
        self.assertAlmostEqual(lufs, rms_db, delta=0.1)

    def test_relative_loudness_matches_amplitude(self):
        loud = integrated_loudness(sine(440.0, 3.0, amp=0.5))
        quiet = integrated_loudness(sine(440.0, 3.0, amp=0.05))
        self.assertAlmostEqual(loud - quiet, 20.0, delta=0.35)

    def test_short_term_window_count(self):
        values = short_term_loudness(sine(300.0, 1.0))
        self.assertGreaterEqual(len(values), 6)
        self.assertTrue(np.all(np.isfinite(values)))

    def test_silence_is_negative_infinity(self):
        self.assertEqual(integrated_loudness(np.zeros(48000)),
                         float("-inf"))

    def test_gain_relation(self):
        linear, gain_db = lufs_to_linear_gain(-20.0, -23.0)
        self.assertAlmostEqual(gain_db, -3.0)
        self.assertAlmostEqual(linear, 10.0 ** (-3.0 / 20.0))


if __name__ == "__main__":
    unittest.main()
