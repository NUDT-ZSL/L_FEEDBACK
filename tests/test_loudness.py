import unittest

import numpy as np

from audiopipeline.config import SAMPLE_RATE
from audiopipeline.loudness import (
    gain_to_reach_target,
    integrated_loudness,
    k_weight,
    true_peak_db,
)


def sine(freq=1000.0, seconds=5.0, amp=0.5, channels=2, sr=SAMPLE_RATE):
    t = np.arange(int(seconds * sr)) / sr
    sig = amp * np.sin(2 * np.pi * freq * t)
    return np.tile(sig[:, None], (1, channels)).astype(np.float64)


class LoudnessTests(unittest.TestCase):
    def test_reference_tone_matches_bs1770(self):
        # A 1 kHz, 0.5 amplitude stereo tone is -6.0 LUFS (FFmpeg reference).
        lufs = integrated_loudness(sine())
        self.assertAlmostEqual(lufs, -6.0, delta=0.2)

    def test_mono_three_lu_below_stereo(self):
        stereo = integrated_loudness(sine(channels=2))
        mono = integrated_loudness(sine(channels=1))
        self.assertAlmostEqual(stereo - mono, 3.01, delta=0.1)

    def test_silence_returns_none(self):
        silence = np.zeros((SAMPLE_RATE, 2), dtype=np.float64)
        self.assertIsNone(integrated_loudness(silence))
        self.assertIsNone(true_peak_db(silence))

    def test_short_event_fallback(self):
        # 200 ms clip is shorter than the 400 ms window but still measured.
        short = sine(seconds=0.2)
        lufs = integrated_loudness(short)
        self.assertIsNotNone(lufs)
        self.assertAlmostEqual(lufs, -6.0, delta=0.5)

    def test_k_weight_is_energy_stable(self):
        audio = sine(seconds=2.0)
        weighted = k_weight(audio)
        self.assertEqual(weighted.shape, audio.shape)
        self.assertTrue(np.isfinite(weighted).all())

    def test_true_peak_detects_intersample(self):
        # Two adjacent opposite full-scale-ish samples create an
        # inter-sample peak larger than the sample peak.
        audio = np.zeros((SAMPLE_RATE, 2), dtype=np.float64)
        audio[100, :] = 0.95
        audio[101, :] = -0.95
        tp = true_peak_db(audio, factor=4)
        self.assertGreater(tp, 20 * np.log10(0.95))

    def test_gain_clamping(self):
        gain = gain_to_reach_target(-40.0, -16.0, max_gain_db=10.0)
        self.assertAlmostEqual(gain, 10 ** (10 / 20), places=5)
        self.assertEqual(gain_to_reach_target(None, -16.0, 10.0), 1.0)


if __name__ == "__main__":
    unittest.main()
