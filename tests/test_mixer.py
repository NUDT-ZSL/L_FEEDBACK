import unittest

import numpy as np

from audio_sync.constants import FRAME_SAMPLES
from audio_sync.mixer import Mixer


class MixerTests(unittest.TestCase):
    def test_equal_power_two_active_streams(self):
        mixer = Mixer("equal_power")
        frames = [
            np.full(FRAME_SAMPLES, 0.4, dtype=np.float32),
            np.full(FRAME_SAMPLES, 0.4, dtype=np.float32),
        ]
        result = mixer.mix(frames)
        self.assertEqual(len(result.gains), 2)
        self.assertTrue(all(result.active))
        # 1/sqrt(2) per active stream.
        self.assertAlmostEqual(result.gains[0], 1 / np.sqrt(2), places=5)
        self.assertLessEqual(result.peak, 32767)

    def test_linear_mode_and_inactive_gating(self):
        mixer = Mixer("linear")
        active = np.full(FRAME_SAMPLES, 0.5, dtype=np.float32)
        silent = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        result = mixer.mix([active, silent])
        self.assertEqual(result.active, (True, False))
        self.assertEqual(result.gains[1], 0.0)
        self.assertAlmostEqual(result.gains[0], 0.5)
        self.assertAlmostEqual(float(result.mixed[0]), 0.25, places=5)

    def test_peak_is_int_and_bounded(self):
        mixer = Mixer()
        loud = [np.full(FRAME_SAMPLES, 1.0, dtype=np.float32) for _ in range(4)]
        result = mixer.mix(loud)
        self.assertIsInstance(result.peak, int)
        self.assertLessEqual(result.peak, 32767)

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            Mixer("cascade")


if __name__ == "__main__":
    unittest.main()
