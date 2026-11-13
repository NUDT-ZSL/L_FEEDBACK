import unittest

import numpy as np

from audiopack.config import Config
from audiopack.normalize import normalize_track


class NormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config(target_loudness_db=-23.0)

    def test_window_length_is_100ms(self) -> None:
        self.assertEqual(self.config.rms_window_frames, 4800)

    def test_gain_within_bounds_hits_target(self) -> None:
        rng = np.random.default_rng(7)
        samples = (rng.normal(0, 0.2, 48000)).astype(np.float32)
        result = normalize_track(samples, self.config)
        self.assertFalse(result.gain_clamped)
        self.assertGreaterEqual(result.applied_gain_db, self.config.gain_min_db)
        self.assertLessEqual(result.applied_gain_db, self.config.gain_max_db)
        # Post-normalization energy of voiced windows should sit near target.
        self.assertTrue(
            abs(result.measured_loudness_db + result.applied_gain_db -
                self.config.target_loudness_db) < 0.01
        )

    def test_quiet_signal_clamps_at_plus_12db_and_warns(self) -> None:
        samples = np.full(48000, 0.003, dtype=np.float32)
        result = normalize_track(samples, self.config)
        self.assertTrue(result.gain_clamped)
        self.assertEqual(result.applied_gain_db, self.config.gain_max_db)
        self.assertGreater(result.requested_gain_db, self.config.gain_max_db)
        self.assertTrue(result.warnings)

    def test_loud_signal_clamps_at_minus_20db(self) -> None:
        samples = np.full(48000, 0.99, dtype=np.float32)
        result = normalize_track(samples, self.config)
        self.assertTrue(result.gain_clamped)
        self.assertEqual(result.applied_gain_db, self.config.gain_min_db)

    def test_silence_is_handled_neutrally(self) -> None:
        samples = np.zeros(48000, dtype=np.float32)
        result = normalize_track(samples, self.config)
        self.assertEqual(result.non_silent_windows, 0)
        self.assertEqual(result.applied_gain_db, 0.0)
        np.testing.assert_array_equal(result.samples, samples)

    def test_out_of_target_intervals_report_frame_ranges(self) -> None:
        samples = np.full(48000, 0.003, dtype=np.float32)
        result = normalize_track(samples, self.config)
        interval = result.warnings[0]
        self.assertEqual(interval.start_frame, 0)
        self.assertEqual(interval.end_frame, 48000)
        self.assertLess(interval.measured_db,
                        self.config.target_loudness_db -
                        self.config.loudness_tolerance_db)


if __name__ == "__main__":
    unittest.main()
