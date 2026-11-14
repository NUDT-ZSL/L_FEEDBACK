import unittest

import numpy as np

from offline_audio_render.random_engine import (
    GAIN_LIMIT_DB,
    PITCH_LIMIT,
    START_LIMIT_S,
    derive_variants,
    event_seed,
)
from offline_audio_render.spatial import (
    ListenerPose,
    distance_gain,
    five_one_gains,
    relative_azimuth_distance,
    stereo_gains,
)
from offline_audio_render.trajectory import ListenerTrajectory, TrajectoryKey


class RandomEngineTests(unittest.TestCase):
    def test_seed_is_stable_and_event_isolated(self):
        first = event_seed(123, 4)
        second = event_seed(123, 4)
        other = event_seed(123, 5)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_variants_are_deterministic_and_in_range(self):
        a = derive_variants(99, 3)
        b = derive_variants(99, 3)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a.seed_value, b.seed_value)
        self.assertTrue(1.0 - PITCH_LIMIT <= a.pitch_factor <= 1.0 + PITCH_LIMIT)
        self.assertTrue(-START_LIMIT_S <= a.start_offset_s <= START_LIMIT_S)
        self.assertTrue(-GAIN_LIMIT_DB <= a.gain_jitter_db <= GAIN_LIMIT_DB)

    def test_distinct_offsets_distinct_streams(self):
        draws = {derive_variants(1, offset).seed_value for offset in range(10)}
        self.assertEqual(len(draws), 10)


class SpatialTests(unittest.TestCase):
    def test_azimuth_left_and_right(self):
        pose = ListenerPose(np.zeros(3), 0.0)
        left_az, _ = relative_azimuth_distance(np.array([0.0, 1.0, 0.0]), pose)
        right_az, _ = relative_azimuth_distance(np.array([0.0, -1.0, 0.0]), pose)
        self.assertLess(left_az, 0.0)
        self.assertGreater(right_az, 0.0)

    def test_stereo_equal_power(self):
        gains = stereo_gains(0.0)
        np.testing.assert_allclose(gains, np.array([np.sqrt(0.5), np.sqrt(0.5)]))
        np.testing.assert_allclose(np.sum(gains**2), 1.0)
        hard_left = stereo_gains(-np.pi / 2)
        np.testing.assert_allclose(hard_left, [1.0, 0.0], atol=1e-12)

    def test_five_one_power_and_lfe_silence(self):
        gains = five_one_gains(np.deg2rad(20.0))
        self.assertEqual(gains.shape, (6,))
        self.assertEqual(gains[3], 0.0)
        np.testing.assert_allclose(np.sum(gains**2), 1.0)

    def test_inverse_square_attenuation_and_min_distance(self):
        self.assertEqual(distance_gain(0.0, min_distance=1.0), 1.0)
        self.assertAlmostEqual(distance_gain(2.0, 1.0), 0.25)
        self.assertAlmostEqual(distance_gain(4.0, 2.0), 0.25)

    def test_trajectory_interpolation_and_endpoint_clamp(self):
        traj = ListenerTrajectory(
            [
                TrajectoryKey(0.0, (0.0, 0.0, 0.0), 0.0),
                TrajectoryKey(2.0, (2.0, 4.0, 0.0), 1.0),
            ]
        )
        middle = traj.pose_at(1.0)
        np.testing.assert_allclose(middle.position, [1.0, 2.0, 0.0])
        self.assertAlmostEqual(middle.yaw, 0.5)
        before = traj.pose_at(-1.0)
        np.testing.assert_allclose(before.position, [0.0, 0.0, 0.0])
        after = traj.pose_at(9.0)
        np.testing.assert_allclose(after.position, [2.0, 4.0, 0.0])


if __name__ == "__main__":
    unittest.main()
