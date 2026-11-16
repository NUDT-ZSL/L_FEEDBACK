import sys
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mediamixer.audio import (
    apply_envelope,
    mix_tracks,
    offset_to_samples,
    place_track,
    total_video_samples,
)
from mediamixer.ffmpeg_utils import frame_timestamp_us
from mediamixer.project import Marker, Track
from mediamixer.video import marker_color, marker_rect


def make_track(index: int, offset_ms: int = 0, envelope=(1.0, 1.0)) -> Track:
    return Track(
        index=index,
        path=Path(f"t{index}.wav"),
        offset_ms=offset_ms,
        envelope=tuple(envelope),
    )


class TimestampTest(unittest.TestCase):
    def test_integer_timestamp_no_float_drift(self):
        rate = Fraction(30, 1)
        self.assertEqual(frame_timestamp_us(0, rate), 0)
        self.assertEqual(frame_timestamp_us(1, rate), 33333)
        self.assertEqual(frame_timestamp_us(30, rate), 1_000_000)

        odd_rate = Fraction(1001, 1000 * 30)  # 29.97 fps style rational
        expected_second = 30 * 1_000_000 * odd_rate.denominator // odd_rate.numerator
        self.assertEqual(frame_timestamp_us(30, odd_rate), expected_second)

    def test_total_samples_integer(self):
        self.assertEqual(total_video_samples(120, Fraction(30, 1)), 48000 * 4)


class MarkerGeometryTest(unittest.TestCase):
    def test_color_is_deterministic(self):
        marker = Marker(frame=10, drift_ms=3)
        self.assertEqual(marker_color(marker), marker_color(marker))
        self.assertTrue(all(0 <= c <= 255 for c in marker_color(marker)))

    def test_rect_shifts_two_pixels_per_ms(self):
        width, height = 320, 240
        center = Marker(frame=0, drift_ms=0)
        shifted = Marker(frame=0, drift_ms=5)
        x0_center, _, _, _ = marker_rect(center, width, height)
        x0_shift, _, _, _ = marker_rect(shifted, width, height)
        self.assertEqual(x0_shift - x0_center, 10)

    def test_rect_clamped_to_borders(self):
        # Narrow frame so the offset rectangle actually reaches the borders.
        width, height = 60, 240
        far_right = Marker(frame=0, drift_ms=10)
        far_left = Marker(frame=0, drift_ms=-10)
        x0, _, x1, _ = marker_rect(far_right, width, height)
        self.assertEqual(x1, width)
        x0, _, x1, _ = marker_rect(far_left, width, height)
        self.assertEqual(x0, 0)


class AudioPlacementTest(unittest.TestCase):
    def test_positive_offset_inserts_silence(self):
        samples = np.ones((10, 2), dtype=np.float32)
        placed = place_track(samples, 10, 100)  # 10 ms = 480 samples
        shift = offset_to_samples(10)
        self.assertEqual(shift, 480)
        self.assertTrue(np.all(placed[:shift] == 0))
        self.assertTrue(np.all(placed[shift:shift + 10] == 1))

    def test_negative_offset_trims_head(self):
        samples = np.arange(1000, dtype=np.float32).reshape(-1, 1)
        samples = np.repeat(samples, 2, axis=1)
        placed = place_track(samples, -10, 1000)
        shift = offset_to_samples(-10)
        # First 480 source samples are dropped; placement starts at 0.
        self.assertTrue(np.all(placed[0] == samples[480]))

    def test_tail_beyond_video_is_truncated(self):
        samples = np.ones((1000, 2), dtype=np.float32)
        placed = place_track(samples, 0, 200)
        self.assertEqual(placed.shape, (200, 2))


class EnvelopeAndMixTest(unittest.TestCase):
    def test_envelope_endpoints_and_interpolation(self):
        samples = np.ones((5, 2), dtype=np.float32)
        shaped = apply_envelope(samples, (0.0, 1.0))
        np.testing.assert_allclose(shaped[0, 0], 0.0)
        np.testing.assert_allclose(shaped[-1, 0], 1.0)
        np.testing.assert_allclose(shaped[2, 0], 0.5)

    def test_peak_normalization_scales_whole_mix(self):
        tracks = (make_track(0), make_track(1))
        decoded = [
            np.ones((10, 2), dtype=np.float32) * 0.8,
            np.ones((10, 2), dtype=np.float32) * 0.8,
        ]
        mix, scale = mix_tracks(tracks, decoded, 10)
        self.assertLess(scale, 1.0)
        self.assertAlmostEqual(float(np.max(np.abs(mix))), 0.99, places=5)

    def test_no_scaling_when_under_peak(self):
        tracks = (make_track(0),)
        decoded = [np.ones((10, 2), dtype=np.float32) * 0.4]
        _, scale = mix_tracks(tracks, decoded, 10)
        self.assertEqual(scale, 1.0)


if __name__ == "__main__":
    unittest.main()
