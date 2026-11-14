from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dofpipe.aperture import KernelSpec, build_aperture_kernel
from dofpipe.config import LensConfig
from dofpipe.depth import parse_depth
from dofpipe.errors import ConfigurationError, FocusTrackError
from dofpipe.focus import (
    FocusTrack,
    Keyframe,
    load_focus_track,
    pchip_interpolate,
)
from dofpipe.lens import ThinLensModel


def make_lens(**overrides: float) -> LensConfig:
    values = {
        "focal_length_mm": 50.0,
        "f_number": 1.4,
        "sensor_width_mm": 36.0,
        "near_plane_m": 0.5,
        "far_plane_m": 20.0,
        "max_blur_radius_px": 8.0,
        "invalid_depth_fallback_m": 20.0,
        "depth_tolerance_m": 0.001,
    }
    values.update(overrides)
    return LensConfig(**values)


class FocusInterpolationTests(unittest.TestCase):
    def test_pchip_endpoints_and_no_overshoot(self) -> None:
        times = np.array([0.0, 0.3, 0.7, 1.0, 1.4])
        values = np.array([8.0, 2.5, 6.0, 3.0, 9.0])
        queries = np.linspace(times[0], times[-1], 1001)
        sampled = pchip_interpolate(times, values, queries)

        self.assertEqual(float(sampled[0]), values[0])
        self.assertEqual(float(sampled[-1]), values[-1])
        for segment in range(len(times) - 1):
            mask = (queries >= times[segment]) & (queries <= times[segment + 1])
            lower = min(values[segment], values[segment + 1])
            upper = max(values[segment], values[segment + 1])
            self.assertGreaterEqual(float(np.min(sampled[mask])), lower - 1e-12)
            self.assertLessEqual(float(np.max(sampled[mask])), upper + 1e-12)

    def test_keyframe_values_exact(self) -> None:
        times = np.array([0.0, 0.25, 0.5])
        values = np.array([1.8, 3.8, 9.0])
        for time, value in zip(times, values):
            self.assertEqual(
                float(pchip_interpolate(times, values, np.asarray(time))), value
            )

    def test_short_track_clamps_and_warns(self) -> None:
        track = FocusTrack(
            [
                Keyframe(0.0, 2.0),
                Keyframe(0.25, 4.0),
                Keyframe(0.5, 8.0),
            ],
            fps=12.0,
            total_frames=8,
        )
        self.assertEqual(track.sample(6), 8.0)
        self.assertEqual(track.sample(7), 8.0)
        self.assertEqual(
            [warning.code for warning in track.warnings],
            ["track_ends_before_sequence"],
        )

    def test_track_starting_late_uses_first_value(self) -> None:
        track = FocusTrack(
            [Keyframe(0.25, 2.0), Keyframe(0.5, 8.0)],
            fps=12.0,
            total_frames=8,
        )
        self.assertEqual(track.sample(0), 2.0)
        self.assertEqual(track.sample(2), 2.0)
        self.assertIn(
            "track_starts_after_sequence",
            [warning.code for warning in track.warnings],
        )

    def test_out_of_order_is_frame_indexed(self) -> None:
        track = FocusTrack(
            [
                Keyframe(0.0, 2.0),
                Keyframe(0.5, 8.0),
                Keyframe(0.25, 4.0),
            ],
            fps=12.0,
            total_frames=8,
        )
        self.assertGreater(track.sample(2), 2.0)
        self.assertLess(track.sample(2), 8.0)
        with self.assertRaises(FocusTrackError) as caught:
            track.sample(3)
        self.assertEqual(caught.exception.frame_index, 3)
        self.assertIn("not strictly after", caught.exception.reason)

    def test_future_timestamp_fails_after_frames(self) -> None:
        track = FocusTrack(
            [Keyframe(0.0, 2.0), Keyframe(0.75, 6.0)],
            fps=12.0,
            total_frames=8,
        )
        self.assertTrue(math.isfinite(track.sample(7)))
        with self.assertRaises(FocusTrackError) as caught:
            track.validate_after_frames()
        self.assertEqual(caught.exception.frame_index, 8)
        self.assertIn("outside sequence range", caught.exception.reason)

    def test_load_json_and_frame_time_unit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "track.json"
            path.write_text(
                json.dumps(
                    {
                        "time_unit": "frames",
                        "keyframes": [
                            {"time": 0, "focus_distance": 2.0},
                            {"time": 6, "focus_distance": 7.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            keyframes = load_focus_track(path, 12.0)
            self.assertEqual(keyframes[1].time_s, 0.5)


class DepthAndLensTests(unittest.TestCase):
    def test_depth_mapping_and_fallbacks(self) -> None:
        lens = make_lens()
        raw = np.array(
            [[1e-12, 0.5], [np.nan, 1.5], [np.inf, -1.0]], dtype=np.float32
        )
        parsed = parse_depth(raw, lens)
        self.assertEqual(parsed.invalid_count, 4)
        self.assertEqual(parsed.clamped_count, 1)
        self.assertEqual(float(parsed.meters[1, 0]), 20.0)
        self.assertAlmostEqual(
            float(parsed.meters[0, 1]), (lens.near_plane_m + lens.far_plane_m) / 2
        )
        self.assertEqual(float(parsed.meters[1, 1]), lens.far_plane_m)
        self.assertTrue(np.all(np.isfinite(parsed.meters)))
        self.assertTrue(np.all(parsed.meters > 0.0))

    def test_coc_zero_at_focus_and_clamped(self) -> None:
        model = ThinLensModel(make_lens(max_blur_radius_px=3.0), image_width=200)
        depth = np.array([1.0, 2.0, 3.0, 10.0, 20.0], dtype=np.float64)
        radius = model.coc_radius_px(depth, 2.0)
        self.assertEqual(float(radius[1]), 0.0)
        self.assertTrue(np.all(radius >= 0.0))
        self.assertTrue(np.all(radius <= 3.0))
        self.assertGreater(float(radius[-1]), float(radius[2]))

    def test_invalid_focus_raises_instead_of_dividing_by_zero(self) -> None:
        model = ThinLensModel(make_lens(), image_width=200)
        with self.assertRaises(ConfigurationError):
            model.coc_radius_px(np.array([2.0]), 0.05)
        with self.assertRaises(ConfigurationError):
            model.coc_radius_px(np.array([0.0]), 2.0)

    def test_extreme_near_focus_saturates_without_overflow(self) -> None:
        model = ThinLensModel(make_lens(max_blur_radius_px=3.0), image_width=200)
        radius = model.coc_radius_px(np.array([2.0]), 0.050000000001)
        self.assertTrue(np.all(np.isfinite(radius)))
        self.assertEqual(float(radius[0]), 3.0)


class ApertureKernelTests(unittest.TestCase):
    def test_circle_is_isotropic_and_normalized(self) -> None:
        kernel = build_aperture_kernel(KernelSpec(blades=0, radius_px=8.0))
        self.assertEqual(kernel.weights.shape, (17, 17))
        self.assertAlmostEqual(float(np.sum(kernel.weights)), 1.0, places=12)
        self.assertLess(float(np.max(np.abs(kernel.weights - kernel.weights.T))), 1e-15)
        self.assertLess(
            float(np.max(np.abs(kernel.weights - kernel.weights[::-1, :]))), 1e-15
        )
        radial_pairs = [(2, 3), (3, 2), (1, 4), (4, 1), (5, 0), (0, 5)]
        center = kernel.half_size
        for dy, dx in radial_pairs:
            self.assertAlmostEqual(
                float(kernel.weights[center + dy, center + dx]),
                float(kernel.weights[center - dy, center - dx]),
                places=12,
            )

    def test_polygon_energy_matches_circle(self) -> None:
        circle = build_aperture_kernel(KernelSpec(blades=0, radius_px=8.0))
        polygon = build_aperture_kernel(KernelSpec(blades=6, radius_px=8.0))
        self.assertAlmostEqual(float(np.sum(polygon.weights)), 1.0, places=12)
        self.assertAlmostEqual(
            float(np.sum(polygon.weights)) / float(np.sum(circle.weights)),
            1.0,
            places=12,
        )
        self.assertEqual(polygon.weights.shape, circle.weights.shape)
        self.assertGreater(float(np.max(polygon.weights)), float(np.max(circle.weights)))

    def test_zero_radius_kernel(self) -> None:
        kernel = build_aperture_kernel(KernelSpec(blades=3, radius_px=0.0))
        self.assertEqual(kernel.weights.shape, (1, 1))
        self.assertEqual(float(kernel.weights[0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
