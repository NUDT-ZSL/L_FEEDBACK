from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dofpipe.config import ApertureConfig, LensConfig, RenderConfig, load_config
from dofpipe.depth import DepthFrame
from dofpipe.errors import FocusTrackError
from dofpipe.pipeline import run_pipeline, verify_slice_consistency
from dofpipe.render import DOFRenderer, image_difference_8bit
from dofpipe.sample import generate_sample


def _renderer(
    *,
    slices: int,
    rounds: int,
    blades: int = 0,
    max_radius: float = 6.0,
) -> DOFRenderer:
    lens = LensConfig(
        focal_length_mm=50.0,
        f_number=1.4,
        sensor_width_mm=36.0,
        near_plane_m=0.5,
        far_plane_m=20.0,
        max_blur_radius_px=max_radius,
        invalid_depth_fallback_m=20.0,
        depth_tolerance_m=0.001,
    )
    aperture = ApertureConfig(blades=blades, rotation_deg=0.0, supersample=8)
    render = RenderConfig(
        depth_slices=slices,
        gather_rounds=rounds,
        png_compression=3,
        output_pattern="frame_{frame:04d}.png",
    )
    return DOFRenderer(lens, aperture, render, width=64, height=36)


def _two_plane_scene() -> tuple[np.ndarray, DepthFrame]:
    color = np.zeros((36, 64, 3), dtype=np.uint8)
    color[:, :, 1] = 255
    color[:, :32] = np.array([0, 0, 255], dtype=np.uint8)
    depth = np.full((36, 64), 12.0, dtype=np.float64)
    depth[:, :32] = 1.0
    frame = DepthFrame(
        meters=depth,
        invalid_mask=np.zeros_like(depth, dtype=bool),
        clamped_mask=np.zeros_like(depth, dtype=bool),
    )
    return color, frame


class ScatterRendererTests(unittest.TestCase):
    def test_background_does_not_bleed_into_focused_foreground(self) -> None:
        color, depth = _two_plane_scene()
        for rounds in (1, 2, 4):
            renderer = _renderer(slices=4, rounds=rounds)
            result = renderer.render(color, depth, focus_m=1.0)
            foreground = result.image_bgr[:, :30]
            # Focused foreground has a zero-radius source kernel; a far source
            # aimed at the foreground is rejected by the destination-depth test.
            np.testing.assert_array_equal(
                foreground,
                np.broadcast_to(np.array([0, 0, 255], dtype=np.uint8), foreground.shape),
            )

    def test_foreground_scatter_reaches_background(self) -> None:
        color, depth = _two_plane_scene()
        renderer = _renderer(slices=4, rounds=2)
        result = renderer.render(color, depth, focus_m=12.0)
        # The closer red plane is defocused and spills across the depth edge.
        self.assertGreater(int(result.image_bgr[18, 32, 2]), 0)

    def test_round_count_does_not_invert_depth_order(self) -> None:
        color, depth = _two_plane_scene()
        images = [
            _renderer(slices=4, rounds=rounds).render(color, depth, focus_m=1.0).image_bgr
            for rounds in (1, 3, 6)
        ]
        self.assertLessEqual(
            image_difference_8bit(images[0], images[1])["max_abs_error_8bit"], 1.0
        )
        self.assertLessEqual(
            image_difference_8bit(images[0], images[2])["max_abs_error_8bit"], 1.0
        )

    def test_full_and_sliced_scatter_are_nearly_identical(self) -> None:
        color, depth = _two_plane_scene()
        full = _renderer(slices=1, rounds=2).render(color, depth, focus_m=12.0)
        sliced = _renderer(slices=16, rounds=2).render(color, depth, focus_m=12.0)
        difference = image_difference_8bit(full.image_bgr, sliced.image_bgr)
        self.assertLessEqual(difference["max_abs_error_8bit"], 1.0)


class PipelineArtifactTests(unittest.TestCase):
    def _generate(self, directory: Path, frames: int = 6) -> Path:
        generate_sample(directory, width=96, height=54, frame_count=frames, seed=77)
        return directory / "config.json"

    @staticmethod
    def _normalized_report(report: dict[str, object]) -> dict[str, object]:
        normalized = copy.deepcopy(report)
        frames = normalized["frames"]
        assert isinstance(frames, list)
        for frame in frames:
            assert isinstance(frame, dict)
            frame["render_time_ms"] = 0.0
        return normalized

    def test_report_fields_and_slice_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            config_path = self._generate(input_dir, frames=6)
            config = load_config(config_path)
            verify = verify_slice_consistency(config, input_dir)
            self.assertEqual(verify["status"], "passed")
            self.assertLessEqual(verify["max_abs_error_8bit"], 2.0)

            summary = run_pipeline(config, input_dir, output_dir)
            report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["total_frames"], 6)
            self.assertEqual(len(report["frames"]), 6)
            self.assertEqual(len(summary.output_files), 6)
            required = {
                "frame_index",
                "focus_distance_m",
                "coc_min_px",
                "coc_max_px",
                "coc_mean_px",
                "aperture_blades",
                "max_blur_radius_px",
                "render_time_ms",
            }
            for frame in report["frames"]:
                self.assertTrue(required.issubset(frame))
                self.assertLessEqual(frame["coc_min_px"], frame["coc_mean_px"])
                self.assertLessEqual(frame["coc_mean_px"], frame["coc_max_px"])
            self.assertGreaterEqual(len(report["warnings"]), 1)
            for output_file in summary.output_files:
                self.assertTrue(output_file.is_file())

    def test_repeat_render_is_byte_deterministic_except_timing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            first_dir = root / "first"
            second_dir = root / "second"
            config = load_config(self._generate(input_dir, frames=4))
            first = run_pipeline(config, input_dir, first_dir)
            second = run_pipeline(config, input_dir, second_dir)

            first_hashes = [
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in first.output_files
            ]
            second_hashes = [
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in second.output_files
            ]
            self.assertEqual(first_hashes, second_hashes)
            first_report = json.loads(
                (first_dir / "report.json").read_text(encoding="utf-8")
            )
            second_report = json.loads(
                (second_dir / "report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                self._normalized_report(first_report),
                self._normalized_report(second_report),
            )

    def _make_invalid_track(self, input_dir: Path, keyframes: list[dict[str, float]]) -> None:
        path = input_dir / "focus_track.json"
        path.write_text(
            json.dumps({"time_unit": "seconds", "keyframes": keyframes}),
            encoding="utf-8",
        )

    def test_out_of_order_track_preserves_partial_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            config = load_config(self._generate(input_dir, frames=6))
            self._make_invalid_track(
                input_dir,
                [
                    {"time": 0.0, "focus_distance": 2.0},
                    {"time": 0.5, "focus_distance": 8.0},
                    {"time": 0.25, "focus_distance": 4.0},
                ],
            )
            with self.assertRaises(FocusTrackError) as caught:
                run_pipeline(config, input_dir, output_dir)
            self.assertEqual(caught.exception.frame_index, 3)
            partial = json.loads(
                (output_dir / "report.partial.json").read_text(encoding="utf-8")
            )
            conventional = json.loads(
                (output_dir / "report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(partial["status"], "failed")
            self.assertEqual(partial["error"]["frame_index"], 3)
            self.assertEqual(partial["total_frames"], 3)
            self.assertEqual(conventional, partial)
            pngs = sorted(output_dir.glob("frame_*.png"))
            self.assertEqual(len(pngs), 3)

    def test_out_of_range_track_keeps_all_rendered_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            config = load_config(self._generate(input_dir, frames=5))
            self._make_invalid_track(
                input_dir,
                [
                    {"time": 0.0, "focus_distance": 2.0},
                    {"time": 0.75, "focus_distance": 6.0},
                ],
            )
            with self.assertRaises(FocusTrackError) as caught:
                run_pipeline(config, input_dir, output_dir)
            self.assertEqual(
                caught.exception.frame_index, config.frame_start + config.frame_count
            )
            partial = json.loads(
                (output_dir / "report.partial.json").read_text(encoding="utf-8")
            )
            self.assertEqual(partial["total_frames"], 5)
            self.assertEqual(len(list(output_dir.glob("frame_*.png"))), 5)


if __name__ == "__main__":
    unittest.main()
