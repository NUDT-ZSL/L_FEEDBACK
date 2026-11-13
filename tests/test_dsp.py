from __future__ import annotations

import array
import math
import unittest

from replay_mixer.audio import load_project
from replay_mixer.dsp import (
    TimeMap,
    detect_silence,
    pan_matrix,
    prepare_project,
    quantize_with_gain,
    render_premix_block,
)
from replay_mixer.schema import Anchor, load_config
from replay_mixer.util import db_to_lin
from tests.helpers import (
    make_project,
    sine_samples,
    temporary_workspace,
    zero_samples,
)


class TimeMapTests(unittest.TestCase):
    def test_piecewise_linear_and_constant_extrapolation(self) -> None:
        time_map = TimeMap.from_anchors(
            (
                Anchor(replay_ms=1000.0, source_ms=500.0),
                Anchor(replay_ms=2000.0, source_ms=1500.0),
                Anchor(replay_ms=4000.0, source_ms=3000.0),
            ),
            sample_rate=1000,
        )
        self.assertAlmostEqual(time_map.source_frame(0.0), -500.0)
        self.assertAlmostEqual(time_map.source_frame(500.0), 0.0)
        self.assertAlmostEqual(time_map.source_frame(1000.0), 500.0)
        self.assertAlmostEqual(time_map.source_frame(2000.0), 1500.0)
        self.assertAlmostEqual(time_map.source_frame(3000.0), 2250.0)
        self.assertAlmostEqual(time_map.source_frame(5000.0), 3750.0)


class SilenceTests(unittest.TestCase):
    def test_sliding_rms_only_trims_long_enough_intervals(self) -> None:
        sample_rate = 8000
        samples = sine_samples(1500, sample_rate, amplitude=0.3)
        samples += zero_samples(1000)
        samples += sine_samples(500, sample_rate, frequency=300.0, amplitude=0.3)
        samples += zero_samples(320)
        samples += sine_samples(180, sample_rate, frequency=260.0, amplitude=0.3)
        intervals = detect_silence(
            array.array("h", samples),
            1,
            sample_rate,
            window_ms=16.0,
            threshold_dbfs=-50.0,
            min_duration_ms=80.0,
        )
        self.assertEqual(len(intervals), 1)
        self.assertGreaterEqual(intervals[0].start_frame, 1500)
        self.assertLessEqual(intervals[0].start_frame, 1600)
        self.assertGreaterEqual(intervals[0].end_frame, 2400)
        self.assertLessEqual(intervals[0].end_frame, 2500)
        self.assertLess(intervals[0].end_frame, 3000)

    def test_remap_preserves_all_non_trimmed_output_samples(self) -> None:
        sample_rate = 8000
        duration_ms = 300.0
        frames = int(round(duration_ms * sample_rate / 1000.0))
        signal = sine_samples(1000, sample_rate, frequency=180.0, amplitude=0.25)
        silence = zero_samples(600)
        tail = sine_samples(
            frames - len(signal) - len(silence),
            sample_rate,
            frequency=240.0,
            amplitude=0.2,
        )
        source = signal + silence + tail
        anchors = [
            {"replay_ms": 0.0, "source_ms": 0.0},
            {"replay_ms": duration_ms, "source_ms": duration_ms},
        ]

        with temporary_workspace("silence-") as root:
            trimmed_dir = root / "trimmed"
            untouched_dir = root / "untouched"
            trimmed_dir.mkdir()
            untouched_dir.mkdir()
            config_kwargs = {
                "sample_rate": sample_rate,
                "duration_ms": duration_ms,
                "block_frames": frames,
            }
            base_commentary = {
                "name": "c",
                "filename": "c.wav",
                "channels": 1,
                "samples": source,
                "anchors": anchors,
                "threshold_dbfs": -50.0,
            }
            trimmed_config = make_project(
                trimmed_dir,
                **config_kwargs,
                commentary=[
                    {
                        **base_commentary,
                        "min_duration_ms": 40.0,
                    }
                ],
            )
            untouched_config = make_project(
                untouched_dir,
                **config_kwargs,
                commentary=[
                    {
                        **base_commentary,
                        "min_duration_ms": 5000.0,
                    }
                ],
            )

            trimmed = prepare_project(load_project(load_config(str(trimmed_config))))
            untouched = prepare_project(
                load_project(load_config(str(untouched_config)))
            )
            mask = trimmed.commentary[0].trim_mask
            self.assertGreater(sum(mask), 0)
            trimmed_audio = render_premix_block(trimmed, 0, frames)
            untouched_audio = render_premix_block(untouched, 0, frames)
            for frame in range(frames):
                if mask[frame]:
                    self.assertEqual(float(trimmed_audio[frame]), 0.0)
                else:
                    self.assertEqual(float(trimmed_audio[frame]), float(untouched_audio[frame]))


class MixingTests(unittest.TestCase):
    def test_mono_stereo_equal_power_pan(self) -> None:
        center = pan_matrix(1, 2, 0.0)
        self.assertAlmostEqual(center[0][0], math.sqrt(0.5), places=12)
        self.assertAlmostEqual(center[1][0], math.sqrt(0.5), places=12)
        left = pan_matrix(1, 2, -1.0)
        self.assertAlmostEqual(left[0][0], 1.0, places=12)
        self.assertAlmostEqual(left[1][0], 0.0, places=12)
        right = pan_matrix(1, 2, 1.0)
        self.assertAlmostEqual(right[0][0], 0.0, places=12)
        self.assertAlmostEqual(right[1][0], 1.0, places=12)

    def test_soft_limit_never_exceeds_configured_peak(self) -> None:
        sample_rate = 8000
        duration_ms = 100.0
        frames = int(round(duration_ms * sample_rate / 1000.0))
        loud = [32767] * frames
        with temporary_workspace("limiter-") as root:
            config_path = make_project(
                root,
                sample_rate=sample_rate,
                duration_ms=duration_ms,
                block_frames=frames,
                peak_limit_dbfs=-1.0,
                commentary=[
                    {
                        "name": "loud",
                        "filename": "loud.wav",
                        "channels": 1,
                        "samples": loud,
                        "anchors": [
                            {"replay_ms": 0.0, "source_ms": 0.0},
                            {
                                "replay_ms": duration_ms,
                                "source_ms": duration_ms,
                            },
                        ],
                        "gain_db": 6.0,
                        "threshold_dbfs": -50.0,
                        "min_duration_ms": 1000.0,
                    }
                ],
            )
            prepared = prepare_project(load_project(load_config(str(config_path))))
            rendered = render_premix_block(prepared, 0, frames)
            ceiling = db_to_lin(-1.0)
            peak = max(abs(float(value)) for value in rendered)
            self.assertLessEqual(peak, ceiling + 1.0e-6)
            self.assertGreater(peak, ceiling * 0.95)
            quantized = quantize_with_gain(rendered, 1.0)
            self.assertTrue(all(-32768 <= value <= 32767 for value in quantized))


class LinearResampleTests(unittest.TestCase):
    def test_positional_alignment_matches_source_ramp(self) -> None:
        sample_rate = 8000
        frames = 800
        duration_ms = frames * 1000.0 / sample_rate
        source = [(index * 37 + 100) % 1200 for index in range(frames)]
        with temporary_workspace("resample-") as root:
            config_path = make_project(
                root,
                sample_rate=sample_rate,
                duration_ms=duration_ms,
                block_frames=frames,
                commentary=[
                    {
                        "name": "ramp",
                        "filename": "ramp.wav",
                        "channels": 1,
                        "samples": source,
                        "anchors": [
                            {"replay_ms": 0.0, "source_ms": 0.0},
                            {
                                "replay_ms": duration_ms,
                                "source_ms": duration_ms,
                            },
                        ],
                        "gain_db": 0.0,
                        "threshold_dbfs": -50.0,
                        "min_duration_ms": 1000.0,
                    }
                ],
            )
            prepared = prepare_project(load_project(load_config(str(config_path))))
            rendered = render_premix_block(prepared, 0, frames)
            for index in range(frames):
                expected = source[index] / 32768.0
                self.assertAlmostEqual(float(rendered[index]), expected, places=7)
