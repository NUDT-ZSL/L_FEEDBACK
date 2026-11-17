from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

import numpy as np
import soundfile as sf

from bgm_mixer.dsp import (
    fade_envelope,
    loop_period_ratio,
    resample_segment,
    resampled_loop_frames,
    scheduled_period_start,
)
from bgm_mixer.synthesis import synthesize_signal

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def write_synth(
    path: Path,
    sample_rate: int,
    duration: float,
    frequencies: Any,
    amplitude: float = 0.5,
    channels: int = 1,
) -> Path:
    samples = synthesize_signal(
        sample_rate=sample_rate,
        duration_seconds=duration,
        frequencies=frequencies,
        amplitudes=amplitude,
        channels=channels,
    )
    sf.write(path, samples, sample_rate, format="WAV", subtype="PCM_24")
    return path


def write_config(path: Path, payload: Dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def run_cli(*arguments: str, timeout: float = 120.0) -> tuple[int, str, str]:
    completed = subprocess.run(
        [PYTHON, str(ROOT / "mixer.py"), *arguments],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return completed.returncode, completed.stdout, completed.stderr


def parse_single_json(stdout: str) -> Dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise AssertionError(f"expected exactly one stdout line, got {len(lines)}: {stdout!r}")
    return json.loads(lines[0])


class ResamplingTests(unittest.TestCase):
    def test_non_integer_length_error_within_one_frame(self) -> None:
        cases = [
            (44100, 48000, 44101),
            (48000, 44100, 47999),
            (192000, 8000, 12345),
            (8000, 192000, 12347),
            (96000, 44100, 33333),
        ]
        for source_rate, target_rate, source_frames in cases:
            with self.subTest(source_rate=source_rate, target_rate=target_rate):
                segment = np.zeros((source_frames, 1), dtype=np.float64)
                expected = resampled_loop_frames(source_frames, source_rate, target_rate)
                actual = resample_segment(segment, source_rate, target_rate, expected)
                theory = source_frames * target_rate / source_rate
                self.assertLessEqual(abs(actual.shape[0] - theory), 1.0)
                self.assertEqual(actual.shape[0], expected)

    def test_chunked_synthesis_matches_contiguous_synthesis(self) -> None:
        contiguous = synthesize_signal(
            48000, 0.5, 1000.0, amplitudes=0.4, start_seconds=1.25
        )
        first = synthesize_signal(
            48000, 0.25, 1000.0, amplitudes=0.4, start_seconds=1.25
        )
        second = synthesize_signal(
            48000, 0.25, 1000.0, amplitudes=0.4, start_seconds=1.5
        )
        np.testing.assert_array_equal(contiguous, np.concatenate((first, second)))

    def test_alias_and_image_rejection(self) -> None:
        cases = [
            (44100, 48000, 15000.0, 18900.0),
            (48000, 44100, 19000.0, 15100.0),
            (48000, 44100, 20000.0, 16100.0),
            (44100, 48000, 20000.0, 16100.0),
        ]
        for source_rate, target_rate, frequency, alias_frequency in cases:
            with self.subTest(source_rate=source_rate, target_rate=target_rate):
                segment = synthesize_signal(
                    source_rate,
                    3.0,
                    frequency,
                    amplitudes=0.5,
                    channels=1,
                )
                expected = resampled_loop_frames(
                    segment.shape[0], source_rate, target_rate
                )
                output = resample_segment(
                    segment, source_rate, target_rate, expected
                )[:, 0]
                trim = int(0.1 * target_rate)
                output = output[trim:-trim]
                self.assertGreater(output.size, target_rate)
                self.assertGreater(self._amplitude(output, target_rate, frequency), 0.1)
                alias = self._amplitude(output, target_rate, alias_frequency)
                wanted = self._amplitude(output, target_rate, frequency)
                rejection_db = 20.0 * math.log10((alias + 1e-30) / wanted)
                self.assertLess(rejection_db, -60.0)

    @staticmethod
    def _amplitude(samples: np.ndarray, sample_rate: int, frequency: float) -> float:
        time_index = np.arange(samples.size, dtype=np.float64)
        carrier = np.exp(-2j * math.pi * frequency * time_index / sample_rate)
        return float(2.0 * abs(np.dot(samples, carrier)) / samples.size)


class LoopMixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.source_rate = 44100
        self.target_rate = 48000
        self.loop_start = 0.123456
        self.loop_end = 1.234567
        self.source = write_synth(
            self.dir / "loop.wav",
            self.source_rate,
            2.0,
            20.0,
            amplitude=0.4,
        )
        self.config = write_config(
            self.dir / "play.json",
            {
                "duration_seconds": 4.0,
                "loop": {
                    "start_seconds": self.loop_start,
                    "end_seconds": self.loop_end,
                },
                "fade": {"in_ms": 10, "out_ms": 10, "curve": "linear"},
                "tracks": [{"volume": 0.9, "offset_seconds": 0.0}],
            },
        )
        self.output = self.dir / "out.wav"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_sample_aligned_loop_seams_duration_and_fades(self) -> None:
        returncode, stdout, stderr = run_cli(
            "--input",
            str(self.source),
            "--target-rate",
            str(self.target_rate),
            "--config",
            str(self.config),
            "--output",
            str(self.output),
        )
        self.assertEqual(returncode, 0, stderr)
        summary = parse_single_json(stdout)
        self.assertEqual(summary["error_code"], None)
        self.assertEqual(summary["output_sample_rate"], self.target_rate)
        self.assertEqual(
            summary["output_frames"], round(4.0 * self.target_rate)
        )

        track = summary["tracks"][0]
        self.assertEqual(
            track["loop_start_sample"],
            int(math.floor(self.loop_start * self.source_rate + 0.5)),
        )
        self.assertEqual(
            track["loop_end_sample"],
            int(math.floor(self.loop_end * self.source_rate + 0.5)),
        )
        self.assertLessEqual(
            abs(track["loop_duration_seconds"] - (self.loop_end - self.loop_start)),
            0.001,
        )

        audio, rate = sf.read(str(self.output), dtype="float64", always_2d=True)
        self.assertEqual(rate, self.target_rate)
        self.assertEqual(audio.shape[0], summary["output_frames"])

        source_frames = track["loop_end_sample"] - track["loop_start_sample"]
        numerator, denominator = loop_period_ratio(
            source_frames, self.source_rate, self.target_rate
        )
        starts = []
        index = 0
        while True:
            start = scheduled_period_start(index, numerator, denominator)
            if start >= audio.shape[0]:
                break
            starts.append(start)
            index += 1
        self.assertEqual(summary["loop_count"], len(starts))
        self.assertGreaterEqual(len(starts), 4)
        for start in starts[1:]:
            self.assertLessEqual(abs(audio[start, 0] - audio[start - 1, 0]), 0.02)

        fade_frames = track["fade_in_frames"]
        self.assertEqual(fade_frames, 480)
        for start in starts:
            fade_in = audio[start : start + fade_frames, 0]
            fade_out_start = start + track["loop_output_frames"] - fade_frames
            fade_out = audio[fade_out_start : fade_out_start + fade_frames, 0]
            if fade_in.size > 1:
                self.assertLessEqual(np.max(np.abs(np.diff(fade_in))), 0.005)
            if fade_out.size > 1:
                self.assertLessEqual(np.max(np.abs(np.diff(fade_out))), 0.005)

    def test_multi_track_clipping_and_offsets(self) -> None:
        left = write_synth(
            self.dir / "left.wav", 44100, 2.0, 1000.0, amplitude=0.8
        )
        right = write_synth(
            self.dir / "right.wav", 32000, 2.0, 1000.0, amplitude=0.8
        )
        config = write_config(
            self.dir / "clip.json",
            {
                "duration_seconds": 2.5,
                "fade_ms": 0,
                "tracks": [
                    {"path": str(left), "volume": 1.0, "offset_seconds": 0.0},
                    {
                        "path": str(right),
                        "volume": 1.0,
                        "offset_seconds": 0.25,
                    },
                ],
            },
        )
        output = self.dir / "clip.wav"
        returncode, stdout, stderr = run_cli(
            "--input",
            str(left),
            "--target-rate",
            "48000",
            "--config",
            str(config),
            "--output",
            str(output),
        )
        self.assertEqual(returncode, 0, stderr)
        summary = parse_single_json(stdout)
        self.assertEqual(summary["output_channels"], 1)
        self.assertGreater(summary["clip_count"], 0)
        self.assertGreater(summary["clipped_frames"], 0)
        self.assertLessEqual(summary["clip_count"], summary["clipped_frames"] * 1)
        self.assertEqual(summary["tracks"][1]["offset_frames"], 12000)
        audio, _ = sf.read(str(output), dtype="float64", always_2d=True)
        self.assertLessEqual(float(np.max(np.abs(audio))), 1.0 + 1e-7)
        self.assertAlmostEqual(summary["peak_after_limit"], 1.0, places=9)

    def test_byte_determinism_and_stable_summary(self) -> None:
        arguments = (
            "--input",
            str(self.source),
            "--target-rate",
            str(self.target_rate),
            "--config",
            str(self.config),
            "--output",
            str(self.output),
        )
        first_code, first_stdout, first_stderr = run_cli(*arguments)
        self.assertEqual(first_code, 0, first_stderr)
        first_bytes = self.output.read_bytes()
        first_hash = hashlib.sha256(first_bytes).hexdigest()

        second_code, second_stdout, second_stderr = run_cli(*arguments)
        self.assertEqual(second_code, 0, second_stderr)
        second_bytes = self.output.read_bytes()
        second_hash = hashlib.sha256(second_bytes).hexdigest()
        self.assertEqual(first_hash, second_hash)
        self.assertEqual(first_stdout, second_stdout)
        self.assertEqual(json.loads(first_stdout), json.loads(second_stdout))

    def test_track_paths_are_relative_to_config(self) -> None:
        nested = self.dir / "nested"
        nested.mkdir()
        source = write_synth(nested / "relative.wav", 44100, 1.0, 100.0)
        config = write_config(
            nested / "relative.json",
            {
                "duration_seconds": 1.0,
                "tracks": [{"path": "relative.wav", "volume": 0.5}],
            },
        )
        output = self.dir / "relative-out.wav"
        code, stdout, stderr = run_cli(
            "--input",
            str(self.source),
            "--target-rate",
            "48000",
            "--config",
            str(config),
            "--output",
            str(output),
        )
        self.assertEqual(code, 0, stderr)
        summary = parse_single_json(stdout)
        self.assertEqual(summary["tracks"][0]["path"], str(source.resolve()))

    def test_mono_track_is_upmixed_for_stereo_output(self) -> None:
        stereo = write_synth(
            self.dir / "stereo.wav",
            48000,
            1.0,
            [220.0, 330.0],
            amplitude=0.2,
            channels=2,
        )
        config = write_config(
            self.dir / "stereo.json",
            {
                "duration_seconds": 1.0,
                "fade_ms": 5,
                "tracks": [
                    {"path": str(stereo), "volume": 0.5},
                    {
                        "path": str(self.source),
                        "volume": 0.5,
                        "offset_seconds": 0.1,
                    },
                ],
            },
        )
        output = self.dir / "stereo-out.wav"
        code, stdout, stderr = run_cli(
            "--input",
            str(self.source),
            "--target-rate",
            "48000",
            "--config",
            str(config),
            "--output",
            str(output),
        )
        self.assertEqual(code, 0, stderr)
        summary = parse_single_json(stdout)
        self.assertEqual(summary["output_channels"], 2)
        audio, rate = sf.read(str(output), dtype="float64", always_2d=True)
        self.assertEqual(rate, 48000)
        self.assertEqual(audio.shape, (48000, 2))


class LongPlaybackTests(unittest.TestCase):
    def test_ten_minute_phase_error_under_one_sample(self) -> None:
        numerator, denominator = loop_period_ratio(49000, 44100, 48000)
        period = numerator / denominator
        self.assertNotEqual(period, round(period))
        for minute in (10, 20, 30):
            timeline = minute * 60 * 48000
            k = int(timeline / period)
            while scheduled_period_start(k, numerator, denominator) > timeline:
                k -= 1
            while scheduled_period_start(k + 1, numerator, denominator) <= timeline:
                k += 1
            actual_start = scheduled_period_start(k, numerator, denominator)
            ideal_position = timeline - k * period
            actual_position = timeline - actual_start
            self.assertLessEqual(abs(actual_position - ideal_position), 0.5)
            self.assertLessEqual(abs(actual_position - ideal_position), 1.0)

    def test_fade_envelope_control_curve_step(self) -> None:
        for curve in ("linear", "equal_power"):
            envelope = fade_envelope(48000, 480, 480, curve)
            max_step = float(np.max(np.abs(np.diff(envelope))))
            self.assertLessEqual(max_step, 0.005, msg=f"{curve}: {max_step}")
            self.assertEqual(envelope[0], 0.0)
            self.assertEqual(envelope[-1], 0.0)


class ErrorHandlingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.good = write_synth(self.dir / "good.wav", 44100, 1.0, 100.0)
        self.output = self.dir / "out.wav"
        self.config = write_config(
            self.dir / "play.json",
            {
                "duration_seconds": 1.0,
                "tracks": [{"volume": 1.0, "offset_seconds": 0.0}],
            },
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def assert_business_failure(
        self,
        expected_code: str,
        *,
        input_path: Path | str | None = None,
        config: Path | str | None = None,
        output: Path | str | None = None,
        target_rate: int = 48000,
    ) -> Dict[str, Any]:
        source = self.good if input_path is None else input_path
        play_config = self.config if config is None else config
        destination = self.output if output is None else output
        code, stdout, stderr = run_cli(
            "--input",
            str(source),
            "--target-rate",
            str(target_rate),
            "--config",
            str(play_config),
            "--output",
            str(destination),
        )
        self.assertNotEqual(code, 0, stderr)
        summary = parse_single_json(stdout)
        self.assertEqual(summary["error_code"], expected_code, summary)
        self.assertEqual(summary["exit_code"], code)
        self.assertFalse(Path(destination).exists())
        leftovers = list(Path(destination).parent.glob(f".{Path(destination).name}.*.tmp"))
        self.assertEqual(leftovers, [])
        return summary

    def test_invalid_source_sample_rate(self) -> None:
        bad = write_synth(self.dir / "low-rate.wav", 4000, 0.5, 100.0)
        self.assert_business_failure("E_INVALID_INPUT", input_path=bad)

    def test_invalid_channel_count(self) -> None:
        mono = synthesize_signal(44100, 0.5, 100.0, 0.4, channels=1)
        bad = self.dir / "three-channels.wav"
        sf.write(
            bad,
            np.repeat(mono, 3, axis=1),
            44100,
            format="WAV",
            subtype="PCM_24",
        )
        self.assert_business_failure("E_INVALID_INPUT", input_path=bad)

    def test_volume_and_negative_offset_validation(self) -> None:
        for track in (
            {"volume": 1.1},
            {"volume": -0.1},
            {"volume": 0.5, "offset_seconds": -0.001},
        ):
            with self.subTest(track=track):
                config = write_config(
                    self.dir / "invalid-track.json",
                    {"duration_seconds": 1.0, "tracks": [track]},
                )
                self.assert_business_failure("E_INVALID_INPUT", config=config)

    def test_malformed_json(self) -> None:
        bad = self.dir / "bad.json"
        bad.write_text('{"duration_seconds": 1.0,', encoding="utf-8")
        self.assert_business_failure("E_INVALID_INPUT", config=bad)

    def test_invalid_target_rate(self) -> None:
        self.assert_business_failure("E_INVALID_INPUT", target_rate=7999)

    def test_missing_output_directory_is_io_error(self) -> None:
        destination = self.dir / "missing" / "out.wav"
        self.assert_business_failure("E_IO", output=destination)

    def test_output_directory_permission_error_is_io(self) -> None:
        if os.name == "nt" or not hasattr(os, "geteuid") or os.geteuid() == 0:
            self.skipTest("POSIX non-root permission check not applicable")
        locked = self.dir / "locked"
        locked.mkdir()
        original_mode = stat.S_IMODE(locked.stat().st_mode)
        os.chmod(locked, 0o500)
        try:
            summary = self.assert_business_failure(
                "E_IO", output=locked / "out.wav"
            )
            self.assertTrue(summary["message"])
        finally:
            os.chmod(locked, original_mode)

    def test_resampled_loop_too_short_uses_resample_code(self) -> None:
        source = write_synth(self.dir / "high-rate.wav", 192000, 0.5, 100.0)
        config = write_config(
            self.dir / "tiny-loop.json",
            {
                "duration_seconds": 1.0,
                "loop": {"start_seconds": 0.0, "end_seconds": 20 / 192000},
                "fade_ms": 0,
                "tracks": [{"volume": 1.0}],
            },
        )
        self.assert_business_failure(
            "E_RESAMPLE", input_path=source, config=config, target_rate=8000
        )

    def test_synthesis_command_is_deterministic(self) -> None:
        first = self.dir / "synth-a.wav"
        second = self.dir / "synth-b.wav"
        args = (
            "synth",
            "--sample-rate",
            "44100",
            "--duration",
            "0.5",
            "--frequency",
            "440",
            "--amplitude",
            "0.25",
        )
        code_a, stdout_a, stderr_a = run_cli(
            *args, "--output", str(first)
        )
        code_b, stdout_b, stderr_b = run_cli(
            *args, "--output", str(second)
        )
        self.assertEqual(code_a, 0, stderr_a)
        self.assertEqual(code_b, 0, stderr_b)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(json.loads(stdout_a)["output_frames"], 22050)


if __name__ == "__main__":
    unittest.main()
