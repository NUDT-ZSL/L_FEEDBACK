from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np

from bpm_grains.analysis import CONFIDENCE_THRESHOLD, detect_bpm
from bpm_grains.artifacts import write_artifacts_atomically
from bpm_grains.pipeline import _octave_aligned_bpm
from bpm_grains.samples import (
    generate_fixture_set,
    make_click_track,
    make_low_confidence_noise,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_cli(*args: str) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, str(ROOT / "bpm_grain.py"), *args],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class BpmAnalysisTests(unittest.TestCase):
    def test_click_tracks_return_expected_tempo_and_high_confidence(self) -> None:
        for expected_bpm in (
            45.0,
            60.0,
            90.0,
            120.0,
            128.0,
            130.0,
            174.0,
            220.0,
            240.0,
        ):
            with self.subTest(bpm=expected_bpm):
                samples = make_click_track(expected_bpm, seconds=10.0)
                result = detect_bpm(samples, 44_100)
                self.assertAlmostEqual(result.bpm, expected_bpm, delta=0.5)
                self.assertGreaterEqual(result.confidence, CONFIDENCE_THRESHOLD)

    def test_unstructured_noise_is_low_confidence(self) -> None:
        samples = make_low_confidence_noise(seconds=6.0)
        result = detect_bpm(samples, 44_100, raise_on_low_confidence=False)
        self.assertLess(result.confidence, CONFIDENCE_THRESHOLD)

    def test_percussion_octave_is_resolved_against_master_tempo(self) -> None:
        self.assertAlmostEqual(_octave_aligned_bpm(239.5, 120.0), 120.0)
        self.assertAlmostEqual(_octave_aligned_bpm(200.0, 120.0), 100.0)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._fixture_temp = tempfile.TemporaryDirectory()
        cls.fixtures = generate_fixture_set(cls._fixture_temp.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._fixture_temp.cleanup()

    def test_outputs_are_valid_bpm_aligned_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            first = temp / "run_a" / "mix.wav"
            second = temp / "run_b" / "mix.wav"

            result_a = _run_cli(
                "--input",
                str(self.fixtures["skill"]),
                "--percussion",
                str(self.fixtures["percussion"]),
                "--output",
                str(first),
                "--seed",
                "4242",
                "--log-level",
                "DEBUG",
            )
            result_b = _run_cli(
                "--input",
                str(self.fixtures["skill"]),
                "--percussion",
                str(self.fixtures["percussion"]),
                "--output",
                str(second),
                "--seed",
                "4242",
                "--log-level",
                "WARNING",
            )

            self.assertEqual(result_a.returncode, 0, result_a.stderr.decode())
            self.assertEqual(result_b.returncode, 0, result_b.stderr.decode())
            self.assertEqual(result_a.stdout, b"")
            self.assertEqual(result_b.stdout, b"")
            self.assertIn(b"soft limiter applied", result_a.stderr)
            self.assertEqual(_sha256(first), _sha256(second))

            with wave.open(str(first), "rb") as handle:
                self.assertEqual(handle.getnchannels(), 2)
                self.assertEqual(handle.getsampwidth(), 2)
                self.assertEqual(handle.getframerate(), 44_100)
                frames = handle.getnframes()
                pcm = np.frombuffer(handle.readframes(frames), dtype="<i2")
                self.assertGreater(frames, 0)
                measured_peak = float(np.max(np.abs(pcm))) / 32768.0

            report = json.loads((first.parent / "report.json").read_text("utf-8"))
            plan = json.loads(
                (first.parent / "mixing_plan.json").read_text("utf-8")
            )
            grains = json.loads((first.parent / "grains.json").read_text("utf-8"))
            grains_a = (first.parent / "grains.json").read_bytes()
            grains_b = (second.parent / "grains.json").read_bytes()

            self.assertEqual(grains_a, grains_b)
            self.assertAlmostEqual(report["bpm"], 120.0, delta=0.5)
            self.assertEqual(plan["bpm"], report["bpm"])
            self.assertEqual(plan["confidence"], report["confidence"])
            expected_beat = round(60.0 * 44_100 / report["bpm"] + 0.5)
            self.assertEqual(plan["beat_samples"], expected_beat)
            self.assertEqual(report["total_samples"], frames)
            self.assertEqual(report["grain_count"], grains["count"])
            self.assertEqual(report["grain_count"], len(grains["grains"]))
            self.assertAlmostEqual(report["peak_level"], measured_peak, places=9)
            self.assertLessEqual(report["peak_dbfs"], 0.0)
            self.assertTrue(report["soft_limiting_applied"])
            self.assertLessEqual(report["limiter_gain_adjustment_db"], 0.0)
            self.assertEqual(plan["alignment_start_sample"], 0)
            self.assertEqual(plan["cloud_loop_samples"], 4 * expected_beat)
            self.assertEqual(report["total_samples"] % (4 * expected_beat), 0)
            with wave.open(str(self.fixtures["skill"]), "rb") as source_handle:
                input_frames = source_handle.getnframes()
            self.assertEqual(
                plan["tracks"]["original"]["input_samples"], input_frames
            )
            self.assertGreaterEqual(
                plan["tracks"]["original"]["loop_samples"], input_frames
            )

            for grain in grains["grains"]:
                self.assertEqual(
                    grain["end_sample"] - grain["start_sample"],
                    grains["grain_length_samples"],
                )
                self.assertGreaterEqual(grain["source_position"], 0)
                self.assertLessEqual(
                    grain["source_position"] + grains["grain_length_samples"],
                    input_frames,
                )
                self.assertIsInstance(grain["start_sample"], int)
                self.assertIsInstance(grain["end_sample"], int)

    def test_stereo_input_without_percussion_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            output = Path(temp_name) / "stereo_mix.wav"
            result = _run_cli(
                "--input",
                str(self.fixtures["stereo"]),
                "--output",
                str(output),
                "--seed",
                "9",
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual(result.stdout, b"")
            with wave.open(str(output), "rb") as handle:
                self.assertEqual(handle.getnchannels(), 2)
                self.assertEqual(handle.getframerate(), 44_100)
                self.assertEqual(handle.getsampwidth(), 2)
            report = json.loads((output.parent / "report.json").read_text("utf-8"))
            self.assertGreaterEqual(report["peak_dbfs"], -120.0)
            self.assertLessEqual(report["peak_dbfs"], 0.0)

    def test_low_confidence_returns_exit_one_and_business_code(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            output = Path(temp_name) / "should_not_exist.wav"
            result = _run_cli(
                "--input",
                str(self.fixtures["low_confidence"]),
                "--output",
                str(output),
            )
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, b"")
            self.assertIn(b"ERR_LOW_CONFIDENCE", result.stderr)
            self.assertFalse(output.exists())

    def test_bad_formats_return_exit_two_and_preserve_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            output = Path(temp_name) / "mix.wav"
            output.write_bytes(b"existing output sentinel")

            for fixture_name in (
                "bad_rate",
                "bad_depth",
                "bad_header",
                "bad_truncated",
            ):
                with self.subTest(fixture=fixture_name):
                    result = _run_cli(
                        "--input",
                        str(self.fixtures[fixture_name]),
                        "--output",
                        str(output),
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertEqual(result.stdout, b"")
                    self.assertIn(b"ERR_BAD_FORMAT", result.stderr)
                    self.assertEqual(
                        output.read_bytes(), b"existing output sentinel"
                    )

    def test_bad_percussion_format_returns_bad_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            output = Path(temp_name) / "mix.wav"
            result = _run_cli(
                "--input",
                str(self.fixtures["skill"]),
                "--percussion",
                str(self.fixtures["bad_rate"]),
                "--output",
                str(output),
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn(b"ERR_BAD_FORMAT", result.stderr)
            self.assertFalse(output.exists())

    def test_wav_write_failure_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            output = Path(temp_name) / "mix.wav"
            output.write_bytes(b"existing output sentinel")

            def fail_after_partial_write(temp_path: Path) -> None:
                temp_path.write_bytes(b"partial invalid wave")
                raise RuntimeError("simulated encoder failure")

            with self.assertRaises(RuntimeError):
                write_artifacts_atomically(
                    output,
                    json_artifacts={},
                    wav_writer=fail_after_partial_write,
                )

            self.assertEqual(output.read_bytes(), b"existing output sentinel")
            leftovers = list(Path(temp_name).glob("*.tmp"))
            self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()

