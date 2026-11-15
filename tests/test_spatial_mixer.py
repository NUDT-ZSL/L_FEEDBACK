"""Deterministic acceptance tests for the offline spatial mixer."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

from spatialmix.audio import fit_to_duration, resample_to_rate
from spatialmix.spatial import (
    compute_distance_gain,
    compute_spatial_gains,
    equal_power_pan,
)
from spatialmix.timeline import samples_to_milliseconds, seconds_to_samples

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "spatial_mixer.py"
SAMPLE_RATE = 48000


def _samples(seconds: float, sample_rate: int = SAMPLE_RATE) -> int:
    return int(math.floor(seconds * sample_rate + 0.5))


def _constant(seconds: float, value: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    return np.full(_samples(seconds, sample_rate), value, dtype=np.float64)


def _sine(
    seconds: float,
    frequency: float,
    amplitude: float,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    count = _samples(seconds, sample_rate)
    time = np.arange(count, dtype=np.float64) / float(sample_rate)
    return amplitude * np.sin(2.0 * math.pi * frequency * time)


def _write_wav(
    path: Path,
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    sf.write(str(path), samples, sample_rate, format="WAV", subtype="PCM_16")


def _event(
    filename: str,
    trigger: float,
    duration: float,
    priority: float,
    policy: str,
    position: Sequence[float],
    fit_strategy: str = "stretch",
) -> Dict[str, Any]:
    return {
        "filename": filename,
        "position": {"x": float(position[0]), "y": float(position[1])},
        "trigger_time": float(trigger),
        "duration": duration,
        "priority": priority,
        "conflict_policy": policy,
        "fit_strategy": fit_strategy,
}


class SpatialGainTests(unittest.TestCase):
    def test_front_and_zero_distance_use_equal_power_center(self) -> None:
        front = compute_spatial_gains(0.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(front.distance_gain, 1.0, places=15)
        self.assertAlmostEqual(front.left_gain, 1.0 / math.sqrt(2.0), places=15)
        self.assertAlmostEqual(front.right_gain, 1.0 / math.sqrt(2.0), places=15)

        zero = compute_spatial_gains(3.0, 4.0, 3.0, 4.0)
        self.assertTrue(math.isfinite(zero.left_gain))
        self.assertTrue(math.isfinite(zero.right_gain))
        self.assertEqual(zero.distance, 0.0)
        self.assertAlmostEqual(zero.distance_gain, 1.0, places=15)
        self.assertAlmostEqual(zero.left_gain, zero.right_gain, places=15)

    def test_left_right_and_rear_pan(self) -> None:
        right = compute_spatial_gains(0.0, 0.0, 1.0, 0.0)
        left = compute_spatial_gains(0.0, 0.0, -1.0, 0.0)
        rear = compute_spatial_gains(0.0, 0.0, 0.0, -1.0)

        self.assertGreater(right.right_gain, right.left_gain)
        self.assertGreater(left.left_gain, left.right_gain)
        self.assertAlmostEqual(right.right_gain, 1.0, places=15)
        self.assertAlmostEqual(left.left_gain, 1.0, places=15)
        self.assertAlmostEqual(rear.left_gain, rear.right_gain, places=15)
        self.assertAlmostEqual(abs(rear.azimuth_degrees), 180.0, places=12)

    def test_inverse_distance_attenuation_and_floor(self) -> None:
        self.assertAlmostEqual(compute_distance_gain(0.5), 1.0, places=15)
        self.assertAlmostEqual(compute_distance_gain(2.0), 0.5, places=15)
        self.assertAlmostEqual(compute_distance_gain(10.0), 0.1, places=15)
        self.assertAlmostEqual(
            compute_distance_gain(1000.0, min_gain=0.05),
            0.05,
            places=15,
        )

    def test_equal_power_pairs(self) -> None:
        for pan in (-1.0, -0.5, 0.0, 0.5, 1.0):
            left, right = equal_power_pan(pan)
            self.assertAlmostEqual(left * left + right * right, 1.0, places=15)


class AudioTransformTests(unittest.TestCase):
    def test_resampling_changes_rate_and_preserves_length(self) -> None:
        source = np.linspace(-0.5, 0.5, 2400, dtype=np.float64)
        result = resample_to_rate(source, 24000, 48000)
        self.assertEqual(result.shape[0], 4800)
        self.assertEqual(result.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_stretch_crop_and_padding_are_exact_length(self) -> None:
        source = np.arange(10, dtype=np.float64)
        stretched, stretch_kind = fit_to_duration(source, 17, "stretch")
        cropped, crop_kind = fit_to_duration(source, 4, "crop")
        padded, pad_kind = fit_to_duration(source, 14, "crop")

        self.assertEqual(stretched.shape[0], 17)
        self.assertEqual(stretch_kind, "stretched")
        self.assertEqual(cropped.shape[0], 4)
        self.assertEqual(crop_kind, "cropped")
        np.testing.assert_array_equal(cropped, source[:4])
        self.assertEqual(padded.shape[0], 14)
        self.assertEqual(pad_kind, "padded")
        np.testing.assert_array_equal(padded[:10], source)
        np.testing.assert_array_equal(padded[10:], np.zeros(4))

    def test_time_rounding_is_half_up(self) -> None:
        self.assertEqual(seconds_to_samples(0.0015), 72)
        self.assertEqual(samples_to_milliseconds(72), 2)
        self.assertEqual(samples_to_milliseconds(144), 3)


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _run_case(
        self,
        name: str,
        events: List[Dict[str, Any]],
        files: Dict[str, Tuple[np.ndarray, int]],
        listener: Optional[Sequence[float]] = None,
        duration: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], np.ndarray, int, Path, Path]:
        case_root = self.root / name
        source_dir = case_root / "sources"
        source_dir.mkdir(parents=True)
        for filename, (samples, sample_rate) in files.items():
            _write_wav(source_dir / filename, samples, sample_rate)

        document: Dict[str, Any] = {
            "listener": {
                "x": float((listener or (0.0, 0.0))[0]),
                "y": float((listener or (0.0, 0.0))[1]),
            },
            "events": events,
        }
        events_path = case_root / "events.json"
        events_path.write_text(
            json.dumps(document, indent=2),
            encoding="utf-8",
        )
        mix_path = case_root / "mix.wav"
        report_path = case_root / "report.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(CLI),
                str(source_dir),
                str(events_path),
                str(mix_path),
                str(report_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg="stdout:\n{}\nstderr:\n{}".format(
                completed.stdout,
                completed.stderr,
            ),
        )
        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        mixed, sample_rate = sf.read(
            str(mix_path),
            dtype="float64",
            always_2d=True,
        )
        if duration is not None:
            self.assertEqual(mixed.shape[0], _samples(duration))
        return report, mixed, sample_rate, mix_path, report_path

    def test_timeline_alignment_and_report(self) -> None:
        event = _event(
            "constant.wav",
            trigger=0.25,
            duration=0.5,
            priority=1,
            policy="mix",
            position=(0.0, 5.0),
        )
        report, mixed, sample_rate, mix_path, _ = self._run_case(
            "alignment",
            [event],
            {"constant.wav": (_constant(0.5, 0.5), SAMPLE_RATE)},
            duration=0.75,
        )

        info = sf.info(str(mix_path))
        self.assertEqual(sample_rate, 48000)
        self.assertEqual(info.samplerate, 48000)
        self.assertEqual(info.channels, 2)
        self.assertEqual(info.subtype, "PCM_16")
        self.assertEqual(mixed.shape, (36000, 2))
        self.assertTrue(np.allclose(mixed[:11999], 0.0, atol=1e-12))
        self.assertGreater(float(np.max(np.abs(mixed[12000:]))), 0.01)

        event_report = report["events"][0]
        self.assertEqual(event_report["start_sample"], 12000)
        self.assertEqual(event_report["end_sample"], 36000)
        self.assertEqual(event_report["start_time_ms"], 250)
        self.assertEqual(event_report["end_time_ms"], 750)
        self.assertEqual(event_report["status"], "played")
        self.assertFalse(event_report["dropped"])

    def test_unsupported_source_rate_is_resampled(self) -> None:
        event = _event(
            "tone24k.wav",
            trigger=0.1,
            duration=0.5,
            priority=1,
            policy="mix",
            position=(0.0, 2.0),
        )
        report, mixed, sample_rate, _, _ = self._run_case(
            "resample",
            [event],
            {"tone24k.wav": (_sine(0.5, 220.0, 0.4, 24000), 24000)},
            duration=0.6,
        )
        self.assertEqual(sample_rate, 48000)
        self.assertEqual(mixed.shape[0], 28800)
        self.assertEqual(report["events"][0]["source_sample_rate"], 24000)
        self.assertEqual(report["events"][0]["start_sample"], 4800)

    def test_audio_is_stretched_to_event_duration(self) -> None:
        event = _event(
            "short.wav",
            trigger=0.0,
            duration=0.5,
            priority=1,
            policy="mix",
            position=(0.0, 2.0),
        )
        report, mixed, _, _, _ = self._run_case(
            "stretch",
            [event],
            {"short.wav": (_constant(0.25, 0.2), SAMPLE_RATE)},
            duration=0.5,
        )
        self.assertEqual(mixed.shape[0], 24000)
        self.assertEqual(report["events"][0]["audio_fit"], "stretched")
        self.assertEqual(report["events"][0]["end_sample"], 24000)

    def test_delay_policy_moves_overlapping_event(self) -> None:
        events = [
            _event("a.wav", 0.0, 0.5, 2, "delay", (0.0, 2.0)),
            _event("b.wav", 0.2, 0.5, 1, "delay", (0.0, 2.0)),
        ]
        report, mixed, _, _, _ = self._run_case(
            "delay",
            events,
            {
                "a.wav": (_constant(0.5, 0.05), SAMPLE_RATE),
                "b.wav": (_constant(0.5, 0.05), SAMPLE_RATE),
            },
            duration=1.0,
        )

        second = report["events"][1]
        self.assertEqual(second["status"], "played")
        self.assertTrue(second["adjusted"])
        self.assertEqual(second["start_sample"], 24000)
        self.assertEqual(second["end_sample"], 48000)
        self.assertEqual(second["start_time_ms"], 500)
        self.assertEqual(second["end_time_ms"], 1000)
        self.assertEqual(second["delay_samples"], 14400)
        self.assertEqual(second["conflict_with"], [0])
        self.assertEqual(mixed.shape[0], 48000)

    def test_drop_policy_drops_lower_priority_event(self) -> None:
        events = [
            _event("low.wav", 0.0, 0.5, 1, "drop", (0.0, 2.0)),
            _event("high.wav", 0.1, 0.5, 5, "drop", (0.0, 2.0)),
        ]
        report, mixed, _, _, _ = self._run_case(
            "drop_replace",
            events,
            {
                "low.wav": (_constant(0.5, 0.1), SAMPLE_RATE),
                "high.wav": (_constant(0.5, 0.1), SAMPLE_RATE),
            },
            duration=0.6,
        )
        self.assertEqual(report["events"][0]["status"], "dropped")
        self.assertTrue(report["events"][0]["adjusted"])
        self.assertIsNone(report["events"][0]["start_time_ms"])
        self.assertEqual(report["events"][1]["status"], "played")
        self.assertEqual(report["summary"]["dropped_events"], 1)
        self.assertEqual(mixed.shape[0], 28800)

    def test_equal_or_lower_later_event_is_dropped(self) -> None:
        events = [
            _event("first.wav", 0.0, 0.5, 5, "drop", (0.0, 2.0)),
            _event("second.wav", 0.1, 0.5, 5, "drop", (0.0, 2.0)),
        ]
        report, _, _, _, _ = self._run_case(
            "drop_tie",
            events,
            {
                "first.wav": (_constant(0.5, 0.1), SAMPLE_RATE),
                "second.wav": (_constant(0.5, 0.1), SAMPLE_RATE),
            },
            duration=0.5,
        )
        self.assertEqual(report["events"][0]["status"], "played")
        self.assertEqual(report["events"][1]["status"], "dropped")
        self.assertIn("dropped by drop policy", report["events"][1]["reason"])

    def test_mix_policy_and_peak_normalization(self) -> None:
        events = [
            _event("loud-a.wav", 0.0, 0.4, 1, "mix", (0.0, 1.0)),
            _event("loud-b.wav", 0.0, 0.4, 1, "mix", (0.0, 1.0)),
        ]
        report, mixed, _, _, _ = self._run_case(
            "normalize",
            events,
            {
                "loud-a.wav": (_constant(0.4, 0.9), SAMPLE_RATE),
                "loud-b.wav": (_constant(0.4, 0.9), SAMPLE_RATE),
            },
            duration=0.4,
        )
        output = report["output"]
        peak = float(np.max(np.abs(mixed)))
        self.assertEqual(report["events"][0]["status"], "played")
        self.assertEqual(report["events"][1]["status"], "played")
        self.assertTrue(output["normalization_required"])
        self.assertLess(output["normalization_gain"], 1.0)
        self.assertLessEqual(peak, 0.98 + 1e-12)
        self.assertGreater(peak, 0.90)
        self.assertLessEqual(output["peak_after_normalization"], 0.98)

    def test_right_panned_source_has_more_right_channel_energy(self) -> None:
        event = _event(
            "right.wav",
            0.0,
            0.25,
            1,
            "mix",
            (1.0, 0.0),
        )
        report, mixed, _, _, _ = self._run_case(
            "pan",
            [event],
            {"right.wav": (_constant(0.25, 0.5), SAMPLE_RATE)},
            duration=0.25,
        )
        left_energy = float(np.sum(mixed[:, 0] * mixed[:, 0]))
        right_energy = float(np.sum(mixed[:, 1] * mixed[:, 1]))
        self.assertGreater(right_energy, left_energy * 1000.0)
        self.assertEqual(report["events"][0]["spatial"]["distance_gain"], 1.0)

    def test_repeated_runs_are_byte_identical(self) -> None:
        case_root = self.root / "determinism"
        source_dir = case_root / "sources"
        source_dir.mkdir(parents=True)
        sine = _sine(0.4, 330.0, 0.25)
        noise = np.random.default_rng(20261116).normal(0.0, 0.1, _samples(0.4))
        _write_wav(source_dir / "sine.wav", sine)
        _write_wav(source_dir / "noise.wav", noise)

        events = [
            _event("sine.wav", 0.0, 0.4, 2, "mix", (-1.0, 1.0)),
            _event("noise.wav", 0.15, 0.4, 1, "mix", (1.0, 2.0)),
        ]
        events_path = case_root / "events.json"
        events_path.write_text(
            json.dumps({"listener": [0.0, 0.0], "events": events}),
            encoding="utf-8",
        )
        mix_path = case_root / "mix.wav"
        report_path = case_root / "report.json"
        command = [
            sys.executable,
            str(CLI),
            str(source_dir),
            str(events_path),
            str(mix_path),
            str(report_path),
        ]

        first = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        first_mix = mix_path.read_bytes()
        first_report = report_path.read_bytes()

        second = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first_mix, mix_path.read_bytes())
        self.assertEqual(first_report, report_path.read_bytes())

    def test_named_cli_arguments_are_supported(self) -> None:
        event = _event("c.wav", 0.0, 0.1, 1, "mix", (0.0, 1.0))
        case_root = self.root / "named_cli"
        source_dir = case_root / "sources"
        source_dir.mkdir(parents=True)
        _write_wav(source_dir / "c.wav", _constant(0.1, 0.1))
        events_path = case_root / "events.json"
        events_path.write_text(
            json.dumps({"listener_position": [0, 0], "events": [event]}),
            encoding="utf-8",
        )
        mix_path = case_root / "mix.wav"
        report_path = case_root / "report.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(CLI),
                "--source-dir",
                str(source_dir),
                "--events-file",
                str(events_path),
                "--output-mix",
                str(mix_path),
                "--report-file",
                str(report_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(mix_path.is_file())
        self.assertTrue(report_path.is_file())


class ErrorHandlingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.source_dir = self.root / "sources"
        self.source_dir.mkdir()
        _write_wav(self.source_dir / "ok.wav", _constant(0.2, 0.1))
        self.events_path = self.root / "events.json"
        self.mix_path = self.root / "mix.wav"
        self.report_path = self.root / "report.json"

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _run_document(self, document: Any) -> subprocess.CompletedProcess:
        self.events_path.write_text(
            json.dumps(document),
            encoding="utf-8",
        )
        return subprocess.run(
            [
                sys.executable,
                str(CLI),
                str(self.source_dir),
                str(self.events_path),
                str(self.mix_path),
                str(self.report_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )

    def _assert_error(
        self,
        completed: subprocess.CompletedProcess,
        expected_fragments: Sequence[str],
    ) -> None:
        self.assertEqual(completed.returncode, 1)
        self.assertTrue(completed.stderr.startswith("ERROR: "))
        self.assertNotIn("Traceback", completed.stderr)
        for fragment in expected_fragments:
            self.assertIn(fragment, completed.stderr)

    def test_negative_duration(self) -> None:
        event = _event("ok.wav", 0.0, -0.1, 1, "mix", (0.0, 1.0))
        completed = self._run_document(
            {"listener": [0, 0], "events": [event]}
        )
        self._assert_error(
            completed,
            ("events[0].duration", "greater than 0.0"),
        )

    def test_unknown_conflict_policy(self) -> None:
        event = _event("ok.wav", 0.0, 0.1, 1, "explode", (0.0, 1.0))
        completed = self._run_document(
            {"listener": [0, 0], "events": [event]}
        )
        self._assert_error(
            completed,
            ("events[0].conflict_policy", "unknown conflict policy"),
        )

    def test_missing_source_file(self) -> None:
        event = _event("missing.wav", 0.0, 0.1, 1, "mix", (0.0, 1.0))
        completed = self._run_document(
            {"listener": [0, 0], "events": [event]}
        )
        self._assert_error(
            completed,
            ("events[0].filename", "missing.wav", "not found"),
        )

    def test_missing_required_field(self) -> None:
        event = _event("ok.wav", 0.0, 0.1, 1, "mix", (0.0, 1.0))
        del event["priority"]
        completed = self._run_document(
            {"listener": [0, 0], "events": [event]}
        )
        self._assert_error(
            completed,
            ("events[0].priority", "missing required field"),
        )

    def test_invalid_json(self) -> None:
        self.events_path.write_text("{not valid json", encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(CLI),
                str(self.source_dir),
                str(self.events_path),
                str(self.mix_path),
                str(self.report_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self._assert_error(completed, ("events_file", "invalid JSON"))

    def test_missing_output_directory(self) -> None:
        event = _event("ok.wav", 0.0, 0.1, 1, "mix", (0.0, 1.0))
        self.events_path.write_text(
            json.dumps({"listener": [0, 0], "events": [event]}),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(CLI),
                str(self.source_dir),
                str(self.events_path),
                str(self.root / "absent" / "mix.wav"),
                str(self.report_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self._assert_error(
            completed,
            ("mix_output", "output directory does not exist"),
        )

    def test_malformed_wav(self) -> None:
        (self.source_dir / "bad.wav").write_text("not a wav", encoding="utf-8")
        event = _event("bad.wav", 0.0, 0.1, 1, "mix", (0.0, 1.0))
        completed = self._run_document(
            {"listener": [0, 0], "events": [event]}
        )
        self._assert_error(
            completed,
            ("events[0].filename", "bad.wav", "cannot read WAV"),
        )

    def test_missing_cli_arguments(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(CLI)],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self._assert_error(completed, ("sources_dir", "missing required argument"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
