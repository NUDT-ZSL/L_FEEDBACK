"""Regression and acceptance tests for the deterministic repair pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audiorepair import loudness, wavio, wsola  # noqa: E402
from audiorepair.batch import load_batch, run_batch  # noqa: E402
from audiorepair.errors import (  # noqa: E402
    AUDIO_UNSUPPORTED_FORMAT,
    INPUT_MISSING,
    STRETCH_UNREACHABLE,
    TIMELINE_INVALID,
    TIMELINE_OUT_OF_RANGE,
    RepairError,
)
from audiorepair.pipeline import run_job  # noqa: E402
from audiorepair.timeline import parse_timeline_document  # noqa: E402
from tools.generate_samples import generate  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pcm24_wav() -> bytes:
    payload = bytes(4800 * 3)
    fmt = struct.pack("<HHIIHH", 1, 1, 48000, 48000 * 3, 3, 24)
    chunks = (
        b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(payload))
        + payload
    )
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


class AudioRepairAcceptanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_value = tempfile.TemporaryDirectory(prefix="audiorepair-tests-")
        cls.temp = Path(cls.temp_value.name)
        cls.samples_dir = cls.temp / "samples"
        cls.generated = generate(str(cls.samples_dir))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_value.cleanup()

    def _output_dir(self, name: str) -> Path:
        path = self.temp / ("out-" + name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_supported_wav_round_trip(self) -> None:
        for sample_rate in (44100, 48000):
            for channels in (1, 2):
                frames = np.arange(257, dtype=np.float64)[:, None] / 512.0
                if channels == 2:
                    frames = np.concatenate((frames, -frames * 0.5), axis=1)
                payload = wavio.encode_wav(frames, sample_rate, channels)
                parsed = wavio.parse_wav_bytes(payload)
                self.assertEqual(parsed.sample_rate, sample_rate)
                self.assertEqual(parsed.channels, channels)
                self.assertEqual(parsed.bits_per_sample, 16)
                self.assertEqual(parsed.frames, frames.shape[0])
                decoded = np.rint(
                    np.clip(frames, -1.0, 32767.0 / 32768.0) * 32768.0
                ) / 32768.0
                np.testing.assert_array_equal(parsed.samples, decoded)

    def test_unsupported_wav_returns_format_code(self) -> None:
        with self.assertRaises(RepairError) as raised:
            wavio.parse_wav_bytes(_pcm24_wav())
        self.assertEqual(raised.exception.code, AUDIO_UNSUPPORTED_FORMAT)

    def test_splice_points_derive_sample_accurate_segments(self) -> None:
        source = Path(self.generated["dialogue_48k_mono"]["wav"])
        audio = wavio.read_wav(str(source))
        source_timeline = json.loads(
            Path(self.generated["dialogue_48k_mono"]["timeline"]).read_text(
                encoding="utf-8"
            )
        )
        for segment in source_timeline["segments"]:
            segment.pop("start_sec")
            segment.pop("end_sec")
        timeline = {
            "target_lufs": source_timeline["target_lufs"],
            "tolerance_lu": source_timeline["tolerance_lu"],
            "crossfade_ms": source_timeline["crossfade_ms"],
            "segments": source_timeline["segments"],
            "splice_points_sec": [2.8],
        }
        parsed = parse_timeline_document(
            timeline, audio.frames, audio.sample_rate
        )
        self.assertEqual(parsed.segments[0].start_sample, 0)
        self.assertEqual(parsed.segments[0].end_sample, 134400)
        self.assertEqual(parsed.segments[1].start_sample, 134400)
        self.assertEqual(parsed.segments[1].end_sample, audio.frames)

    def test_end_to_end_constraints_and_report_remeasurement(self) -> None:
        for name in ("dialogue_48k_mono", "dialogue_44k_stereo"):
            with self.subTest(name=name):
                source = Path(self.generated[name]["wav"])
                timeline = Path(self.generated[name]["timeline"])
                output_dir = self._output_dir(name)
                output = output_dir / "repaired.wav"
                report_path = output_dir / "report.json"
                result = run_job(
                    str(source), str(timeline), str(output), str(report_path)
                )
                self.assertEqual(result["status"], "ok")
                report = json.loads(report_path.read_text(encoding="utf-8"))
                repaired = wavio.read_wav(str(output))
                source_audio = wavio.read_wav(str(source))
                self.assertEqual(repaired.sample_rate, source_audio.sample_rate)
                self.assertEqual(repaired.channels, source_audio.channels)

                measured_lufs = loudness.integrated_loudness(
                    repaired.samples, repaired.sample_rate
                )
                measured_peak = loudness.peak_dbfs(repaired.samples)
                measured_dc = loudness.dc_offset(repaired.samples)
                self.assertLessEqual(abs(measured_lufs - (-23.0)), 0.5)
                self.assertLessEqual(measured_peak, -1.0)
                self.assertLess(abs(measured_dc), 0.001)
                self.assertAlmostEqual(
                    report["verification"]["integrated_loudness_lufs"],
                    measured_lufs,
                    delta=1e-5,
                )
                self.assertAlmostEqual(
                    report["verification"]["peak_dbfs"], measured_peak, delta=1e-5
                )
                self.assertAlmostEqual(
                    report["verification"]["dc_offset"], measured_dc, delta=1e-8
                )
                self.assertEqual(
                    report["output"]["sha256"], _sha256(output)
                )
                self.assertEqual(report["status"], "complete")

                all_expected = []
                for segment in report["segments"]:
                    self.assertEqual(
                        segment["output_length_samples"],
                        int(
                            round(
                                (
                                    segment["original_end_sample"]
                                    - segment["original_start_sample"]
                                )
                                * segment["stretch_ratio"]
                            )
                        ),
                    )
                    lower, upper = segment["reachable_range"]
                    for ratio in segment["span_stretch_ratios"]:
                        self.assertGreaterEqual(ratio, lower)
                        self.assertLessEqual(ratio, upper)
                    for boundary in segment["boundaries"]:
                        self.assertLessEqual(
                            abs(boundary["error_ms"]), 20.0
                        )
                        all_expected.append(boundary["expected_sample"])

                    xfade_in = (
                        segment["crossfade_in"]["duration_samples"]
                        if segment["crossfade_in"]
                        else 0
                    )
                    xfade_out = (
                        segment["crossfade_out"]["duration_samples"]
                        if segment["crossfade_out"]
                        else 0
                    )
                    region = repaired.samples[
                        segment["output_start_sample"] + xfade_in :
                        segment["output_start_sample"]
                        + segment["output_length_samples"]
                        - xfade_out
                    ]
                    region_lufs = loudness.integrated_loudness(
                        region, repaired.sample_rate
                    )
                    self.assertAlmostEqual(
                        segment["normalized_loudness_lufs"],
                        region_lufs,
                        delta=2e-5,
                    )

                remeasured_boundaries = wsola.detect_boundaries(
                    repaired.samples, repaired.sample_rate, all_expected
                )
                report_actual = [
                    boundary["actual_sample"]
                    for segment in report["segments"]
                    for boundary in segment["boundaries"]
                ]
                self.assertEqual(
                    report_actual,
                    [boundary["actual_sample"] for boundary in remeasured_boundaries],
                )
                for crossfade in report["crossfades"]:
                    expected_samples = int(
                        round(20.0 * repaired.sample_rate / 1000.0)
                    )
                    self.assertEqual(
                        crossfade["duration_samples"], expected_samples
                    )
                    start = crossfade["start_sample"]
                    end = crossfade["end_sample"]
                    self.assertLessEqual(
                        float(np.max(np.abs(repaired.samples[start:end]))), 1.0
                    )

    def test_business_error_codes_and_no_partial_output(self) -> None:
        source = Path(self.generated["dialogue_48k_mono"]["wav"])
        timeline = json.loads(
            Path(self.generated["dialogue_48k_mono"]["timeline"]).read_text(
                encoding="utf-8"
            )
        )
        cases = []
        cases.append(
            (
                INPUT_MISSING,
                str(self.temp / "does-not-exist.wav"),
                str(self.generated["dialogue_48k_mono"]["timeline"]),
            )
        )
        invalid_json = self.temp / "invalid.json"
        invalid_json.write_text("{broken", encoding="utf-8")
        cases.append((TIMELINE_INVALID, str(source), str(invalid_json)))
        out_of_range = json.loads(json.dumps(timeline))
        out_of_range["segments"][0]["start_sec"] = -0.5
        out_of_range_path = self.temp / "out-of-range.json"
        out_of_range_path.write_text(json.dumps(out_of_range), encoding="utf-8")
        cases.append((TIMELINE_OUT_OF_RANGE, str(source), str(out_of_range_path)))
        unreachable = json.loads(json.dumps(timeline))
        unreachable["segments"][0]["stretch_ratio"] = 1.6
        unreachable_path = self.temp / "unreachable.json"
        unreachable_path.write_text(json.dumps(unreachable), encoding="utf-8")
        cases.append((STRETCH_UNREACHABLE, str(source), str(unreachable_path)))

        for index, (code, input_path, timeline_path) in enumerate(cases):
            with self.subTest(code=code):
                output_dir = self._output_dir("error-{}".format(index))
                output = output_dir / "repaired.wav"
                report = output_dir / "report.json"
                with self.assertRaises(RepairError) as raised:
                    run_job(input_path, timeline_path, str(output), str(report))
                self.assertEqual(raised.exception.code, code)
                self.assertFalse(output.exists())
                self.assertFalse(report.exists())
                self.assertFalse(Path(str(output) + ".incomplete").exists())
                self.assertEqual(
                    [path for path in output_dir.iterdir()
                     if path.name.startswith(".audiorepair-")],
                    [],
                )

    def test_preexisting_output_untouched_on_input_error(self) -> None:
        output_dir = self._output_dir("preserve")
        output = output_dir / "repaired.wav"
        report = output_dir / "report.json"
        output.write_bytes(b"old-wav")
        report.write_bytes(b"old-report")
        invalid = output_dir / "bad.json"
        invalid.write_text("not json", encoding="utf-8")
        with self.assertRaises(RepairError) as raised:
            run_job(
                self.generated["dialogue_48k_mono"]["wav"],
                str(invalid),
                str(output),
                str(report),
            )
        self.assertEqual(raised.exception.code, TIMELINE_INVALID)
        self.assertEqual(output.read_bytes(), b"old-wav")
        self.assertEqual(report.read_bytes(), b"old-report")

    def test_batch_thread_determinism_and_clean_shutdown(self) -> None:
        before = {
            thread.ident for thread in threading.enumerate()
            if thread.name.startswith("audiorepair")
        }
        first_dir = self._output_dir("batch-one")
        second_dir = self._output_dir("batch-four")
        jobs_one = load_batch(
            self.generated["batch"], output_dir=str(first_dir)
        )
        jobs_four = load_batch(
            self.generated["batch"], output_dir=str(second_dir)
        )
        summary_one = run_batch(
            jobs_one, 1, str(first_dir / "batch-summary.json")
        )
        summary_four = run_batch(
            jobs_four, 4, str(second_dir / "batch-summary.json")
        )
        self.assertEqual(summary_one["summary"]["failed"], 0)
        self.assertEqual(summary_four["summary"]["failed"], 0)
        for job_id in ("mono48", "stereo44"):
            wav_one = first_dir / (job_id + ".wav")
            wav_four = second_dir / (job_id + ".wav")
            report_one = first_dir / (job_id + ".report.json")
            report_four = second_dir / (job_id + ".report.json")
            self.assertEqual(wav_one.read_bytes(), wav_four.read_bytes())
            self.assertEqual(report_one.read_bytes(), report_four.read_bytes())
            self.assertEqual(_sha256(wav_one), _sha256(wav_four))
        after = {
            thread.ident for thread in threading.enumerate()
            if thread.name.startswith("audiorepair")
        }
        self.assertEqual(after - before, set())


if __name__ == "__main__":
    unittest.main()

