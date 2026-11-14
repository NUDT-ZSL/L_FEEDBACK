from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from telemetry_synth.errors import SynthError
from telemetry_synth.util import deterministic_noise, soft_limit
from telemetry_synth.wav_io import validate_pcm16_wav
from tools.generate_samples import construct_telemetry, write_example_files


ROOT = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "telemetry_synth.cli"]


def run_cli(*args: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*CLI, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


class PipelineAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory()
        cls.temp_root = Path(cls._temp.name)
        cls.sample_dir = cls.temp_root / "samples"
        cls.telemetry_path, cls.params_path = write_example_files(cls.sample_dir)
        cls.output_dir = cls.temp_root / "output"
        cls.output_path = cls.output_dir / "race.wav"
        cls.report_path = cls.output_dir / "race.report.json"
        cls.process = run_cli(
            "--telemetry",
            str(cls.telemetry_path),
            "--params",
            str(cls.params_path),
            "--output",
            str(cls.output_path),
            "--seed",
            "20261115",
            "--block-size",
            "4096",
        )
        cls.report = (
            json.loads(cls.report_path.read_text(encoding="utf-8"))
            if cls.report_path.exists()
            else None
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def test_success_and_memory_line(self) -> None:
        self.assertEqual(self.process.returncode, 0, self.process.stderr)
        self.assertIn("peak_memory_estimate_bytes=", self.process.stderr)
        self.assertIn("method=modeled_streaming_working_set", self.process.stderr)
        self.assertIsNotNone(self.report)
        self.assertEqual(self.report["status"], "ok")

    def test_wav_header_and_exact_alignment(self) -> None:
        telemetry = construct_telemetry()
        expected_samples = round(
            telemetry["frame_count"]
            * telemetry["frame_interval_ms"]
            / 1000.0
            * 48000
        )
        with wave.open(str(self.output_path), "rb") as handle:
            self.assertEqual(handle.getnchannels(), 2)
            self.assertEqual(handle.getsampwidth(), 2)
            self.assertEqual(handle.getframerate(), 48000)
            self.assertEqual(handle.getnframes(), expected_samples)
            raw = handle.readframes(handle.getnframes())

        with self.output_path.open("rb") as handle:
            header = handle.read(44)
        riff, riff_size, wave_tag = struct.unpack("<4sI4s", header[:12])
        self.assertEqual(riff, b"RIFF")
        self.assertEqual(wave_tag, b"WAVE")
        self.assertEqual(riff_size, self.output_path.stat().st_size - 8)
        fmt_tag, fmt_size, audio_format, channels, sample_rate, byte_rate, align, bits = (
            struct.unpack("<4sIHHIIHH", header[12:36])
        )
        data_tag, data_size = struct.unpack("<4sI", header[36:44])
        self.assertEqual((fmt_tag, fmt_size), (b"fmt ", 16))
        self.assertEqual((audio_format, channels, sample_rate, bits), (1, 2, 48000, 16))
        self.assertEqual((byte_rate, align), (192000, 4))
        self.assertEqual(data_tag, b"data")
        self.assertEqual(data_size, expected_samples * 4)
        self.assertEqual(len(raw), data_size)

        pcm = np.frombuffer(raw, dtype="<i2").reshape(-1, 2)
        self.assertEqual(pcm.shape[0], expected_samples)
        self.assertGreaterEqual(int(pcm.min()), -32768)
        self.assertLessEqual(int(pcm.max()), 32767)
        self.assertEqual(self.report["timeline"]["frame_count"], 2400)
        self.assertEqual(self.report["timeline"]["written_sample_count"], expected_samples)
        self.assertTrue(self.report["timeline"]["sample_count_match"])
        self.assertEqual(self.report["wav_header"]["data_size_bytes"], expected_samples * 4)

    def test_report_peaks_and_events(self) -> None:
        self.assertEqual(
            set(self.report["bus_peaks"]),
            {"engine", "tire", "events"},
        )
        for peak in self.report["bus_peaks"].values():
            self.assertGreater(peak, 0.0)
            self.assertTrue(math.isfinite(peak))
        self.assertLessEqual(self.report["mix"]["post_limit_peak"], 1.0)
        self.assertGreater(self.report["mix"]["pre_limit_peak"], 0.0)
        self.assertEqual(self.report["events"]["trigger_count"], 13)
        self.assertEqual(self.report["events"]["counts_by_type"]["shift_alert"], 3)
        self.assertEqual(self.report["events"]["engine_shift_count"], 3)
        self.assertEqual(self.report["events"]["attenuated_count"], 3)
        self.assertGreater(self.report["events"]["tire_state_switch_count"], 0)

    def test_repeat_run_is_byte_identical(self) -> None:
        second_dir = self.temp_root / "output_repeat"
        second_wav = second_dir / "race.wav"
        second_report = second_dir / "race.report.json"
        process = run_cli(
            "--telemetry",
            str(self.telemetry_path),
            "--params",
            str(self.params_path),
            "--output",
            str(second_wav),
            "--seed",
            "20261115",
            "--block-size",
            "4096",
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        first_bytes = self.output_path.read_bytes()
        second_bytes = second_wav.read_bytes()
        self.assertEqual(first_bytes, second_bytes)
        repeat_report = json.loads(second_report.read_text(encoding="utf-8"))
        self.assertEqual(
            self.report["outputs"]["wav_sha256"],
            repeat_report["outputs"]["wav_sha256"],
        )

    def test_block_size_does_not_change_audio(self) -> None:
        small_block_dir = self.temp_root / "output_small_blocks"
        small_block_wav = small_block_dir / "race.wav"
        process = run_cli(
            "--telemetry",
            str(self.telemetry_path),
            "--params",
            str(self.params_path),
            "--output",
            str(small_block_wav),
            "--seed",
            "20261115",
            "--block-size",
            "257",
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(self.output_path.read_bytes(), small_block_wav.read_bytes())

    def test_no_temporary_outputs_after_success(self) -> None:
        leftovers = list(self.output_dir.glob(".*.tmp"))
        self.assertEqual(leftovers, [])

    def test_corrupt_wav_header_is_rejected(self) -> None:
        data = self.output_path.read_bytes()
        corrupt_path = self.temp_root / "corrupt.wav"
        corrupt_path.write_bytes(data[:-1])
        with self.assertRaises(SynthError) as caught:
            validate_pcm16_wav(corrupt_path, 576000)
        self.assertEqual(caught.exception.code, "E_WAV_HEADER")

        with self.assertRaises(SynthError) as caught:
            validate_pcm16_wav(self.output_path, 576001)
        self.assertEqual(caught.exception.code, "E_WAV_HEADER")


class ValidationFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory()
        cls.temp_root = Path(cls._temp.name)
        cls.sample_dir = cls.temp_root / "samples"
        cls.telemetry_path, cls.params_path = write_example_files(cls.sample_dir)
        cls.telemetry = construct_telemetry()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def _write_telemetry_variant(self, name: str, document: object) -> Path:
        path = self.temp_root / name
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def _assert_rejected_without_output(
        self,
        *,
        telemetry_path: Path,
        expected_code: str,
        params_path: Path | None = None,
    ) -> None:
        case_dir = self.temp_root / f"case_{expected_code}_{len(list(self.temp_root.glob('case_*')))}"
        output_path = case_dir / "race.wav"
        report_path = case_dir / "race.report.json"
        process = run_cli(
            "--telemetry",
            str(telemetry_path),
            "--params",
            str(params_path or self.params_path),
            "--output",
            str(output_path),
            "--seed",
            "3",
            "--block-size",
            "512",
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn(f"error_code={expected_code}", process.stderr)
        self.assertFalse(output_path.exists())
        self.assertFalse(report_path.exists())
        if case_dir.exists():
            self.assertEqual(list(case_dir.glob(".*.tmp")), [])

    def test_missing_frame_rejected(self) -> None:
        document = json.loads(json.dumps(self.telemetry))
        del document["frames"][100]
        path = self._write_telemetry_variant("missing_frame.json", document)
        self._assert_rejected_without_output(
            telemetry_path=path,
            expected_code="E_TELEMETRY_MISSING_FRAME",
        )

    def test_missing_field_rejected(self) -> None:
        document = json.loads(json.dumps(self.telemetry))
        del document["frames"][10]["slip"]
        path = self._write_telemetry_variant("missing_field.json", document)
        self._assert_rejected_without_output(
            telemetry_path=path,
            expected_code="E_TELEMETRY_FIELD",
        )

    def test_frame_index_gap_rejected(self) -> None:
        document = json.loads(json.dumps(self.telemetry))
        for index, frame in enumerate(document["frames"][100:], start=100):
            frame["frame_index"] = index + 1
            frame["time_ms"] = (index + 1) * 5.0
        path = self._write_telemetry_variant("frame_gap.json", document)
        self._assert_rejected_without_output(
            telemetry_path=path,
            expected_code="E_TELEMETRY_MISSING_FRAME",
        )

    def test_nonmonotonic_timestamp_rejected(self) -> None:
        document = json.loads(json.dumps(self.telemetry))
        document["frames"][10]["time_ms"] = document["frames"][9]["time_ms"]
        path = self._write_telemetry_variant("bad_time.json", document)
        self._assert_rejected_without_output(
            telemetry_path=path,
            expected_code="E_TELEMETRY_TIMESTAMP",
        )

    def test_out_of_range_field_rejected(self) -> None:
        document = json.loads(json.dumps(self.telemetry))
        document["frames"][10]["rpm"] = 99999.0
        path = self._write_telemetry_variant("bad_rpm.json", document)
        self._assert_rejected_without_output(
            telemetry_path=path,
            expected_code="E_TELEMETRY_RANGE",
        )

    def test_unknown_surface_rejected(self) -> None:
        document = json.loads(json.dumps(self.telemetry))
        document["frames"][10]["surface"] = "unobtainium"
        path = self._write_telemetry_variant("bad_surface.json", document)
        self._assert_rejected_without_output(
            telemetry_path=path,
            expected_code="E_TELEMETRY_RANGE",
        )

    def test_malformed_json_rejected(self) -> None:
        path = self.temp_root / "broken.json"
        path.write_text('{"schema_version": 1,', encoding="utf-8")
        self._assert_rejected_without_output(
            telemetry_path=path,
            expected_code="E_TELEMETRY_JSON",
        )

    def test_invalid_parameter_rejected(self) -> None:
        params = json.loads(self.params_path.read_text(encoding="utf-8"))
        params["engine"]["rpm_max"] = "not-a-number"
        params_path = self.temp_root / "bad_params.json"
        params_path.write_text(json.dumps(params), encoding="utf-8")
        self._assert_rejected_without_output(
            telemetry_path=self.telemetry_path,
            expected_code="E_PARAMS_SCHEMA",
            params_path=params_path,
        )

    def test_invalid_block_size_is_usage_error(self) -> None:
        output_path = self.temp_root / "usage" / "race.wav"
        process = run_cli(
            "--telemetry",
            str(self.telemetry_path),
            "--params",
            str(self.params_path),
            "--output",
            str(output_path),
            "--seed",
            "1",
            "--block-size",
            "0",
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("error_code=E_USAGE", process.stderr)
        self.assertFalse(output_path.exists())

    def test_existing_target_is_not_modified_on_validation_error(self) -> None:
        case_dir = self.temp_root / "preserve"
        case_dir.mkdir()
        output_path = case_dir / "race.wav"
        output_path.write_bytes(b"sentinel")
        document = json.loads(json.dumps(self.telemetry))
        document["frames"][3]["throttle"] = 4.0
        path = self._write_telemetry_variant("bad_throttle.json", document)
        process = run_cli(
            "--telemetry",
            str(path),
            "--params",
            str(self.params_path),
            "--output",
            str(output_path),
            "--seed",
            "1",
            "--block-size",
            "512",
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("error_code=E_TELEMETRY_RANGE", process.stderr)
        self.assertEqual(output_path.read_bytes(), b"sentinel")
        self.assertFalse((case_dir / "race.report.json").exists())


class DspUnitTests(unittest.TestCase):
    def test_noise_is_addressed_by_global_sample_index(self) -> None:
        whole = deterministic_noise(123, "tire_noise", 1000, 31, 2)
        split = np.vstack(
            [
                deterministic_noise(123, "tire_noise", 1000, 7, 2),
                deterministic_noise(123, "tire_noise", 1007, 24, 2),
            ]
        )
        self.assertTrue(np.array_equal(whole, split))
        self.assertFalse(
            np.array_equal(whole, deterministic_noise(123, "other", 1000, 31, 2))
        )
        self.assertGreaterEqual(float(whole.min()), -1.0)
        self.assertLess(float(whole.max()), 1.0)

    def test_soft_limiter_bounds_extreme_inputs(self) -> None:
        samples = np.array([[-100.0, -2.0], [0.0, 0.5], [2.0, 100.0]])
        limited = soft_limit(samples, 0.9)
        self.assertLessEqual(float(np.max(np.abs(limited))), 1.0)
        self.assertEqual(float(limited[1, 1]), 0.5)


if __name__ == "__main__":
    unittest.main()

