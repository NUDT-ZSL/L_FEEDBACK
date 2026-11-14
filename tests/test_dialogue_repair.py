from __future__ import annotations

import json
import math
import os
import re
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dialogue_repair.errors import RepairFailure
from dialogue_repair.manifest import SAMPLE_RATE
from dialogue_repair.processor import process_files
from dialogue_repair.samples import generate_sample_bundle
from dialogue_repair.selftest import run_selftest
from dialogue_repair.wavio import encode_wav, quantize_int, read_wav


class RepairToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="repair-test-")
        self.directory = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_wav(
        self, name: str, samples: np.ndarray, *, bits: int = 24
    ) -> Path:
        path = self.directory / name
        encoded, _ = encode_wav(samples, quantization_bits=bits)
        path.write_bytes(encoded)
        return path

    def write_manifest(self, name: str, value: dict) -> Path:
        path = self.directory / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def base_manifest(self, operations: list[dict], *, bits: int = 24) -> dict:
        return {"quantization_bits": bits, "operations": operations}

    def test_gain_scales_only_target_and_preserves_length(self) -> None:
        index = np.arange(1000, dtype=np.float64)
        source = 0.2 * np.sin(2.0 * math.pi * 121.0 * index / SAMPLE_RATE)
        input_path = self.write_wav("input.wav", source)
        manifest_path = self.write_manifest(
            "manifest.json",
            self.base_manifest(
                [
                    {
                        "index": 0,
                        "type": "gain",
                        "start_sample": 100,
                        "end_sample": 200,
                        "gain": 0.4,
                    }
                ]
            ),
        )
        output_path = self.directory / "output.wav"
        report_path = self.directory / "report.json"
        result = process_files(input_path, manifest_path, output_path, report_path)

        before = read_wav(input_path).mono_samples
        after = read_wav(output_path).mono_samples
        self.assertEqual(result.sample_count, 1000)
        np.testing.assert_array_equal(after[:100], before[:100])
        np.testing.assert_array_equal(after[200:], before[200:])
        self.assertFalse(np.array_equal(after[100:200], before[100:200]))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(
            report["operations"][0]["actual_interval"],
            {"start_sample": 100, "end_sample": 200},
        )
        self.assertEqual(report["operations"][0]["fade_length_samples"], 0)

    def test_replace_crossfade_has_exact_original_boundary_samples(self) -> None:
        count = 9600
        index = np.arange(count, dtype=np.float64)
        source = 0.15 * np.sin(2.0 * math.pi * 97.0 * index / SAMPLE_RATE)
        input_path = self.write_wav("input.wav", source)
        start, end = 1200, 6000
        manifest_path = self.write_manifest(
            "manifest.json",
            self.base_manifest(
                [
                    {
                        "index": 0,
                        "type": "replace",
                        "start_sample": start,
                        "end_sample": end,
                        "frequency_hz": 700.0,
                        "phase_radians": 0.25,
                        "amplitude": 0.5,
                        "fade_ms": 1.0,
                    }
                ]
            ),
        )
        output_path = self.directory / "output.wav"
        process_files(
            input_path,
            manifest_path,
            output_path,
            self.directory / "report.json",
        )
        before = read_wav(input_path).mono_samples
        after = read_wav(output_path).mono_samples
        np.testing.assert_array_equal(after[:start], before[:start])
        np.testing.assert_array_equal(after[end:], before[end:])
        self.assertEqual(after[start], before[start])
        self.assertEqual(after[end - 1], before[end - 1])
        self.assertFalse(np.array_equal(after[start + 200 : end - 200], before[start + 200 : end - 200]))

    def test_splice_report_and_determinism(self) -> None:
        source = np.zeros(2000, dtype=np.float64)
        reference = 0.2 * np.sin(
            2.0 * math.pi * 330.0 * np.arange(600, dtype=np.float64) / SAMPLE_RATE
        )
        input_path = self.write_wav("input.wav", source)
        reference_path = self.write_wav("reference.wav", reference)
        manifest_path = self.write_manifest(
            "manifest.json",
            self.base_manifest(
                [
                    {
                        "index": 0,
                        "type": "splice",
                        "start_sample": 400,
                        "end_sample": 1000,
                        "reference_file": reference_path.name,
                        "reference_start": 0,
                        "reference_end": 600,
                        "fade_ms": 2.0,
                    }
                ],
                bits=24,
            ),
        )
        output_path = self.directory / "output.wav"
        report_path = self.directory / "report.json"
        first = process_files(input_path, manifest_path, output_path, report_path)
        wav_bytes = output_path.read_bytes()
        report_bytes = report_path.read_bytes()
        second = process_files(input_path, manifest_path, output_path, report_path)
        self.assertEqual(wav_bytes, output_path.read_bytes())
        self.assertEqual(report_bytes, report_path.read_bytes())
        alternative_output = self.directory / "output-alternative.wav"
        alternative_report = self.directory / "report-alternative.json"
        third = process_files(
            input_path,
            manifest_path,
            alternative_output,
            alternative_report,
        )
        self.assertEqual(wav_bytes, alternative_output.read_bytes())
        self.assertEqual(report_bytes, alternative_report.read_bytes())
        self.assertEqual(first.wav_sha256, second.wav_sha256)
        self.assertEqual(first.wav_sha256, third.wav_sha256)
        self.assertEqual(first.report_sha256, second.report_sha256)
        self.assertEqual(first.report_sha256, third.report_sha256)

        report = json.loads(report_bytes)
        operation = report["operations"][0]
        self.assertEqual(operation["fade_length_samples"], 96)
        self.assertEqual(
            operation["splice"]["reference_interval"],
            {"start_sample": 0, "end_sample": 600},
        )
        self.assertIsInstance(operation["actual_interval"]["start_sample"], int)
        self.assertIsInstance(operation["levels"]["before"]["rms_linear"], float)
        self.assertRegex(
            report_bytes.decode("utf-8"),
            re.compile(r'"rms_linear": -?\d+\.\d{12}'),
        )

    def test_manifest_errors_are_collected_and_targets_are_preserved(self) -> None:
        input_path = self.write_wav(
            "input.wav", np.zeros(1000, dtype=np.float64)
        )
        manifest_path = self.write_manifest(
            "manifest.json",
            self.base_manifest(
                [
                    {
                        "index": 0,
                        "type": "gain",
                        "start_sample": 100,
                        "end_sample": 300,
                        "gain": -1.0,
                    },
                    {
                        "index": 1,
                        "type": "gain",
                        "start_sample": 200,
                        "end_sample": 400,
                        "gain": 0.5,
                    },
                    {
                        "index": 2,
                        "type": "unknown",
                        "start_sample": 500,
                        "end_sample": 600,
                    },
                    {
                        "index": 3,
                        "type": "replace",
                        "start_sample": 2000,
                        "end_sample": 2100,
                        "frequency_hz": 440.0,
                        "phase_radians": 0.0,
                        "fade_ms": -1.0,
                    },
                ]
            ),
        )
        output_path = self.directory / "output.wav"
        report_path = self.directory / "report.json"
        output_path.write_bytes(b"OLD-WAV")
        report_path.write_bytes(b"OLD-REPORT")

        with self.assertRaises(RepairFailure) as captured:
            process_files(input_path, manifest_path, output_path, report_path)
        categories = {error["category"] for error in captured.exception.errors}
        self.assertTrue(
            {
                "negative_gain",
                "overlapping_intervals",
                "unknown_operation_type",
                "interval_out_of_bounds",
                "invalid_fade_length",
            }.issubset(categories)
        )
        self.assertEqual(output_path.read_bytes(), b"OLD-WAV")
        self.assertEqual(report_path.read_bytes(), b"OLD-REPORT")
        self.assertEqual(
            list(self.directory.glob(".*.stage-*")),
            [],
        )
        self.assertEqual(
            list(self.directory.glob(".*.backup-*")),
            [],
        )

    def test_missing_and_nonconforming_reference_are_structured_errors(self) -> None:
        input_path = self.write_wav(
            "input.wav", np.zeros(1000, dtype=np.float64)
        )
        bad_reference, _ = encode_wav(
            np.zeros(100, dtype=np.float64),
            quantization_bits=16,
            sample_rate=44100,
        )
        (self.directory / "bad.wav").write_bytes(bad_reference)
        for reference_name, expected_category in (
            ("missing.wav", "missing_reference"),
            ("bad.wav", "reference_format_mismatch"),
        ):
            manifest_path = self.write_manifest(
                f"{reference_name}.json",
                self.base_manifest(
                    [
                        {
                            "index": 0,
                            "type": "splice",
                            "start_sample": 100,
                            "end_sample": 200,
                            "reference_file": reference_name,
                            "reference_start": 0,
                            "reference_end": 100,
                            "fade_ms": 1.0,
                        }
                    ]
                ),
            )
            output_path = self.directory / f"{reference_name}.out.wav"
            report_path = self.directory / f"{reference_name}.report.json"
            with self.assertRaises(RepairFailure) as captured:
                process_files(input_path, manifest_path, output_path, report_path)
            self.assertTrue(
                any(
                    error["category"] == expected_category
                    and error["operation_index"] == 0
                    for error in captured.exception.errors
                )
            )
            self.assertFalse(output_path.exists())
            self.assertFalse(report_path.exists())

    def test_reference_out_of_bounds_is_a_structured_error(self) -> None:
        input_path = self.write_wav(
            "input.wav", np.zeros(1000, dtype=np.float64)
        )
        self.write_wav("reference.wav", np.zeros(100, dtype=np.float64))
        manifest_path = self.write_manifest(
            "manifest.json",
            self.base_manifest(
                [
                    {
                        "index": 0,
                        "type": "splice",
                        "start_sample": 100,
                        "end_sample": 200,
                        "reference_file": "reference.wav",
                        "reference_start": 50,
                        "reference_end": 150,
                        "fade_ms": 1.0,
                    }
                ]
            ),
        )
        output_path = self.directory / "output.wav"
        report_path = self.directory / "report.json"
        with self.assertRaises(RepairFailure) as captured:
            process_files(input_path, manifest_path, output_path, report_path)
        self.assertTrue(
            any(
                error["category"] == "reference_out_of_bounds"
                and error["operation_index"] == 0
                for error in captured.exception.errors
            )
        )
        self.assertFalse(output_path.exists())
        self.assertFalse(report_path.exists())

    def test_staging_failure_preserves_existing_output(self) -> None:
        input_path = self.write_wav(
            "input.wav", np.zeros(1000, dtype=np.float64)
        )
        manifest_path = self.write_manifest(
            "manifest.json",
            self.base_manifest(
                [
                    {
                        "index": 0,
                        "type": "gain",
                        "start_sample": 100,
                        "end_sample": 200,
                        "gain": 0.5,
                    }
                ]
            ),
        )
        output_path = self.directory / "output.wav"
        output_path.write_bytes(b"OLD-WAV")
        report_path = self.directory / "missing-directory" / "report.json"
        with self.assertRaises(RepairFailure) as captured:
            process_files(input_path, manifest_path, output_path, report_path)
        self.assertEqual(captured.exception.errors[0]["category"], "output_write_failed")
        self.assertEqual(output_path.read_bytes(), b"OLD-WAV")
        self.assertEqual(list(self.directory.glob(".*.stage-*")), [])
        self.assertEqual(list(self.directory.glob(".*.backup-*")), [])

    def test_commit_failure_rolls_back_existing_output(self) -> None:
        source = np.zeros(1000, dtype=np.float64)
        input_path = self.write_wav("input.wav", source)
        manifest_path = self.write_manifest(
            "manifest.json",
            self.base_manifest(
                [
                    {
                        "index": 0,
                        "type": "gain",
                        "start_sample": 100,
                        "end_sample": 200,
                        "gain": 0.5,
                    }
                ]
            ),
        )
        output_path = self.directory / "output.wav"
        report_path = self.directory / "report.json"
        output_path.write_bytes(b"OLD-WAV")
        real_replace = os.replace

        def failing_replace(source_path, destination_path):
            if os.path.abspath(destination_path) == os.path.abspath(report_path):
                raise OSError("injected report replacement failure")
            return real_replace(source_path, destination_path)

        with mock.patch("dialogue_repair.atomic.os.replace", side_effect=failing_replace):
            with self.assertRaises(RepairFailure) as captured:
                process_files(input_path, manifest_path, output_path, report_path)
        self.assertEqual(captured.exception.errors[0]["category"], "output_write_failed")
        self.assertEqual(output_path.read_bytes(), b"OLD-WAV")
        self.assertFalse(report_path.exists())
        self.assertEqual(list(self.directory.glob(".*.stage-*")), [])
        self.assertEqual(list(self.directory.glob(".*.backup-*")), [])

    def test_quantization_saturates_and_rounds_deterministically(self) -> None:
        values = np.array([2.0, -2.0, 0.5 / 32768.0, -0.5 / 32768.0])
        quantized = quantize_int(values, 16)
        self.assertEqual(quantized.tolist(), [32767, -32768, 1, -1])

    def test_wave_format_extensible_pcm_is_readable(self) -> None:
        source = 0.1 * np.sin(
            2.0 * math.pi * 200.0 * np.arange(480, dtype=np.float64) / SAMPLE_RATE
        )
        standard, _ = encode_wav(source, quantization_bits=24)
        extensible_payload = (
            struct.pack(
                "<HHIIHHHHI",
                0xFFFE,
                1,
                SAMPLE_RATE,
                SAMPLE_RATE * 3,
                3,
                24,
                22,
                24,
                0x4,
            )
            + bytes.fromhex("0100000000001000800000aa00389b71")
        )
        data_chunk = standard[36:]
        body = (
            b"WAVE"
            + b"fmt "
            + struct.pack("<I", len(extensible_payload))
            + extensible_payload
            + data_chunk
        )
        extensible = b"RIFF" + struct.pack("<I", len(body)) + body
        path = self.directory / "extensible.wav"
        path.write_bytes(extensible)
        decoded = read_wav(path)
        self.assertEqual(decoded.sample_rate, SAMPLE_RATE)
        self.assertEqual(decoded.channels, 1)
        self.assertEqual(decoded.sample_count, 480)
        self.assertEqual(decoded.bit_depth, 24)
        expected = read_wav(self.write_wav("x.wav", source)).mono_samples
        self.assertTrue(np.array_equal(decoded.mono_samples, expected))

    def test_builtin_selftest(self) -> None:
        result = run_selftest()
        self.assertTrue(result["ok"])
        self.assertEqual(
            [check["name"] for check in result["checks"]],
            [
                "track_length_preserved",
                "bitwise_reproducible",
                "error_paths_produce_no_files",
            ],
        )

    def test_sample_bundle_is_byte_stable(self) -> None:
        first = generate_sample_bundle(self.directory / "first")
        second = generate_sample_bundle(self.directory / "second")
        for first_path, second_path in (
            (first.dialogue_wav, second.dialogue_wav),
            (first.reference_wav, second.reference_wav),
            (first.manifest_json, second.manifest_json),
        ):
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())


if __name__ == "__main__":
    unittest.main()

