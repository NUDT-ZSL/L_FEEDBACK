from __future__ import annotations

import contextlib
import io
import json
import math
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_mixer.cli import main
from audio_mixer.core import (
    build_envelope,
    linear_resample,
    load_project,
    mix_project,
    peak_dbfs,
    seconds_to_samples,
)


def find_ffmpeg() -> str | None:
    candidates = [
        os.environ.get("FFMPEG_BINARY"),
        shutil.which("ffmpeg"),
        "/home/liang/tools/ffmpeg",
        "/tmp/ffmpeg-static/ffmpeg",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return None


class MixerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="audio_mixer_test_")
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_wav(
        self,
        name: str,
        samples: np.ndarray,
        sample_rate: int,
        subtype: str = "FLOAT",
    ) -> Path:
        path = self.root / name
        sf.write(str(path), samples, sample_rate, subtype=subtype, format="WAV")
        return path

    def write_project(
        self,
        clips: list[dict],
        *,
        sample_rate: int = 1000,
        channels: int = 1,
        total_duration_sec: float = 0.01,
        output_format: object = "wav16",
        name: str = "project.json",
        extra: dict | None = None,
    ) -> Path:
        data = {
            "sample_rate": sample_rate,
            "channels": channels,
            "output_format": output_format,
            "total_duration_sec": total_duration_sec,
            "output_name": "mix",
            "clips": clips,
        }
        if extra:
            data.update(extra)
        path = self.root / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path


class ConversionTests(MixerTestCase):
    def test_seconds_to_samples_uses_half_up_rounding(self) -> None:
        self.assertEqual(seconds_to_samples(0.0005, 1000), 1)
        self.assertEqual(seconds_to_samples(0.00149, 1000), 1)
        self.assertEqual(seconds_to_samples(0.0015, 1000), 2)
        self.assertEqual(seconds_to_samples(1.2345, 1000), 1235)
        self.assertEqual(seconds_to_samples(-0.0005, 1000), -1)

    def test_linear_resampling_produces_expected_positions(self) -> None:
        source = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        resampled = linear_resample(source, 4, 8)
        expected = np.array(
            [[0.0], [0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(resampled, expected, atol=1e-7)

    def test_envelope_is_bounded_and_scaled(self) -> None:
        envelope, warning = build_envelope(10, 8, 8)
        self.assertIsNotNone(warning)
        self.assertEqual(warning["code"], "FADE_SCALED")
        self.assertEqual(envelope.shape, (10,))
        self.assertGreaterEqual(float(envelope.min()), 0.0)
        self.assertLessEqual(float(envelope.max()), 1.0)
        self.assertAlmostEqual(float(envelope[0]), 0.0, places=7)
        self.assertAlmostEqual(float(envelope[-1]), 0.0, places=7)

    def test_equal_power_sine_shape(self) -> None:
        envelope, warning = build_envelope(4, 4, 0)
        self.assertIsNone(warning)
        expected = np.sin(0.5 * math.pi * np.arange(4) / 4)
        np.testing.assert_allclose(envelope, expected, atol=1e-7)


class MixingTests(MixerTestCase):
    def test_timeline_alignment_and_success_status(self) -> None:
        self.write_wav("click.wav", np.array([0.1, 0.2, 0.3]), 1000)
        project_path = self.write_project(
            [{"source": "click.wav", "start_sec": 0.002, "duration_sec": 0.003}]
        )
        result = mix_project(load_project(project_path))
        expected = np.zeros(10, dtype=np.float32)
        expected[2:5] = [0.1, 0.2, 0.3]
        np.testing.assert_allclose(result.audio[:, 0], expected, atol=1e-7)
        self.assertEqual(result.clip_reports[0]["status"], "success")
        self.assertEqual(result.clip_reports[0]["start_sample"], 2)
        self.assertEqual(result.clip_reports[0]["end_sample"], 5)

    def test_channel_fill_and_trim(self) -> None:
        self.write_wav("mono.wav", np.array([0.25, 0.5]), 1000)
        stereo_project = self.write_project(
            [
                {
                    "source": "mono.wav",
                    "start_sec": 0.0,
                    "duration_sec": 0.002,
                }
            ],
            channels=2,
        )
        stereo_result = mix_project(load_project(stereo_project))
        np.testing.assert_allclose(
            stereo_result.audio[:, 0], stereo_result.audio[:, 1]
        )

        source = np.column_stack(
            (np.array([0.1, 0.2]), np.array([0.3, 0.4]))
        )
        self.write_wav("stereo.wav", source, 1000)
        mono_project = self.write_project(
            [
                {
                    "source": "stereo.wav",
                    "start_sec": 0.0,
                    "duration_sec": 0.002,
                }
            ],
            channels=1,
        )
        mono_result = mix_project(load_project(mono_project))
        np.testing.assert_allclose(
            mono_result.audio[:2, 0], [0.1, 0.2], atol=1e-7
        )

    def test_truncation_and_invalid_clips(self) -> None:
        self.write_wav("tail.wav", np.array([0.2, 0.3, 0.4, 0.5]), 1000)
        project_path = self.write_project(
            [
                {
                    "source": "tail.wav",
                    "start_sec": 0.003,
                    "duration_sec": 0.005,
                },
                {
                    "source": "tail.wav",
                    "start_sec": -0.001,
                    "duration_sec": 0.002,
                },
                {
                    "source": "tail.wav",
                    "start_sec": 0.0,
                    "duration_sec": 0.0,
                },
                {
                    "source": "missing.wav",
                    "start_sec": 0.0,
                    "duration_sec": 0.002,
                },
            ],
            total_duration_sec=0.005,
        )
        result = mix_project(load_project(project_path))
        np.testing.assert_allclose(
            result.audio[3:5, 0], [0.2, 0.3], atol=1e-7
        )
        self.assertEqual(result.clip_reports[0]["status"], "truncated")
        self.assertIn("TIMELINE_TRUNCATED", result.clip_reports[0]["warning_codes"])
        self.assertEqual(
            result.clip_reports[1]["error_code"], "INVALID_START"
        )
        self.assertEqual(
            result.clip_reports[2]["error_code"], "INVALID_DURATION"
        )
        self.assertEqual(
            result.clip_reports[3]["error_code"], "SOURCE_NOT_FOUND"
        )

    def test_damaged_media_is_skipped(self) -> None:
        bad = self.root / "bad.wav"
        bad.write_bytes(b"not a wave file")
        project_path = self.write_project(
            [{"source": "bad.wav", "start_sec": 0.0, "duration_sec": 0.002}]
        )
        result = mix_project(load_project(project_path))
        self.assertEqual(result.clip_reports[0]["status"], "skipped")
        self.assertEqual(
            result.clip_reports[0]["error_code"], "SOURCE_DECODE_ERROR"
        )
        self.assertEqual(peak_dbfs(result.audio), -120.0)

    def test_resampling_warning_is_reported(self) -> None:
        self.write_wav(
            "low-rate.wav", np.array([0.0, 0.2, 0.4, 0.6]), 500
        )
        project_path = self.write_project(
            [
                {
                    "source": "low-rate.wav",
                    "start_sec": 0.0,
                    "duration_sec": 0.008,
                }
            ],
            sample_rate=1000,
            total_duration_sec=0.008,
        )
        result = mix_project(load_project(project_path))
        np.testing.assert_allclose(
            result.audio[:, 0],
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6],
            atol=1e-7,
        )
        self.assertIn("RESAMPLED", result.clip_reports[0]["warning_codes"])
        self.assertEqual(result.clip_reports[0]["status"], "warning")

    def test_soft_limiter_caps_overlap(self) -> None:
        self.write_wav("loud.wav", np.full(4, 0.9), 1000)
        clip = {
            "source": "loud.wav",
            "start_sec": 0.0,
            "duration_sec": 0.004,
        }
        project_path = self.write_project([clip, dict(clip)])
        result = mix_project(load_project(project_path))
        self.assertLessEqual(float(np.max(np.abs(result.audio))), 0.95 + 1e-6)
        self.assertEqual(result.normalization_gain_db, 0.0)

    def test_non_overlapping_peak_is_normalized_to_minus_point_one(self) -> None:
        self.write_wav("loud.wav", np.full(4, 0.9), 1000)
        project_path = self.write_project(
            [
                {
                    "source": "loud.wav",
                    "start_sec": 0.0,
                    "duration_sec": 0.004,
                    "gain_db": 6.020599913,
                }
            ]
        )
        result = mix_project(load_project(project_path))
        self.assertAlmostEqual(result.normalization_gain_db, -5.2054, places=3)
        self.assertAlmostEqual(result.peak_dbfs, -0.1, places=2)


class CliIntegrationTests(MixerTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.ffmpeg = find_ffmpeg()

    @contextlib.contextmanager
    def capture_cli(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            yield stdout, stderr

    def run_cli(self, *args: object) -> tuple[int, str, str]:
        with self.capture_cli() as (stdout, stderr):
            code = main([str(arg) for arg in args])
        return code, stdout.getvalue(), stderr.getvalue()

    @unittest.skipUnless(find_ffmpeg(), "FFmpeg is required for integration tests")
    def test_wav24_output_and_report_match_decoded_audio(self) -> None:
        self.write_wav("tone.wav", np.full(12, 0.4), 1000)
        project_path = self.write_project(
            [
                {
                    "source": "tone.wav",
                    "start_sec": 0.001,
                    "duration_sec": 0.008,
                    "fade_in_sec": 0.002,
                    "fade_out_sec": 0.002,
                }
            ],
            output_format="wav24",
        )
        output_dir = self.root / "out"
        code, stdout, stderr = self.run_cli(
            project_path, output_dir, "--ffmpeg", self.ffmpeg
        )
        self.assertEqual(code, 0, stderr)
        self.assertIn("output:", stdout)
        output = output_dir / "mix.wav"
        report_path = output_dir / "mix.report.json"
        self.assertTrue(output.is_file())
        self.assertTrue(report_path.is_file())

        info = sf.info(str(output))
        self.assertEqual(info.samplerate, 1000)
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.subtype, "PCM_24")
        decoded, _ = sf.read(str(output), dtype="float32", always_2d=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "success")
        self.assertEqual(report["sample_count"], decoded.shape[0])
        self.assertAlmostEqual(
            report["duration_sec"], round(decoded.shape[0] / info.samplerate, 3)
        )
        self.assertAlmostEqual(
            report["peak_dbfs"], round(peak_dbfs(decoded), 3), places=3
        )
        self.assertIn("ffmpeg", report)
        self.assertIn("command_summary", report["ffmpeg"])

    @unittest.skipUnless(find_ffmpeg(), "FFmpeg is required for integration tests")
    def test_flac_output(self) -> None:
        self.write_wav("tone.wav", np.full(20, 0.2), 2000)
        project_path = self.write_project(
            [
                {
                    "source": "tone.wav",
                    "start_sec": 0.0,
                    "duration_sec": 0.008,
                }
            ],
            sample_rate=4000,
            output_format="flac",
            total_duration_sec=0.01,
        )
        code, _, stderr = self.run_cli(
            project_path, self.root / "out-flac", "--ffmpeg", self.ffmpeg
        )
        self.assertEqual(code, 0, stderr)
        output = self.root / "out-flac" / "mix.flac"
        info = sf.info(str(output))
        self.assertEqual(info.format, "FLAC")
        self.assertEqual(info.samplerate, 4000)

    @unittest.skipUnless(find_ffmpeg(), "FFmpeg is required for integration tests")
    def test_wav16_output(self) -> None:
        self.write_wav("tone.wav", np.full(10, 0.2), 1000)
        project_path = self.write_project(
            [
                {
                    "source": "tone.wav",
                    "start_sec": 0.0,
                    "duration_sec": 0.005,
                }
            ],
            output_format="wav16",
        )
        code, _, stderr = self.run_cli(
            project_path, self.root / "out-wav16", "--ffmpeg", self.ffmpeg
        )
        self.assertEqual(code, 0, stderr)
        info = sf.info(str(self.root / "out-wav16" / "mix.wav"))
        self.assertEqual(info.format, "WAV")
        self.assertEqual(info.subtype, "PCM_16")

    def test_ffmpeg_failure_writes_report_and_removes_audio(self) -> None:
        self.write_wav("tone.wav", np.full(10, 0.1), 1000)
        project_path = self.write_project(
            [{"source": "tone.wav", "start_sec": 0.0, "duration_sec": 0.005}]
        )
        fake = self.root / "fake-ffmpeg"
        fake.write_text(
            "#!/bin/sh\n"
            "echo 'synthetic encoder failure' >&2\n"
            "exit 7\n",
            encoding="utf-8",
        )
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        output_dir = self.root / "failed-out"
        code, _, stderr = self.run_cli(
            project_path, output_dir, "--ffmpeg", fake
        )
        self.assertEqual(code, 6)
        self.assertIn("synthetic encoder failure", stderr)
        self.assertFalse((output_dir / "mix.wav").exists())
        report = json.loads(
            (output_dir / "mix.report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["code"], "FFMPEG_ENCODE_FAILED")
        self.assertEqual(report["ffmpeg"]["returncode"], 7)
        self.assertIn(
            "synthetic encoder failure", report["error"]["ffmpeg_stderr"]
        )

    def test_global_error_exit_codes(self) -> None:
        code, _, stderr = self.run_cli(self.root / "absent.json", self.root / "out")
        self.assertEqual(code, 2)
        self.assertIn("not found", stderr)

        invalid = self.root / "invalid.json"
        invalid.write_text("{broken", encoding="utf-8")
        code, _, stderr = self.run_cli(invalid, self.root / "out")
        self.assertEqual(code, 3)
        self.assertIn("invalid project JSON", stderr)

        valid = self.write_project([])
        code, _, stderr = self.run_cli(
            valid, self.root / "out", "--ffmpeg", self.root / "no-ffmpeg"
        )
        self.assertEqual(code, 5)
        self.assertIn("FFmpeg", stderr)

    @unittest.skipUnless(find_ffmpeg(), "FFmpeg is required for integration tests")
    def test_clip_level_errors_still_return_success(self) -> None:
        self.write_wav("good.wav", np.full(10, 0.2), 1000)
        bad = self.root / "bad.wav"
        bad.write_bytes(b"broken")
        project_path = self.write_project(
            [
                {
                    "source": "good.wav",
                    "start_sec": 0.0,
                    "duration_sec": 0.003,
                },
                {
                    "source": "missing.wav",
                    "start_sec": 0.002,
                    "duration_sec": 0.002,
                },
                {
                    "source": "good.wav",
                    "start_sec": -0.001,
                    "duration_sec": 0.002,
                },
                {
                    "source": "bad.wav",
                    "start_sec": 0.0,
                    "duration_sec": 0.002,
                },
            ]
        )
        code, _, stderr = self.run_cli(
            project_path, self.root / "partial-out", "--ffmpeg", self.ffmpeg
        )
        self.assertEqual(code, 0, stderr)
        report = json.loads(
            (self.root / "partial-out" / "mix.report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["clips"][0]["status"], "success")
        self.assertEqual(report["clips"][1]["error_code"], "SOURCE_NOT_FOUND")
        self.assertEqual(report["clips"][2]["error_code"], "INVALID_START")
        self.assertEqual(
            report["clips"][3]["error_code"], "SOURCE_DECODE_ERROR"
        )


if __name__ == "__main__":
    unittest.main()
