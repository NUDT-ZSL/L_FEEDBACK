"""Fast unit tests for fixed rules, reporting, CLI, and cleanup."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from mxf_r128 import cli
from mxf_r128.constants import (
    E_ARGUMENT,
    E_DIR_NOT_FOUND,
    E_LOUDNESS_FAIL,
    E_MEDIA_CORRUPT,
    E_REPORT_WRITE,
    E_SYNC_FAIL,
    E_TIMEOUT,
    E_UNEXPECTED,
    ERROR_CODES,
    W_AUDIO_DELAY,
    W_FPS_DEFAULT,
    W_FPS_INFERRED,
    WARNING_CODES,
)
from mxf_r128.loudness import (
    LoudnessMeasurement,
    loudness_passes,
    normalization_gain_db,
)
from mxf_r128.pipeline import _audio_filter
from mxf_r128.process import run_command
from mxf_r128.rates import resolve_frame_rate
from mxf_r128.report import build_report, empty_file_record, write_report_atomic


class RuleTests(unittest.TestCase):
    def test_fps_inferred_from_nominal_rate(self) -> None:
        resolved = resolve_frame_rate(
            {"avg_frame_rate": "0/0", "r_frame_rate": "25/1"}
        )
        self.assertEqual(resolved.text, "25/1")
        self.assertEqual(resolved.warning, W_FPS_INFERRED)

    def test_fps_inferred_from_time_base(self) -> None:
        resolved = resolve_frame_rate(
            {
                "avg_frame_rate": "0/0",
                "r_frame_rate": "0/0",
                "time_base": "1/25",
            }
        )
        self.assertEqual(resolved.text, "25/1")
        self.assertEqual(resolved.warning, W_FPS_INFERRED)

    def test_fps_default_when_no_inference(self) -> None:
        resolved = resolve_frame_rate(
            {
                "avg_frame_rate": "0/0",
                "r_frame_rate": "0/0",
                "time_base": "1/90000",
            }
        )
        self.assertEqual(resolved.text, "25/1")
        self.assertEqual(resolved.warning, W_FPS_DEFAULT)

    def test_loudness_gain_and_thresholds(self) -> None:
        measurement = LoudnessMeasurement(-19.0, 3.0, -8.0)
        self.assertAlmostEqual(normalization_gain_db(measurement) or 0.0, -4.0)
        self.assertTrue(
            loudness_passes(LoudnessMeasurement(-23.2, 1.0, -2.0))
        )
        self.assertFalse(
            loudness_passes(LoudnessMeasurement(-22.4, 1.0, -2.0))
        )
        self.assertFalse(
            loudness_passes(LoudnessMeasurement(-23.0, 1.0, -0.9))
        )

    def test_transient_gain_is_peak_limited(self) -> None:
        measurement = LoudnessMeasurement(-26.95, 0.0, -0.91)
        gain = normalization_gain_db(measurement)
        self.assertIsNotNone(gain)
        self.assertAlmostEqual(gain or 0.0, -0.09)

    def test_51_pan_is_explicit_and_not_downmixed(self) -> None:
        audio_filter = _audio_filter(6, 0.0)
        self.assertIn("pan=5.1|FL=c0|FR=c1|FC=c2|LFE=c3|BL=c4|BR=c5", audio_filter)
        self.assertNotIn("stereo", audio_filter)

    def test_symbols_are_fixed(self) -> None:
        for symbol in (
            E_MEDIA_CORRUPT,
            E_LOUDNESS_FAIL,
            E_SYNC_FAIL,
            E_TIMEOUT,
            E_DIR_NOT_FOUND,
        ):
            self.assertIn(symbol, ERROR_CODES)
        self.assertEqual(
            {W_FPS_INFERRED, W_FPS_DEFAULT, W_AUDIO_DELAY}, set(WARNING_CODES)
        )


class ReportTests(unittest.TestCase):
    def test_records_are_sorted_and_schema_is_complete(self) -> None:
        first = empty_file_record("b.mxf")
        second = empty_file_record("a.mxf")
        report = build_report(
            input_dir=Path("/in"),
            output_dir=Path("/out"),
            max_workers=2,
            records=[first, second],
        )
        self.assertEqual(
            [entry["source_file"] for entry in report["files"]],
            ["a.mxf", "b.mxf"],
        )
        required = {
            "source_file",
            "status",
            "output_path",
            "video_codec",
            "video_profile",
            "pix_fmt",
            "frame_rate",
            "loudness_lufs",
            "loudness_lra",
            "true_peak_dbtp",
            "channel_layout",
            "warnings",
            "error_code",
        }
        self.assertLessEqual(required, set(report["files"][0]))

    def test_report_replacement_is_atomic_and_leaves_no_temp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "nested" / "qc.json"
            first = {"version": 1, "files": []}
            second = {
                "version": 2,
                "files": [{"source_file": "a.mxf"}],
            }
            write_report_atomic(report_path, first)
            write_report_atomic(report_path, second)
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded, second)
            leftovers = list(report_path.parent.glob("*.tmp"))
            self.assertEqual([], leftovers)


class ProcessCleanupTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "process-group assertion uses POSIX signals")
    def test_timeout_kills_entire_process_group(self) -> None:
        script = (
            "import subprocess,sys,time\n"
            "child = subprocess.Popen("
            "[sys.executable,'-c','import time;time.sleep(60)'])\n"
            "print(child.pid, flush=True)\n"
            "time.sleep(60)\n"
        )
        result = run_command(
            [sys.executable, "-c", script],
            timeout=0.6,
        )
        self.assertTrue(result.timed_out)
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        self.assertTrue(lines)
        child_pid = int(lines[0])
        deadline = time.monotonic() + 3.0
        alive = True
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                alive = False
                break
            time.sleep(0.05)
        self.assertFalse(alive)


class CliExitCodeTests(unittest.TestCase):
    def test_missing_input_directory_is_exit_2(self) -> None:
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary:
            with redirect_stderr(stream):
                code = cli.main(
                    [
                        "--input-dir",
                        str(Path(temporary) / "missing"),
                        "--output-dir",
                        str(Path(temporary) / "out"),
                        "--report",
                        str(Path(temporary) / "qc.json"),
                    ]
                )
        self.assertEqual(code, 2)
        self.assertIn(E_DIR_NOT_FOUND, stream.getvalue())

    def test_invalid_workers_is_exit_2(self) -> None:
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary:
            with redirect_stderr(stream):
                code = cli.main(
                    [
                        "--input-dir",
                        temporary,
                        "--output-dir",
                        str(Path(temporary) / "out"),
                        "--report",
                        str(Path(temporary) / "qc.json"),
                        "--max-workers",
                        "0",
                    ]
                )
        self.assertEqual(code, 2)
        self.assertIn(E_ARGUMENT, stream.getvalue())

    def test_uncaught_error_is_exit_3_and_uses_business_code(self) -> None:
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch(
                "mxf_r128.cli.run_batch", side_effect=RuntimeError("secret")
            ):
                with mock.patch(
                    "mxf_r128.cli.resolve_tool", return_value="tool"
                ):
                    with redirect_stderr(stream):
                        code = cli.main(
                            [
                                "--input-dir",
                                temporary,
                                "--output-dir",
                                str(Path(temporary) / "out"),
                                "--report",
                                str(Path(temporary) / "qc.json"),
                            ]
                        )
        self.assertEqual(code, 3)
        self.assertIn(E_UNEXPECTED, stream.getvalue())
        self.assertNotIn("secret", stream.getvalue())

    def test_report_write_failure_is_exit_3(self) -> None:
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch(
                "mxf_r128.cli.run_batch", return_value=[]
            ), mock.patch(
                "mxf_r128.cli.resolve_tool", return_value="tool"
            ), mock.patch(
                "mxf_r128.cli.write_report_atomic",
                side_effect=OSError("disk"),
            ), redirect_stderr(stream):
                code = cli.main(
                    [
                        "--input-dir",
                        temporary,
                        "--output-dir",
                        str(Path(temporary) / "out"),
                        "--report",
                        str(Path(temporary) / "qc.json"),
                    ]
                )
        self.assertEqual(code, 3)
        self.assertIn(E_REPORT_WRITE, stream.getvalue())

    def test_missing_tools_are_exit_2(self) -> None:
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch(
                "mxf_r128.cli.resolve_tool", return_value=None
            ), redirect_stderr(stream):
                code = cli.main(
                    [
                        "--input-dir",
                        temporary,
                        "--output-dir",
                        str(Path(temporary) / "out"),
                        "--report",
                        str(Path(temporary) / "qc.json"),
                    ]
                )
        self.assertEqual(code, 2)
        self.assertIn("E_FFMPEG_NOT_FOUND", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
