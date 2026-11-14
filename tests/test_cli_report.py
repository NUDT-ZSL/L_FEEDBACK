import json
import os
import subprocess
import sys
import tempfile
import unittest

from subalign.cli import main
from subalign.report import iso_duration, read_report
from subalign.subtitles import load_subtitle

from tests.samples import (
    linear_drift,
    reference_events,
    segmented_drift,
    translated_events,
    write_bytes,
    write_subtitle,
)


class CliReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = self.temp.name
        self.events = reference_events(count=20)
        self.reference = os.path.join(self.root, "reference.srt")
        write_subtitle(self.reference, self.events)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _target(self, name: str, events, fmt: str = "srt", encoding: str = "utf-8") -> str:
        path = os.path.join(self.root, name)
        write_subtitle(path, events, fmt=fmt, encoding=encoding)
        return path

    def test_batch_run_writes_aligned_files_and_parseable_report(self) -> None:
        linear = self._target("en_us.srt", linear_drift(self.events, 1.03, 900))
        chinese = self._target(
            "zh_cn.srt", linear_drift(translated_events(self.events), 1.0, 650)
        )
        ass_target = self._target(
            "jp.ass", linear_drift(self.events, 1.0, 300), fmt="ass"
        )
        utf16_target = self._target(
            "ru.srt", linear_drift(self.events, 1.0, 1200), encoding="utf-16-le"
        )
        output_dir = os.path.join(self.root, "out")
        report_path = os.path.join(self.root, "report.json")

        exit_code = main(
            [
                "--reference",
                self.reference,
                "--targets",
                linear,
                chinese,
                ass_target,
                utf16_target,
                "--output-dir",
                output_dir,
                "--report",
                report_path,
            ]
        )
        self.assertEqual(exit_code, 0)

        report = read_report(report_path)
        self.assertEqual(report["schema_version"], "1.0")
        self.assertEqual(len(report["files"]), 4)
        self.assertEqual(report["errors"], [])

        expected_names = {"en_us_aligned.srt", "zh_cn_aligned.srt", "jp_aligned.ass", "ru_aligned.srt"}
        self.assertEqual({os.path.basename(item["output_path"]) for item in report["files"]}, expected_names)

        for entry in report["files"]:
            aligned = load_subtitle(entry["output_path"])
            self.assertEqual(len(aligned.events), 20)
            for original, event in zip(self.events, aligned.events):
                self.assertLess(abs(event.start - original[0]), 10)
                self.assertEqual(event.end - event.start, original[1] - original[0])
            self.assertIn("drift_model", entry)
            self.assertIn("match_statistics", entry)
            self.assertIsInstance(entry["warnings"], list)
            self.assertTrue(entry["processing_time"].startswith("PT"))
            self.assertIsInstance(entry["processing_time_ms"], int)

        ass_entry = next(item for item in report["files"] if item["output_path"].endswith(".ass"))
        aligned_ass = load_subtitle(ass_entry["output_path"])
        self.assertEqual(aligned_ass.fmt, "ass")
        self.assertTrue(all(event.style == "Main" for event in aligned_ass.events))

        reloaded = json.dumps(report, ensure_ascii=False)
        self.assertIn("generated_at", reloaded)

    def test_segmented_sample_through_cli(self) -> None:
        boundary = self.events[10][0]
        drifted = segmented_drift(self.events, boundary, 1.0, 700, 1.025, -1200)
        target = self._target("segmented.srt", drifted)
        report_path = os.path.join(self.root, "segmented_report.json")
        exit_code = main(
            [
                "-r",
                self.reference,
                "-t",
                target,
                "--report",
                report_path,
            ]
        )
        self.assertEqual(exit_code, 0)
        report = read_report(report_path)
        model = report["files"][0]["drift_model"]
        self.assertEqual(model["type"], "segmented")
        self.assertGreaterEqual(len(model["segments"]), 2)
        aligned = load_subtitle(report["files"][0]["output_path"])
        for original, event in zip(self.events, aligned.events):
            self.assertLess(abs(event.start - original[0]), 10)

    def test_missing_target_returns_non_zero_with_structured_error(self) -> None:
        report_path = os.path.join(self.root, "bad_report.json")
        exit_code = main(
            [
                "-r",
                self.reference,
                "-t",
                os.path.join(self.root, "ghost.srt"),
                "--report",
                report_path,
            ]
        )
        self.assertEqual(exit_code, 2)

    def test_unsupported_extension_returns_non_zero(self) -> None:
        bad = os.path.join(self.root, "notes.txt")
        write_bytes(bad, b"not a subtitle")
        exit_code = main(
            [
                "-r",
                self.reference,
                "-t",
                bad,
                "--report",
                os.path.join(self.root, "x.json"),
            ]
        )
        self.assertEqual(exit_code, 2)

    def test_corrupt_target_processing_keeps_other_files_and_exits_non_zero(self) -> None:
        good = self._target("good.srt", linear_drift(self.events, 1.0, 500))
        bad = os.path.join(self.root, "broken.srt")
        write_bytes(bad, b"\x80\x81 invalid")
        report_path = os.path.join(self.root, "mixed_report.json")
        exit_code = main(
            [
                "-r",
                self.reference,
                "-t",
                good,
                bad,
                "--report",
                report_path,
            ]
        )
        self.assertEqual(exit_code, 1)
        report = read_report(report_path)
        self.assertEqual(len(report["files"]), 1)
        self.assertEqual(len(report["errors"]), 1)
        self.assertEqual(report["errors"][0]["error_type"], "EncodingError")

    def test_module_invocation(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "subalign", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("--reference", completed.stdout)

    def test_iso_duration_formatting(self) -> None:
        self.assertEqual(iso_duration(0), "PT0S")
        self.assertEqual(iso_duration(1500), "PT1.500S")
        self.assertEqual(iso_duration(65000), "PT1M5S")


if __name__ == "__main__":
    unittest.main()
