"""End-to-end acceptance tests using deterministic generated fixtures."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from audio_norm.cli import main as cli_main  # noqa: E402
from audio_norm.processor import FatalConfigError, run_batch  # noqa: E402
from audio_norm.wavio import read_wav  # noqa: E402
from generate_samples import generate  # noqa: E402


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.mkdtemp(prefix="audio_norm_test_")
        cls.input_dir = os.path.join(cls.workspace, "input")
        cls.output_dir = os.path.join(cls.workspace, "output")
        generate(cls.input_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workspace, ignore_errors=True)

    def _run(self, parallelism=2):
        return cli_main([self.input_dir, self.output_dir, "-j",
                         str(parallelism)])

    def test_partial_run_exit_code_and_status(self):
        code = self._run(parallelism=2)
        self.assertEqual(code, 1)
        report_path = os.path.join(
            self.output_dir, "processing_report.json"
        )
        with open(report_path, encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["overall_status"], "partial")
        self.assertIn(report["overall_status"],
                      ("success", "partial", "failed"))
        self.assertEqual(report["total_tracks"], 9)
        self.assertEqual(report["failed_tracks"], 4)

    def test_output_order_matches_sorted_input(self):
        self._run()
        with open(os.path.join(self.output_dir, "processing_report.json"),
                  encoding="utf-8") as handle:
            report = json.load(handle)
        names = [track["file_name"] for track in report["tracks"]]
        self.assertEqual(names, sorted(names))

    def test_outputs_are_48k_24bit_and_loudness_on_target(self):
        self._run()
        with open(os.path.join(self.output_dir, "processing_report.json"),
                  encoding="utf-8") as handle:
            report = json.load(handle)
        for track in report["tracks"]:
            if track["status"] != "success":
                continue
            path = os.path.join(
                self.output_dir,
                track["file_name"].replace(".wav", "_normalized.wav"),
            )
            data = read_wav(path)
            self.assertEqual(data.sample_rate, 48000)
            self.assertEqual(data.sample_width, 3)
            self.assertEqual(
                data.n_channels, track["source"]["channels"]
            )
            self.assertLessEqual(
                track["loudness"]["loudness_error_lu"], 0.5
            )
            self.assertLessEqual(
                track["peak"]["final_true_peak_dbtp"], -1.0
            )

    def test_limiter_engages_for_dense_track(self):
        self._run()
        with open(os.path.join(self.output_dir, "processing_report.json"),
                  encoding="utf-8") as handle:
            report = json.load(handle)
        dense = next(
            t for t in report["tracks"]
            if t["file_name"] == "dialogue_dense_loud.wav"
        )
        self.assertTrue(dense["limiter"]["limiter_applied"])
        self.assertGreater(dense["limiter"]["trigger_count"], 0)
        self.assertGreater(dense["limiter"]["max_gain_reduction_db"], 0.0)

    def test_subtitle_correction_details_recorded(self):
        self._run()
        with open(os.path.join(self.output_dir, "processing_report.json"),
                  encoding="utf-8") as handle:
            report = json.load(handle)
        stereo = next(
            t for t in report["tracks"]
            if t["file_name"] == "dialogue_stereo.wav"
        )
        adjustments = stereo["subtitles"]["adjustments"]
        self.assertGreaterEqual(len(adjustments), 2)
        first = adjustments[0]
        self.assertEqual(
            set(first),
            {"index", "original_start", "original_end", "new_start",
             "new_end", "silence_start", "silence_end"},
        )
        # Duration must remain unchanged.
        self.assertAlmostEqual(
            (first["new_end"] - first["new_start"]),
            (first["original_end"] - first["original_start"]),
            places=3,
        )
        srt_path = os.path.join(
            self.output_dir, "dialogue_stereo_normalized.srt"
        )
        with open(srt_path, encoding="utf-8") as handle:
            self.assertIn("00:00:01,000", handle.read())

    def test_error_classification(self):
        self._run()
        with open(os.path.join(self.output_dir, "processing_report.json"),
                  encoding="utf-8") as handle:
            report = json.load(handle)
        failed = {
            t["file_name"]: " ".join(t["errors"])
            for t in report["tracks"] if t["status"] == "error"
        }
        self.assertIn("dialogue_corrupt.wav", failed)
        self.assertIn("dialogue_quad.wav", failed)
        self.assertIn("dialogue_32k.wav", failed)
        self.assertIn("dialogue_bad_srt.wav", failed)
        self.assertTrue(report["retained_temp_dirs"])

    def test_resampling_metadata(self):
        self._run()
        with open(os.path.join(self.output_dir, "processing_report.json"),
                  encoding="utf-8") as handle:
            report = json.load(handle)
        mono = next(
            t for t in report["tracks"]
            if t["file_name"] == "dialogue_mono_quiet.wav"
        )
        self.assertTrue(mono["resampling"]["performed"])
        self.assertEqual(
            mono["resampling"]["input_sample_rate_hz"], 44100
        )
        self.assertEqual(
            mono["resampling"]["output_sample_rate_hz"], 48000
        )

    def test_missing_input_is_fatal(self):
        with self.assertRaises(FatalConfigError):
            run_batch(os.path.join(self.workspace, "missing"),
                      os.path.join(self.workspace, "x"))


if __name__ == "__main__":
    unittest.main()
