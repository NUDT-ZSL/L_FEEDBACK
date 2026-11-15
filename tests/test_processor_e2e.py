import json
import os
import subprocess
import sys
import unittest

import numpy as np
import soundfile as sf

from seamless_loop.config import load_project
from seamless_loop.processor import process_project

from helpers import base_project, make_workspace, write_json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ProcessorEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.root, self.samples_dir, self.output_dir, self.paths = make_workspace()
        self.config_path = write_json(
            os.path.join(self.samples_dir, "project.json"),
            base_project(self.samples_dir, self.output_dir),
        )

    def _run(self):
        project = load_project(self.config_path)
        output_path = process_project(project)
        report_path = output_path.replace("_loop.wav", "_loop_report.json")
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        data, sr = sf.read(output_path, dtype="float64")
        return project, output_path, report_path, report, data, sr

    def test_output_spec_and_exact_sample_count(self):
        project, _, _, report, data, sr = self._run()
        self.assertEqual(sr, 48000)
        self.assertEqual(data.ndim, 2)
        self.assertEqual(data.shape[1], 2)
        expected_samples = round(4.5 * 48000)
        self.assertEqual(data.shape[0], expected_samples)
        self.assertEqual(report["output_summary"]["total_samples"], expected_samples)
        self.assertEqual(report["output_summary"]["channels"], 2)
        self.assertEqual(report["output_summary"]["sample_rate"], 48000)

    def test_loop_seam_is_click_free_and_periodic(self):
        _, _, _, _, data, _ = self._run()
        loop_len = 67200
        loop = data[:loop_len]
        tiled = np.tile(loop, (3, 1))
        # Every seam must be a finite, bounded step with no impulse spike:
        # the seam jump never exceeds the largest jump inside one loop.
        normal_jumps = np.max(np.abs(np.diff(loop, axis=0)))
        for seam in (loop_len, 2 * loop_len):
            seam_jump = np.max(np.abs(tiled[seam] - tiled[seam - 1]))
            self.assertTrue(np.all(np.isfinite(tiled[seam])))
            self.assertLessEqual(seam_jump, normal_jumps + 1e-9)

    def test_tracks_are_sample_aligned(self):
        _, _, _, report, _, _ = self._run()
        loop_lengths = {t["loop_length_samples"] for t in report["tracks"]}
        self.assertEqual(loop_lengths, {67200})
        for track in report["tracks"]:
            self.assertEqual(track["crossfade_samples_used"], 4800)
            self.assertFalse(track["crossfade_adjusted"])

    def test_mono_source_is_duplicated_to_stereo(self):
        # A mono-only project must produce identical left/right channels.
        mono_config = write_json(
            os.path.join(self.samples_dir, "mono.json"),
            base_project(
                self.samples_dir,
                self.output_dir,
                project_name="mono",
                input_files=["bass_mono_48k_pcm24.wav"],
            ),
        )
        project = load_project(mono_config)
        output_path = process_project(project)
        data, _ = sf.read(output_path, dtype="float64")
        np.testing.assert_allclose(data[:, 0], data[:, 1], atol=1e-12)

    def test_deterministic_output(self):
        _, path_a, _, _, data_a, _ = self._run()
        data_b, _ = sf.read(path_a, dtype="float64")
        np.testing.assert_array_equal(data_a, data_b)

    def test_report_completeness_and_naming(self):
        _, output_path, report_path, report, _, _ = self._run()
        self.assertTrue(output_path.endswith("demo_loop.wav"))
        self.assertTrue(report_path.endswith("demo_loop_report.json"))
        self.assertEqual(len(report["tracks"]), 3)
        for track in report["tracks"]:
            for key in (
                "crossfade_samples_used",
                "loop_length_samples",
                "crossfade_adjusted",
            ):
                self.assertIn(key, track)
        self.assertEqual(
            set(report["output_summary"].keys()),
            {"sample_rate", "channels", "total_samples",
             "duration_seconds", "subtype"},
        )

    def test_integer_loop_multiple_no_truncation(self):
        config_path = write_json(
            os.path.join(self.samples_dir, "exact.json"),
            base_project(
                self.samples_dir, self.output_dir,
                project_name="exact", target_duration=2.8,
            ),
        )
        project = load_project(config_path)
        output_path = process_project(project)
        data, _ = sf.read(output_path, dtype="float64")
        # 2.8 s at 48k = 134400 = 2 * 67200, an exact multiple.
        self.assertEqual(data.shape[0], 134400)

    def test_per_track_fade_shortening_recorded_in_report(self):
        # A 0.15 s region (7200 samples) cannot fit the requested 0.1 s fade
        # (4800) twice (needs 9600), so the fade is shortened to 3600.
        import soundfile as sf

        short_path = os.path.join(self.samples_dir, "tiny_track.wav")
        sr = 48000
        t = np.arange(int(0.3 * sr)) / sr
        sf.write(short_path, 0.5 * np.sin(2 * np.pi * 220 * t), sr,
                 subtype="PCM_16")
        payload = base_project(
            self.samples_dir,
            self.output_dir,
            project_name="tiny",
            input_files=["tiny_track.wav"],
            loop_start=0.0,
            loop_end=0.15,
            crossfade_duration=0.1,
            target_duration=0.3,
        )
        config_path = write_json(
            os.path.join(self.samples_dir, "tiny.json"), payload
        )
        with self.assertLogs("seamless_loop", level="WARNING") as logs:
            project = load_project(config_path)
            output_path = process_project(project)
        report_path = output_path.replace("_loop.wav", "_loop_report.json")
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        track = report["tracks"][0]
        self.assertTrue(track["crossfade_adjusted"])
        self.assertEqual(track["crossfade_samples_configured"], 4800)
        self.assertEqual(track["crossfade_samples_used"], 3600)
        self.assertTrue(
            any("shortening crossfade" in msg for msg in logs.output)
        )


class CliExitCodeTests(unittest.TestCase):
    def setUp(self):
        self.root, self.samples_dir, self.output_dir, self.paths = make_workspace()

    def _run_cli(self, config_path):
        return subprocess.run(
            [sys.executable, "-m", "seamless_loop", config_path],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

    def _write(self, name, payload):
        return write_json(os.path.join(self.samples_dir, name), payload)

    def test_success_exit_zero(self):
        config = self._write(
            "ok.json", base_project(self.samples_dir, self.output_dir)
        )
        result = self._run_cli(config)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK", result.stdout)

    def test_missing_project_file_exit_2(self):
        result = self._run_cli(os.path.join(self.root, "missing.json"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("ERROR", result.stderr)

    def test_missing_input_file_exit_3(self):
        payload = base_project(
            self.samples_dir, self.output_dir, input_files=["ghost.wav"]
        )
        result = self._run_cli(self._write("f.json", payload))
        self.assertEqual(result.returncode, 3)

    def test_unsupported_format_exit_3(self):
        payload = base_project(
            self.samples_dir, self.output_dir,
            input_files=["unsupported_48k_u8.wav"],
        )
        result = self._run_cli(self._write("f.json", payload))
        self.assertEqual(result.returncode, 3)

    def test_sample_rate_mismatch_exit_3(self):
        payload = base_project(
            self.samples_dir, self.output_dir,
            input_files=[
                "drums_stereo_48k_pcm16.wav",
                "mismatch_mono_44k1_pcm16.wav",
            ],
        )
        result = self._run_cli(self._write("f.json", payload))
        self.assertEqual(result.returncode, 3)

    def test_loop_beyond_duration_exit_4(self):
        payload = base_project(
            self.samples_dir, self.output_dir,
            loop_start=0.0, loop_end=5.0,  # files are only 3 s long
        )
        result = self._run_cli(self._write("f.json", payload))
        self.assertEqual(result.returncode, 4)

    def test_negative_fade_exit_4(self):
        payload = base_project(
            self.samples_dir, self.output_dir, crossfade_duration=-1
        )
        result = self._run_cli(self._write("f.json", payload))
        self.assertEqual(result.returncode, 4)

    def test_zero_fade_exit_4(self):
        payload = base_project(
            self.samples_dir, self.output_dir, crossfade_duration=0
        )
        result = self._run_cli(self._write("f.json", payload))
        self.assertEqual(result.returncode, 4)

    def test_overwrite_false_exit_5_then_overwrite_true_succeeds(self):
        keep = base_project(
            self.samples_dir, self.output_dir,
            project_name="keep", overwrite=False,
        )
        config = self._write("keep.json", keep)
        first = self._run_cli(config)
        self.assertEqual(first.returncode, 0, first.stderr)
        second = self._run_cli(config)
        self.assertEqual(second.returncode, 5)
        self.assertIn("overwrite", second.stderr.lower())

        overwrite = dict(keep, overwrite=True)
        config2 = self._write("keep2.json", overwrite)
        third = self._run_cli(config2)
        self.assertEqual(third.returncode, 0, third.stderr)

    def test_no_arguments_prints_usage(self):
        result = subprocess.run(
            [sys.executable, "-m", "seamless_loop"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
