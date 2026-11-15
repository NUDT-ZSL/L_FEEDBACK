from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from scene_mix.batch import run_batch
from scene_mix.dsp import (
    K_STAGE1_A,
    K_STAGE1_B,
    K_STAGE2_A,
    K_STAGE2_B,
    make_gain_envelope,
    peak_dbfs,
)
from scene_mix.pipeline import mix_group
from scene_mix.synth import generate_group

ROOT = Path(__file__).resolve().parents[1]


def _direct_biquad(samples: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    output = np.zeros_like(samples)
    for index in range(samples.size):
        value = b[0] * samples[index]
        if index >= 1:
            value += b[1] * samples[index - 1] - a[1] * output[index - 1]
        if index >= 2:
            value += b[2] * samples[index - 2] - a[2] * output[index - 2]
        output[index] = value / a[0]
    return output


class DspTests(unittest.TestCase):
    def test_k_filter_matches_direct_iir(self) -> None:
        from scene_mix.dsp import k_filter_48k

        rng = np.random.default_rng(123)
        samples = rng.standard_normal(4096)
        direct = _direct_biquad(samples, K_STAGE1_B, K_STAGE1_A)
        direct = _direct_biquad(direct, K_STAGE2_B, K_STAGE2_A)
        fft_result = k_filter_48k(samples[:, None])[:, 0]
        self.assertLess(float(np.max(np.abs(direct - fft_result))), 1e-9)

    def test_gain_envelope_has_linear_50ms_crossfade(self) -> None:
        boundary = 48_000
        fade = 2_400
        envelope = make_gain_envelope(
            [6.0, -6.0],
            [boundary],
            output_length=96_000,
            sample_rate=48_000,
            fade_seconds=0.050,
        )
        ramp = envelope[boundary - fade : boundary + fade]
        self.assertAlmostEqual(float(envelope[boundary - fade]), 10 ** (6.0 / 20.0), places=9)
        expected_midpoint = (
            10 ** (6.0 / 20.0) + 10 ** (-6.0 / 20.0)
        ) / 2.0
        self.assertAlmostEqual(float(ramp[fade]), expected_midpoint, places=2)
        steps = np.diff(ramp)
        self.assertLess(float(np.max(steps) - np.min(steps)), 1e-10)
        self.assertAlmostEqual(float(envelope[boundary + fade]), 10 ** (-6.0 / 20.0), places=9)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory(prefix="scene_mix_tests_")
        cls.root = Path(cls._temp.name)
        cls.input_root = cls.root / "inputs"
        cls.group_dir = cls.input_root / "seed_101"
        cls.timeline = generate_group(
            seed=101,
            output_dir=cls.group_dir,
            channels=6,
            silent_track=2,
            clipped_track=3,
        )
        cls.output_dir = cls.root / "outputs" / "seed_101"
        cls.report = mix_group(
            cls.group_dir / "timeline.json",
            cls.output_dir,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def test_synth_is_sample_deterministic_and_has_required_faults(self) -> None:
        second = self.root / "repeat" / "seed_101"
        timeline = generate_group(
            seed=101,
            output_dir=second,
            channels=6,
            silent_track=2,
            clipped_track=3,
        )
        self.assertEqual(
            (self.group_dir / "timeline.json").read_bytes(),
            (second / "timeline.json").read_bytes(),
        )
        self.assertEqual(timeline["tracks"], self.timeline["tracks"])
        for track in self.timeline["tracks"]:
            self.assertEqual(
                (self.group_dir / track["file"]).read_bytes(),
                (second / track["file"]).read_bytes(),
            )
            self.assertLessEqual(abs(int(track["offset_samples"])), 14_400)

        self.assertEqual(self.timeline["reference_track"], "cam_01.wav")
        silent_data, _ = sf.read(
            str(self.group_dir / "cam_02.wav"), dtype="float64", always_2d=True
        )
        self.assertEqual(float(np.max(np.abs(silent_data))), 0.0)
        clipped_data, _ = sf.read(
            str(self.group_dir / "cam_03.wav"), dtype="float64", always_2d=True
        )
        self.assertGreaterEqual(
            int(np.count_nonzero(np.abs(clipped_data) >= 0.999)), 8
        )
        for track in self.timeline["tracks"]:
            if track["measured_lufs"] is not None:
                self.assertGreaterEqual(float(track["measured_lufs"]), -30.1)
                self.assertLessEqual(float(track["measured_lufs"]), -5.95)
        self.assertFalse((self.group_dir / "timeline.json").read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_alignment_loudness_gain_warnings_and_output(self) -> None:
        report = self.report
        warnings = set(report["warnings"])
        self.assertEqual(
            warnings,
            {"SILENT_TRACK", "CLIPPED_INPUT", "GAIN_CLAMPED", "LIMITER_ACTIVE"},
        )

        by_id = {track["id"]: track for track in report["tracks"]}
        self.assertIsNone(by_id["cam_02"]["offset_samples"])
        self.assertEqual(by_id["cam_02"]["business_code"], "SILENT_TRACK")
        self.assertTrue(by_id["cam_03"]["clipped_input"])
        self.assertIn("CLIPPED_INPUT", by_id["cam_03"]["warnings"])
        for track in report["tracks"]:
            if track["status"] == "ok":
                self.assertLessEqual(int(track["offset_error_samples"]), 5)
                self.assertTrue(track["offset_error_within_5_samples"])

        self.assertEqual(len(report["scenes"]), 4)
        for scene in report["scenes"]:
            self.assertIsNotNone(scene["loudness_lufs"])
            self.assertGreaterEqual(float(scene["applied_gain_db"]), -12.0)
            self.assertLessEqual(float(scene["applied_gain_db"]), 12.0)
        self.assertTrue(any(scene["gain_clamped"] for scene in report["scenes"]))
        self.assertTrue(any(scene["applied_gain_db"] == -12.0 for scene in report["scenes"]))
        self.assertGreater(float(report["pre_normalization_peak_dbfs"]), 0.0)
        self.assertTrue(report["limiter_active"])
        self.assertAlmostEqual(float(report["final_output_peak_dbfs"]), -1.0, places=4)

        output = self.output_dir / "mix.wav"
        info = sf.info(str(output))
        self.assertEqual(info.samplerate, 48_000)
        self.assertEqual(info.channels, 2)
        self.assertEqual(info.subtype, "PCM_16")
        samples, _ = sf.read(str(output), dtype="float64", always_2d=True)
        self.assertAlmostEqual(peak_dbfs(samples), -1.0, delta=0.02)
        self.assertFalse(
            (self.output_dir / "report.json").read_bytes().startswith(b"\xef\xbb\xbf")
        )

    def test_batch_isolation_and_jobs_cap(self) -> None:
        batch_root = self.root / "batch"
        batch_inputs = batch_root / "inputs"
        generate_group(seed=210, output_dir=batch_inputs / "good", channels=2)
        generate_group(seed=211, output_dir=batch_inputs / "bad", channels=2)
        bad_timeline = batch_inputs / "bad" / "timeline.json"
        bad_timeline.write_text("{not valid json", encoding="utf-8")

        output_root = batch_root / "outputs"
        summary = run_batch(batch_inputs, output_root, jobs_requested=999)
        self.assertLessEqual(summary["jobs_used"], max(1, os.cpu_count() or 1))
        statuses = {group["name"]: group["status"] for group in summary["groups"]}
        self.assertEqual(statuses["good"], "success")
        self.assertEqual(statuses["bad"], "failed")
        self.assertEqual(summary["success"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertTrue((output_root / "good" / "report.json").is_file())
        self.assertTrue((output_root / "good" / "mix.wav").is_file())
        self.assertTrue((output_root / "bad" / ".progress").is_file())

        second = run_batch(batch_inputs, output_root, jobs_requested=2)
        second_statuses = {
            group["name"]: group["status"] for group in second["groups"]
        }
        self.assertEqual(second_statuses["good"], "skipped_completed")
        self.assertEqual(second_statuses["bad"], "failed")

    @unittest.skipIf(os.name == "nt", "POSIX process-group kill test")
    def test_strong_kill_recovery_accumulates_retry_count(self) -> None:
        batch_root = self.root / "kill_recovery"
        inputs = batch_root / "inputs"
        outputs = batch_root / "outputs"
        generate_group(seed=770, output_dir=inputs / "seed_770", channels=2)

        env = dict(os.environ)
        env["SCENE_MIX_TEST_DELAY_MS"] = "8000"
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "main.py"),
                "batch",
                "--input-root",
                str(inputs),
                "--output-root",
                str(outputs),
                "--jobs",
                "1",
            ],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            text=True,
        )
        progress = outputs / "seed_770" / ".progress"
        deadline = time.time() + 20.0
        while time.time() < deadline and not progress.is_file():
            time.sleep(0.05)
        self.assertTrue(progress.is_file(), "worker did not persist .progress")
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait(timeout=10)
        self.assertTrue(progress.is_file(), "progress must survive a hard kill")

        env["SCENE_MIX_TEST_DELAY_MS"] = "0"
        recovered = subprocess.run(
            [
                sys.executable,
                str(ROOT / "main.py"),
                "batch",
                "--input-root",
                str(inputs),
                "--output-root",
                str(outputs),
                "--jobs",
                "1",
            ],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(recovered.returncode, 0, recovered.stderr)
        report = json.loads(
            (outputs / "seed_770" / "report.json").read_text(encoding="utf-8")
        )
        self.assertGreaterEqual(int(report["retry_count"]), 1)
        self.assertFalse(progress.exists())
        summary = json.loads(
            (outputs / "batch_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["groups"][0]["status"], "success")
        self.assertGreaterEqual(int(summary["groups"][0]["retry_count"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
