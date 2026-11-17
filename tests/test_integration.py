from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dialogue_mixer.audio_io import decode_wav_float32
from dialogue_mixer.engine import (
    EXIT_FATAL,
    EXIT_PARTIAL,
    EXIT_SUCCESS,
    _equal_power_fades,
    run_project,
)
from dialogue_mixer.samplegen import generate_all


ROOT = Path(__file__).resolve().parents[1]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="dialogue-mixer-tests-")
        cls.sample_root = Path(cls.temporary.name) / "samples"
        cls.projects = generate_all(cls.sample_root)
        cls.valid = run_project(cls.projects["valid"])
        cls.invalid = run_project(cls.projects["invalid"])
        cls.fatal = run_project(cls.projects["fatal"])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def read_json(self, path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    def test_valid_project_outputs_and_metrics(self) -> None:
        self.assertEqual(self.valid.exit_code, EXIT_SUCCESS)
        self.assertEqual(self.valid.status, "success")
        self.assertIsNotNone(self.valid.audio_path)
        self.assertIsNotNone(self.valid.index_path)
        log = self.read_json(self.valid.log_path)
        index = self.read_json(self.valid.index_path)
        audio, sample_rate = decode_wav_float32(self.valid.audio_path)

        self.assertEqual(sample_rate, 48000)
        self.assertEqual(audio.size, 9 * 48000)
        self.assertEqual(index["total_samples"], audio.size)
        self.assertLessEqual(float(abs(audio).max()), 1.0)
        self.assertLessEqual(float(abs(audio).max()), 0.9772372209558107 + 1e-7)
        self.assertLessEqual(index["alignment"]["max_alignment_error_samples"], 1.0)
        self.assertTrue(index["alignment"]["within_tolerance"])

        self.assertEqual(log["counts"]["duck_events"], 8)
        self.assertEqual(log["counts"]["overlap_events"], 1)
        self.assertEqual(log["counts"]["time_stretched_segments"], 4)
        self.assertEqual(log["counts"]["crossfades_applied"], 1)
        self.assertGreaterEqual(log["counts"]["dialogues"], 4)
        self.assertTrue(log["normalization"]["applied"])
        self.assertLess(log["normalization"]["normalization_factor"], 1.0)
        self.assertLessEqual(log["outputs"]["output_peak_dbfs"], 0.0)
        self.assertEqual(
            log["ducking"]["merge_rule"], "per_sample_minimum"
        )
        self.assertTrue(log["crossfades"]["independent_of_ducking"])
        self.assertEqual(log["crossfades"]["transitions"][0]["curve"], "equal_power")

    def test_stretched_segment_matches_dialogue_interval_exactly(self) -> None:
        index = self.read_json(self.valid.index_path)
        dialogue_track = next(
            track for track in index["tracks"] if track["id"] == "dialogue"
        )
        by_dialogue = {
            segment["dialogue_id"]: segment
            for segment in dialogue_track["segments"]
        }
        first = by_dialogue["line_zh_01"]
        self.assertEqual(first["source_kind"], "stretched")
        self.assertEqual(first["start_sample"], 48000)
        self.assertEqual(first["duration_samples"], 57600)
        self.assertEqual(first["end_sample"], 105600)
        self.assertAlmostEqual(first["actual_start_seconds"], 1.0)
        self.assertAlmostEqual(first["actual_duration_seconds"], 1.2)
        self.assertEqual(first["stretch"]["algorithm"], "wsola")
        self.assertAlmostEqual(first["stretch"]["actual_ratio"], 1.2)
        self.assertLessEqual(first["timeline_alignment_error_samples"], 1.0)
        self.assertEqual(by_dialogue["line_cn_04"]["source_kind"], "original")

    def test_equal_power_crossfade_curve(self) -> None:
        log = self.read_json(self.valid.log_path)
        transition = log["crossfades"]["transitions"][0]
        self.assertEqual(transition["fade_out"], "cos(pi/2 * p)")
        self.assertEqual(transition["fade_in"], "sin(pi/2 * p)")
        self.assertEqual(transition["duration_samples"], 24000)
        self.assertTrue(transition["independent_of_ducking"])
        fade_out, fade_in = _equal_power_fades(24000)
        self.assertAlmostEqual(float(fade_out[0]), 1.0)
        self.assertAlmostEqual(float(fade_in[0]), 0.0)
        self.assertAlmostEqual(float(fade_out[-1]), 0.0)
        self.assertAlmostEqual(float(fade_in[-1]), 1.0)
        self.assertTrue(
            np.allclose(
                fade_out * fade_out + fade_in * fade_in,
                1.0,
                atol=1e-12,
            )
        )

    def test_partial_project_returns_nonzero_and_keeps_outputs(self) -> None:
        self.assertEqual(self.invalid.exit_code, EXIT_PARTIAL)
        self.assertEqual(self.invalid.status, "partial")
        self.assertTrue(self.invalid.audio_path.is_file())
        self.assertTrue(self.invalid.index_path.is_file())
        log = self.read_json(self.invalid.log_path)
        codes = set(log["business_error_codes"])
        self.assertIn("MISSING_AUDIO_FILE", codes)
        self.assertIn("MISSING_DUCK_TARGET", codes)
        self.assertIn("UNMERGEABLE_OVERLAP", codes)
        audio, _ = decode_wav_float32(self.invalid.audio_path)
        self.assertEqual(audio.size, 6 * 48000)
        self.assertLessEqual(float(abs(audio).max()), 1.0)
        index = self.read_json(self.invalid.index_path)
        dialogue_segments = next(
            track for track in index["tracks"] if track["id"] == "dialogue"
        )["segments"]
        self.assertEqual(len(dialogue_segments), 1)
        self.assertEqual(dialogue_segments[0]["dialogue_id"], "line_first")

    def test_invalid_sample_rate_is_logged_without_audio(self) -> None:
        self.assertEqual(self.fatal.exit_code, EXIT_FATAL)
        self.assertEqual(self.fatal.status, "failed")
        self.assertIsNone(self.fatal.audio_path)
        self.assertTrue(self.fatal.log_path.is_file())
        log = self.read_json(self.fatal.log_path)
        self.assertIn("INVALID_SAMPLE_RATE", log["business_error_codes"])
        self.assertFalse(
            (self.fatal.log_path.parent / "should_not_exist.wav").exists()
        )

    def test_rerun_is_byte_deterministic(self) -> None:
        audio_hash = file_hash(self.valid.audio_path)
        index_hash = file_hash(self.valid.index_path)
        rerun = run_project(self.projects["valid"])
        self.assertEqual(rerun.exit_code, EXIT_SUCCESS)
        self.assertEqual(file_hash(rerun.audio_path), audio_hash)
        self.assertEqual(file_hash(rerun.index_path), index_hash)

    def test_cli_entrypoint(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "dialogue_mixer",
                "mix",
                str(self.projects["valid"]),
            ],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("status=success", completed.stdout)


if __name__ == "__main__":
    unittest.main()

