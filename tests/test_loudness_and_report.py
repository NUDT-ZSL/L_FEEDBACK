from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tests.support import manifest, read_audio, run_cli, write_audio, write_manifest


class LoudnessAndReportTests(unittest.TestCase):
    def test_loudness_and_true_peak_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_audio(root / "assets" / "a.wav", 44_100, 1, 3.0, amplitude=0.08)
            write_audio(
                root / "assets" / "b.wav", 48_000, 2, 3.0, frequency=330.0, amplitude=0.07
            )
            value = manifest(
                root,
                [
                    {
                        "id": "a",
                        "path": "assets/a.wav",
                        "sample_rate": 44_100,
                        "channels": 1,
                        "start_ms": 150.0,
                        "gain_db": -3.0,
                        "role": "dialogue",
                        "priority": 20,
                    },
                    {
                        "id": "b",
                        "path": "assets/b.wav",
                        "sample_rate": 48_000,
                        "channels": 2,
                        "start_ms": 1_000.0,
                        "gain_db": -4.0,
                        "role": "dialogue",
                        "priority": 10,
                    },
                ],
                duration_ms=4_000.0,
                target_lufs=-17.5,
                true_peak_db=-2.0,
            )
            manifest_path = write_manifest(root / "manifest.json", value)
            completed = run_cli(manifest_path)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((root / "out" / "report.json").read_text())
            self.assertLessEqual(abs(report["loudness"]["final_encoded_lufs"] - (-17.5)), 5.0)
            self.assertLessEqual(report["loudness"]["final_encoded_true_peak_dbtp"], -2.0)
            self.assertTrue(report["loudness"]["within_target_tolerance_5_lu"])
            self.assertTrue(report["loudness"]["below_true_peak_ceiling"])
            self.assertEqual(report["output"]["sample_rate"], 48_000)
            self.assertEqual(report["output"]["channels"], 2)
            audio, sample_rate = read_audio(root / "out" / "master.wav")
            self.assertEqual(sample_rate, 48_000)
            self.assertEqual(audio.shape, (4_000 * 48, 2))
            self.assertEqual(report["output"]["frames"], audio.shape[0])
            actual_sha = hashlib.sha256(
                (root / "out" / "master.wav").read_bytes()
            ).hexdigest()
            self.assertEqual(report["output"]["sha256"], actual_sha)
            self.assertTrue(report["verification"]["gain_segments_cover_each_track"])
            for track in report["tracks"]:
                self.assertIn("first_valid_sample", track)
                self.assertIn("duck_intervals", track)
                self.assertIn("loudness_contribution", track)
                self.assertIsNotNone(track["contribution_float32_sha256"])

    def test_true_peak_guard_reduces_global_gain_when_needed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_audio(
                root / "assets" / "quiet.wav",
                48_000,
                2,
                2.0,
                frequency=220.0,
                amplitude=0.015,
                impulse_at=24_000,
            )
            value = manifest(
                root,
                [
                    {
                        "id": "quiet",
                        "path": "assets/quiet.wav",
                        "sample_rate": 48_000,
                        "channels": 2,
                        "start_ms": 0.0,
                        "gain_db": 0.0,
                        "role": "sfx",
                    }
                ],
                duration_ms=2_000.0,
                target_lufs=-26.0,
                true_peak_db=-6.0,
            )
            manifest_path = write_manifest(root / "manifest.json", value)
            completed = run_cli(manifest_path)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((root / "out" / "report.json").read_text())
            self.assertLessEqual(report["loudness"]["final_encoded_true_peak_dbtp"], -6.0)
            self.assertLessEqual(abs(report["loudness"]["final_encoded_lufs"] - (-26.0)), 5.0)
            self.assertTrue(report["loudness"]["limiter"]["applied"])


if __name__ == "__main__":
    unittest.main()
