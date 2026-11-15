from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.support import manifest, read_audio, run_cli, write_audio, write_manifest


def _base_tracks() -> list[dict[str, object]]:
    return [
        {
            "id": "a",
            "path": "assets/a.wav",
            "sample_rate": 48_000,
            "channels": 1,
            "start_ms": 100.0,
            "gain_db": -2.0,
            "role": "dialogue",
            "priority": 20,
        },
        {
            "id": "b",
            "path": "assets/b.wav",
            "sample_rate": 44_100,
            "channels": 2,
            "start_ms": 1_100.0,
            "gain_db": -3.0,
            "role": "dialogue",
            "priority": 10,
        },
        {
            "id": "amb",
            "path": "assets/amb.wav",
            "sample_rate": 22_050,
            "channels": 2,
            "start_ms": 0.0,
            "gain_db": -18.0,
            "role": "ambience",
        },
    ]


class RecoveryIncrementErrorTests(unittest.TestCase):
    def _create_project(self, root: Path) -> tuple[Path, dict[str, object]]:
        write_audio(root / "assets" / "a.wav", 48_000, 1, 2.0, amplitude=0.1)
        write_audio(root / "assets" / "b.wav", 44_100, 2, 2.0, frequency=330.0, amplitude=0.08)
        write_audio(root / "assets" / "amb.wav", 22_050, 2, 2.5, frequency=90.0, amplitude=0.05)
        value = manifest(root, _base_tracks(), duration_ms=3_500.0)
        path = write_manifest(root / "manifest.json", value)
        return path, value

    def test_incremental_version_matches_clean_full_render(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, value = self._create_project(root)
            state_incremental = root / "state-incremental"
            self.assertEqual(run_cli(manifest_path, state_dir=state_incremental).returncode, 0)

            changed = copy.deepcopy(value)
            changed["project"]["version"] = "v2"
            changed["tracks"][1]["start_ms"] = 1_237.37
            manifest_path = write_manifest(root / "manifest.json", changed)
            completed = run_cli(manifest_path, state_dir=state_incremental)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            incremental_audio, incremental_rate = read_audio(root / "out" / "master.wav")
            incremental_bytes = (root / "out" / "master.wav").read_bytes()
            report = json.loads((root / "out" / "report.json").read_text())
            self.assertEqual(report["project"]["version"], "v2")
            self.assertEqual(report["recovery"]["action"], "manifest_changed_full_rebuild")
            self.assertGreater(report["recovery"]["cache_hits"], 0)

            state_full = root / "state-full"
            completed = run_cli(manifest_path, state_dir=state_full)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            full_audio, full_rate = read_audio(root / "out" / "master.wav")
            full_bytes = (root / "out" / "master.wav").read_bytes()
            self.assertEqual(incremental_rate, full_rate)
            self.assertTrue(np.array_equal(incremental_audio, full_audio))
            self.assertEqual(incremental_bytes, full_bytes)

    def test_forced_kill_resumes_without_partial_master(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = self._create_project(root)
            state_dir = root / "state"
            crashed = run_cli(
                manifest_path,
                state_dir=state_dir,
                env_extra={"CUTMASTER_FAULT_INJECT": "after_track"},
            )
            self.assertEqual(crashed.returncode, 75, crashed.stderr)
            self.assertFalse((root / "out" / "master.wav").exists())
            checkpoints = list(state_dir.glob("projects/*/checkpoint.json"))
            self.assertEqual(len(checkpoints), 1)
            checkpoint = json.loads(checkpoints[0].read_text())
            self.assertEqual(checkpoint["status"], "in_progress")

            completed = run_cli(manifest_path, state_dir=state_dir)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((root / "out" / "report.json").read_text())
            self.assertEqual(report["recovery"]["previous_status"], "in_progress")
            self.assertEqual(report["recovery"]["action"], "resume_from_checkpoint")
            self.assertGreaterEqual(report["recovery"]["cache_hits"], 1)

            resumed, _ = read_audio(root / "out" / "master.wav")
            clean_state = root / "state-clean"
            self.assertEqual(run_cli(manifest_path, state_dir=clean_state).returncode, 0)
            clean, _ = read_audio(root / "out" / "master.wav")
            self.assertTrue(np.array_equal(resumed, clean))

    def test_failure_after_mix_does_not_publish_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = self._create_project(root)
            state_dir = root / "state"
            crashed = run_cli(
                manifest_path,
                state_dir=state_dir,
                env_extra={"CUTMASTER_FAULT_INJECT": "after_mix"},
            )
            self.assertEqual(crashed.returncode, 75, crashed.stderr)
            self.assertFalse((root / "out" / "master.wav").exists())
            self.assertFalse(list((root / "out").glob("*.partial")) if (root / "out").exists() else [])
            completed = run_cli(manifest_path, state_dir=state_dir)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((root / "out" / "master.wav").exists())

    def test_structured_errors_and_no_partial_master(self) -> None:
        cases = [
            ("E_MISSING_TRACK", 3),
            ("E_BAD_RATE", 2),
            ("E_BAD_CHANNELS", 2),
            ("E_BAD_START", 2),
            ("E_START_OUT_OF_RANGE", 4),
        ]
        for expected_code, expected_exit in cases:
            with self.subTest(code=expected_code), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest_path, value = self._create_project(root)
                changed = copy.deepcopy(value)
                if expected_code == "E_MISSING_TRACK":
                    (root / "assets" / "a.wav").unlink()
                elif expected_code == "E_BAD_RATE":
                    changed["tracks"][0]["sample_rate"] = 0
                elif expected_code == "E_BAD_CHANNELS":
                    changed["tracks"][0]["channels"] = 0
                elif expected_code == "E_BAD_START":
                    changed["tracks"][0]["start_ms"] = -1.0
                else:
                    changed["tracks"][0]["start_ms"] = 3_500.0
                write_manifest(manifest_path, changed)
                completed = run_cli(manifest_path)
                self.assertEqual(completed.returncode, expected_exit, completed.stderr)
                payload = json.loads(completed.stderr)
                self.assertEqual(payload["error"]["code"], expected_code)
                self.assertFalse((root / "out" / "master.wav").exists())
                self.assertFalse((root / "out" / "report.json").exists())

    def test_actual_metadata_mismatch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, value = self._create_project(root)
            changed = copy.deepcopy(value)
            changed["tracks"][0]["sample_rate"] = 44_100
            write_manifest(manifest_path, changed)
            completed = run_cli(manifest_path)
            self.assertEqual(completed.returncode, 3, completed.stderr)
            self.assertEqual(json.loads(completed.stderr)["error"]["code"], "E_METADATA_MISMATCH")
            self.assertFalse((root / "out" / "master.wav").exists())


if __name__ == "__main__":
    unittest.main()
