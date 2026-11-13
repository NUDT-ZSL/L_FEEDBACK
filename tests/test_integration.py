from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from frame_render.sample import write_sample


ROOT = Path(__file__).resolve().parent.parent


def run_cli(
    arguments: list[str],
    *,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    if env_overrides:
        environment.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "frame_render", *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def parse_stdout(result: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(result.stdout)


def parse_stderr(result: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(result.stderr)


def make_scene(directory: Path, name: str, frames: int = 8) -> Path:
    path = directory / name
    write_sample(path, frame_count=frames, width=28, height=18, samples_per_axis=4)
    return path


def pixel_digest(sequence_dir: Path, frame_start: int, frame_end: int) -> str:
    import hashlib

    digest = hashlib.sha256()
    for frame_number in range(frame_start, frame_end + 1):
        image = np.asarray(
            Image.open(sequence_dir / f"frame_{frame_number:06d}.png").convert("RGB")
        )
        digest.update(frame_number.to_bytes(8, "little", signed=False))
        digest.update(image.tobytes())
    return digest.hexdigest()


def chunk_id(frame_start: int, frame_end: int) -> str:
    return f"chunk_{frame_start:06d}_{frame_end:06d}"


class IntegrationTests(unittest.TestCase):
    def test_concurrency_resume_corrupt_checkpoint_and_missing_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = make_scene(root, "scene.json")
            serial = run_cli(
                [
                    "run",
                    "--input",
                    str(scene),
                    "--work-dir",
                    str(root / "serial_work"),
                    "--output",
                    str(root / "serial_frames"),
                    "--chunk-size",
                    "2",
                    "--jobs",
                    "1",
                    "--max-retries",
                    "1",
                ]
            )
            self.assertEqual(serial.returncode, 0, serial.stderr)
            parallel = run_cli(
                [
                    "run",
                    "--input",
                    str(scene),
                    "--work-dir",
                    str(root / "parallel_work"),
                    "--output",
                    str(root / "parallel_frames"),
                    "--chunk-size",
                    "2",
                    "--jobs",
                    "3",
                    "--max-retries",
                    "1",
                ]
            )
            self.assertEqual(parallel.returncode, 0, parallel.stderr)
            self.assertEqual(
                pixel_digest(root / "serial_frames", 0, 7),
                pixel_digest(root / "parallel_frames", 0, 7),
            )

            resumed = run_cli(
                [
                    "render",
                    "--input",
                    str(scene),
                    "--work-dir",
                    str(root / "parallel_work"),
                    "--chunk-size",
                    "2",
                    "--jobs",
                    "3",
                    "--max-retries",
                    "1",
                ]
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            resumed_summary = parse_stdout(resumed)
            self.assertEqual(resumed_summary["chunks_reused"], 4)
            self.assertEqual(resumed_summary["chunks_rendered"], 0)

            invalid_chunk = chunk_id(4, 5)
            checkpoint = (
                root / "parallel_work" / "checkpoints" / f"{invalid_chunk}.json"
            )
            record = json.loads(checkpoint.read_text(encoding="utf-8"))
            record["checkpoint_version"] = 999
            checkpoint.write_text(json.dumps(record), encoding="utf-8")
            corrupted = run_cli(
                [
                    "run",
                    "--input",
                    str(scene),
                    "--work-dir",
                    str(root / "parallel_work"),
                    "--output",
                    str(root / "parallel_frames"),
                    "--chunk-size",
                    "2",
                    "--jobs",
                    "3",
                    "--max-retries",
                    "1",
                ]
            )
            self.assertEqual(corrupted.returncode, 0, corrupted.stderr)
            corrupted_summary = parse_stdout(corrupted)["render"]
            self.assertEqual(corrupted_summary["chunks_rendered"], 1)
            self.assertEqual(corrupted_summary["rendered_chunks"], [invalid_chunk])
            self.assertEqual(
                corrupted_summary["invalid_checkpoints"][0]["chunk_id"],
                invalid_chunk,
            )
            self.assertEqual(
                pixel_digest(root / "parallel_frames", 0, 7),
                pixel_digest(root / "serial_frames", 0, 7),
            )

            malformed_chunk = chunk_id(2, 3)
            malformed_checkpoint = (
                root / "parallel_work" / "checkpoints" / f"{malformed_chunk}.json"
            )
            malformed_checkpoint.write_text("{not valid json", encoding="utf-8")
            malformed = run_cli(
                [
                    "render",
                    "--input",
                    str(scene),
                    "--work-dir",
                    str(root / "parallel_work"),
                    "--chunk-size",
                    "2",
                    "--jobs",
                    "3",
                    "--max-retries",
                    "1",
                ]
            )
            self.assertEqual(malformed.returncode, 0, malformed.stderr)
            malformed_summary = parse_stdout(malformed)
            self.assertEqual(malformed_summary["rendered_chunks"], [malformed_chunk])
            self.assertIn(
                "checkpoint_unreadable",
                malformed_summary["invalid_checkpoints"][0]["reason"],
            )

            frame_file = (
                root
                / "parallel_work"
                / "chunks"
                / invalid_chunk
                / "frames"
                / "frame_000004.png"
            )
            os.replace(frame_file, frame_file.with_suffix(".saved"))
            missing = run_cli(
                [
                    "assemble",
                    "--input",
                    str(scene),
                    "--work-dir",
                    str(root / "parallel_work"),
                    "--output",
                    str(root / "missing_frames"),
                    "--chunk-size",
                    "2",
                ]
                + [],
            )
            self.assertEqual(missing.returncode, 4)
            missing_summary = parse_stderr(missing)
            self.assertEqual(missing_summary["status"], "error")
            self.assertEqual(missing_summary["successful_frames"], 6)
            self.assertEqual(missing_summary["missing_frames"], [4, 5])
            self.assertEqual(
                missing_summary["corrupted_chunks"][0]["chunk_id"], invalid_chunk
            )
            self.assertFalse((root / "missing_frames").exists())

    def test_forced_and_recoverable_worker_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = make_scene(root, "scene.json", frames=6)
            target_chunk = chunk_id(0, 1)
            hard = run_cli(
                [
                    "render",
                    "--input",
                    str(scene),
                    "--work-dir",
                    str(root / "hard_work"),
                    "--chunk-size",
                    "2",
                    "--jobs",
                    "1",
                    "--max-retries",
                    "0",
                ],
                env_overrides={
                    "OFR_TEST_FAULT": json.dumps(
                        {
                            "chunk_id": target_chunk,
                            "mode": "hard_exit",
                            "after_frames": 1,
                            "attempt": 1,
                            "exit_code": 23,
                        }
                    )
                },
            )
            self.assertEqual(hard.returncode, 3)
            hard_summary = parse_stderr(hard)
            self.assertEqual(
                hard_summary["forced_terminations"][0]["chunk_id"], target_chunk
            )
            self.assertFalse(
                (
                    root
                    / "hard_work"
                    / "reports"
                    / f"{target_chunk}_attempt_001.json"
                ).exists()
            )
            resumed = run_cli(
                [
                    "render",
                    "--input",
                    str(scene),
                    "--work-dir",
                    str(root / "hard_work"),
                    "--chunk-size",
                    "2",
                    "--jobs",
                    "2",
                    "--max-retries",
                    "1",
                ]
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            resumed_summary = parse_stdout(resumed)
            self.assertEqual(resumed_summary["chunks_reused"], 2)
            self.assertEqual(resumed_summary["chunks_rendered"], 1)
            self.assertEqual(resumed_summary["retry_counts"], {target_chunk: 1})

            recoverable_target = chunk_id(2, 3)
            recoverable = run_cli(
                [
                    "render",
                    "--input",
                    str(scene),
                    "--work-dir",
                    str(root / "recoverable_work"),
                    "--chunk-size",
                    "2",
                    "--jobs",
                    "1",
                    "--max-retries",
                    "0",
                ],
                env_overrides={
                    "OFR_TEST_FAULT": json.dumps(
                        {
                            "chunk_id": recoverable_target,
                            "mode": "recoverable",
                            "after_frames": 1,
                            "attempt": 1,
                            "reason": "synthetic recoverable render failure",
                        }
                    )
                },
            )
            self.assertEqual(recoverable.returncode, 3)
            recoverable_summary = parse_stderr(recoverable)
            failure = recoverable_summary["recoverable_failures"][0]
            self.assertEqual(failure["chunk_id"], recoverable_target)
            self.assertEqual(failure["completed_frames"], [2])
            report = json.loads(Path(failure["report"]).read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertTrue(report["recoverable"])
            self.assertFalse(report["chunk_completed"])
            self.assertEqual(report["completed_chunks"], [chunk_id(0, 1)])
            again = run_cli(
                [
                    "render",
                    "--input",
                    str(scene),
                    "--work-dir",
                    str(root / "recoverable_work"),
                    "--chunk-size",
                    "2",
                    "--jobs",
                    "1",
                    "--max-retries",
                    "1",
                ]
            )
            self.assertEqual(again.returncode, 0, again.stderr)

    def test_geometry_fingerprint_change_forces_rerender(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = make_scene(root, "scene.json", frames=4)
            first = run_cli(
                [
                    "render",
                    "--input",
                    str(scene),
                    "--work-dir",
                    str(root / "work"),
                    "--chunk-size",
                    "2",
                    "--jobs",
                    "2",
                    "--max-retries",
                    "1",
                ]
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            document = json.loads(scene.read_text(encoding="utf-8"))
            document["scene"]["spheres"][0]["albedo"] = [0.1, 0.2, 0.9]
            scene.write_text(json.dumps(document), encoding="utf-8")
            changed = run_cli(
                [
                    "render",
                    "--input",
                    str(scene),
                    "--work-dir",
                    str(root / "work"),
                    "--chunk-size",
                    "2",
                    "--jobs",
                    "2",
                    "--max-retries",
                    "1",
                ]
            )
            self.assertEqual(changed.returncode, 0, changed.stderr)
            changed_summary = parse_stdout(changed)
            self.assertEqual(changed_summary["chunks_rendered"], 2)
            self.assertEqual(changed_summary["chunks_reused"], 0)
            self.assertEqual(len(changed_summary["invalid_checkpoints"]), 2)


if __name__ == "__main__":
    unittest.main()

