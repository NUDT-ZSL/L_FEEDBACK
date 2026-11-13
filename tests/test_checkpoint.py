from __future__ import annotations

import hashlib
import unittest

from replay_mixer.errors import EXIT_CHECKPOINT_INVALID, EXIT_INTERRUPTED
from tests.helpers import (
    load_json,
    make_project,
    parse_error,
    parse_success,
    run_cli,
    sine_samples,
    temporary_workspace,
    write_json,
)


def _project(directory) -> "object":
    sample_rate = 8000
    duration_ms = 400.0
    frames = int(round(duration_ms * sample_rate / 1000.0))
    samples = sine_samples(
        frames,
        sample_rate,
        frequency=180.0,
        amplitude=0.1,
        channels=1,
    )
    return make_project(
        directory,
        sample_rate=sample_rate,
        duration_ms=duration_ms,
        output_channels=2,
        block_frames=128,
        commentary=[
            {
                "name": "c",
                "filename": "c.wav",
                "channels": 1,
                "samples": samples,
                "anchors": [
                    {"replay_ms": 0.0, "source_ms": 0.0},
                    {"replay_ms": duration_ms, "source_ms": duration_ms},
                ],
                "gain_db": -1.5,
                "pan": -0.2,
                "threshold_dbfs": -50.0,
                "min_duration_ms": 1000.0,
            }
        ],
    )


class ResumeTests(unittest.TestCase):
    def test_measure_and_final_resume_are_byte_identical(self) -> None:
        with temporary_workspace("resume-") as root:
            one_shot_dir = root / "one"
            measure_dir = root / "measure"
            final_dir = root / "final"
            one_shot_dir.mkdir()
            measure_dir.mkdir()
            final_dir.mkdir()
            one_shot = _project(one_shot_dir)
            measure_config = _project(measure_dir)
            final_config = _project(final_dir)

            one_stats = parse_success(run_cli(one_shot))
            one_bytes = (one_shot_dir / "out" / "mix.wav").read_bytes()
            self.assertEqual(
                one_stats["output_sha256"],
                hashlib.sha256(one_bytes).hexdigest(),
            )

            interrupted_measure = run_cli(
                measure_config,
                "--stop-after-blocks",
                "3",
                "--stop-after-phase",
                "measure",
            )
            self.assertEqual(interrupted_measure.returncode, EXIT_INTERRUPTED)
            self.assertEqual(parse_error(interrupted_measure)["code"], "EXPORT_INTERRUPTED")
            measure_stats = parse_success(run_cli(measure_config))
            self.assertEqual(measure_stats["resumed_blocks"], 3)
            self.assertEqual(measure_stats["resumed_phase"], "measure")
            self.assertEqual(measure_stats["measure_resumed_blocks"], 3)
            measure_bytes = (measure_dir / "out" / "mix.wav").read_bytes()
            self.assertEqual(measure_bytes, one_bytes)

            interrupted_final = run_cli(
                final_config,
                "--stop-after-blocks",
                "2",
                "--stop-after-phase",
                "final",
            )
            self.assertEqual(interrupted_final.returncode, EXIT_INTERRUPTED)
            self.assertEqual(parse_error(interrupted_final)["context"]["phase"], "final")
            final_stats = parse_success(run_cli(final_config))
            self.assertEqual(final_stats["resumed_blocks"], 2)
            self.assertEqual(final_stats["resumed_phase"], "final")
            self.assertEqual(final_stats["final_resumed_blocks"], 2)
            final_bytes = (final_dir / "out" / "mix.wav").read_bytes()
            self.assertEqual(final_bytes, one_bytes)


class CheckpointValidationTests(unittest.TestCase):
    def test_corrupt_checkpoint_strict_error_then_auto_restart(self) -> None:
        with temporary_workspace("bad-ckpt-") as root:
            config_path = _project(root)
            interrupted = run_cli(
                config_path,
                "--stop-after-blocks",
                "2",
                "--stop-after-phase",
                "measure",
            )
            self.assertEqual(interrupted.returncode, EXIT_INTERRUPTED)
            checkpoint = root / "out" / "mix.wav.ckpt.json"
            payload = load_json(checkpoint)
            payload["data_sha256"] = "0" * 64
            write_json(checkpoint, payload)

            strict = run_cli(config_path, "--strict-checkpoint")
            self.assertEqual(strict.returncode, EXIT_CHECKPOINT_INVALID)
            error = parse_error(strict)
            self.assertEqual(error["code"], "CHECKPOINT_INVALID")
            self.assertTrue(error["context"]["reason"])

            restarted = parse_success(run_cli(config_path))
            self.assertEqual(restarted["resumed_blocks"], 0)
            self.assertIsNotNone(restarted["checkpoint_reset_reason"])
            self.assertTrue((root / "out" / "mix.wav").is_file())

    def test_changed_data_digest_is_detected(self) -> None:
        with temporary_workspace("raw-digest-") as root:
            config_path = _project(root)
            interrupted = run_cli(
                config_path,
                "--stop-after-blocks",
                "2",
                "--stop-after-phase",
                "measure",
            )
            self.assertEqual(interrupted.returncode, EXIT_INTERRUPTED)
            raw_path = root / "out" / "mix.wav.measure.raw"
            with raw_path.open("r+b") as handle:
                handle.seek(0)
                first = handle.read(1)
                handle.seek(0)
                handle.write(bytes([first[0] ^ 0xFF]))
            strict = run_cli(config_path, "--strict-checkpoint")
            self.assertEqual(strict.returncode, EXIT_CHECKPOINT_INVALID)
            self.assertIn("digest", parse_error(strict)["message"])

    def test_config_digest_mismatch_restarts_or_errors_strictly(self) -> None:
        with temporary_workspace("config-digest-") as root:
            config_path = _project(root)
            interrupted = run_cli(
                config_path,
                "--stop-after-blocks",
                "2",
                "--stop-after-phase",
                "measure",
            )
            self.assertEqual(interrupted.returncode, EXIT_INTERRUPTED)
            config = load_json(config_path)
            config["mix"]["target_rms_dbfs"] = -19.0
            write_json(config_path, config)

            strict = run_cli(config_path, "--strict-checkpoint")
            self.assertEqual(strict.returncode, EXIT_CHECKPOINT_INVALID)
            self.assertEqual(
                parse_error(strict)["context"]["reason"],
                "config_digest_mismatch",
            )
            restarted = parse_success(run_cli(config_path))
            self.assertEqual(restarted["resumed_blocks"], 0)
            self.assertIn("config_digest_mismatch", restarted["checkpoint_reset_reason"])
