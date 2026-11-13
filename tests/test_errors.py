from __future__ import annotations

import unittest
from pathlib import Path

from replay_mixer.errors import (
    EXIT_CHANNEL_COUNT,
    EXIT_INVALID_CONFIG,
    EXIT_MISSING_INPUT,
    EXIT_UNSUPPORTED_SAMPLE_RATE,
)
from tests.helpers import (
    load_json,
    make_project,
    parse_error,
    run_cli,
    sine_samples,
    temporary_workspace,
    write_wav,
    write_json,
)


def _base_project(root: Path) -> Path:
    sample_rate = 8000
    duration_ms = 250.0
    frames = int(round(duration_ms * sample_rate / 1000.0))
    samples = sine_samples(frames, sample_rate, amplitude=0.1)
    return make_project(
        root,
        sample_rate=sample_rate,
        duration_ms=duration_ms,
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
                "threshold_dbfs": -50.0,
                "min_duration_ms": 1000.0,
            }
        ],
    )


class ErrorContractTests(unittest.TestCase):
    def test_missing_input_has_distinct_exit_code_and_json_error(self) -> None:
        with temporary_workspace("missing-") as root:
            config_path = _base_project(root)
            (root / "inputs" / "c.wav").unlink()
            result = run_cli(config_path)
            self.assertEqual(result.returncode, EXIT_MISSING_INPUT)
            error = parse_error(result)
            self.assertEqual(error["code"], "MISSING_INPUT")
            self.assertEqual(error["exit_code"], EXIT_MISSING_INPUT)
            self.assertEqual(result.stdout, "")
            self.assertFalse((root / "out").exists())

    def test_missing_json_field_is_rejected_before_mixing(self) -> None:
        with temporary_workspace("schema-") as root:
            config_path = _base_project(root)
            config = load_json(config_path)
            del config["tracks"]["commentary"][0]["pan"]
            write_json(config_path, config)
            result = run_cli(config_path)
            self.assertEqual(result.returncode, EXIT_INVALID_CONFIG)
            error = parse_error(result)
            self.assertEqual(error["code"], "INVALID_CONFIG")
            self.assertIn("pan", error["context"]["missing"])
            self.assertFalse((root / "out").exists())

    def test_invalid_field_type_is_rejected_before_mixing(self) -> None:
        with temporary_workspace("type-") as root:
            config_path = _base_project(root)
            config = load_json(config_path)
            config["mix"]["block_frames"] = "128"
            write_json(config_path, config)
            result = run_cli(config_path)
            self.assertEqual(result.returncode, EXIT_INVALID_CONFIG)
            self.assertEqual(parse_error(result)["code"], "INVALID_CONFIG")
            self.assertFalse((root / "out").exists())

    def test_non_monotonic_anchor_is_rejected(self) -> None:
        with temporary_workspace("anchor-") as root:
            config_path = _base_project(root)
            config = load_json(config_path)
            anchors = config["tracks"]["commentary"][0]["anchors"]
            anchors[1]["replay_ms"] = anchors[0]["replay_ms"]
            write_json(config_path, config)
            result = run_cli(config_path)
            self.assertEqual(result.returncode, EXIT_INVALID_CONFIG)
            self.assertIn("strictly increasing", parse_error(result)["message"])

    def test_unsupported_sample_rate_has_distinct_exit_code(self) -> None:
        with temporary_workspace("rate-") as root:
            config_path = _base_project(root)
            config = load_json(config_path)
            config["sample_rate"] = 12345
            write_json(config_path, config)
            result = run_cli(config_path)
            self.assertEqual(result.returncode, EXIT_UNSUPPORTED_SAMPLE_RATE)
            self.assertEqual(parse_error(result)["code"], "UNSUPPORTED_SAMPLE_RATE")

    def test_channel_count_out_of_range_has_distinct_exit_code(self) -> None:
        with temporary_workspace("channels-") as root:
            config_path = _base_project(root)
            config = load_json(config_path)
            config["channels"] = 3
            write_json(config_path, config)
            result = run_cli(config_path)
            self.assertEqual(result.returncode, EXIT_CHANNEL_COUNT)
            self.assertEqual(
                parse_error(result)["code"],
                "CHANNEL_COUNT_OUT_OF_RANGE",
            )

    def test_track_channel_count_out_of_range_has_distinct_exit_code(self) -> None:
        with temporary_workspace("track-channels-") as root:
            config_path = _base_project(root)
            config = load_json(config_path)
            config["tracks"]["commentary"][0]["channels"] = 0
            write_json(config_path, config)
            result = run_cli(config_path)
            self.assertEqual(result.returncode, EXIT_CHANNEL_COUNT)
            self.assertEqual(
                parse_error(result)["code"],
                "CHANNEL_COUNT_OUT_OF_RANGE",
            )

    def test_input_wav_channel_count_out_of_range_has_distinct_exit_code(self) -> None:
        with temporary_workspace("wav-channels-") as root:
            config_path = _base_project(root)
            config = load_json(config_path)
            sample_rate = config["sample_rate"]
            frames = int(
                round(
                    config["replay"]["duration_ms"]
                    * sample_rate
                    / 1000.0
                )
            )
            write_wav(
                root / "inputs" / "c.wav",
                sample_rate,
                3,
                [100] * (frames * 3),
            )
            result = run_cli(config_path)
            self.assertEqual(result.returncode, EXIT_CHANNEL_COUNT)
            self.assertEqual(
                parse_error(result)["code"],
                "CHANNEL_COUNT_OUT_OF_RANGE",
            )
