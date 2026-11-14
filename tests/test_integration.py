from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from radio_pipeline.config import load_config
from radio_pipeline.ffmpeg_io import find_tool
from radio_pipeline.pipeline import run_pipeline
from radio_pipeline.stats import build_report, load_processing_log
from radio_pipeline.verify import verify_run


ROOT = Path(__file__).resolve().parents[1]


def _ffmpeg_available() -> bool:
    try:
        find_tool("ffmpeg")
        find_tool("ffprobe")
        return True
    except Exception:
        return False


@unittest.skipUnless(_ffmpeg_available(), "FFmpeg and ffprobe are required")
class IntegrationTests(unittest.TestCase):
    def test_lossy_run_report_and_verify(self) -> None:
        config = load_config(ROOT / "config" / "default.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "lossy"
            summary = run_pipeline(config, run_dir)
            self.assertEqual(summary["counts"]["total_packets"], 400)
            self.assertGreater(summary["counts"]["plc_frames"], 0)
            self.assertGreater(summary["counts"]["timeout_packets"], 0)
            self.assertGreater(summary["counts"]["duplicate_arrivals"], 0)
            self.assertGreater(summary["counts"]["invalid_arrivals"], 0)
            self.assertTrue(summary["degradation"]["triggered"])

            log = load_processing_log(run_dir / "logs" / "processing_log.json")
            rebuilt = build_report(log)
            stored = json.loads((run_dir / "report.json").read_text("utf-8"))
            self.assertEqual(rebuilt, stored)

            verification, code = verify_run(run_dir)
            self.assertEqual(code, 0, verification)
            self.assertTrue(verification["ok"])
            self.assertEqual(
                verification["metrics"]["wav_duration_error_samples"], 0
            )
            self.assertEqual(
                verification["metrics"]["ogg_duration_error_samples"], 0
            )

    def test_audio_outputs_are_binary_deterministic(self) -> None:
        config = load_config(ROOT / "config" / "sequential.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first"
            second = Path(temp_dir) / "second"
            first_summary = run_pipeline(config, first)
            second_summary = run_pipeline(config, second)
            self.assertEqual(
                first_summary["output"]["pcm_sha256"],
                second_summary["output"]["pcm_sha256"],
            )
            self.assertEqual(
                first_summary["output"]["wav_sha256"],
                second_summary["output"]["wav_sha256"],
            )
            self.assertEqual(
                first_summary["output"]["ogg_sha256"],
                second_summary["output"]["ogg_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
