from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from broadcast_master.ffmpeg_tools import analyze_file, write_wav_24
from broadcast_master.mastering import normalize_master
from broadcast_master.wav import probe_wav
from tests.sample_factory import corrupt_wav_block_alignment, create_scenario


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "broadcast_master", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BroadcastMasterIntegrationTests(unittest.TestCase):
    def test_full_pipeline_report_sync_fallback_and_delivery_format(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bm_full_") as temporary:
            root = Path(temporary)
            scenario = create_scenario(root / "sample")
            result = run_cli(
                "--manifest",
                str(scenario.manifest_path),
                "--output",
                str(scenario.output_path),
                "--report",
                str(scenario.report_path),
                "--temp-root",
                str(root / "work"),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(scenario.output_path.is_file())
            self.assertTrue(scenario.report_path.is_file())

            report = json.loads(scenario.report_path.read_text(encoding="utf-8"))
            for key in (
                "status",
                "manifest",
                "timeline",
                "languages",
                "segments",
                "fallbacks",
                "sync",
                "normalization",
                "verification",
                "output",
            ):
                self.assertIn(key, report)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["verification"]["passed"], True)
            self.assertLessEqual(
                abs(report["verification"]["integrated_lufs"] - (-23.0)), 0.2
            )
            self.assertLessEqual(report["verification"]["true_peak_dbtp"], -1.0)

            languages = {item["id"]: item for item in report["languages"]}
            self.assertEqual(set(languages), {"zh-CN", "en-US", "ja-JP"})
            self.assertTrue(languages["ja-JP"]["silence_risk"])
            self.assertGreaterEqual(
                languages["zh-CN"]["loudness"]["true_peak_violation_count"], 2
            )
            violation = languages["zh-CN"]["loudness"]["true_peak_violations"][0]
            self.assertRegex(violation["timecode"], r"^\d{2}:\d{2}:\d{2}:\d{2}$")
            violation_seconds = {
                item["window_start_seconds"]
                for item in languages["zh-CN"]["loudness"]["true_peak_violations"]
            }
            self.assertIn(1.2, violation_seconds)
            self.assertIn(4.1, violation_seconds)
            self.assertGreater(
                languages["zh-CN"]["loudness"]["short_term"]["window_count"], 0
            )

            self.assertEqual(len(report["fallbacks"]), 1)
            fallback = report["fallbacks"][0]
            self.assertEqual(fallback["requested_language"], "ja-JP")
            self.assertEqual(fallback["fallback_language"], "en-US")
            self.assertEqual(fallback["reason"], "frame_out_of_track_range")
            self.assertEqual(len(report["segments"]), 2)
            self.assertEqual(report["segments"][0]["fade_out_samples"], 720)
            self.assertEqual(report["segments"][1]["fade_in_samples"], 720)

            sync = report["sync"]
            self.assertEqual(sync["status"], "compensated")
            self.assertAlmostEqual(sync["estimated_offset_ms"], 30.0, delta=2.0)
            self.assertGreater(sync["applied_shift_samples"], 0)
            self.assertLessEqual(abs(sync["applied_shift_ms"]), 500.0)

            normalization = report["normalization"]
            self.assertIn("before", normalization)
            self.assertIn("after", normalization)
            self.assertIn("gain_db", normalization)
            self.assertEqual(normalization["verified_on_final_wav"], True)

            info = probe_wav(scenario.output_path, validate_alignment=True)
            self.assertEqual(info.sample_rate, 48000)
            self.assertEqual(info.channels, 2)
            self.assertEqual(info.bits_per_sample, 24)
            self.assertEqual(info.data_frames, 8 * 48000)
            self.assertEqual(report["output"]["single_audio_stream"], True)

    def test_business_validation_is_single_line_json_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bm_bad_") as temporary:
            root = Path(temporary)
            scenario = create_scenario(root / "sample")
            manifest = json.loads(scenario.manifest_path.read_text(encoding="utf-8"))
            del manifest["audio"]["bit_depth"]
            bad_manifest = root / "bad.json"
            bad_manifest.write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "should-not-exist.wav"
            report = root / "should-not-exist.json"
            result = run_cli(
                "--manifest",
                str(bad_manifest),
                "--output",
                str(output),
                "--report",
                str(report),
            )
            self.assertEqual(result.returncode, 2)
            self.assertEqual(len(result.stderr.splitlines()), 1)
            payload = json.loads(result.stderr)
            self.assertIsInstance(payload["code"], int)
            self.assertIsInstance(payload["message"], str)
            self.assertFalse(output.exists())
            self.assertFalse(report.exists())
            dry_result = run_cli("--manifest", str(bad_manifest), "--dry-run")
            self.assertEqual(dry_result.returncode, 2)
            self.assertEqual(len(dry_result.stderr.splitlines()), 1)
            self.assertIsInstance(json.loads(dry_result.stderr)["code"], int)

    def test_media_metadata_mismatch_is_business_error(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bm_meta_") as temporary:
            root = Path(temporary)
            scenario = create_scenario(root / "sample")
            manifest = json.loads(scenario.manifest_path.read_text(encoding="utf-8"))
            manifest["audio"]["sample_rate"] = 44100
            for language in manifest["languages"]:
                language["sample_rate"] = 44100
            bad_manifest = scenario.manifest_path.parent / "bad-rate.json"
            bad_manifest.write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "bad.wav"
            report = root / "bad-report.json"
            result = run_cli(
                "--manifest",
                str(bad_manifest),
                "--output",
                str(output),
                "--report",
                str(report),
            )
            self.assertEqual(result.returncode, 2)
            payload = json.loads(result.stderr)
            self.assertEqual(payload["code"], 1005)
            self.assertFalse(output.exists())
            self.assertFalse(report.exists())

    def test_dry_run_has_stage_timings_and_writes_no_delivery_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bm_dry_") as temporary:
            root = Path(temporary)
            scenario = create_scenario(root / "sample")
            result = run_cli(
                "--manifest", str(scenario.manifest_path), "--dry-run"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["mode"], "dry-run")
            self.assertEqual(payload["sync"]["status"], "compensated")
            self.assertEqual(len(payload["segments"]), 2)
            timings = payload["stage_timings_ms"]
            for key in (
                "parse_manifest_ms",
                "probe_media_ms",
                "simulate_timeline_ms",
                "sync_estimate_ms",
                "total_ms",
            ):
                self.assertIn(key, timings)
                self.assertGreaterEqual(timings[key], 0)
            self.assertFalse(scenario.output_path.exists())
            self.assertFalse(scenario.report_path.exists())

    def test_out_of_range_sync_preserves_alignment(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bm_range_") as temporary:
            root = Path(temporary)
            scenario = create_scenario(
                root / "sample", reference_delay_ms=750.0
            )
            result = run_cli(
                "--manifest",
                str(scenario.manifest_path),
                "--output",
                str(scenario.output_path),
                "--report",
                str(scenario.report_path),
                "--temp-root",
                str(root / "work"),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(scenario.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["sync"]["status"], "out_of_range")
            self.assertEqual(report["sync"]["applied_shift_samples"], 0)
            self.assertEqual(report["sync"]["applied_shift_ms"], 0.0)
            self.assertAlmostEqual(
                report["sync"]["estimated_offset_ms"], 750.0, delta=5.0
            )

    def test_resume_skips_completed_language_and_matches_baseline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bm_resume_") as temporary:
            root = Path(temporary)
            scenario = create_scenario(root / "sample")
            work = root / "work"
            baseline = root / "baseline.wav"
            baseline_report = root / "baseline.json"
            baseline_run = run_cli(
                "--manifest",
                str(scenario.manifest_path),
                "--output",
                str(baseline),
                "--report",
                str(baseline_report),
                "--temp-root",
                str(work),
            )
            self.assertEqual(baseline_run.returncode, 0, baseline_run.stderr)

            second = scenario.language_paths["en-US"]
            original = corrupt_wav_block_alignment(second)
            failed_output = root / "failed.wav"
            failed_report = root / "failed.json"
            failed = run_cli(
                "--manifest",
                str(scenario.manifest_path),
                "--output",
                str(failed_output),
                "--report",
                str(failed_report),
                "--temp-root",
                str(work),
            )
            self.assertEqual(failed.returncode, 3)
            self.assertEqual(len(failed.stderr.splitlines()), 1)
            self.assertFalse(failed_output.exists())
            self.assertFalse(failed_report.exists())
            markers = list(work.rglob("progress.json"))
            self.assertEqual(len(markers), 1)
            marker = json.loads(markers[0].read_text(encoding="utf-8"))
            self.assertIn("zh-CN", marker["completed_languages"])
            staging = markers[0].parent / "staging"
            self.assertFalse(staging.exists())

            second.write_bytes(original)
            resumed = root / "resumed.wav"
            resumed_report = root / "resumed.json"
            resumed_run = run_cli(
                "--manifest",
                str(scenario.manifest_path),
                "--output",
                str(resumed),
                "--report",
                str(resumed_report),
                "--temp-root",
                str(work),
            )
            self.assertEqual(resumed_run.returncode, 0, resumed_run.stderr)
            self.assertEqual(sha256(baseline), sha256(resumed))
            report = json.loads(resumed_report.read_text(encoding="utf-8"))
            self.assertEqual(report["recovery"]["resumed"], True)
            self.assertIn(
                "zh-CN", report["recovery"]["skipped_language_analyses"]
            )

    def test_video_without_audio_uses_no_reference_status(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bm_noref_") as temporary:
            root = Path(temporary)
            scenario = create_scenario(
                root / "sample", include_reference=False
            )
            result = run_cli(
                "--manifest",
                str(scenario.manifest_path),
                "--output",
                str(scenario.output_path),
                "--report",
                str(scenario.report_path),
                "--temp-root",
                str(root / "work"),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(scenario.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["sync"]["status"], "no_reference")
            self.assertIsNone(report["sync"]["reference_stream_index"])
            self.assertEqual(report["sync"]["applied_shift_samples"], 0)

    def test_dynamic_loudnorm_handles_hot_spikes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bm_hot_") as temporary:
            root = Path(temporary)
            sample_rate = 48000
            channels = 2
            count = sample_rate * 8
            rng = np.random.default_rng(5)
            audio = (
                rng.standard_normal((count, channels))
                * (10.0 ** (-48.0 / 20.0))
            ).astype(np.float32)
            for index in range(0, count, 12000):
                audio[index + 2, :] = 0.999
            mastered, details = normalize_master(
                audio, sample_rate, channels, 25.0
            )
            self.assertEqual(details["method"], "two_pass_loudnorm")
            self.assertLessEqual(abs(details["after"]["integrated_lufs"] + 23.0), 0.2)
            self.assertLessEqual(details["after"]["true_peak_dbtp"], -1.0)
            output = root / "hot.wav"
            write_wav_24(output, mastered, sample_rate, channels)
            final = analyze_file(output, 25.0)
            self.assertLessEqual(abs(final.integrated_lufs + 23.0), 0.2)
            self.assertLessEqual(final.true_peak_dbtp, -1.0)


if __name__ == "__main__":
    unittest.main()

