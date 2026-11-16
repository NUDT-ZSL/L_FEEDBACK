from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import soundfile as sf

from tools.make_demo_assets import generate


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "localize_dialogue.py"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="dialogue-localizer-")
        cls.asset_dir = Path(cls.temp_dir.name) / "assets"
        cls.summary = generate(cls.asset_dir)
        cls.manifest_path = Path(str(cls.summary["manifest"]))
        cls.manifest = json.loads(
            cls.manifest_path.read_text(encoding="utf-8")
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def _write_manifest(self, manifest: dict[str, Any], name: str) -> Path:
        path = Path(self.temp_dir.name) / name
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def _absolute_manifest(self) -> dict[str, Any]:
        manifest = json.loads(json.dumps(self.manifest))
        base = self.asset_dir.resolve()
        manifest["video"]["path"] = str(
            (base / manifest["video"]["path"]).resolve()
        )
        manifest["dialogue_track"]["path"] = str(
            (base / manifest["dialogue_track"]["path"]).resolve()
        )
        for segment in manifest["segments"]:
            for target in segment["targets"].values():
                target["path"] = str(
                    (base / target["path"]).resolve()
                )
        return manifest

    def _run_zh(
        self,
        output: Path,
        report: Path,
        *,
        offset_ms: int = 0,
        manifest: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return run_cli(
            "--manifest",
            str(manifest or self.manifest_path),
            "--target-language",
            "zh",
            "--output",
            str(output),
            "--report",
            str(report),
            "--offset-ms",
            str(offset_ms),
        )

    def test_zh_timeline_loudness_peak_and_flags(self) -> None:
        output = Path(self.temp_dir.name) / "zh.wav"
        report_path = Path(self.temp_dir.name) / "zh.json"
        completed = self._run_zh(output, report_path)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        info = sf.info(str(output))
        self.assertEqual(info.samplerate, 48_000)
        self.assertEqual(info.channels, 2)
        self.assertEqual(info.subtype, "PCM_16")
        self.assertEqual(info.frames, 480_000)

        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["target_language"], "zh")
        self.assertEqual(report["output"]["samples"], 480_000)
        self.assertEqual(len(report["segments"]), 4)
        for segment in report["segments"]:
            self.assertIsNotNone(segment["integrated_loudness_lufs"])
            self.assertLessEqual(
                abs(segment["integrated_loudness_lufs"] + 23.0),
                1.0,
                segment,
            )
            self.assertLessEqual(segment["peak_dbfs"], -1.0, segment)

        by_id = {segment["id"]: segment for segment in report["segments"]}
        line2 = by_id["scene01_line02"]
        line3 = by_id["scene01_line03"]
        line4 = by_id["scene01_line04"]
        self.assertTrue(line2["caused_shift"])
        self.assertTrue(line3["shifted"])
        self.assertTrue(line4["shifted"])
        self.assertTrue(line4["truncated"])
        self.assertGreater(line3["crossfade_ms"], 0.0)
        self.assertLessEqual(line3["crossfade_ms"], 10.0)
        self.assertAlmostEqual(line4["actual_end_sec"], 10.0, places=6)
        self.assertGreaterEqual(line3["actual_start_sec"], line2["actual_end_sec"] - 0.010001)

    def test_second_language_is_independent(self) -> None:
        output = Path(self.temp_dir.name) / "en.wav"
        report_path = Path(self.temp_dir.name) / "en.json"
        completed = run_cli(
            "--manifest",
            str(self.manifest_path),
            "--target-language",
            "en",
            "--output",
            str(output),
            "--report",
            str(report_path),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["target_language"], "en")
        self.assertEqual(report["segments"][0]["subtitle"], self.manifest["segments"][0]["text"]["en"])
        info = sf.info(str(output))
        self.assertEqual(info.frames, 480_000)

    def test_positive_global_offset_moves_timeline(self) -> None:
        output = Path(self.temp_dir.name) / "offset.wav"
        report_path = Path(self.temp_dir.name) / "offset.json"
        completed = self._run_zh(output, report_path, offset_ms=50)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        first = report["segments"][0]
        self.assertAlmostEqual(first["actual_start_sec"], 0.55, places=6)
        self.assertFalse(first["shifted"])
        self.assertEqual(report["global_offset_ms"], 50)

    def test_negative_offset_before_zero_is_rejected(self) -> None:
        output = Path(self.temp_dir.name) / "negative.wav"
        report_path = Path(self.temp_dir.name) / "negative.json"
        completed = self._run_zh(output, report_path, offset_ms=-600)
        self.assertEqual(completed.returncode, 4)
        payload = json.loads(completed.stderr)
        self.assertEqual(payload["code"], "OFFSET_OUT_OF_RANGE")
        self.assertEqual(payload["exit_code"], 4)
        self.assertFalse(output.exists())
        self.assertFalse(report_path.exists())

    def test_sample_rate_mismatch_has_path_code_and_cleanup(self) -> None:
        manifest = self._absolute_manifest()
        manifest["segments"][0]["targets"]["zh"]["sample_rate"] = 48_000
        manifest_path = self._write_manifest(manifest, "rate-mismatch.json")
        output = Path(self.temp_dir.name) / "rate.wav"
        report_path = Path(self.temp_dir.name) / "rate.json"
        completed = self._run_zh(
            output, report_path, manifest=manifest_path
        )
        self.assertEqual(completed.returncode, 3)
        payload = json.loads(completed.stderr)
        self.assertEqual(payload["code"], "SAMPLE_RATE_MISMATCH")
        self.assertTrue(payload["file"].endswith("segment_01.wav"))
        self.assertEqual(payload["details"]["actual_sample_rate"], 44_100)
        self.assertFalse(output.exists())
        self.assertFalse(report_path.exists())

    def test_missing_target_file_has_path_code_and_cleanup(self) -> None:
        manifest = self._absolute_manifest()
        missing = Path(self.temp_dir.name) / "does-not-exist.wav"
        manifest["segments"][2]["targets"]["zh"]["path"] = str(missing)
        manifest_path = self._write_manifest(manifest, "missing.json")
        output = Path(self.temp_dir.name) / "missing.wav"
        report_path = Path(self.temp_dir.name) / "missing-report.json"
        completed = self._run_zh(
            output, report_path, manifest=manifest_path
        )
        self.assertEqual(completed.returncode, 3)
        payload = json.loads(completed.stderr)
        self.assertEqual(payload["code"], "AUDIO_FILE_MISSING")
        self.assertEqual(payload["file"], str(missing.resolve()))
        self.assertFalse(output.exists())
        self.assertFalse(report_path.exists())

    def test_decode_failure_reports_file_and_cleans_outputs(self) -> None:
        manifest = self._absolute_manifest()
        corrupt = Path(self.temp_dir.name) / "corrupt.wav"
        corrupt.write_bytes(b"not-a-wave-file")
        manifest["segments"][1]["targets"]["zh"]["path"] = str(corrupt)
        manifest_path = self._write_manifest(manifest, "corrupt.json")
        output = Path(self.temp_dir.name) / "corrupt-out.wav"
        report_path = Path(self.temp_dir.name) / "corrupt-report.json"
        completed = self._run_zh(
            output, report_path, manifest=manifest_path
        )
        self.assertEqual(completed.returncode, 3)
        payload = json.loads(completed.stderr)
        self.assertEqual(payload["code"], "AUDIO_DECODE_FAILED")
        self.assertEqual(payload["file"], str(corrupt.resolve()))
        self.assertFalse(output.exists())
        self.assertFalse(report_path.exists())

    def test_missing_manifest_field_is_structured_json(self) -> None:
        manifest = json.loads(json.dumps(self.manifest))
        del manifest["segments"][0]["targets"]["zh"]["path"]
        manifest_path = self._write_manifest(manifest, "missing-field.json")
        output = Path(self.temp_dir.name) / "field.wav"
        report_path = Path(self.temp_dir.name) / "field.json"
        completed = self._run_zh(
            output, report_path, manifest=manifest_path
        )
        self.assertEqual(completed.returncode, 2)
        payload = json.loads(completed.stderr)
        self.assertEqual(payload["code"], "MISSING_FIELD")
        self.assertIn("segments[0].targets.zh.path", payload["details"]["field"])
        self.assertFalse(output.exists())
        self.assertFalse(report_path.exists())

    def test_inverted_time_range_is_rejected(self) -> None:
        manifest = json.loads(json.dumps(self.manifest))
        manifest["segments"][0]["end"] = manifest["segments"][0]["start"]
        manifest_path = self._write_manifest(manifest, "inverted.json")
        output = Path(self.temp_dir.name) / "inverted.wav"
        report_path = Path(self.temp_dir.name) / "inverted-report.json"
        completed = self._run_zh(
            output, report_path, manifest=manifest_path
        )
        self.assertEqual(completed.returncode, 2)
        payload = json.loads(completed.stderr)
        self.assertEqual(payload["code"], "INVALID_TIME_RANGE")
        self.assertFalse(output.exists())
        self.assertFalse(report_path.exists())

    def test_failure_does_not_remove_previous_valid_outputs(self) -> None:
        output = Path(self.temp_dir.name) / "previous.wav"
        report_path = Path(self.temp_dir.name) / "previous.json"
        first = self._run_zh(output, report_path)
        self.assertEqual(first.returncode, 0, first.stderr)
        original_wav = output.read_bytes()
        original_report = report_path.read_bytes()

        manifest = self._absolute_manifest()
        manifest["segments"][0]["targets"]["zh"]["path"] = str(
            Path(self.temp_dir.name) / "missing-again.wav"
        )
        manifest_path = self._write_manifest(manifest, "previous-fail.json")
        failed = self._run_zh(
            output, report_path, manifest=manifest_path
        )
        self.assertEqual(failed.returncode, 3)
        self.assertEqual(output.read_bytes(), original_wav)
        self.assertEqual(report_path.read_bytes(), original_report)

    def test_report_numbers_are_strict_json(self) -> None:
        output = Path(self.temp_dir.name) / "strict.wav"
        report_path = Path(self.temp_dir.name) / "strict.json"
        completed = self._run_zh(output, report_path)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        text = report_path.read_text(encoding="utf-8")
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        report = json.loads(text)
        for segment in report["segments"]:
            for key, value in segment.items():
                if isinstance(value, float):
                    self.assertTrue(math.isfinite(value), (key, value))

    def test_unicode_and_space_paths(self) -> None:
        manifest = self._absolute_manifest()
        source_target = Path(
            manifest["segments"][0]["targets"]["zh"]["path"]
        )
        unicode_dir = Path(self.temp_dir.name) / "本地化 素材"
        unicode_dir.mkdir()
        unicode_target = unicode_dir / "第一段 对白.wav"
        shutil.copy2(source_target, unicode_target)
        manifest["segments"][0]["targets"]["zh"]["path"] = str(
            unicode_target
        )
        manifest_path = self._write_manifest(manifest, "unicode.json")
        output = Path(self.temp_dir.name) / "输出 目录" / "对白 轨.wav"
        report_path = (
            Path(self.temp_dir.name) / "输出 目录" / "报告 文件.json"
        )
        completed = self._run_zh(
            output, report_path, manifest=manifest_path
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(output.is_file())
        self.assertTrue(report_path.is_file())
        self.assertEqual(sf.info(str(output)).frames, 480_000)


if __name__ == "__main__":
    unittest.main()

