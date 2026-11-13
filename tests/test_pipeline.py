from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from audiopipe.manifest import parse_manifest
from audiopipe.model import Job
from audiopipe.util import probe_wav, write_pcm16_wav_atomic


REPO = Path(__file__).resolve().parents[1]


class WavAssemblyTests(unittest.TestCase):
    def test_header_and_data_sizes_are_exact(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as temp:
            root = Path(temp)
            pcm = root / "raw.pcm"
            pcm.write_bytes(b"\x00\x01\x02\x03" * 1200)
            output = root / "out.wav"
            info = write_pcm16_wav_atomic(pcm, output)
            self.assertEqual(info["data_size"], pcm.stat().st_size)
            self.assertEqual(info["file_size"], pcm.stat().st_size + 44)
            self.assertEqual(info["riff_size"], info["file_size"])
            validated = probe_wav(output)
            self.assertEqual(validated["sample_rate"], 48000)
            self.assertEqual(validated["channels"], 2)
            self.assertEqual(validated["bits_per_sample"], 16)

    def test_truncated_data_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as temp:
            root = Path(temp)
            pcm = root / "raw.pcm"
            pcm.write_bytes(b"\x00\x00\x00\x00" * 100)
            output = root / "out.wav"
            write_pcm16_wav_atomic(pcm, output)
            payload = output.read_bytes()
            output.write_bytes(payload[:-4])
            with self.assertRaises(ValueError):
                probe_wav(output)


class ManifestTests(unittest.TestCase):
    def test_nested_tracks_target_alias_and_fade_aliases(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as temp:
            show = Path(temp) / "show"
            show.mkdir()
            for name in ("d.wav", "m.wav", "a.wav"):
                (show / name).write_bytes(b"")
            manifest_path = show / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "tracks": {"dialogue": "d.wav", "music": "m.wav", "ambience": "a.wav"},
                        "dialogue_target_lufs": -16.0,
                        "music_fade": {"start_sec": 1.0, "duration_sec": 2.0},
                        "music_gain_db": -6.0,
                        "environment_gain_db": -12.0,
                    }
                ),
                encoding="utf-8",
            )
            job = Job("show", show, manifest_path, "show/manifest.json")
            parsed = parse_manifest(job)
            self.assertEqual(parsed.target_lufs, -16.0)
            self.assertEqual(parsed.music_fade_start_sec, 1.0)
            self.assertEqual(parsed.music_fade_duration_sec, 2.0)
            self.assertEqual(parsed.max_attenuation_db, 24.0)
            self.assertEqual(parsed.max_boost_db, 12.0)

    def test_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as temp:
            show = Path(temp) / "show"
            show.mkdir()
            manifest_path = show / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "dialogue_path": "../outside.wav",
                        "music_path": "m.wav",
                        "ambience_path": "a.wav",
                        "target_lufs": -16.0,
                        "music_fade_start_sec": 1.0,
                        "music_fade_duration_sec": 1.0,
                        "music_gain_db": -6.0,
                        "ambience_gain_db": -12.0,
                    }
                ),
                encoding="utf-8",
            )
            job = Job("show", show, manifest_path, "show/manifest.json")
            with self.assertRaises(ValueError):
                parse_manifest(job)


@unittest.skipUnless(
    os.environ.get("FFMPEG") and os.environ.get("FFPROBE"),
    "set FFMPEG and FFPROBE to run end-to-end tests",
)
class EndToEndTests(unittest.TestCase):
    def test_builtin_self_test_and_cli_summary(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as temp:
            selftest_dir = Path(temp) / "selftest"
            command = [
                sys.executable,
                str(REPO / "audio_pipeline.py"),
                "self-test",
                "--dir",
                str(selftest_dir),
                "--ffmpeg",
                os.environ["FFMPEG"],
                "--ffprobe",
                os.environ["FFPROBE"],
                "--jobs",
                "3",
            ]
            completed = subprocess.run(
                command,
                cwd=REPO,
                text=True,
                capture_output=True,
                timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = json.loads((selftest_dir / "selftest_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["ok"])
            self.assertEqual(report["passed_checks"], report["total_checks"])
            self.assertEqual(report["first_pipeline_exit_code"], 1)

            result_path = selftest_dir / "cli_result.json"
            cli = subprocess.run(
                [
                    sys.executable,
                    str(REPO / "audio_pipeline.py"),
                    "run",
                    "--root",
                    str(selftest_dir / "shows"),
                    "--ffmpeg",
                    os.environ["FFMPEG"],
                    "--ffprobe",
                    os.environ["FFPROBE"],
                    "--concurrency",
                    "3",
                    "--work-dir",
                    str(selftest_dir / "work"),
                    "--result",
                    str(result_path),
                ],
                cwd=REPO,
                text=True,
                capture_output=True,
                timeout=120,
            )
            self.assertEqual(cli.returncode, 1)
            lines = cli.stderr.strip().splitlines()
            self.assertEqual(len(lines), 1, cli.stderr)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            for key in ("total", "completed", "skipped", "failed", "interrupted"):
                self.assertIn(f"{key}={result['counts'][key]}", lines[0])
            self.assertIn(f"exit={result['exit_code']}", lines[0])


if __name__ == "__main__":
    unittest.main()
