from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

from dialogue_sync.srt import parse_srt_text
from dialogue_sync.state import inspect_state
from tests.fixtures import create_project

REPO = Path(__file__).resolve().parents[1]


def run_cli(args: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "dialogue_sync", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def ffprobe_json(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            (
                "stream=index,codec_name,codec_type,sample_rate,channels,"
                "bits_per_sample,bits_per_raw_sample:stream_tags=language,title"
            ),
            "-show_chapters",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def ebu_r128(path: Path) -> tuple[float, float]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(path),
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    summary = result.stderr[result.stderr.rfind("Summary:") :]
    integrated_match = re.search(r"I:\s+(-?\d+(?:\.\d+)?)\s+LUFS", summary)
    peak_match = re.search(r"Peak:\s+(-?\d+(?:\.\d+)?)\s+dBFS", summary)
    if not integrated_match or not peak_match:
        raise AssertionError("could not parse ebur128 summary")
    return float(integrated_match.group(1)), float(peak_match.group(1))


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="dialogue-sync-e2e-")
        cls.root = Path(cls.temp.name)
        cls.fixture = create_project(cls.root / "fixture")
        cls.out = cls.root / "out"
        cls.build_result = run_cli(
            [
                "build",
                "--timeline",
                str(cls.fixture["timeline"]),
                "--output-dir",
                str(cls.out),
                "--quiet",
            ]
        )
        cls.report_path = cls.out / "report.json"
        cls.report = (
            json.loads(cls.report_path.read_text(encoding="utf-8"))
            if cls.report_path.is_file()
            else {}
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_01_command_succeeds_with_partial_entry_status(self) -> None:
        self.assertEqual(self.build_result.returncode, 0, self.build_result.stderr)
        self.assertEqual(self.report["status"], "partial")
        self.assertEqual(self.report["summary"]["entries_total"], 5)
        self.assertEqual(self.report["summary"]["entries_succeeded"], 4)
        self.assertEqual(self.report["summary"]["entries_skipped_or_failed"], 1)

    def test_02_operations_and_error_codes(self) -> None:
        entries = {entry["index"]: entry for entry in self.report["entries"]}
        self.assertEqual(entries[1]["operation"], "stretch")
        self.assertAlmostEqual(entries[1]["stretch_rate"], 1.5 / 1.4, places=6)
        self.assertEqual(entries[2]["operation"], "crop")
        self.assertEqual(entries[3]["operation"], "pad_silence")
        self.assertEqual(entries[4]["operation"], "skip")
        self.assertIn("E_RANGE", entries[4]["error_codes"])
        self.assertEqual(entries[5]["operation"], "none")
        self.assertIn("E_CLIP", entries[2]["error_codes"])
        self.assertIn("E_CLIP", entries[5]["error_codes"])
        for entry in entries.values():
            self.assertIsInstance(entry["error_codes"], list)
            self.assertTrue(all(isinstance(code, str) for code in entry["error_codes"]))
            self.assertIn("applied_gain_db", entry)
            self.assertIn("source_path", entry)
            self.assertIn("source_timecode", entry)
            self.assertIn("target_timecode", entry)
        self.assertLess(self.report["mix"]["mix_gain_db"], 0)

    def test_03_subtitles_are_shifted_ordered_and_overlap_warned(self) -> None:
        srt_path = self.out / "dialogue_mix.srt"
        document = parse_srt_text(srt_path.read_text(encoding="utf-8"))
        self.assertEqual([cue.index for cue in document.cues], [1, 2, 3, 4, 5])
        expected = {
            1: (1_000, 2_500),
            2: (4_000, 5_500),
            3: (7_000, 8_500),
            4: (9_500, 11_000),
            5: (4_500, 5_500),
        }
        for cue in document.cues:
            self.assertEqual((cue.start_ms, cue.end_ms), expected[cue.index])
            self.assertGreaterEqual(cue.start_ms, 0)
            self.assertGreater(cue.end_ms, cue.start_ms)
        overlaps = self.report["subtitles"]["overlaps"]
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0]["code"], "W_SUBTITLE_OVERLAP")

    def test_04_audio_profile_loudness_and_duration(self) -> None:
        wav_path = self.out / "dialogue_mix.wav"
        with wave.open(str(wav_path), "rb") as handle:
            self.assertEqual(handle.getframerate(), 48_000)
            self.assertEqual(handle.getnchannels(), 2)
            self.assertEqual(handle.getsampwidth(), 3)
            self.assertEqual(handle.getnframes(), 576_000)
        integrated, peak = ebu_r128(wav_path)
        self.assertGreaterEqual(integrated, -23.5)
        self.assertLessEqual(integrated, -22.5)
        self.assertLessEqual(peak, -1.0)
        self.assertTrue(self.report["loudness"]["within_tolerance"])
        self.assertLessEqual(
            abs(self.report["loudness"]["after"]["integrated_lufs"] + 23.0), 0.5
        )
        self.assertLessEqual(self.report["loudness"]["after"]["true_peak_dbtp"], -1.0)
        self.assertTrue(self.report["loudness"]["true_peak_exceeded_before"])
        self.assertIn("before", self.report["loudness"])
        self.assertIn("after", self.report["loudness"])

    def test_05_mkv_language_chapters_and_duration(self) -> None:
        probe = ffprobe_json(self.out / "dialogue_mix.mkv")
        streams = probe["streams"]
        audio = [stream for stream in streams if stream["codec_type"] == "audio"][0]
        subtitle = [stream for stream in streams if stream["codec_type"] == "subtitle"][0]
        self.assertEqual(audio["codec_name"], "pcm_s24le")
        self.assertEqual(audio["sample_rate"], "48000")
        self.assertEqual(audio["channels"], 2)
        self.assertEqual(audio["bits_per_raw_sample"], "24")
        self.assertEqual(audio["tags"]["language"], "eng")
        self.assertEqual(subtitle["codec_name"], "subrip")
        self.assertEqual(subtitle["tags"]["language"], "eng")
        chapters = probe["chapters"]
        self.assertEqual(len(chapters), 5)
        for chapter in chapters:
            self.assertRegex(chapter["tags"]["title"], r"Dialogue \d+")
        duration = float(probe["format"]["duration"])
        self.assertLessEqual(abs(duration - 12.0), 0.05)
        self.assertLessEqual(self.report["container"]["duration_error_ms"], 50)
        self.assertTrue(self.report["container"]["verified"])

    def test_06_atomic_outputs_leave_no_partial_files(self) -> None:
        partials = [
            path
            for path in self.out.rglob("*")
            if ".partial" in path.name or path.name.endswith(".tmp")
        ]
        self.assertEqual(partials, [])
        for name in ("dialogue_mix.wav", "dialogue_mix.srt", "dialogue_mix.mkv", "report.json"):
            self.assertTrue((self.out / name).is_file())

    def test_07_mono_override_and_language_tags(self) -> None:
        mono_out = self.root / "mono"
        result = run_cli(
            [
                "build",
                "--timeline",
                str(self.fixture["timeline"]),
                "--output-dir",
                str(mono_out),
                "--channels",
                "mono",
                "--language",
                "jpn",
                "--subtitle-language",
                "jpn",
                "--quiet",
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with wave.open(str(mono_out / "dialogue_mix.wav"), "rb") as handle:
            self.assertEqual(handle.getnchannels(), 1)
            self.assertEqual(handle.getframerate(), 48_000)
            self.assertEqual(handle.getsampwidth(), 3)
        probe = ffprobe_json(mono_out / "dialogue_mix.mkv")
        audio = [stream for stream in probe["streams"] if stream["codec_type"] == "audio"][0]
        subtitle = [stream for stream in probe["streams"] if stream["codec_type"] == "subtitle"][0]
        self.assertEqual(audio["channels"], 1)
        self.assertEqual(audio["tags"]["language"], "jpn")
        self.assertEqual(subtitle["tags"]["language"], "jpn")


class RecoveryTests(unittest.TestCase):
    def test_invalid_entry_timecode_is_reported_without_failing_run(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dialogue-sync-etimcode-") as temp:
            root = Path(temp)
            fixture = create_project(root / "fixture")
            timeline_path = fixture["timeline"]
            timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
            timeline["dialogue_entries"].append(
                {
                    "id": "bad_timecode",
                    "source": "media/dialog_03_48k_mono24.wav",
                    "source_start": "BAD",
                    "target_start": "00:00:11.000",
                    "subtitle_index": 6,
                }
            )
            timeline_path.write_text(
                json.dumps(timeline, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with fixture["srt"].open("a", encoding="utf-8") as handle:
                handle.write(
                    "\n6\n00:00:00,000 --> 00:00:01,000\nBad timecode line\n"
                )
            out = root / "out"
            result = run_cli(
                [
                    "build",
                    "--timeline",
                    str(timeline_path),
                    "--output-dir",
                    str(out),
                    "--quiet",
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            bad = report["entries"][-1]
            self.assertEqual(bad["status"], "skipped")
            self.assertEqual(bad["error_code"], "E_TIMECODE")
            self.assertEqual(
                report["summary"]["entries_skipped_or_failed"], 2
            )

    def test_abrupt_exit_then_resume_from_checksummed_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dialogue-sync-resume-") as temp:
            root = Path(temp)
            fixture = create_project(root / "fixture")
            out = root / "out"
            work = out / ".dialogue_sync_work"
            base = [
                "build",
                "--timeline",
                str(fixture["timeline"]),
                "--output-dir",
                str(out),
            ]
            aborted = run_cli([*base, "--simulate-abort-after", "2"])
            self.assertEqual(aborted.returncode, 75, aborted.stderr)
            self.assertTrue((work / "state.json").is_file())
            self.assertFalse((out / "report.json").exists())

            first_state = inspect_state(work)
            self.assertTrue(first_state["valid"])
            self.assertEqual(first_state["entry_reports"], 2)
            self.assertEqual(len(first_state["fragments"]), 2)
            self.assertTrue(all(item["checksum_ok"] for item in first_state["fragments"]))
            fragment_hashes_before = {
                item["entry"]: item["checksum_ok"] for item in first_state["fragments"]
            }
            self.assertTrue(all(fragment_hashes_before.values()))

            resumed = run_cli(base)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertIn("[3/5]", resumed.stdout)
            self.assertNotIn("[1/5]", resumed.stdout)
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["state"]["phase"], "complete")
            self.assertEqual(report["summary"]["entries_total"], 5)
            final_state = inspect_state(work)
            self.assertEqual(final_state["phase"], "complete")
            self.assertEqual(len(final_state["fragments"]), 4)
            self.assertTrue(all(item["checksum_ok"] for item in final_state["fragments"]))

    def test_corrupt_state_is_detected_and_force_recovers(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dialogue-sync-corrupt-") as temp:
            root = Path(temp)
            fixture = create_project(root / "fixture")
            out = root / "out"
            work = out / ".dialogue_sync_work"
            base = [
                "build",
                "--timeline",
                str(fixture["timeline"]),
                "--output-dir",
                str(out),
                "--quiet",
            ]
            aborted = run_cli([*base, "--simulate-abort-after", "1"])
            self.assertEqual(aborted.returncode, 75)
            state_path = work / "state.json"
            with state_path.open("ab") as handle:
                handle.write(b"\n{truncated")

            corrupt = run_cli([*base, "--quiet"])
            self.assertEqual(corrupt.returncode, 2)
            self.assertIn("E_STATE_INTEGRITY", corrupt.stderr)
            self.assertFalse((out / "report.json").exists())

            recovered = run_cli([*base, "--force", "--quiet"])
            self.assertEqual(recovered.returncode, 0, recovered.stderr)
            probe = inspect_state(work)
            self.assertTrue(probe["valid"])
            self.assertEqual(probe["phase"], "complete")


if __name__ == "__main__":
    unittest.main()

