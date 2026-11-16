from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lipalign.synth import generate_samples
from lipalign.wavio import read_wav, write_pcm16_wav


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "lipalign"] + list(arguments),
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        timeout=120,
    )


def one_json_line(stderr: str) -> dict:
    lines = [line for line in stderr.splitlines() if line.strip()]
    if len(lines) != 1:
        raise AssertionError("expected one stderr line, got {!r}".format(stderr))
    return json.loads(lines[0])


class ErrorHandlingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="lipalign errors ")
        cls.root = Path(cls.temp.name)
        cls.sample_dir = cls.root / "sample data"
        generate_samples(cls.sample_dir, seed=777)
        cls.timeline = cls.sample_dir / "timeline.json"
        with cls.timeline.open("r", encoding="utf-8") as handle:
            cls.timeline_payload = json.load(handle)
        cls.cache = cls.root / "error cache"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_invalid_json_has_single_line_error(self) -> None:
        bad_json = self.root / "bad timeline.json"
        bad_json.write_text("{bad json\n", encoding="utf-8")
        result = run_cli(
            "align",
            "--timeline",
            str(bad_json),
            "--audio-dir",
            str(self.sample_dir / "audio"),
            "--output",
            str(self.root / "bad json output.wav"),
            "--cache-dir",
            str(self.cache),
        )
        self.assertNotEqual(result.returncode, 0)
        payload = one_json_line(result.stderr)
        self.assertEqual(payload["code"], "E_JSON_INVALID")

    def test_corrupt_wav_has_single_line_error(self) -> None:
        broken = self.root / "broken.wav"
        broken.write_bytes(b"RIFF\x10\x00\x00\x00WAVEfmt ")
        result = run_cli(
            "align",
            "--timeline",
            str(self.timeline),
            "--audio",
            "en={}".format(broken),
            "--output",
            str(self.root / "broken output.wav"),
            "--cache-dir",
            str(self.cache),
        )
        self.assertNotEqual(result.returncode, 0)
        payload = one_json_line(result.stderr)
        self.assertEqual(payload["code"], "E_WAV_CORRUPT")

    def test_rate_mismatch_has_single_line_error(self) -> None:
        wrong_rate = self.root / "wrong rate.wav"
        write_pcm16_wav(
            wrong_rate,
            np.zeros(44100, dtype=np.int16),
            sample_rate=44100,
            channels=1,
        )
        result = run_cli(
            "align",
            "--timeline",
            str(self.timeline),
            "--audio",
            "en={}".format(wrong_rate),
            "--output",
            str(self.root / "rate output.wav"),
            "--cache-dir",
            str(self.cache),
        )
        self.assertNotEqual(result.returncode, 0)
        payload = one_json_line(result.stderr)
        self.assertEqual(payload["code"], "E_RATE_MISMATCH")

    def test_missing_sentence_keeps_other_results_and_cache(self) -> None:
        source = read_wav(self.sample_dir / "audio" / "en.wav")
        missing_sentence = self.timeline_payload["sentences"][2]
        source.samples[
            missing_sentence["start_sample"] - 1000 :
            missing_sentence["end_sample"] + 30000,
            0,
        ] = 0
        damaged = self.root / "missing sentence.wav"
        write_pcm16_wav(damaged, source.samples, source.sample_rate, 1)
        output = self.root / "partial output.wav"
        report_path = self.root / "partial report.json"
        partial_cache = self.root / "partial cache"
        result = run_cli(
            "align",
            "--timeline",
            str(self.timeline),
            "--audio",
            "en={}".format(damaged),
            "--output",
            str(output),
            "--report",
            str(report_path),
            "--cache-dir",
            str(partial_cache),
        )
        self.assertEqual(result.returncode, 6)
        payload = one_json_line(result.stderr)
        self.assertEqual(payload["code"], "E_PARTIAL_FAILURE")
        self.assertTrue(output.is_file())
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["totals"]["sentences_failed"], 1)
        total_sentences = len(self.timeline_payload["sentences"])
        self.assertEqual(report["totals"]["sentences_processed"], total_sentences - 1)
        failed = [
            sentence
            for sentence in report["languages"][0]["sentences"]
            if sentence["status"] == "failed"
        ]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["id"], missing_sentence["id"])
        cache_entries = list((partial_cache / "stretched").glob("*.json"))
        self.assertEqual(len(cache_entries), total_sentences - 1)


if __name__ == "__main__":
    unittest.main()

