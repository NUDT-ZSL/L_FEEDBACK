from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from lipalign.synth import generate_samples
from lipalign.wavio import read_wav, sha256_file, write_pcm16_wav


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "lipalign"] + list(arguments),
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        timeout=120,
    )


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="lipalign test ")
        cls.root = Path(cls.temp.name)
        cls.sample_dir = cls.root / "sample data"
        cls.generated = generate_samples(cls.sample_dir, seed=424242)
        cls.timeline_path = cls.sample_dir / "timeline.json"
        with cls.timeline_path.open("r", encoding="utf-8") as handle:
            cls.timeline = json.load(handle)
        cls.cache_dir = cls.root / "cache with spaces"
        cls.output_dir = cls.root / "output with spaces"
        cls.output = cls.output_dir / "multi track.wav"
        cls.report = cls.output_dir / "alignment report.json"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_01_generated_sample_layout(self) -> None:
        timeline = self.timeline
        self.assertEqual(timeline["sample_rate"], 48000)
        self.assertGreaterEqual(len(timeline["sentences"]), 1)
        self.assertEqual(len(self.generated["languages"]), 3)
        for sentence in timeline["sentences"]:
            self.assertLess(sentence["start_sample"], sentence["end_sample"])
            self.assertTrue(sentence["phonemes"])
            for phoneme in sentence["phonemes"]:
                self.assertTrue(phoneme["phoneme"])
                self.assertTrue(phoneme["mouth"])
        for language in ("en", "zh", "es"):
            with wave.open(str(self.sample_dir / "audio" / (language + ".wav")), "rb") as wav:
                self.assertEqual(wav.getnchannels(), 1)
                self.assertEqual(wav.getframerate(), 48000)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertEqual(wav.getnframes(), timeline["duration_samples"])

    def test_02_generation_is_deterministic(self) -> None:
        second_dir = self.root / "sample deterministic"
        generate_samples(second_dir, seed=424242)
        for relative in (
            "timeline.json",
            "audio/en.wav",
            "audio/zh.wav",
            "audio/es.wav",
            "background.wav",
            "manifest.json",
        ):
            with self.subTest(relative=relative):
                self.assertEqual(
                    file_digest(self.sample_dir / relative),
                    file_digest(second_dir / relative),
                )

    def test_03_wsola_alignment_and_report(self) -> None:
        result = run_cli(
            "align",
            "--timeline",
            str(self.timeline_path),
            "--audio-dir",
            str(self.sample_dir / "audio"),
            "--background",
            str(self.sample_dir / "background.wav"),
            "--algorithm",
            "wsola",
            "--output",
            str(self.output),
            "--report",
            str(self.report),
            "--cache-dir",
            str(self.cache_dir),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cache_hits=0", result.stdout)
        self.assertEqual(result.stderr, "")

        with wave.open(str(self.output), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 4)
            self.assertEqual(wav.getframerate(), 48000)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getnframes(), self.timeline["duration_samples"])

        audio = read_wav(self.output)
        self.assertEqual(audio.samples.shape, (self.timeline["duration_samples"], 4))
        voice_rms = np.sqrt(
            np.mean((audio.samples[:, :3].astype(np.float64) / 32768.0) ** 2, axis=0)
        )
        background_rms = float(
            np.sqrt(
                np.mean(
                    (audio.samples[:, 3].astype(np.float64) / 32768.0) ** 2
                )
            )
        )
        self.assertTrue(np.all(voice_rms > 0.01))
        self.assertGreater(background_rms, 0.002)
        self.assertLess(background_rms, 0.02)

        with self.report.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["output"]["channel_count"], 4)
        self.assertTrue(report["output"]["all_channels_same_sample_count"])
        self.assertLessEqual(
            abs(report["output"]["duration_error_sec"]), 0.005
        )
        total_sentences = len(self.timeline["sentences"]) * 3
        self.assertEqual(report["totals"]["sentences_processed"], total_sentences)
        self.assertEqual(report["totals"]["cache_recomputes"], total_sentences)
        for language in report["languages"]:
            self.assertEqual(
                len(language["sentences"]), len(self.timeline["sentences"])
            )
            for sentence in language["sentences"]:
                self.assertEqual(sentence["status"], "ok")
                self.assertIsNotNone(sentence["source_start_sample"])
                self.assertIsNotNone(sentence["source_end_sample"])
                self.assertEqual(sentence["algorithm"], "wsola")
                self.assertEqual(
                    sentence["actual_start_sample"], sentence["target_start_sample"]
                )
                self.assertEqual(
                    sentence["actual_end_sample"], sentence["target_end_sample"]
                )
                expected_factor = (
                    sentence["target_samples"] / sentence["source_samples"]
                )
                self.assertAlmostEqual(
                    sentence["stretch_factor"], expected_factor, places=7
                )

    def test_04_cache_reuses_all_unchanged_fragments(self) -> None:
        second_output = self.output_dir / "multi track second.wav"
        second_report = self.output_dir / "alignment report second.json"
        result = run_cli(
            "align",
            "--timeline",
            str(self.timeline_path),
            "--audio-dir",
            str(self.sample_dir / "audio"),
            "--background",
            str(self.sample_dir / "background.wav"),
            "--output",
            str(second_output),
            "--report",
            str(second_report),
            "--cache-dir",
            str(self.cache_dir),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        total_sentences = len(self.timeline["sentences"]) * 3
        self.assertIn("cache_hits={}".format(total_sentences), result.stdout)
        self.assertEqual(sha256_file(self.output), sha256_file(second_output))
        with second_report.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["totals"]["cache_hits"], total_sentences)
        self.assertEqual(report["totals"]["cache_recomputes"], 0)

    def test_05_only_changed_fragment_is_recomputed(self) -> None:
        source_path = self.sample_dir / "audio" / "en.wav"
        modified_path = self.root / "en modified.wav"
        audio = read_wav(source_path)
        sentence = self.timeline["sentences"][0]
        sample_index = (sentence["start_sample"] + sentence["end_sample"]) // 2
        original = int(audio.samples[sample_index, 0])
        audio.samples[sample_index, 0] = np.int16(
            max(-32768, min(32767, original + 257))
        )
        write_pcm16_wav(modified_path, audio.samples, audio.sample_rate, 1)

        output = self.output_dir / "incremental.wav"
        report_path = self.output_dir / "incremental report.json"
        result = run_cli(
            "align",
            "--timeline",
            str(self.timeline_path),
            "--audio",
            "en={}".format(modified_path),
            "--audio",
            "es={}".format(self.sample_dir / "audio" / "es.wav"),
            "--audio",
            "zh={}".format(self.sample_dir / "audio" / "zh.wav"),
            "--background",
            str(self.sample_dir / "background.wav"),
            "--output",
            str(output),
            "--report",
            str(report_path),
            "--cache-dir",
            str(self.cache_dir),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        total_sentences = len(self.timeline["sentences"]) * 3
        self.assertIn("cache_hits={}".format(total_sentences - 1), result.stdout)
        self.assertIn("cache_recomputes=1", result.stdout)
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        recomputed = [
            (language["language"], sentence["id"])
            for language in report["languages"]
            for sentence in language["sentences"]
            if sentence["cache"] == "recomputed"
        ]
        self.assertEqual(recomputed, [("en", sentence["id"])])

    def test_06_algorithm_change_invalidates_cache(self) -> None:
        output = self.output_dir / "phase vocoder.wav"
        report_path = self.output_dir / "phase vocoder report.json"
        result = run_cli(
            "align",
            "--timeline",
            str(self.timeline_path),
            "--audio-dir",
            str(self.sample_dir / "audio"),
            "--background",
            str(self.sample_dir / "background.wav"),
            "--algorithm",
            "phase_vocoder",
            "--output",
            str(output),
            "--report",
            str(report_path),
            "--cache-dir",
            str(self.cache_dir),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cache_hits=0", result.stdout)
        total_sentences = len(self.timeline["sentences"]) * 3
        self.assertIn("cache_recomputes={}".format(total_sentences), result.stdout)
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["settings"]["algorithm"], "phase_vocoder")
        self.assertTrue(
            all(
                sentence["algorithm"] == "phase_vocoder"
                for language in report["languages"]
                for sentence in language["sentences"]
            )
        )

    def test_07_report_command_summarizes_report_and_cache(self) -> None:
        result = run_cli(
            "report",
            "--report",
            str(self.report),
            "--cache-dir",
            str(self.cache_dir),
            "--json",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["report"]["status"], "ok")
        sidecars = list((self.cache_dir / "stretched").glob("*.json"))
        self.assertEqual(summary["cache"]["entry_count"], len(sidecars))
        self.assertEqual(summary["cache"]["invalid_sidecar_count"], 0)

    def test_08_sample_rate_change_invalidates_cache(self) -> None:
        sample_44k = self.root / "sample 44k"
        generate_samples(sample_44k, seed=424242, sample_rate=44100)
        with (sample_44k / "timeline.json").open("r", encoding="utf-8") as handle:
            timeline_44k = json.load(handle)
        output = self.output_dir / "rate 44k.wav"
        report_path = self.output_dir / "rate 44k report.json"
        result = run_cli(
            "align",
            "--timeline",
            str(sample_44k / "timeline.json"),
            "--audio",
            "en={}".format(sample_44k / "audio" / "en.wav"),
            "--output",
            str(output),
            "--report",
            str(report_path),
            "--cache-dir",
            str(self.cache_dir),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cache_hits=0", result.stdout)
        self.assertIn(
            "cache_recomputes={}".format(len(timeline_44k["sentences"])),
            result.stdout,
        )
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["timeline"]["sample_rate"], 44100)
        self.assertEqual(report["totals"]["cache_hits"], 0)


if __name__ == "__main__":
    unittest.main()

