import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import audio_timeline_tool as tool  # noqa: E402
import generate_samples  # noqa: E402


def _run_cli(timeline: Path, audio_dir: Path, output_dir: Path):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "audio_timeline_tool.py"),
            "--timeline",
            str(timeline),
            "--audio-dir",
            str(audio_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _read_pcm(path: Path):
    with wave.open(str(path), "rb") as reader:
        params = reader.getparams()
        data = np.frombuffer(
            reader.readframes(reader.getnframes()), dtype="<i2"
        ).astype(np.float64)
    return params, data


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.sample_dir = cls.root / "samples"
        cls.output_dir = cls.root / "output"
        generate_samples.generate(cls.sample_dir)
        cls.result = _run_cli(
            cls.sample_dir / "timeline.json",
            cls.sample_dir / "audio",
            cls.output_dir,
        )
        cls.timeline = json.loads(
            (cls.output_dir / "processed_timeline.json").read_text(
                encoding="utf-8"
            )
        )
        cls.report = json.loads(
            (cls.output_dir / "report.json").read_text(encoding="utf-8")
        )
        cls.audio_by_id = {item["id"]: item for item in cls.timeline["audio"]}
        cls.report_by_id = {item["id"]: item for item in cls.report["audio"]}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_cli_succeeds_and_writes_expected_primary_files(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr)
        self.assertTrue((self.output_dir / "processed_timeline.json").is_file())
        self.assertTrue((self.output_dir / "report.json").is_file())
        self.assertTrue((self.output_dir / "impact_processed.wav").is_file())
        self.assertTrue((self.output_dir / "impact_loop_0.wav").is_file())

    def test_timeline_status_fields_and_original_fields_are_preserved(self):
        impact = self.audio_by_id["impact"]
        self.assertEqual(impact["events"][0]["name"], "first_hit")
        self.assertEqual(impact["events"][0]["status"], "OK")
        self.assertEqual(impact["events"][2]["status"], "EVENT_OUT_OF_RANGE")
        self.assertEqual(impact["events"][2]["aligned_time"], 1.5)
        self.assertEqual(impact["loops"][0]["loop_status"], "OK")
        self.assertEqual(impact["loops"][0]["loop_file"], "impact_loop_0.wav")
        self.assertEqual(impact["loops"][1]["loop_status"], "LOOP_TOO_SHORT")
        self.assertIsNone(impact["loops"][1]["loop_file"])
        self.assertEqual(
            self.timeline["description"],
            "Deterministic animation-sound timeline fixture",
        )

    def test_unsupported_and_missing_inputs_are_reported_and_skipped(self):
        stereo = self.report_by_id["stereo_skip"]
        self.assertEqual(stereo["processing_status"], "SKIPPED")
        self.assertEqual(stereo["error"]["code"], "UNSUPPORTED_FORMAT")
        self.assertAlmostEqual(stereo["original_duration"], 0.4)

        floating = self.report_by_id["float_skip"]
        self.assertEqual(floating["error"]["code"], "UNSUPPORTED_FORMAT")
        self.assertAlmostEqual(floating["original_duration"], 0.3)

        missing = self.report_by_id["missing"]
        self.assertEqual(missing["error"]["code"], "AUDIO_READ_ERROR")
        self.assertEqual(
            self.audio_by_id["missing"]["events"][0]["status"], "SKIPPED"
        )
        self.assertEqual(
            self.audio_by_id["missing"]["loops"][0]["loop_status"], "SKIPPED"
        )
        self.assertFalse((self.output_dir / "stereo_skip_processed.wav").exists())
        self.assertFalse((self.output_dir / "missing_processed.wav").exists())

    def test_time_scale_and_invalid_scale_behavior(self):
        stretch = self.report_by_id["stretch"]
        self.assertAlmostEqual(stretch["original_duration"], 0.6, places=6)
        self.assertAlmostEqual(stretch["processed_duration"], 1.2, places=6)
        self.assertEqual(stretch["time_scale"], 2.0)

        invalid = self.report_by_id["invalid_scale"]
        self.assertEqual(invalid["time_scale"], 1.0)
        self.assertIn(
            "INVALID_TIME_SCALE",
            [item["code"] for item in invalid["diagnostics"]],
        )
        self.assertAlmostEqual(invalid["processed_duration"], 0.5, places=6)

    def test_event_alignment_uses_exact_peak_sample(self):
        impact = self.report_by_id["impact"]
        event = impact["events"][0]
        self.assertEqual(event["status"], "OK")
        self.assertGreater(event["aligned_sample"], 0)

        params, pcm = _read_pcm(self.output_dir / "impact_processed.wav")
        sample_rate = params.framerate
        aligned = event["aligned_sample"]
        time = self.audio_by_id["impact"]["events"][0]["time"]
        left = max(0, int(math.floor((time - 0.010) * sample_rate)))
        right = min(
            pcm.size - 1,
            int(math.ceil((time + 0.010) * sample_rate)),
        )
        expected_peak = float(np.max(np.abs(pcm[left : right + 1])))
        self.assertEqual(abs(pcm[aligned]), expected_peak)
        self.assertAlmostEqual(
            event["aligned_time"], aligned / float(sample_rate), places=12
        )
        self.assertTrue(-10.0 <= event["offset_ms"] <= 10.0)
        self.assertGreater(event["offset_ms"], 0.0)

    def test_outputs_are_48khz_mono_16bit_and_peak_limited(self):
        for path in self.output_dir.glob("*.wav"):
            params, pcm = _read_pcm(path)
            self.assertEqual(params.framerate, 48000, path.name)
            self.assertEqual(params.nchannels, 1, path.name)
            self.assertEqual(params.sampwidth, 2, path.name)
            self.assertLessEqual(
                int(np.max(np.abs(pcm))), tool.MAX_PEAK_PCM, path.name
            )
            peak_dbfs = 20.0 * math.log10(
                max(1, int(np.max(np.abs(pcm)))) / 32768.0
            )
            self.assertLessEqual(peak_dbfs, -1.0, path.name)

    def test_loudness_target_is_applied_when_peak_allows(self):
        _, pcm = _read_pcm(self.output_dir / "stretch_processed.wav")
        rms = float(np.sqrt(np.mean(np.square(pcm / 32768.0))))
        self.assertAlmostEqual(rms, 0.1, delta=0.001)
        self.assertAlmostEqual(
            self.report_by_id["stretch"]["loudness"]["rms_after"],
            rms,
            delta=0.0001,
        )

    def test_loop_edges_are_zero_and_duration_is_preserved(self):
        params, pcm = _read_pcm(self.output_dir / "impact_loop_0.wav")
        self.assertEqual(params.framerate, 48000)
        self.assertEqual(pcm.size, 12000)
        self.assertEqual(int(pcm[0]), 0)
        self.assertEqual(int(pcm[-1]), 0)
        self.assertGreater(int(np.max(np.abs(pcm))), 0)
        self.assertFalse((self.output_dir / "impact_loop_1.wav").exists())


class ErrorAndCompatibilityTests(unittest.TestCase):
    def test_loudness_limiter_warning_branch(self):
        samples = np.full(48000, 2.0, dtype=np.float64)
        adjusted, info, diagnostics = tool._normalize_loudness(samples)
        self.assertIn(
            "LOUDNESS_LIMITED", [item["code"] for item in diagnostics]
        )
        self.assertAlmostEqual(info["gain_db"], -20.0, places=6)
        self.assertLessEqual(
            float(np.max(np.abs(adjusted))), tool.MAX_PEAK_LINEAR
        )

    def test_generator_is_deterministic(self):
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first = Path(first_tmp)
            second = Path(second_tmp)
            generate_samples.generate(first)
            generate_samples.generate(second)
            first_files = sorted(
                path.relative_to(first) for path in first.rglob("*") if path.is_file()
            )
            second_files = sorted(
                path.relative_to(second)
                for path in second.rglob("*")
                if path.is_file()
            )
            self.assertEqual(first_files, second_files)
            for relative in first_files:
                first_hash = hashlib.sha256(
                    (first / relative).read_bytes()
                ).hexdigest()
                second_hash = hashlib.sha256(
                    (second / relative).read_bytes()
                ).hexdigest()
                self.assertEqual(first_hash, second_hash, str(relative))

    def test_malformed_timeline_returns_exit_1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timeline = root / "timeline.json"
            timeline.write_text("{not valid json", encoding="utf-8")
            result = _run_cli(timeline, root, root / "output")
            self.assertEqual(result.returncode, 1)
            self.assertIn("error:", result.stderr.lower())

    def test_non_directory_output_path_returns_exit_1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_samples.generate(root / "samples")
            output_file = root / "not_a_directory"
            output_file.write_text("occupied", encoding="utf-8")
            result = _run_cli(
                root / "samples" / "timeline.json",
                root / "samples" / "audio",
                output_file,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("error:", result.stderr.lower())

    def test_numeric_events_are_normalized_to_objects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = root / "samples"
            generate_samples.generate(samples)
            timeline_data = {
                "audio": [
                    {
                        "id": "numeric",
                        "file": "impact_48k.wav",
                        "events": [0.200],
                        "loops": [[0.1, 0.35]],
                    }
                ]
            }
            timeline_path = root / "numeric.json"
            timeline_path.write_text(
                json.dumps(timeline_data), encoding="utf-8"
            )
            result = _run_cli(
                timeline_path, samples / "audio", root / "output"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            processed = json.loads(
                (root / "output" / "processed_timeline.json").read_text(
                    encoding="utf-8"
                )
            )
            event = processed["audio"][0]["events"][0]
            loop = processed["audio"][0]["loops"][0]
            self.assertEqual(event["time"], 0.2)
            self.assertEqual(event["status"], "OK")
            self.assertIn("aligned_time", event)
            self.assertEqual(loop["loop_status"], "OK")
            self.assertEqual(loop["loop_file"], "numeric_loop_0.wav")


if __name__ == "__main__":
    unittest.main()
