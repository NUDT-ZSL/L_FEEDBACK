import json
import tempfile
import unittest
from pathlib import Path

from lipsync_tool import mux
from lipsync_tool.config import Config
from lipsync_tool.pipeline import run_pipeline
from lipsync_tool.samples import generate_sample_tree


@unittest.skipUnless(mux.ffmpeg_available(), "ffmpeg/ffprobe required for end-to-end tests")
class PipelineEndToEndTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.sample_paths = generate_sample_tree(self.root / "samples")
        self.metadata_path = self.sample_paths["metadata"]
        self.video_path = self.sample_paths["video"]
        self.audio_dir = self.root / "samples" / "audio"
        self.output_dir = self.root / "out"

    def tearDown(self):
        self._temp.cleanup()

    def _run(self, config=None, languages=("en", "ja", "zh")):
        return run_pipeline(
            metadata_path=self.metadata_path,
            audio_dir=self.audio_dir,
            output_dir=self.output_dir,
            languages=list(languages),
            config=config or Config(),
            config_notes=["unit test defaults"],
            video_path=self.video_path,
        )

    def test_full_pipeline_outputs_and_report_consistency(self):
        report = self._run()
        self.assertEqual(report["failed_segments"], [])
        self.assertEqual(report["mux_fallback_segments"], [])

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        for entry in report["language_reports"]:
            language = entry["language"]
            lip_path = self.output_dir / f"lipsync_{language}.json"
            lip_payload = json.loads(lip_path.read_text(encoding="utf-8"))
            # Report event count must equal the emitted lip JSON event count.
            self.assertEqual(entry["event_count"], lip_payload["event_count"])
            self.assertEqual(entry["event_count"], len(lip_payload["lip_events"]))
            self.assertEqual(entry["event_count"], len(metadata["lip_events"]))

            mp4_path = self.output_dir / f"cutscene_{language}.mp4"
            wav_path = self.output_dir / f"dub_{language}_stretched.wav"
            self.assertTrue(mp4_path.is_file())
            self.assertTrue(wav_path.is_file())

            streams = mux.probe_streams(mp4_path)
            codecs = {(stream["codec_type"], stream["codec_name"]) for stream in streams}
            self.assertIn(("video", "h264"), codecs)
            self.assertTrue(any(kind == "audio" and codec == "aac" for kind, codec in codecs))
            for event in lip_payload["lip_events"]:
                self.assertLessEqual(abs(event["offset_ms"]), 50.0 + 1e-6)

    def test_stretch_bounds_are_clamped(self):
        report = self._run()
        by_language = {entry["language"]: entry for entry in report["language_reports"]}
        self.assertEqual(by_language["en"]["stretch_factor"], 1.25)
        self.assertEqual(by_language["ja"]["stretch_factor"], 0.75)
        self.assertTrue(
            any("clamped" in warning for warning in by_language["en"]["warnings"])
        )
        self.assertTrue(
            any("clamped" in warning for warning in by_language["ja"]["warnings"])
        )
        self.assertEqual(by_language["zh"]["stretch_factor"], 1.0)

    def test_silence_markers_emitted_for_english(self):
        report = self._run(languages=("en",))
        entry = report["language_reports"][0]
        self.assertTrue(entry["silence_segments"])
        lip_payload = json.loads((self.output_dir / "lipsync_en.json").read_text(encoding="utf-8"))
        self.assertTrue(lip_payload["silence_markers"])
        self.assertTrue(any("silence" in warning for warning in entry["warnings"]))

    def test_mux_failure_falls_back_to_wav_and_json(self):
        report = self._run(config=Config(simulate_mux_failure=True), languages=("zh",))
        entry = report["language_reports"][0]
        self.assertEqual(entry["status"], "mux_fallback")
        self.assertEqual(report["mux_fallback_segments"], ["zh"])
        self.assertFalse((self.output_dir / "cutscene_zh.mp4").exists())
        self.assertTrue((self.output_dir / "dub_zh_stretched.wav").is_file())
        self.assertTrue((self.output_dir / "lipsync_zh.json").is_file())
        self.assertTrue(any("fallback" in error for error in entry["errors"]))

    def test_missing_language_is_recorded_not_raised(self):
        report = self._run(languages=("zh", "fr"))
        statuses = {entry["language"]: entry["status"] for entry in report["language_reports"]}
        self.assertEqual(statuses["fr"], "failed")
        self.assertIn("fr", report["failed_segments"])
        self.assertEqual(statuses["zh"], "ok")

    def test_stretched_wav_keeps_sample_rate_and_bit_depth(self):
        import wave

        self._run(languages=("zh",))
        with wave.open(str(self.output_dir / "dub_zh_stretched.wav")) as stretched:
            self.assertEqual(stretched.getframerate(), 32000)
            self.assertEqual(stretched.getnchannels(), 1)
            self.assertEqual(stretched.getsampwidth(), 2)

    def test_low_coverage_emits_uncovered_gap_markers(self):
        metadata_path = self.root / "sparse_metadata.json"
        sparse_metadata = {
            "frame_rate": 24,
            "duration_seconds": 6.0,
            "target_sample_rate": 32000,
            "lip_events": [
                {"id": "only", "phoneme": "AA", "start_frame": 0, "duration_frames": 12}
            ],
        }
        metadata_path.write_text(json.dumps(sparse_metadata), encoding="utf-8")
        report = run_pipeline(
            metadata_path=metadata_path,
            audio_dir=self.audio_dir,
            output_dir=self.output_dir / "sparse",
            languages=["zh"],
            config=Config(coverage_min_ratio=0.5),
            video_path=self.video_path,
        )
        entry = report["language_reports"][0]
        self.assertTrue(any("coverage" in warning for warning in entry["warnings"]))
        self.assertTrue(entry["uncovered_segments"])
        lip_payload = json.loads(
            (self.output_dir / "sparse" / "lipsync_zh.json").read_text(encoding="utf-8")
        )
        self.assertTrue(lip_payload["uncovered_gap_markers"])


if __name__ == "__main__":
    unittest.main()
