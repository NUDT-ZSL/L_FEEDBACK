import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mediamixer.cli import main
from mediamixer.project import load_project
from mediamixer.pipeline import execute

from tests.helpers import base_project, extract_frame, ffprobe_json, make_track, write_project
from tests.mediagen import make_sine, make_video


class EndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        make_video(self.dir, fps=30, duration=2.0)
        make_sine(self.dir, "a.wav", frequency=440, duration=2.0)
        make_sine(self.dir, "b.wav", frequency=660, duration=3.0)
        self.markers = [
            {"frame": 10, "drift_ms": -5},
            {"frame": 40, "drift_ms": 8},
        ]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, payload=None, name="project.json"):
        payload = payload or base_project(
            "video.mp4",
            [
                make_track("a.wav", offset_ms=0, envelope=[1.0, 0.5]),
                make_track("b.wav", offset_ms=200, envelope=[0.4, 0.4]),
            ],
            markers=self.markers,
        )
        return write_project(self.dir, payload, name)

    def _run_cli(self, project_path):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main([str(project_path)])
        events = [
            json.loads(line)
            for line in buffer.getvalue().splitlines()
            if line.strip()
        ]
        return code, events

    def test_full_run_output_streams_and_duration(self):
        project_path = self._write()
        code, events = self._run_cli(project_path)
        self.assertEqual(code, 0)

        output = self.dir / "out.mp4"
        info = ffprobe_json(output)
        duration = float(info["format"]["duration"])
        self.assertAlmostEqual(duration, 2.0, delta=0.05)

        streams = {(s["codec_type"], s["codec_name"]) for s in info["streams"]}
        self.assertIn(("video", "h264"), streams)
        self.assertIn(("audio", "aac"), streams)
        audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
        self.assertEqual(int(audio["sample_rate"]), 48000)

        state_path = output.with_name("out.mp4.state.json")
        self.assertTrue(state_path.exists())
        intermediate = self.dir / "out.intermediate.mp4"
        self.assertTrue(intermediate.exists())
        self.assertTrue(any(e["event"] == "peak_normalization" for e in events))

    def test_markers_rendered_at_distinct_positions(self):
        self._run_cli(self._write())
        frame_a = np.frombuffer(
            extract_frame(self.dir / "out.mp4", 10), dtype=np.uint8
        ).reshape(240, 320, 3)
        frame_b = np.frombuffer(
            extract_frame(self.dir / "out.mp4", 40), dtype=np.uint8
        ).reshape(240, 320, 3)

        # Marker A centered at 160-10, marker B at 160+16 -> 26 px apart.
        center_a = frame_a[108:132, 130:170]
        center_b = frame_b[108:132, 156:196]
        self.assertGreater(float(center_a.std()), 0)
        self.assertGreater(float(center_b.std()), 0)

        # Unmarked frame has no injected rectangle at the marker rows.
        frame_plain = np.frombuffer(
            extract_frame(self.dir / "out.mp4", 5), dtype=np.uint8
        ).reshape(240, 320, 3)
        self.assertFalse(
            np.array_equal(frame_plain[108:132, 130:170], center_a)
        )

    def test_incremental_reuses_intermediate_video(self):
        code, _ = self._run_cli(self._write())
        self.assertEqual(code, 0)
        intermediate = self.dir / "out.intermediate.mp4"
        digest_before = intermediate.read_bytes()

        payload = json.loads((self.dir / "project.json").read_text())
        payload["audio_tracks"][0]["offset_ms"] = 150
        project_path = self._write(payload)
        code, events = self._run_cli(project_path)
        self.assertEqual(code, 0)

        modes = [e for e in events if e["event"] == "incremental_reuse"]
        self.assertEqual(len(modes), 1)
        self.assertEqual(modes[0]["changed_tracks"], [0])
        self.assertEqual(intermediate.read_bytes(), digest_before)

        info = ffprobe_json(self.dir / "out.mp4")
        self.assertAlmostEqual(
            float(info["format"]["duration"]), 2.0, delta=0.05
        )

    def test_envelope_change_also_incremental(self):
        self._run_cli(self._write())
        payload = json.loads((self.dir / "project.json").read_text())
        payload["audio_tracks"][1]["volume_envelope"] = [0.1, 0.9, 0.2]
        code, events = self._run_cli(self._write(payload))
        self.assertEqual(code, 0)
        self.assertTrue(
            any(e["event"] == "incremental_reuse" for e in events)
        )

    def test_video_change_forces_full_rebuild(self):
        self._run_cli(self._write())
        intermediate = self.dir / "out.intermediate.mp4"
        old_digest = intermediate.read_bytes()

        # Replace the video content deterministically with a different pattern.
        make_video(self.dir, fps=30, duration=2.0, source="color=c=red")

        code, events = self._run_cli(self._write())
        self.assertEqual(code, 0)
        rebuilds = [e for e in events if e["event"] == "full_rebuild"]
        self.assertEqual(len(rebuilds), 1)
        self.assertTrue(intermediate.exists())

    def test_intermediate_digest_tamper_forces_rebuild(self):
        self._run_cli(self._write())
        intermediate = self.dir / "out.intermediate.mp4"
        with open(intermediate, "ab") as handle:
            handle.write(b"tampered")
        code, events = self._run_cli(self._write())
        self.assertEqual(code, 0)
        self.assertTrue(
            any(e["event"] == "full_rebuild" for e in events)
        )

    def test_stale_temp_state_cleaned_without_exit_handler(self):
        self._run_cli(self._write())
        stale = self.dir / "out.mp4.state.json.tmp"
        stale.write_text('{"incomplete": true}', encoding="utf-8")

        code, _ = self._run_cli(self._write())
        self.assertEqual(code, 0)
        self.assertFalse(stale.exists())

    def test_missing_output_directory_is_created(self):
        payload = base_project(
            "video.mp4",
            [make_track("a.wav")],
            output="nested/dir/out.mp4",
        )
        code, _ = self._run_cli(self._write(payload))
        self.assertEqual(code, 0)
        self.assertTrue((self.dir / "nested/dir/out.mp4").exists())

    def test_peak_normalization_is_logged_when_clipping(self):
        # Two full-scale sine tracks sum above 1.0 and must be scaled down.
        make_sine(self.dir, "loud_a.wav", amplitude=1.0, duration=1.0)
        make_sine(self.dir, "loud_b.wav", amplitude=1.0, duration=1.0)
        payload = base_project(
            "video.mp4",
            [
                make_track("loud_a.wav", envelope=[1.0, 1.0]),
                make_track("loud_b.wav", envelope=[1.0, 1.0]),
            ],
            output="loud.mp4",
        )
        code, events = self._run_cli(self._write(payload, "loud.json"))
        self.assertEqual(code, 0)
        normalization = next(
            event for event in events if event["event"] == "peak_normalization"
        )
        self.assertLess(normalization["scale"], 1.0)
        self.assertAlmostEqual(normalization["peak_target"], 0.99)

    def test_ffmpeg_failure_exits_1_with_json_error(self):
        project = load_project(self._write())
        import mediamixer.pipeline as pipeline

        def boom(_path):
            raise RuntimeError("simulated decode failure")

        original = pipeline.ffmpeg_utils.probe_video
        pipeline.ffmpeg_utils.probe_video = boom
        try:
            buffer = io.StringIO()
            import contextlib

            with contextlib.redirect_stderr(buffer):
                code = main([str(self.dir / "project.json")])
        finally:
            pipeline.ffmpeg_utils.probe_video = original
        self.assertEqual(code, 1)
        error = json.loads(buffer.getvalue().strip())
        self.assertEqual(error["code"], "PROCESSING_ERROR")

    def test_invalid_ffmpeg_path_exits_1(self):
        self._write()
        previous = os.environ.get("FFMPEG_PATH")
        os.environ["FFMPEG_PATH"] = str(self.dir / "no_such_bin")
        buffer = io.StringIO()
        try:
            import contextlib

            with contextlib.redirect_stderr(buffer):
                code = main([str(self.dir / "project.json")])
        finally:
            if previous is None:
                os.environ.pop("FFMPEG_PATH", None)
            else:
                os.environ["FFMPEG_PATH"] = previous
        self.assertEqual(code, 1)
        error = json.loads(buffer.getvalue().strip())
        self.assertEqual(error["code"], "FFMPEG_ERROR")


if __name__ == "__main__":
    unittest.main()
