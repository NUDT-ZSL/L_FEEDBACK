import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mediamixer.cli import main
from mediamixer.errors import ValidationError
from mediamixer.project import load_project

from tests.helpers import base_project, make_track, write_project
from tests.mediagen import make_sine, make_video


class ValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.video = make_video(self.dir)
        self.audio = make_sine(self.dir, "a.wav")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _project(self, **overrides):
        payload = base_project(
            "video.mp4", [make_track("a.wav")], markers=[]
        )
        payload.update(overrides)
        return payload

    def test_valid_project_loads(self):
        path = write_project(self.dir, self._project())
        project = load_project(path)
        self.assertEqual(project.tracks[0].offset_ms, 0)

    def test_missing_video_file(self):
        payload = self._project(video_path="nope.mp4")
        with self.assertRaises(ValidationError):
            load_project(write_project(self.dir, payload))

    def test_unsupported_video_container(self):
        payload = self._project(video_path="a.wav")
        with self.assertRaises(ValidationError):
            load_project(write_project(self.dir, payload))

    def test_track_count_bounds(self):
        payload = self._project(audio_tracks=[])
        with self.assertRaises(ValidationError):
            load_project(write_project(self.dir, payload))

        payload = self._project(
            audio_tracks=[make_track("a.wav") for _ in range(9)]
        )
        with self.assertRaises(ValidationError):
            load_project(write_project(self.dir, payload))

    def test_offset_bounds(self):
        payload = self._project(
            audio_tracks=[make_track("a.wav", offset_ms=-5001)]
        )
        with self.assertRaises(ValidationError):
            load_project(write_project(self.dir, payload))

    def test_envelope_bounds_and_length(self):
        payload = self._project(
            audio_tracks=[make_track("a.wav", envelope=[1.0])]
        )
        with self.assertRaises(ValidationError):
            load_project(write_project(self.dir, payload))

        payload = self._project(
            audio_tracks=[make_track("a.wav", envelope=[0.5, 1.5])]
        )
        with self.assertRaises(ValidationError):
            load_project(write_project(self.dir, payload))

    def test_marker_drift_bounds(self):
        payload = self._project(
            frame_markers=[{"frame": 0, "drift_ms": 11}]
        )
        with self.assertRaises(ValidationError):
            load_project(write_project(self.dir, payload))

    def test_too_many_markers(self):
        payload = self._project(
            frame_markers=[
                {"frame": i, "drift_ms": 0} for i in range(4)
            ]
        )
        with self.assertRaises(ValidationError):
            load_project(write_project(self.dir, payload))

    def test_invalid_json_exits_2_with_json_error(self):
        path = self.dir / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        import io
        from contextlib import redirect_stderr

        buffer = io.StringIO()
        with redirect_stderr(buffer):
            code = main([str(path)])
        self.assertEqual(code, 2)
        error = json.loads(buffer.getvalue().strip())
        self.assertIn("code", error)
        self.assertIn("message", error)


if __name__ == "__main__":
    unittest.main()
