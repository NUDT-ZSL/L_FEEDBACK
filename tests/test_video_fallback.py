import os
import tempfile
import unittest
from unittest import mock

from subalign.cli import main
from subalign.report import read_report
from subalign.subtitles import load_subtitle
from subalign.video import FFmpegUnavailable, extract_reference_events

from tests.samples import linear_drift, reference_events, write_subtitle


class VideoFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = self.temp.name
        self.events = reference_events(count=12)
        self.reference = os.path.join(self.root, "reference.srt")
        write_subtitle(self.reference, self.events)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_missing_ffmpeg_raises_typed_error(self) -> None:
        video = os.path.join(self.root, "movie.mp4")
        with open(video, "wb") as handle:
            handle.write(b"fake")
        with mock.patch("subalign.video.shutil.which", return_value=None):
            with self.assertRaises(FFmpegUnavailable):
                extract_reference_events(video)

    def test_video_reference_falls_back_to_subtitle_with_warning(self) -> None:
        # A .mp4 path that is actually subtitle content exercises the fallback.
        target = os.path.join(self.root, "target.srt")
        write_subtitle(target, linear_drift(self.events, 1.0, 750))
        report_path = os.path.join(self.root, "report.json")

        with mock.patch("subalign.video.shutil.which", return_value=None):
            exit_code = main(
                [
                    "-r",
                    self.reference,
                    "-t",
                    target,
                    "--report",
                    report_path,
                ]
            )
        self.assertEqual(exit_code, 0)
        report = read_report(report_path)
        self.assertEqual(report["reference"]["warnings"], [])

    def test_packet_clustering_builds_reference_cues(self) -> None:
        packets = [10.0, 30.0, 60.0, 900.0, 940.0, 2100.0]
        with mock.patch("subalign.video.shutil.which", return_value="ffprobe"):
            with mock.patch("subalign.video._run_ffprobe", return_value=packets):
                events = extract_reference_events(os.path.join(self.root, "clip.mkv"))
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0].start, 10.0)
        self.assertEqual(events[1].start, 900.0)
        self.assertEqual(events[2].start, 2100.0)

    def test_video_path_without_ffmpeg_and_invalid_subtitle_errors(self) -> None:
        video = os.path.join(self.root, "movie.mp4")
        with open(video, "wb") as handle:
            handle.write(b"\x80\x81 binary garbage")
        target = os.path.join(self.root, "target.srt")
        write_subtitle(target, linear_drift(self.events, 1.0, 200))
        report_path = os.path.join(self.root, "report.json")
        with mock.patch("subalign.video.shutil.which", return_value=None):
            exit_code = main(
                ["-r", video, "-t", target, "--report", report_path]
            )
        self.assertEqual(exit_code, 1)
        report = read_report(report_path)
        self.assertTrue(
            any(w["code"] == "FFMPEG_UNAVAILABLE" for w in report["reference"]["warnings"])
        )
        self.assertEqual(report["errors"][0]["error_type"], "UnsupportedFormat")
        self.assertEqual(report["files"], [])

    def test_ffprobe_packet_failure_is_structured(self) -> None:
        video = os.path.join(self.root, "clip.mp4")
        with open(video, "wb") as handle:
            handle.write(b"fake")
        with mock.patch("subalign.video.shutil.which", return_value="ffprobe"):
            with mock.patch(
                "subalign.video._run_ffprobe",
                side_effect=RuntimeError("boom"),
            ):
                with self.assertRaises(RuntimeError):
                    extract_reference_events(video)


if __name__ == "__main__":
    unittest.main()
