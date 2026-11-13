"""Integration tests using the real local FFmpeg 6.x installation."""

from __future__ import annotations

import io
import json
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from mxf_r128 import cli
from mxf_r128.constants import (
    E_LOUDNESS_FAIL,
    E_MEDIA_CORRUPT,
    ERROR_CODES,
    GOP_SIZE,
)
from mxf_r128.probe import keyframe_times, probe_media
from mxf_r128.rates import parse_rate
from mxf_r128.samples import generate_samples
from tests.helpers import find_tools


TOOLS = find_tools()


def _channel_frequencies(
    ffmpeg: str, path: Path, channels: int, duration: float
) -> list[float]:
    raw_path = path.parent / f".{path.stem}.f32"
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-c:a",
        "pcm_f32le",
        "-f",
        "f32le",
        str(raw_path),
    ]
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError("audio decode failed")
    data = raw_path.read_bytes()
    raw_path.unlink(missing_ok=True)
    frames = len(data) // (4 * channels)
    start_frame = int(frames * 0.2)
    end_frame = int(frames * 0.8)
    frequencies: list[float] = []
    for channel in range(channels):
        crossings = 0
        previous = 0.0
        for frame in range(start_frame, end_frame):
            offset = (frame * channels + channel) * 4
            (value,) = struct.unpack_from("<f", data, offset)
            if frame == start_frame:
                previous = value
                continue
            if previous <= 0.0 < value or previous >= 0.0 > value:
                crossings += 1
            previous = value
        elapsed = (end_frame - start_frame) / (frames / duration)
        frequencies.append(crossings / (2.0 * elapsed))
    return frequencies


@unittest.skipUnless(TOOLS, "local FFmpeg and ffprobe are required")
class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert TOOLS is not None
        cls.ffmpeg, cls.ffprobe = TOOLS
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.samples_dir = cls.root / "samples"
        cls.output_dir = cls.root / "output"
        cls.report_path = cls.root / "qc.json"
        generate_samples(
            cls.samples_dir,
            ffmpeg=cls.ffmpeg,
            ffprobe=cls.ffprobe,
            duration=2.4,
            include_subtitle=True,
        )
        stderr = io.StringIO()
        with mock.patch.dict(
            os.environ,
            {"FFMPEG_BIN": cls.ffmpeg, "FFPROBE_BIN": cls.ffprobe},
        ), redirect_stderr(stderr):
            cls.exit_code = cli.main(
                [
                    "--input-dir",
                    str(cls.samples_dir),
                    "--output-dir",
                    str(cls.output_dir),
                    "--report",
                    str(cls.report_path),
                    "--max-workers",
                    "3",
                ]
            )
        cls.stderr = stderr.getvalue()
        cls.report = json.loads(cls.report_path.read_text(encoding="utf-8"))
        cls.records = {
            record["source_file"]: record for record in cls.report["files"]
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_batch_exit_code_and_business_stderr(self) -> None:
        self.assertEqual(self.exit_code, 1)
        self.assertNotIn("ffmpeg version", self.stderr.lower())
        self.assertIn(E_LOUDNESS_FAIL, self.stderr)
        self.assertIn(E_MEDIA_CORRUPT, self.stderr)

    def test_report_is_sorted_with_fixed_symbols(self) -> None:
        names = [record["source_file"] for record in self.report["files"]]
        self.assertEqual(names, sorted(names))
        self.assertEqual(self.report["max_workers"], 3)
        for record in self.report["files"]:
            self.assertIn(record["error_code"], ERROR_CODES | {None})
            for warning in record["warnings"]:
                self.assertTrue(warning.startswith("W_"))

    def test_valid_mpeg2_and_avcintra_contract(self) -> None:
        for name, fps in (
            ("01_valid_mpeg2_25_stereo.mxf", "25/1"),
            ("02_valid_avcintra_2997_stereo.mxf", "30000/1001"),
        ):
            record = self.records[name]
            self.assertEqual(record["status"], "success")
            self.assertIsNone(record["error_code"])
            self.assertEqual(record["video_codec"], "h264")
            self.assertEqual(record["video_profile"], "High")
            self.assertEqual(record["video_profile_level"], "High 4.1")
            self.assertEqual(record["pix_fmt"], "yuv420p")
            self.assertEqual(record["frame_rate"], fps)
            self.assertEqual(record["gop_size"], GOP_SIZE)
            self.assertAlmostEqual(record["loudness_lufs"], -23.0, delta=0.5)
            self.assertLessEqual(record["true_peak_dbtp"], -1.0)
            self.assertLessEqual(record["output_start_delta_ms"], 10.0)

    def test_ffprobe_contract_and_gop(self) -> None:
        record = self.records["01_valid_mpeg2_25_stereo.mxf"]
        info = probe_media(self.ffprobe, Path(record["output_path"]))
        self.assertEqual([s["codec_type"] for s in info.streams], ["video", "audio"])
        video = info.video or {}
        self.assertEqual(video["codec_name"], "h264")
        self.assertEqual(video["profile"], "High")
        self.assertEqual(int(video["level"]), 41)
        self.assertEqual(video["pix_fmt"], "yuv420p")
        self.assertEqual(parse_rate(video["avg_frame_rate"]).numerator, 25)
        times = keyframe_times(self.ffprobe, Path(record["output_path"]))
        intervals = [
            (current - previous) * 25.0
            for previous, current in zip(times, times[1:])
        ]
        self.assertTrue(intervals)
        self.assertLessEqual(max(intervals), GOP_SIZE + 1.0)

    def test_51_layout_and_channel_order(self) -> None:
        record = self.records["03_valid_5_1.mxf"]
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["audio_channels"], 6)
        self.assertEqual(record["channel_layout"], "5.1")
        frequencies = _channel_frequencies(
            self.ffmpeg, Path(record["output_path"]), 6, 2.4
        )
        expected = [220, 330, 440, 110, 660, 770]
        for measured, wanted in zip(frequencies, expected):
            self.assertLessEqual(abs(measured - wanted), 15.0)

    def test_delay_compensation_and_warning(self) -> None:
        record = self.records["04_timestamp_offset_400ms.mxf"]
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["warnings"], [])
        self.assertAlmostEqual(record["source_audio_delay_sec"], 0.378, delta=0.02)
        self.assertLessEqual(record["output_start_delta_ms"], 10.0)

        late = self.records["05_timestamp_offset_1200ms.mxf"]
        self.assertEqual(late["status"], "success")
        self.assertEqual(late["warnings"], ["W_AUDIO_DELAY"])
        self.assertEqual(late["source_audio_delay_sec"], 1.178)
        self.assertLessEqual(late["output_start_delta_ms"], 10.0)

    def test_loudness_failure_retains_mp4(self) -> None:
        record = self.records["06_loudness_nonconvergent.mxf"]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error_code"], E_LOUDNESS_FAIL)
        self.assertTrue(Path(record["output_path"]).is_file())
        self.assertLessEqual(record["true_peak_dbtp"], -1.0)
        self.assertGreater(
            abs(record["loudness_lufs"] - (-23.0)), 0.5
        )

    def test_corrupt_files_do_not_stop_batch(self) -> None:
        for name in ("09_corrupt_zero.mxf", "10_corrupt_binary.mxf"):
            record = self.records[name]
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["error_code"], E_MEDIA_CORRUPT)
            self.assertIsNone(record["output_path"])
        following = self.records["08_missing_fps_index.mxf"]
        self.assertNotEqual(following["error_code"], E_MEDIA_CORRUPT)

    def test_missing_fps_warning_is_not_a_direct_failure(self) -> None:
        record = self.records["08_missing_fps_index.mxf"]
        self.assertEqual(record["warnings"], ["W_FPS_INFERRED"])
        self.assertEqual(record["frame_rate"], "25/1")
        self.assertNotIn(
            record["error_code"], {E_MEDIA_CORRUPT, "E_TIMEOUT"}
        )

    def test_subtitle_stream_is_mapped_to_mov_text(self) -> None:
        record = self.records["07_subtitle_preserved.mxf"]
        self.assertEqual(record["status"], "success")
        info = probe_media(self.ffprobe, Path(record["output_path"]))
        subtitles = [s for s in info.streams if s["codec_type"] == "subtitle"]
        self.assertEqual(len(subtitles), 1)
        self.assertEqual(subtitles[0]["codec_name"], "mov_text")

    def test_no_temporary_or_partial_files(self) -> None:
        leftovers = [
            path.name
            for path in self.output_dir.iterdir()
            if path.suffix in {".tmp", ".partial"} or ".partial." in path.name
        ]
        self.assertEqual([], leftovers)
        report_temps = list(self.report_path.parent.glob("*.tmp"))
        self.assertEqual([], report_temps)

    def test_valid_only_batch_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            input_dir.mkdir()
            for name in (
                "01_valid_mpeg2_25_stereo.mxf",
                "02_valid_avcintra_2997_stereo.mxf",
            ):
                shutil.copy2(self.samples_dir / name, input_dir / name)
            stderr = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {"FFMPEG_BIN": self.ffmpeg, "FFPROBE_BIN": self.ffprobe},
            ), redirect_stderr(stderr):
                code = cli.main(
                    [
                        "--input-dir",
                        str(input_dir),
                        "--output-dir",
                        str(root / "output"),
                        "--report",
                        str(root / "qc.json"),
                        "--max-workers",
                        "2",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(stderr.getvalue(), "")
            report = json.loads((root / "qc.json").read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["success"], 2)
            self.assertEqual(report["summary"]["failed"], 0)

    def test_warning_only_batch_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            input_dir.mkdir()
            shutil.copy2(
                self.samples_dir / "05_timestamp_offset_1200ms.mxf",
                input_dir / "05_timestamp_offset_1200ms.mxf",
            )
            stderr = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {"FFMPEG_BIN": self.ffmpeg, "FFPROBE_BIN": self.ffprobe},
            ), redirect_stderr(stderr):
                code = cli.main(
                    [
                        "--input-dir",
                        str(input_dir),
                        "--output-dir",
                        str(root / "output"),
                        "--report",
                        str(root / "qc.json"),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(stderr.getvalue(), "")
            report = json.loads((root / "qc.json").read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["success"], 1)
            self.assertEqual(report["files"][0]["warnings"], ["W_AUDIO_DELAY"])


if __name__ == "__main__":
    unittest.main()
