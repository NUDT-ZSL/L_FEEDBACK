from __future__ import annotations

import json
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from generate_demo import FS, generate_demo
from podcast_post.pipeline import process_manifest
from podcast_post.wavio import write_wav

from .helpers import manifest_document, write_manifest


ROOT = Path(__file__).resolve().parents[1]


def _riff_fields(payload: bytes) -> tuple[tuple[int, ...], bytes]:
    if payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise AssertionError("not RIFF/WAVE")
    offset = 12
    fmt = None
    data = None
    while offset + 8 <= len(payload):
        chunk_id = payload[offset : offset + 4]
        size = struct.unpack_from("<I", payload, offset + 4)[0]
        body = payload[offset + 8 : offset + 8 + size]
        if chunk_id == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", body, 0)
        elif chunk_id == b"data":
            data = body
        offset += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise AssertionError("missing fmt or data")
    return fmt, data


def _run_cli(manifest: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "postproduce.py"),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.manifest = generate_demo(cls.root / "demo")
        cls.first = cls.root / "out1"
        cls.second = cls.root / "out2"
        cls.result1 = process_manifest(cls.manifest, cls.first)
        cls.result2 = process_manifest(cls.manifest, cls.second)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_output_is_byte_deterministic(self) -> None:
        for name in ("episode.wav", "chapters.json", "report.json"):
            self.assertEqual(
                (self.first / name).read_bytes(),
                (self.second / name).read_bytes(),
                name,
            )

    def test_wav_header_and_length(self) -> None:
        fmt, data = _riff_fields((self.first / "episode.wav").read_bytes())
        self.assertEqual(fmt, (1, 2, FS, 192000, 4, 16))
        self.assertEqual(len(data), 864000 * 2 * 2)

    def test_report_and_chapters(self) -> None:
        report = json.loads((self.first / "report.json").read_text(encoding="utf-8"))
        chapters = json.loads((self.first / "chapters.json").read_text(encoding="utf-8"))
        self.assertEqual(report["ducking_trigger_count"], 13)
        self.assertEqual(report["chapter_alignment_adjustment_count"], 3)
        self.assertEqual(report["crossfades"]["count"], 10)
        self.assertEqual(report["crossfades"]["truncated_count"], 1)
        self.assertAlmostEqual(report["final_integrated_lufs"], -16.0, delta=0.03)
        self.assertLessEqual(report["final_true_peak_dbtp"], -1.0)
        self.assertEqual(set(report["track_peaks"]), {"host", "guest", "music"})

        spans = chapters["chapters"]
        self.assertEqual(spans[0]["start_sample"], 0)
        self.assertEqual(spans[-1]["end_sample"], 864000)
        for left, right in zip(spans, spans[1:]):
            self.assertEqual(left["end_sample"], right["start_sample"])
            self.assertGreater(left["duration_samples"], 0)
        self.assertEqual(spans[1]["start_sample"], 144000)
        self.assertEqual(spans[3]["start_sample"], 432000)
        self.assertEqual(spans[4]["start_sample"], 576000)

    def test_ffmpeg_independent_loudness_and_peak(self) -> None:
        process = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-i",
                str(self.first / "episode.wav"),
                "-af",
                "ebur128=peak=true",
                "-f",
                "null",
                "-",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if process.returncode != 0:
            self.skipTest("ffmpeg not available")
        loudness_match = re.search(
            r"Integrated loudness:\s+I:\s+(-?\d+(?:\.\d+)?) LUFS",
            process.stderr,
        )
        peak_match = re.search(
            r"True peak:\s+Peak:\s+(-?\d+(?:\.\d+)?) dBFS",
            process.stderr,
        )
        self.assertIsNotNone(loudness_match)
        self.assertIsNotNone(peak_match)
        report = self.result1.report
        self.assertAlmostEqual(
            report["final_integrated_lufs"], float(loudness_match.group(1)), delta=0.1
        )
        self.assertAlmostEqual(
            report["final_true_peak_dbtp"], float(peak_match.group(1)), delta=0.2
        )

    def test_missing_asset_business_error_and_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            document = manifest_document(
                1000,
                [
                    {
                        "id": "h",
                        "track": "host",
                        "asset": "missing.wav",
                        "source_in": 0,
                        "source_out": 1000,
                        "timeline_in": 0,
                        "timeline_out": 1000,
                        "gain_db": 0.0,
                    }
                ],
            )
            for track in document["tracks"]:
                track["asset"] = "missing.wav"
            manifest = write_manifest(root / "manifest.json", document)
            output = root / "out"
            process = _run_cli(manifest, output)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("E_ASSET_MISSING", process.stderr)
            self.assertFalse(output.exists())

    def test_unsupported_format_business_error_and_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            samples = np.zeros((4000, 1), dtype=np.float64)
            write_wav(root / "asset.wav", samples, FS, 16)
            document = manifest_document(
                1000,
                [
                    {
                        "id": "h",
                        "track": "host",
                        "asset": "asset.wav",
                        "source_in": 0,
                        "source_out": 1000,
                        "timeline_in": 0,
                        "timeline_out": 1000,
                        "gain_db": 0.0,
                    }
                ],
            )
            manifest = write_manifest(root / "manifest.json", document)
            output = root / "out"
            process = _run_cli(manifest, output)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("E_FORMAT_UNSUPPORTED", process.stderr)
            self.assertFalse(output.exists())

    def test_clip_range_business_error(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_wav(root / "asset.wav", np.zeros((1000, 1)), FS, 24)
            document = manifest_document(
                2000,
                [
                    {
                        "id": "h",
                        "track": "host",
                        "asset": "asset.wav",
                        "source_in": 0,
                        "source_out": 1500,
                        "timeline_in": 0,
                        "timeline_out": 1500,
                        "gain_db": 0.0,
                    }
                ],
            )
            manifest = write_manifest(root / "manifest.json", document)
            output = root / "out"
            process = _run_cli(manifest, output)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("E_CLIP_RANGE", process.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

