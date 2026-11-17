"""End-to-end tests for the avswitch ABR muxer.

Run from the repo root:  python -m unittest discover -s tests -v
Samples are (re)generated deterministically if missing.
"""
import json
import subprocess
import unittest
from pathlib import Path

from avswitch import cli, gop, probe

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
OUT = ROOT / "out" / "tests"
FPS = 24
TOTAL_FRAMES = 30 * FPS


def ensure_samples():
    expected = [
        SAMPLES / "video_low_800k.mp4",
        SAMPLES / "video_mid_2500k.mp4",
        SAMPLES / "video_high_5000k.mp4",
        SAMPLES / "audio_main.aac",
        SAMPLES / "video_mid_corrupt.mp4",
    ]
    if not all(p.exists() for p in expected):
        subprocess.run(["python3", str(ROOT / "create_samples.py")], check=True,
                       cwd=str(ROOT))


class MuxerTestBase(unittest.TestCase):
    manifest = None
    output_name = None

    @classmethod
    def setUpClass(cls):
        ensure_samples()
        OUT.mkdir(parents=True, exist_ok=True)
        cls.output = OUT / cls.output_name
        cls.report = cli.run(ROOT / cls.manifest, cls.output)
        cls.verification = cls.report["verification"]


class TestNormalRun(MuxerTestBase):
    manifest = "manifest.json"
    output_name = "normal.ts"

    def test_output_parseable_by_ffmpeg(self):
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(self.output), "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stderr.strip(), "")

    def test_two_streams_h264_aac(self):
        codecs = sorted(s["codec_name"] for s in self.verification["streams"])
        self.assertEqual(codecs, ["aac", "h264"])

    def test_distinct_pids_and_pes_lengths(self):
        ts = self.verification["ts_structure"]
        self.assertNotEqual(ts["video_pid"], ts["audio_pid"])
        self.assertEqual(ts["video_pes_count"], TOTAL_FRAMES)
        self.assertEqual(ts["pes_length_violations"], [])

    def test_sync_within_20ms(self):
        sync = self.verification["sync"]
        self.assertTrue(sync["within_tolerance"])
        self.assertLessEqual(sync["max_abs_ms"], 20.0)

    def test_monotonic_dts_and_continuity(self):
        self.assertTrue(self.verification["video_dts_monotonic"])
        self.assertTrue(self.verification["audio_pts_monotonic"])
        continuity = self.verification["continuity"]
        self.assertTrue(continuity["continuous"])
        self.assertEqual(continuity["frames"], TOTAL_FRAMES)

    def test_switches_only_at_keyframes(self):
        manifest = json.loads((ROOT / self.manifest).read_text())
        keyframes = {}
        for stream in manifest["video_streams"]:
            path = ROOT / stream["path"]
            streams = probe.probe_streams(path)
            v_idx = probe.stream_index_by_type(streams, "video")
            keyframes[stream["id"]] = gop.keyframe_times(
                probe.probe_packets(path, v_idx))
        for segment in self.report["segments"]:
            self.assertTrue(
                gop.is_keyframe_time(keyframes[segment["stream"]], segment["src_in"]),
                f"segment start {segment['src_in']} is not a keyframe of "
                f"{segment['stream']}")

    def test_switches_occurred_and_logged(self):
        self.assertGreaterEqual(len(self.report["switches"]), 3)
        for switch in self.report["switches"]:
            self.assertIn(switch["reason"],
                          ("bandwidth_drop", "bandwidth_headroom", "buffer_underrun"))


class TestFallbackRun(MuxerTestBase):
    manifest = "manifest_fallback.json"
    output_name = "fallback.ts"

    def test_fallback_logged_with_reason(self):
        fallbacks = self.report["fallbacks"]
        self.assertEqual(len(fallbacks), 1)
        entry = fallbacks[0]
        self.assertEqual(entry["target_stream"], "mid")
        self.assertEqual(entry["fallback_stream"], "low")
        self.assertEqual(entry["time"], 12.0)
        self.assertTrue(entry["reason"])

    def test_output_still_valid(self):
        self.assertTrue(self.verification["video_dts_monotonic"])
        self.assertTrue(self.verification["sync"]["within_tolerance"])
        self.assertTrue(self.verification["continuity"]["continuous"])
        self.assertEqual(self.verification["continuity"]["frames"], TOTAL_FRAMES)

    def test_corrupt_stream_not_used(self):
        for segment in self.report["segments"]:
            self.assertNotEqual(segment["stream"], "mid")


if __name__ == "__main__":
    unittest.main()
