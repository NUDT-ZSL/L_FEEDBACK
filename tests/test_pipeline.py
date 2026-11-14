import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audiomixer.config import parse_scene
from audiomixer.pipeline import run_scene
from audiomixer.report import verify_report
from audiomixer.wavio import read_wav
from tests.fixtures import make_event, make_scene, make_track


class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="audiomixer_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _run(self, raw_scene):
        scene = parse_scene(raw_scene)
        return run_scene(scene, self.tmp)

    def _read_report(self, result):
        return json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))

    def test_end_to_end_and_verification(self):
        raw = make_scene(
            "happy",
            tracks=[
                make_track(
                    "dialogue",
                    [
                        make_event(
                            "e1", 0.5, 1.5, frequency=220, amplitude=0.5,
                            pan=-0.5, sample_rate=44100,
                            fades={"fade_in": 0.1, "fade_out": 0.1, "curve": "linear"},
                        )
                    ],
                )
            ],
        )
        result = self._run(raw)
        self.assertEqual(result["status"], "ok")
        verification = verify_report(result["report_path"])
        self.assertTrue(verification["checks"]["all_passed"])

        samples, sample_rate, _ = read_wav(result["output_path"])
        self.assertEqual(sample_rate, 48000)
        self.assertEqual(samples.shape, (192000, 2))

        report = self._read_report(result)
        event = report["events"][0]
        # 0.5 s * 48000 = 24000; 1.5 s = 72000 frames.
        self.assertEqual(event["project_start_frame"], 24000)
        self.assertEqual(event["project_end_frame_exclusive"], 96000)
        self.assertEqual(event["aligned_frames"], 72000)
        resampling = event["resampling"][0]
        self.assertEqual(resampling["src_rate"], 44100)
        self.assertEqual(resampling["dst_rate"], 48000)
        self.assertEqual(resampling["input_frames"], 66150)
        self.assertEqual(resampling["final_output_frames"], 72000)
        self.assertEqual(
            resampling["expected_output_frames"],
            round(66150 * 48000 / 44100),
        )
        self.assertIn(resampling["engine"], {"ffmpeg_soxr", "ffmpeg_swr", "numpy_windowed_sinc"})

        kinds = {entry["kind"] for entry in event["fades"]}
        self.assertEqual(kinds, {"fade_in", "fade_out"})
        for entry in event["fades"]:
            self.assertLessEqual(
                entry["event_relative_end_frame"], event["aligned_frames"] - 1
            )

    def test_deterministic_outputs_are_bit_identical(self):
        raw = make_scene(
            "determinism",
            tracks=[
                make_track(
                    "a",
                    [make_event("e1", 0.0, 1.0, frequency=300, amplitude=0.4, sample_rate=32000)],
                )
            ],
        )
        first = self._run(raw)
        first_hash = json.loads(Path(first["report_path"]).read_text())["output"]["file_sha256"]
        shutil.rmtree(self.tmp / "assets")
        second = self._run(raw)
        second_hash = json.loads(Path(second["report_path"]).read_text())["output"]["file_sha256"]
        self.assertEqual(first_hash, second_hash)

    def test_pan_maps_mono_to_stereo(self):
        raw = make_scene(
            "pan",
            duration=1.0,
            tracks=[
                make_track(
                    "hard_right",
                    [make_event("e", 0.0, 1.0, frequency=440, amplitude=0.5, pan=1.0)],
                )
            ],
        )
        result = self._run(raw)
        samples, _, _ = read_wav(result["output_path"])
        # Hard-right pan: left channel is silent, right carries the signal.
        self.assertTrue(np.all(np.abs(samples[:, 0]) < 1e-9))
        self.assertGreater(np.max(np.abs(samples[:, 1])), 0.1)
        report = self._read_report(result)
        mapping = report["events"][0]["channel_mapping"]
        self.assertEqual(mapping["input_channels"], 1)
        self.assertEqual(mapping["output_channels"], 2)
        self.assertAlmostEqual(mapping["left_gain"], 0.0, places=9)
        self.assertAlmostEqual(mapping["right_gain"], 1.0, places=9)

    def test_muted_track_is_silent_but_recorded(self):
        raw = make_scene(
            "muted",
            duration=1.0,
            tracks=[
                make_track(
                    "muted_track",
                    [make_event("e", 0.0, 1.0, amplitude=0.6)],
                    muted=True,
                )
            ],
        )
        result = self._run(raw)
        # Silence cannot be loudness-normalised, so this must surface as a
        # business error rather than a misleading success report.
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "E_PEAK_LIMIT")
        self.assertFalse(Path(result["output_path"]).exists())

    def test_track_gain_changes_level(self):
        def render(gain_db):
            raw = make_scene(
                f"gain_{gain_db}",
                duration=1.0,
                tracks=[
                    make_track(
                        "t",
                        [make_event("e", 0.0, 1.0, amplitude=0.4)],
                        gain_db=gain_db,
                    )
                ],
            )
            result = self._run(raw)
            report = self._read_report(result)
            return report["loudness"]["loudness_before_lufs"]

        loud = render(0.0)
        quiet = render(-12.0)
        self.assertAlmostEqual(loud - quiet, 12.0, delta=0.2)

    def test_crossfade_interval_recorded(self):
        raw = make_scene(
            "xfade",
            duration=4.0,
            tracks=[
                make_track(
                    "dialogue",
                    [
                        make_event("a", 0.0, 2.0, amplitude=0.4),
                        make_event(
                            "b", 2.0, 2.0, amplitude=0.4,
                            fades={"crossfade": 0.5, "crossfade_curve": "equal_power"},
                        ),
                    ],
                )
            ],
        )
        result = self._run(raw)
        self.assertEqual(result["status"], "ok")
        report = self._read_report(result)
        event_b = next(event for event in report["events"] if event["event_id"] == "b")
        crossfade = next(entry for entry in event_b["fades"] if entry["kind"] == "crossfade_in")
        self.assertEqual(crossfade["curve"], "equal_power")
        self.assertEqual(crossfade["frame_count"], 24000)
        self.assertEqual(crossfade["project_start_frame"], 96000)
        self.assertEqual(crossfade["project_end_frame"], 119999)

    def test_stereo_asset_supported(self):
        raw = make_scene(
            "stereo_asset",
            duration=1.0,
            tracks=[
                make_track(
                    "music",
                    [make_event("bed", 0.0, 1.0, amplitude=0.3, channels=2, pan=0.0)],
                )
            ],
        )
        result = self._run(raw)
        self.assertEqual(result["status"], "ok")
        report = self._read_report(result)
        self.assertEqual(report["events"][0]["channel_mapping"]["input_channels"], 2)


if __name__ == "__main__":
    unittest.main()
