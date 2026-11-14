import json
import shutil
import tempfile
import unittest
from pathlib import Path

from audiomixer.config import parse_scene
from audiomixer.errors import MixerError
from audiomixer.generator import generate_assets
from audiomixer.pipeline import run_scene
from audiomixer.wavio import write_wav_atomic
from tests.fixtures import make_event, make_scene, make_track


class ErrorChainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="audiomixer_err_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _run(self, raw, generate=True):
        scene = parse_scene(raw)
        if generate:
            generate_assets(scene, self.tmp)
        return run_scene(scene, self.tmp, generate=False)

    def _assert_clean_failure(self, result, code):
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], code)
        self.assertFalse(Path(result["output_path"]).exists(), "no partial WAV may remain")
        report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
        self.assertEqual(report["error_code"], code)
        self.assertFalse(report["output_wav_present"])

    def test_missing_asset(self):
        raw = make_scene(
            "missing",
            tracks=[make_track("t", [make_event("ghost", 0.0, 1.0)])],
        )
        scene = parse_scene(raw)
        result = run_scene(scene, self.tmp, generate=False)
        self._assert_clean_failure(result, "E_MISSING_ASSET")

    def test_generate_flag_disabled_keeps_assets_external(self):
        raw = make_scene(
            "external_only",
            tracks=[make_track("t", [make_event("e", 0.0, 1.0)])],
        )
        raw["generate"] = False
        scene = parse_scene(raw)
        self.assertFalse(scene.generate_assets)
        result = run_scene(scene, self.tmp)
        self._assert_clean_failure(result, "E_MISSING_ASSET")
        self.assertFalse((self.tmp / "assets").exists())

    def test_unsupported_format_extension(self):
        raw = make_scene(
            "bad_format",
            tracks=[
                make_track(
                    "t",
                    [
                        make_event(
                            "e", 0.0, 1.0,
                            asset="assets/pretend.mp3",
                        )
                    ],
                )
            ],
        )
        asset_dir = self.tmp / "assets"
        asset_dir.mkdir(parents=True)
        (asset_dir / "pretend.mp3").write_bytes(b"ID3\x03not-a-wav")
        result = self._run(raw, generate=False)
        self._assert_clean_failure(result, "E_UNSUPPORTED_FORMAT")

    def test_unsupported_channel_count(self):
        raw = make_scene(
            "six_channels",
            tracks=[make_track("t", [make_event("e", 0.0, 1.0, channels=6)])],
        )
        with self.assertRaises(MixerError) as context:
            parse_scene(raw)
        self.assertEqual(context.exception.code, "E_UNSUPPORTED_FORMAT")

    def test_event_outside_project_duration(self):
        raw = make_scene(
            "too_long",
            duration=2.0,
            tracks=[make_track("t", [make_event("e", 1.5, 1.0)])],
        )
        with self.assertRaises(MixerError) as context:
            parse_scene(raw)
        self.assertEqual(context.exception.code, "E_OUT_OF_RANGE")
        self.assertEqual(context.exception.details["event_id"], "e")

    def test_fade_longer_than_event(self):
        raw = make_scene(
            "fade_overshoot",
            duration=2.0,
            tracks=[
                make_track(
                    "t",
                    [
                        make_event(
                            "e", 0.0, 0.5,
                            fades={"fade_in": 0.4, "fade_out": 0.4, "curve": "linear"},
                        )
                    ],
                )
            ],
        )
        with self.assertRaises(MixerError) as context:
            parse_scene(raw)
        self.assertEqual(context.exception.code, "E_OUT_OF_RANGE")

    def test_peak_limit_cannot_be_satisfied(self):
        raw = make_scene(
            "peak_impossible",
            duration=2.0,
            target_lufs=-10.0,
            peak_limit=-30.0,
            tolerance=0.2,
            tracks=[make_track("t", [make_event("e", 0.0, 2.0, amplitude=0.9)])],
        )
        result = self._run(raw)
        self._assert_clean_failure(result, "E_PEAK_LIMIT")

    def test_stale_output_removed_on_failure(self):
        raw = make_scene(
            "stale",
            duration=2.0,
            target_lufs=-10.0,
            peak_limit=-30.0,
            tolerance=0.2,
            tracks=[make_track("t", [make_event("e", 0.0, 2.0, amplitude=0.9)])],
        )
        scene = parse_scene(raw)
        output_path = self.tmp / scene.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_wav_atomic(output_path, __import__("numpy").zeros((10, 2)), 48000, 4)
        self.assertTrue(output_path.exists())
        result = run_scene(scene, self.tmp)
        self.assertEqual(result["error_code"], "E_PEAK_LIMIT")
        self.assertFalse(output_path.exists(), "stale WAV must be removed on failure")

    def test_corrupt_wav_payload(self):
        raw = make_scene(
            "corrupt",
            tracks=[
                make_track("t", [make_event("e", 0.0, 1.0, asset="assets/broken.wav")])
            ],
        )
        asset_dir = self.tmp / "assets"
        asset_dir.mkdir(parents=True)
        (asset_dir / "broken.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk")
        result = self._run(raw, generate=False)
        self._assert_clean_failure(result, "E_UNSUPPORTED_FORMAT")


if __name__ == "__main__":
    unittest.main()
