from __future__ import annotations

import contextlib
import io
import json
import math
import shutil
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dialogue_baker.audio import read_pcm16_wav, read_riff_metadata, true_peak_dbtp
from dialogue_baker.cli import main as cli_main
from dialogue_baker.errors import BakerError
from dialogue_baker.pipeline import run_bake
from dialogue_baker.timebase import Timebase
from tools.generate_demo import generate_demo


class DemoFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory(prefix="dialogue-baker-tests-")
        cls.temp_root = Path(cls._temp.name)
        cls.base_project = generate_demo(cls.temp_root / "base_project")
        cls.base_dir = cls.base_project.parent

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def copy_project(self, name: str) -> Path:
        destination = self.temp_root / name / "project"
        shutil.copytree(self.base_dir, destination)
        return destination / "project.json"

    def read_json(self, path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    def write_json(self, path: Path, value) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )


class TimebaseTests(unittest.TestCase):
    def test_rational_frame_boundaries_roundtrip_exactly(self) -> None:
        for fps in (
            Fraction(24, 1),
            Fraction(25, 1),
            Fraction(60, 1),
            Fraction(24000, 1001),
            Fraction(30000, 1001),
            Fraction(23, 1),
        ):
            with self.subTest(fps=fps):
                timebase = Timebase(fps, 48000)
                self.assertTrue(timebase.validate_roundtrip(1000))
                previous = -1
                for frame in range(1001):
                    sample = timebase.frame_to_sample(frame)
                    self.assertEqual(timebase.sample_to_frame(sample), frame)
                    self.assertGreater(sample, previous)
                    previous = sample

    def test_fractional_frame_active_boundaries_use_sample_intervals(self) -> None:
        timebase = Timebase(Fraction(30000, 1001), 48000)
        start_sample = timebase.frame_to_sample(10)
        end_sample = timebase.frame_to_sample(35)
        active = []
        for frame in range(60):
            frame_start, frame_end = timebase.frame_sample_span(frame)
            if start_sample < frame_end and end_sample > frame_start:
                active.append(frame)
        self.assertEqual(active, list(range(10, 35)))

    def test_true_peak_of_faded_half_scale_signal(self) -> None:
        samples = np.full((19200, 2), 0.5, dtype=np.float64)
        ramp = np.linspace(0.0, 1.0, 240, endpoint=False)
        samples[:240] *= ramp[:, None]
        samples[-240:] *= ramp[::-1, None]
        measured = true_peak_dbtp(samples, 4)
        self.assertIsNotNone(measured)
        self.assertLessEqual(measured, -6.005)
        self.assertGreater(measured, -6.10)


class BakeIntegrationTests(DemoFixture):
    def test_end_to_end_alignment_limiting_and_determinism(self) -> None:
        project = self.copy_project("determinism")
        out_a = self.temp_root / "determinism" / "out_a"
        out_b = self.temp_root / "determinism" / "out_b"
        result_a = run_bake(project, out_a)
        result_b = run_bake(project, out_b)
        self.assertEqual(result_a.status, "complete")
        self.assertEqual(result_b.status, "complete")

        for filename in ("dialogue.wav", "bake_report.json", "subtitles_frames.json"):
            self.assertEqual(
                (out_a / filename).read_bytes(),
                (out_b / filename).read_bytes(),
                f"{filename} is not byte-identical",
            )

        report = self.read_json(out_a / "bake_report.json")
        subtitles = self.read_json(out_a / "subtitles_frames.json")
        self.assertTrue(report["timebase"]["boundary_roundtrip_ok"])
        self.assertEqual(report["project"]["duration_samples"], 240000)
        self.assertGreaterEqual(report["mix"]["overlap_pair_count"], 4)
        self.assertGreater(
            report["loudness"]["post_gain_pre_limiter_clipped_sample_count"], 0
        )
        processing = report["loudness"]["processing"]
        self.assertTrue(processing["true_peak_limited"])
        self.assertGreater(processing["limiter"]["limited_sample_count"], 0)
        self.assertTrue(report["subtitles"]["alignment"]["ok"])
        self.assertEqual(report["subtitles"]["frame_count"], 120)
        self.assertEqual(subtitles["frame_count"], 120)
        self.assertEqual(
            report["loudness"]["measurement_parameters"]["block_samples"], 19200
        )
        self.assertEqual(
            report["loudness"]["measurement_parameters"]["hop_samples"], 4800
        )
        self.assertEqual(
            report["loudness"]["measurement_parameters"]["overlap"], 0.75
        )

        encoded_tp = report["output"]["encoded_audio"]["true_peak_dbtp"]
        self.assertLessEqual(
            encoded_tp, report["project"]["true_peak_dbtp"] + 1e-6
        )
        self.assertTrue(report["output"]["format_validation"]["ok"])
        self.assertEqual(report["output"]["encoded_audio"]["sample_rate"], 48000)
        self.assertEqual(report["output"]["encoded_audio"]["channels"], 2)
        self.assertEqual(
            report["output"]["encoded_audio"]["sample_width_bytes"], 2
        )

        wav_metadata = read_riff_metadata(out_a / "dialogue.wav")
        self.assertEqual(wav_metadata["fps"], "24")
        self.assertEqual(wav_metadata["sample_rate"], 48000)
        self.assertEqual(wav_metadata["target_lufs"], -10.0)
        encoded_samples, wav_info = read_pcm16_wav(out_a / "dialogue.wav")
        self.assertEqual(wav_info["frame_count"], 240000)
        self.assertEqual(encoded_samples.shape, (240000, 2))

        lines = {line["id"]: line for line in report["dialogues"]}
        self.assertTrue(lines["d02_alice"]["source"]["resampled"])
        self.assertEqual(
            lines["d02_alice"]["source"]["resample_length_error_final_samples"], 0
        )
        self.assertGreater(
            lines["d02_alice"]["spatial"]["frame_segment_count"], 1
        )
        self.assertNotEqual(
            lines["d02_alice"]["spatial"]["pan_min"],
            lines["d02_alice"]["spatial"]["pan_max"],
        )
        self.assertFalse(
            math.isclose(
                lines["d03_bob"]["spatial"]["doppler_factor_min"],
                lines["d03_bob"]["spatial"]["doppler_factor_max"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        for line in report["dialogues"]:
            self.assertEqual(
                line["mix"]["post_clipped_sample_count"], 0, line["id"]
            )
            active_frames = [
                frame["frame"]
                for frame in subtitles["frames"]
                if line["id"] in frame["active_ids"]
            ]
            self.assertEqual(
                active_frames,
                list(range(line["start_frame"], line["end_frame"])),
                line["id"],
            )

        timebase = Timebase(
            Fraction(report["project"]["fps"]), report["project"]["sample_rate"]
        )
        for frame in subtitles["frames"]:
            expected = [
                line["id"]
                for line in report["dialogues"]
                if line["start_sample"] < frame["end_sample"]
                and line["end_sample"] > frame["start_sample"]
            ]
            self.assertEqual(frame["active_ids"], expected, frame["frame"])
            self.assertEqual(
                frame["start_sample"], timebase.frame_to_sample(frame["frame"])
            )
            self.assertEqual(
                frame["end_sample"], timebase.frame_to_sample(frame["frame"] + 1)
            )

    def test_resume_from_completed_stage_and_snapshot_validation(self) -> None:
        project = self.copy_project("resume")
        out = self.temp_root / "resume" / "out"
        first = run_bake(project, out, stop_after="spatialize")
        self.assertEqual(first.status, "stopped")
        self.assertIsNone(first.report_path)
        self.assertFalse((out / "bake_report.json").exists())

        second = run_bake(project, out)
        self.assertEqual(second.status, "complete")
        self.assertEqual(second.stage_actions["parse"], "reused")
        self.assertEqual(second.stage_actions["spatialize"], "reused")
        self.assertEqual(second.stage_actions["mix"], "executed")

        spatial_manifest = out / ".bake_state" / "spatialize" / "manifest.json"
        manifest = self.read_json(spatial_manifest)
        manifest["version"] = 999
        self.write_json(spatial_manifest, manifest)
        third = run_bake(project, out, stop_after="spatialize")
        self.assertEqual(third.stage_actions["parse"], "reused")
        self.assertEqual(third.stage_actions["spatialize"], "executed")

        spatial_array = out / ".bake_state" / "spatialize" / "line_000_stereo.npy"
        data = bytearray(spatial_array.read_bytes())
        data[-1] ^= 0x01
        spatial_array.write_bytes(data)
        fourth = run_bake(project, out, stop_after="spatialize")
        self.assertEqual(fourth.stage_actions["parse"], "reused")
        self.assertEqual(fourth.stage_actions["spatialize"], "executed")

        script_path = project.parent / "script.json"
        script = self.read_json(script_path)
        script["dialogues"][0]["text"] = "Changed text invalidates the parse digest."
        self.write_json(script_path, script)
        fifth = run_bake(project, out, stop_after="parse")
        self.assertEqual(fifth.stage_actions["parse"], "executed")

    def test_business_error_codes(self) -> None:
        cases = []

        def project_for(name: str) -> Path:
            return self.copy_project(name)

        missing_audio_project = project_for("missing_audio")
        script_path = missing_audio_project.parent / "script.json"
        script = self.read_json(script_path)
        script["dialogues"][0]["audio"] = "voices/not_there.wav"
        self.write_json(script_path, script)
        cases.append((missing_audio_project, "E_MISSING_INPUT"))

        fps_project = project_for("fps_mismatch")
        scene_path = fps_project.parent / "scene.json"
        scene = self.read_json(scene_path)
        scene["fps"] = "25"
        self.write_json(scene_path, scene)
        cases.append((fps_project, "E_FPS_MISMATCH"))

        track_project = project_for("missing_track")
        scene_path = track_project.parent / "scene.json"
        scene = self.read_json(scene_path)
        del scene["tracks"]["bob"]
        self.write_json(scene_path, scene)
        cases.append((track_project, "E_MISSING_TRACK"))

        range_project = project_for("frame_range")
        script_path = range_project.parent / "script.json"
        script = self.read_json(script_path)
        script["dialogues"][-1]["end_frame"] = 121
        self.write_json(script_path, script)
        cases.append((range_project, "E_FRAME_OUT_OF_RANGE"))

        overlap_project = project_for("same_speaker_overlap")
        script_path = overlap_project.parent / "script.json"
        script = self.read_json(script_path)
        script["dialogues"][3]["start_frame"] = 79
        self.write_json(script_path, script)
        cases.append((overlap_project, "E_SAME_SPEAKER_OVERLAP"))

        for index, (project, expected_code) in enumerate(cases):
            with self.subTest(code=expected_code):
                with self.assertRaises(BakerError) as caught:
                    run_bake(project, self.temp_root / "errors" / f"out_{index}")
                self.assertEqual(caught.exception.code, expected_code)

    def test_cli_writes_stable_error_report(self) -> None:
        project = self.copy_project("cli_error")
        script_path = project.parent / "script.json"
        script = self.read_json(script_path)
        script["dialogues"][0]["audio"] = "voices/missing_cli_audio.wav"
        self.write_json(script_path, script)
        out = self.temp_root / "cli_error" / "out"
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = cli_main(
                ["bake", "--project", str(project), "--out", str(out)]
            )
        self.assertEqual(code, 2)
        payload = json.loads(stderr.getvalue().splitlines()[0])
        self.assertEqual(payload["error"]["code"], "E_MISSING_INPUT")
        error_report = self.read_json(out / "bake_error.json")
        self.assertEqual(error_report["error"]["code"], "E_MISSING_INPUT")


if __name__ == "__main__":
    unittest.main()

