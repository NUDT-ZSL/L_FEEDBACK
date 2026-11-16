from __future__ import annotations

import io
import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from spatial_audio_renderer.errors import RenderRuntimeError
from spatial_audio_renderer.demo import generate_demo
from spatial_audio_renderer.loudness import LoudnessMeter
from spatial_audio_renderer.render import render_scene
from spatial_audio_renderer.scene import load_scene
from tests.helpers import (
    SAMPLE_RATE,
    base_scene,
    clone_scene,
    make_standard_scene,
    write_scene,
)


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "spatial_audio_renderer", *arguments],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class EndToEndTests(unittest.TestCase):
    def test_missing_and_damaged_tracks_are_skipped_and_stderr_mentions_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            scene_path = make_standard_scene(
                directory, include_missing=True, damaged_track=True
            )
            output = directory / "mix.wav"
            result = run_cli(str(scene_path), str(output))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("missing_narration", result.stderr)
            self.assertIn("damaged_radio", result.stderr)
            self.assertIn("duration=2.000 s", result.stderr)
            self.assertIn("peak=", result.stderr)
            self.assertGreaterEqual(result.stderr.count("progress:"), 2)
            info = sf.info(str(output))
            self.assertEqual(info.channels, 2)
            self.assertEqual(info.samplerate, SAMPLE_RATE)
            self.assertEqual(info.subtype, "PCM_16")
            self.assertEqual(info.frames, 96000)
            samples, _ = sf.read(str(output))
            self.assertTrue(np.all(np.isfinite(samples)))
            self.assertGreater(float(np.max(np.abs(samples))), 0.0)
            self.assertEqual(
                list(directory.glob(".spatial-render-*")), []
            )

    def test_nonmonotonic_scene_is_parameter_error_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            scene = clone_scene(base_scene())
            scene["sources"][0]["position_keys"].insert(
                1, {"time": 50, "position": [-1.0, 0.0, 2.0]}
            )
            scene["sources"][0]["position_keys"][2]["time"] = 50
            scene_path = write_scene(directory, scene)
            output = directory / "must_not_exist.wav"
            output.write_bytes(b"preexisting-sentinel")
            result = run_cli(str(scene_path), str(output))
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("strictly increasing", result.stderr)
            self.assertEqual(output.read_bytes(), b"preexisting-sentinel")

    def test_bad_arguments_and_bad_compression_are_parameter_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            scene_path = make_standard_scene(directory)
            output = directory / "never.wav"
            bad_ratio = run_cli(
                str(scene_path),
                str(output),
                "--compress",
                "--ratio",
                "0.5",
            )
            self.assertEqual(bad_ratio.returncode, 1)
            self.assertFalse(output.exists())
            bad_output = run_cli(str(scene_path), str(directory / "mix.mp3"))
            self.assertEqual(bad_output.returncode, 1)

    def test_write_failure_returns_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            scene_path = make_standard_scene(directory)
            output = directory / "missing_parent" / "mix.wav"
            result = run_cli(str(scene_path), str(output))
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("runtime error:", result.stderr)
            self.assertFalse(output.exists())
            self.assertEqual(
                list(directory.glob(".spatial-render-*")), []
            )

    def test_loudness_normalization_lands_near_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            scene_path = make_standard_scene(directory)
            output = directory / "normalized.wav"
            result = run_cli(
                str(scene_path),
                str(output),
                "--normalize-lufs",
                "-24",
                "--compress",
                "--threshold-db",
                "-18",
                "--ratio",
                "3",
                "--attack-ms",
                "8",
                "--release-ms",
                "100",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            samples, sample_rate = sf.read(str(output))
            meter = LoudnessMeter(sample_rate)
            meter.process(samples)
            measured = meter.integrated_loudness()
            self.assertIsNotNone(measured)
            self.assertLess(abs(measured - (-24.0)), 0.6)

    def test_all_tracks_missing_renders_silence_successfully(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            scene = clone_scene(base_scene(duration_samples=24000))
            for index, source in enumerate(scene["sources"]):
                source["file"] = "absent-%d.wav" % index
                source["position_keys"] = [
                    {"time": 0, "position": [1.0, 0.0, 1.0]}
                ]
            scene_path = write_scene(directory, scene)
            output = directory / "silence.wav"
            result = run_cli(str(scene_path), str(output))
            self.assertEqual(result.returncode, 0, result.stderr)
            samples, _ = sf.read(str(output))
            self.assertEqual(samples.shape, (24000, 2))
            self.assertEqual(float(np.max(np.abs(samples))), 0.0)

    def test_direct_render_cleans_temp_when_final_writer_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            scene_path = make_standard_scene(directory, duration_samples=4800)
            scene = load_scene(scene_path)
            output = directory / "failure.wav"
            with mock.patch(
                "spatial_audio_renderer.render._write_final_wav",
                side_effect=RenderRuntimeError("simulated disk failure"),
            ):
                with self.assertRaises(RenderRuntimeError):
                    render_scene(
                        scene, output, stderr=io.StringIO()
                    )
            self.assertFalse(output.exists())
            self.assertEqual(
                list(directory.glob(".spatial-render-*")), []
            )

    def test_insufficient_disk_space_is_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            scene_path = make_standard_scene(directory, duration_samples=4800)
            scene = load_scene(scene_path)
            output = directory / "no_space.wav"
            fake_usage = type(
                "Usage",
                (),
                {"total": 100, "used": 99, "free": 1},
            )()
            with mock.patch(
                "spatial_audio_renderer.render.shutil.disk_usage",
                return_value=fake_usage,
            ):
                with self.assertRaises(RenderRuntimeError):
                    render_scene(
                        scene, output, stderr=io.StringIO()
                    )
            self.assertFalse(output.exists())

    def test_project_json_round_trip_and_output_not_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            scene_path = make_standard_scene(directory)
            source_path = directory / "left.wav"
            result = run_cli(str(scene_path), str(source_path))
            self.assertEqual(result.returncode, 1)
            data = json.loads(scene_path.read_text(encoding="utf-8"))
            self.assertEqual(data["format"], "spatial-scene-v1")

    def test_demo_inputs_are_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_scene = generate_demo(Path(first))
            second_scene = generate_demo(Path(second))
            self.assertEqual(
                first_scene.read_bytes(), second_scene.read_bytes()
            )
            for relative in (
                "assets/dialogue.wav",
                "assets/ambience_loop.wav",
                "assets/flyby.wav",
            ):
                first_hash = hashlib.sha256(
                    (Path(first) / relative).read_bytes()
                ).hexdigest()
                second_hash = hashlib.sha256(
                    (Path(second) / relative).read_bytes()
                ).hexdigest()
                self.assertEqual(first_hash, second_hash)

    def test_render_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            scene_path = make_standard_scene(directory, duration_samples=24000)
            scene = load_scene(scene_path)
            first = directory / "first.wav"
            second = directory / "second.wav"
            render_scene(scene, first, stderr=io.StringIO())
            render_scene(scene, second, stderr=io.StringIO())
            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
