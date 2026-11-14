from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from immersive_audio.container import ContainerReader
from immersive_audio.errors import PipelineError
from immersive_audio.renderer import (
    distance_attenuation,
    render_block,
    vbap_gains,
)

from tests.fixtures import (
    audio_object,
    ffmpeg_path,
    ffprobe_path,
    keyframe,
    run_cli,
    tone,
    write_samples,
    write_scene,
)


class RendererTests(unittest.TestCase):
    def test_vbap_ring_positions(self) -> None:
        np.testing.assert_allclose(vbap_gains(0.0), np.eye(8)[0], atol=1e-12)
        np.testing.assert_allclose(vbap_gains(45.0), np.eye(8)[1], atol=1e-12)
        center = vbap_gains(22.5)
        self.assertAlmostEqual(float(center[0]), 1 / np.sqrt(2))
        self.assertAlmostEqual(float(center[1]), 1 / np.sqrt(2))
        self.assertAlmostEqual(float(np.linalg.norm(center)), 1.0)
        np.testing.assert_allclose(vbap_gains(360.0), np.eye(8)[0], atol=1e-12)

    def test_distance_and_elevation(self) -> None:
        self.assertAlmostEqual(distance_attenuation(1.0), 1.0)
        self.assertAlmostEqual(distance_attenuation(2.0), 0.25)
        self.assertAlmostEqual(distance_attenuation(0.0, 0.1), 100.0)
        self.assertTrue(np.all(vbap_gains(90.0, 90.0) == 0.0))

    def test_render_block_is_linear_in_object_audio(self) -> None:
        audio = np.array([[1.0, 2.0], [0.5, -0.5]], dtype=np.float32)
        metadata = np.array(
            [
                [0.0, 0.0, 1.0, 1.0, 0.0],
                [180.0, 0.0, 2.0, 0.5, 0.0],
            ],
            dtype=np.float32,
        )
        rendered = render_block(audio, metadata)
        self.assertEqual(rendered.shape, (2, 8))
        np.testing.assert_allclose(rendered[0, 0], 1.0)
        np.testing.assert_allclose(rendered[0, 4], 2.0 * 0.25 * 0.5)

    def test_extreme_coordinates_remain_finite(self) -> None:
        audio = np.ones((4, 3), dtype=np.float32)
        metadata = np.array(
            [
                [1e9, 1e9, 0.0, 1e6, 0.0],
                [-1e9, -1e9, 1e9, -2.0, 0.0],
                [720.25, -720.25, 0.0, 0.0, 1e9],
            ],
            dtype=np.float32,
        )
        rendered = render_block(audio, metadata)
        self.assertTrue(np.all(np.isfinite(rendered)))

    def test_render_matches_container_metadata_and_wav(self) -> None:
        with tempfile.TemporaryDirectory(prefix="iapano-render-") as temporary:
            root = Path(temporary)
            write_samples(root / "a.wav", tone(8_000, 0.2, 440.0), 8_000)
            write_samples(root / "b.wav", tone(8_000, 0.2, 880.0), 8_000)
            scene_path = write_scene(
                root / "scene.json",
                [
                    audio_object(
                        "a",
                        [keyframe(0.0, 0.0), keyframe(0.2, 90.0)],
                        [{"path": "a.wav", "start": 0.0, "loops": 1}],
                    ),
                    audio_object(
                        "b",
                        [keyframe(0.0, 180.0), keyframe(0.2, 270.0)],
                        [{"path": "b.wav", "start": 0.05, "loops": 1}],
                    ),
                ],
                sample_rate=8_000,
                frame_length=400,
                duration=0.2,
            )
            container = root / "scene.iap"
            wav = root / "scene.wav"
            packed = run_cli("pack", str(scene_path), str(container))
            self.assertEqual(packed.returncode, 0, msg=packed.stderr)
            rendered = run_cli("render", str(container), str(wav))
            self.assertEqual(rendered.returncode, 0, msg=rendered.stderr)

            reader = ContainerReader(container)
            expected_blocks = [
                render_block(audio, metadata)
                for _, metadata, audio in reader.iter_blocks()
            ]
            expected = np.concatenate(expected_blocks, axis=0)
            actual, sample_rate = sf.read(wav, dtype="float32", always_2d=True)
            self.assertEqual(sample_rate, 8_000)
            self.assertEqual(actual.shape, expected.shape)
            np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)

    @unittest.skipUnless(ffprobe_path(), "ffprobe is not available in PATH")
    def test_ffprobe_accepts_eight_channel_wav(self) -> None:
        with tempfile.TemporaryDirectory(prefix="iapano-ffprobe-") as temporary:
            root = Path(temporary)
            write_samples(root / "a.wav", tone(48_000, 0.1), 48_000)
            scene_path = write_scene(
                root / "scene.json",
                [
                    audio_object(
                        "a",
                        [keyframe(0.0, 0.0), keyframe(0.1, 90.0)],
                        [{"path": "a.wav", "start": 0.0, "loops": 1}],
                    )
                ],
                sample_rate=48_000,
                frame_length=512,
                duration=0.1,
            )
            container = root / "scene.iap"
            wav = root / "scene.wav"
            self.assertEqual(
                run_cli("pack", str(scene_path), str(container)).returncode, 0
            )
            self.assertEqual(
                run_cli("render", str(container), str(wav)).returncode, 0
            )
            command = [
                ffprobe_path() or "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,sample_rate,channels,duration",
                "-of",
                "json",
                str(wav),
            ]
            probe = subprocess.run(command, capture_output=True, text=True, check=True)
            stream = json.loads(probe.stdout)["streams"][0]
            self.assertEqual(stream["codec_name"], "pcm_f32le")
            self.assertEqual(stream["sample_rate"], "48000")
            self.assertEqual(stream["channels"], 8)
            self.assertAlmostEqual(float(stream["duration"]), 0.1, places=6)
            if ffmpeg_path():
                decode = subprocess.run(
                    [
                        ffmpeg_path() or "ffmpeg",
                        "-v",
                        "error",
                        "-i",
                        str(wav),
                        "-f",
                        "null",
                        "-",
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(decode.returncode, 0, msg=decode.stderr)

    def test_invalid_metadata_is_rejected(self) -> None:
        audio = np.zeros((4, 1), dtype=np.float32)
        metadata = np.array([[np.nan, 0.0, 1.0, 1.0, 0.0]], dtype=np.float32)
        with self.assertRaises(PipelineError) as raised:
            render_block(audio, metadata)
        self.assertEqual(raised.exception.code, "E4002")
