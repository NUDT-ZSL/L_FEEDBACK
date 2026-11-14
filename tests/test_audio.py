from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from immersive_audio.audio import (
    analyze_scene_audio,
    build_object_bus,
    effective_range,
    mix_object_bus,
    read_object_bus,
)
from immersive_audio.errors import PipelineError
from immersive_audio.scene import load_scene

from tests.fixtures import audio_object, keyframe, write_samples, write_scene


class AudioMixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="iapano-audio-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_overlapping_clips_are_added_and_cropped(self) -> None:
        write_samples(self.root / "first.wav", np.full(800, 0.25, dtype=np.float32), 8_000)
        write_samples(self.root / "second.wav", np.full(800, 0.50, dtype=np.float32), 8_000)
        scene_path = write_scene(
            self.root / "scene.json",
            [
                audio_object(
                    "a",
                    [keyframe(0.0)],
                    [
                        {"path": "first.wav", "start": 0.0, "loops": 1},
                        {"path": "second.wav", "start": 0.05, "loops": 1},
                    ],
                )
            ],
            sample_rate=8_000,
            frame_length=800,
            duration=0.25,
        )
        scene = load_scene(scene_path)
        bus = mix_object_bus(scene, scene.objects[0], 2_000)
        np.testing.assert_allclose(bus[:400], 0.25)
        np.testing.assert_allclose(bus[400:800], 0.75)
        np.testing.assert_allclose(bus[800:1200], 0.50)
        np.testing.assert_allclose(bus[1200:], 0.0)
        self.assertEqual(effective_range(bus), (0, 1200))

    def test_loop_count_and_zero_length_clip(self) -> None:
        write_samples(self.root / "empty.wav", np.array([], dtype=np.float32), 8_000)
        write_samples(
            self.root / "pulse.wav",
            np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            8_000,
        )
        scene_path = write_scene(
            self.root / "scene.json",
            [
                audio_object(
                    "a",
                    [keyframe(0.0)],
                    [
                        {"path": "empty.wav", "start": 0.0, "loops": 3},
                        {"path": "pulse.wav", "start": 0.0, "loops": 3},
                    ],
                )
            ],
            sample_rate=8_000,
            frame_length=4,
            duration=0.0015,
        )
        scene = load_scene(scene_path)
        bus = mix_object_bus(scene, scene.objects[0], 12)
        expected = np.tile(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), 3)
        np.testing.assert_allclose(bus, expected)

    def test_stereo_downmix_and_linear_resample(self) -> None:
        stereo = np.column_stack(
            [
                np.ones(800, dtype=np.float32),
                -np.ones(800, dtype=np.float32) * 0.5,
            ]
        )
        write_samples(self.root / "stereo.wav", stereo, 8_000)
        scene_path = write_scene(
            self.root / "scene.json",
            [
                audio_object(
                    "a",
                    [keyframe(0.0)],
                    [{"path": "stereo.wav", "start": 0.0, "loops": 1}],
                )
            ],
            sample_rate=16_000,
            frame_length=1_600,
            duration=0.1,
        )
        scene = load_scene(scene_path)
        analysis = analyze_scene_audio(scene)
        self.assertEqual(analysis.objects[0].clips[0].target_frames, 1_600)
        bus = mix_object_bus(scene, scene.objects[0], analysis.total_frames)
        self.assertEqual(bus.shape, (1_600,))
        np.testing.assert_allclose(bus, 0.25, atol=1e-6)

    def test_valid_frame_range_is_output(self) -> None:
        write_samples(
            self.root / "tone.wav",
            np.linspace(-0.5, 0.5, 800, dtype=np.float32),
            8_000,
        )
        scene_path = write_scene(
            self.root / "scene.json",
            [
                audio_object(
                    "a",
                    [keyframe(0.0)],
                    [{"path": "tone.wav", "start": 0.05, "loops": 1}],
                )
            ],
            sample_rate=8_000,
            frame_length=400,
            duration=0.2,
        )
        scene = load_scene(scene_path)
        analysis = analyze_scene_audio(scene)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bus.f32"
            info = build_object_bus(scene, scene.objects[0], analysis, path)
            self.assertEqual(info.valid_start_frame, 400)
            self.assertEqual(info.valid_end_frame, 1200)
            self.assertEqual(read_object_bus(info).shape, (1_600,))

    def test_missing_audio_file(self) -> None:
        scene_path = write_scene(
            self.root / "scene.json",
            [
                audio_object(
                    "a",
                    [keyframe(0.0)],
                    [{"path": "missing.wav", "start": 0.0, "loops": 1}],
                )
            ],
        )
        with self.assertRaises(PipelineError) as raised:
            analyze_scene_audio(load_scene(scene_path))
        self.assertEqual(raised.exception.code, "E2001")
