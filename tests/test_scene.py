from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from immersive_audio.errors import PipelineError
from immersive_audio.scene import load_scene

from tests.fixtures import audio_object, keyframe, write_scene, write_samples, tone


class SceneParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="iapano-scene-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _expect_code(self, path: Path, code: str) -> None:
        with self.assertRaises(PipelineError) as raised:
            load_scene(path)
        self.assertEqual(raised.exception.code, code)

    def test_parses_relative_clips_and_gain_curve(self) -> None:
        write_samples(self.root / "voice.wav", tone(48_000, 0.1))
        obj = {
            "id": "voice",
            "position_keyframes": [
                {
                    "t": 0.0,
                    "az": 0.0,
                    "el": 0.0,
                    "radius": 1.0,
                    "size": 0.2,
                },
                {
                    "t": 1.0,
                    "az": 90.0,
                    "el": 20.0,
                    "radius": 2.0,
                    "size": 0.4,
                },
            ],
            "gain_curve": [{"time": 0.0, "value": 1.0}, {"time": 1.0, "value": 0.5}],
            "clips": [{"file": "voice.wav", "start_time": 0.1, "loop_count": 2}],
        }
        scene_path = write_scene(self.root / "scene.json", [obj])
        scene = load_scene(scene_path)
        self.assertEqual(scene.sample_rate, 48_000)
        self.assertEqual(scene.frame_length, 1024)
        self.assertEqual(scene.object_count, 1)
        self.assertEqual(scene.objects[0].clips[0].path, (self.root / "voice.wav").resolve())
        self.assertEqual(scene.objects[0].clips[0].loops, 2)
        self.assertEqual(len(scene.objects[0].keyframes), 2)
        self.assertAlmostEqual(scene.objects[0].keyframes[1].gain, 0.5)
        self.assertAlmostEqual(scene.objects[0].keyframes[1].size, 0.4)

    def test_minimum_and_maximum_object_counts(self) -> None:
        for count in (1, 64):
            objects = [
                audio_object(
                    f"o{index}",
                    [keyframe(0.0, azimuth=float(index * 5))],
                )
                for index in range(count)
            ]
            path = write_scene(self.root / f"scene-{count}.json", objects, duration=0.01)
            self.assertEqual(load_scene(path).object_count, count)

    def test_invalid_json(self) -> None:
        path = self.root / "bad.json"
        path.write_text("{broken", encoding="utf-8")
        self._expect_code(path, "E1002")

    def test_invalid_sample_rate(self) -> None:
        path = write_scene(
            self.root / "scene.json",
            [audio_object("a", [keyframe(0.0)])],
            sample_rate=0,
        )
        self._expect_code(path, "E1004")

    def test_invalid_frame_length(self) -> None:
        path = write_scene(
            self.root / "scene.json",
            [audio_object("a", [keyframe(0.0)])],
            frame_length=0,
        )
        self._expect_code(path, "E1005")

    def test_invalid_object_counts(self) -> None:
        path = write_scene(self.root / "empty.json", [], duration=0.1)
        self._expect_code(path, "E1006")
        objects = [
            audio_object(f"o{index}", [keyframe(0.0)]) for index in range(65)
        ]
        path = write_scene(self.root / "many.json", objects, duration=0.1)
        self._expect_code(path, "E1006")

    def test_non_monotonic_keyframes(self) -> None:
        path = write_scene(
            self.root / "scene.json",
            [audio_object("a", [keyframe(0.5), keyframe(0.5)])],
        )
        self._expect_code(path, "E1007")

    def test_invalid_duration(self) -> None:
        path = write_scene(
            self.root / "scene.json",
            [audio_object("a", [keyframe(0.0)])],
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        data["duration_seconds"] = -1.0
        path.write_text(json.dumps(data), encoding="utf-8")
        self._expect_code(path, "E1008")

    def test_duplicate_object_id(self) -> None:
        path = write_scene(
            self.root / "scene.json",
            [
                audio_object("same", [keyframe(0.0)]),
                audio_object("same", [keyframe(0.0)]),
            ],
        )
        self._expect_code(path, "E1009")

    def test_missing_scene_and_schema(self) -> None:
        self._expect_code(self.root / "missing.json", "E1001")
        path = self.root / "schema.json"
        path.write_text(json.dumps({"sample_rate": 48_000}), encoding="utf-8")
        self._expect_code(path, "E1003")
