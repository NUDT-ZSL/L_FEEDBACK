from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from cinemix.errors import SceneValidationError
from cinemix.scene import load_scene


def _silence(path: Path, seconds: float = 2.0, rate: int = 48000) -> None:
    sf.write(
        str(path),
        np.zeros((int(seconds * rate), 1), dtype=np.float64),
        rate,
        subtype="PCM_16",
    )


def _base_scene() -> dict:
    return {
        "scene_id": "unit",
        "duration": 2.0,
        "ambience": {
            "file": "ambience.wav",
            "loop_start": 0.1,
            "loop_end": 1.5,
            "gain": 0.5,
        },
        "samples": {"hit": "hit.wav"},
        "events": [{"time": 0.25, "sample": "hit", "gain": 0.5}],
        "music_layers": [
            {"file": "music.wav", "start": 0.0, "loops": 1, "gain": 0.2}
        ],
        "dialogue_intervals": [
            {"start": 0.4, "end": 0.9, "duck_depth_db": 6.0}
        ],
        "camera_cuts": [1.0],
    }


class SceneValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "scene"
        self.root.mkdir()
        _silence(self.root / "ambience.wav")
        _silence(self.root / "music.wav")
        _silence(self.root / "hit.wav", seconds=0.2)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write(self, data: dict) -> Path:
        path = self.root / "scene.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_valid_scene(self) -> None:
        scene = load_scene(self._write(_base_scene()))
        self.assertEqual(scene.scene_id, "unit")
        self.assertEqual(scene.events[0].name, "hit")
        self.assertAlmostEqual(scene.dialogue_intervals[0].duck_gain, 10 ** (-6 / 20))

    def test_missing_required_field(self) -> None:
        data = _base_scene()
        del data["dialogue_intervals"]
        with self.assertRaisesRegex(SceneValidationError, "missing required field"):
            load_scene(self._write(data))

    def test_invalid_time_value(self) -> None:
        data = _base_scene()
        data["dialogue_intervals"][0]["end"] = 0.2
        with self.assertRaisesRegex(SceneValidationError, "greater than start"):
            load_scene(self._write(data))

    def test_non_finite_time_value(self) -> None:
        data = _base_scene()
        data["camera_cuts"] = [float("nan")]
        with self.assertRaisesRegex(SceneValidationError, "finite"):
            load_scene(self._write(data))

    def test_unknown_sample_name(self) -> None:
        data = _base_scene()
        data["events"][0]["sample"] = "missing"
        with self.assertRaisesRegex(SceneValidationError, "unknown sample"):
            load_scene(self._write(data))

    def test_sample_mapping_array_and_parent_relative_path(self) -> None:
        shared = self.root.parent / "shared_hit.wav"
        _silence(shared, seconds=0.2)
        data = _base_scene()
        data["samples"] = [{"name": "hit", "file": "../shared_hit.wav"}]
        scene = load_scene(self._write(data))
        self.assertEqual(scene.events[0].file, shared.resolve())

    def test_effects_alias_for_event_list(self) -> None:
        data = _base_scene()
        data["effects"] = data.pop("events")
        scene = load_scene(self._write(data))
        self.assertEqual(len(scene.events), 1)


if __name__ == "__main__":
    unittest.main()
