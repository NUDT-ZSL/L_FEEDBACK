import json
import subprocess
import sys
import unittest
from pathlib import Path

import yaml

from offline_audio_render.errors import (
    E_BUS_CONFIG,
    E_CURVE,
    E_EVENT_OVERLAP,
    E_INVALID_LAYOUT,
    E_MISSING_FIELD,
    E_TRAJECTORY,
)
from offline_audio_render.scene import load_scene


ROOT = Path(__file__).resolve().parents[1]


def minimal_scene(**overrides):
    scene = {
        "sample_rate": 48000,
        "layout": "stereo",
        "global_seed": 7,
        "events": [
            {
                "id": "a",
                "time": 0.0,
                "source": "sources/a.wav",
                "position": [0, 0, 0],
            }
        ],
    }
    scene.update(overrides)
    return scene


class SceneValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.id().replace(".", "_"))
        self.base = Path("/tmp") / ("uid49_" + self._testMethodName)
        self.base.mkdir(parents=True, exist_ok=True)

    def write(self, payload):
        path = self.base / "scene.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return path

    def test_missing_top_level_field(self):
        scene = minimal_scene()
        del scene["sample_rate"]
        with self.assertRaises(Exception) as ctx:
            load_scene(self.write(scene))
        self.assertEqual(ctx.exception.code, E_MISSING_FIELD)

    def test_invalid_layout(self):
        with self.assertRaises(Exception) as ctx:
            load_scene(self.write(minimal_scene(layout="atmos")))
        self.assertEqual(ctx.exception.code, E_INVALID_LAYOUT)

    def test_overlap_same_voice(self):
        scene = minimal_scene(
            events=[
                {
                    "id": "a",
                    "time": 1.0,
                    "source": "x.wav",
                    "position": [0, 0, 0],
                    "voice": "v1",
                },
                {
                    "id": "b",
                    "time": 1.0,
                    "source": "y.wav",
                    "position": [0, 0, 0],
                    "voice": "v1",
                },
            ]
        )
        with self.assertRaises(Exception) as ctx:
            load_scene(self.write(scene))
        self.assertEqual(ctx.exception.code, E_EVENT_OVERLAP)

    def test_trajectory_must_be_monotonic(self):
        scene = minimal_scene(
            trajectory=[
                {"time": 1.0, "position": [0, 0, 0], "yaw": 0},
                {"time": 0.5, "position": [1, 0, 0], "yaw": 0},
            ]
        )
        with self.assertRaises(Exception) as ctx:
            load_scene(self.write(scene))
        self.assertEqual(ctx.exception.code, E_TRAJECTORY)

    def test_curve_must_span_normalised_range(self):
        scene = minimal_scene(
            events=[
                {
                    "id": "a",
                    "time": 0.0,
                    "source": "a.wav",
                    "position": [0, 0, 0],
                    "gain_curve": [
                        {"t": 0.1, "value": -1.0},
                        {"t": 1.0, "value": 0.0},
                    ],
                }
            ]
        )
        with self.assertRaises(Exception) as ctx:
            load_scene(self.write(scene))
        self.assertEqual(ctx.exception.code, E_CURVE)

    def test_bus_cycle_rejected(self):
        scene = minimal_scene(
            buses=[
                {"name": "a", "route_to": "b"},
                {"name": "b", "route_to": "a"},
            ],
            events=[
                {
                    "id": "a",
                    "time": 0.0,
                    "source": "a.wav",
                    "position": [0, 0, 0],
                    "bus": "a",
                }
            ],
        )
        with self.assertRaises(Exception) as ctx:
            load_scene(self.write(scene))
        self.assertEqual(ctx.exception.code, E_BUS_CONFIG)

    def test_cli_emits_json_stderr_and_nonzero_exit(self):
        path = self.write(minimal_scene(layout="bad"))
        proc = subprocess.run(
            [sys.executable, "-m", "offline_audio_render", str(path), "-o", str(self.base / "out")],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        payload = json.loads(proc.stderr)
        self.assertTrue(payload["error"])
        self.assertEqual(payload["code"], E_INVALID_LAYOUT)


if __name__ == "__main__":
    unittest.main()
