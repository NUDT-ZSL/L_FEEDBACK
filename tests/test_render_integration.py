import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from offline_audio_render.renderer import Renderer
from offline_audio_render.errors import E_EVENT_OVERLAP
from offline_audio_render.scene import load_scene

ROOT = Path(__file__).resolve().parents[1]


def make_tone(path, sample_rate=48000, frames=4800, frequency=440.0, channels=1, seed=1):
    t = np.arange(frames, dtype=np.float64) / sample_rate
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    if channels == 1:
        data = wave[:, None]
    elif channels == 2:
        data = np.stack(
            [wave, 0.5 * np.sin(2 * np.pi * frequency * t + np.pi / 3)], axis=1
        )
    else:
        shifts = np.linspace(0, np.pi, channels)
        data = 0.5 * np.sin(2 * np.pi * frequency * t[:, None] + shifts[None, :])
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    data = data + rng.normal(scale=0.005, size=data.shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data.astype(np.float32), sample_rate, subtype="FLOAT")


class RenderIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="uid49_render_"))
        make_tone(self.dir / "a.wav", seed=1)
        make_tone(self.dir / "b.wav", frequency=660.0, seed=2)
        make_tone(self.dir / "stereo.wav", channels=2, seed=3)
        make_tone(self.dir / "surround.wav", channels=6, seed=4)
        make_tone(self.dir / "rate24.wav", sample_rate=24000, frames=2400, seed=5)
        (self.dir / "broken.wav").write_bytes(b"not-a-wav" * 20)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def scene_path(self, payload, name="scene.yaml"):
        path = self.dir / name
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return path

    def base_scene(self, **overrides):
        scene = {
            "sample_rate": 48000,
            "layout": "stereo",
            "global_seed": 4242,
            "block_size": 4096,
            "duration": 2.0,
            "trajectory": [
                {"time": 0.0, "position": [0.0, 0.0, 0.0], "yaw": 0.0},
                {"time": 2.0, "position": [2.0, 0.0, 0.0], "yaw": 0.0},
            ],
            "events": [
                {
                    "id": "e1",
                    "time": 0.1,
                    "source": "a.wav",
                    "position": [0.0, 2.0, 0.0],
                    "seed_offset": 1,
                    "gain_curve": [
                        {"t": 0.0, "value": -6.0},
                        {"t": 1.0, "value": 0.0},
                    ],
                    "lowpass_curve": [
                        {"t": 0.0, "value": 400.0},
                        {"t": 1.0, "value": 8000.0},
                    ],
                },
                {
                    "id": "e2",
                    "time": 0.6,
                    "source": "b.wav",
                    "position": [0.0, -2.0, 0.0],
                    "seed_offset": 2,
                },
            ],
        }
        scene.update(overrides)
        return scene

    def render(self, payload, out_name, block_size=None):
        path = self.scene_path(payload)
        scene = load_scene(path)
        if block_size is not None:
            scene.block_size = block_size
        return Renderer(scene, self.dir / out_name).render()

    def test_deterministic_byte_identical_repeat(self):
        first = self.render(self.base_scene(), "out1")
        second = self.render(self.base_scene(), "out2")
        one = Path(first.output_files["master"]).read_bytes()
        two = Path(second.output_files["master"]).read_bytes()
        self.assertEqual(one, two)
        first_report = dict(first.report)
        second_report = dict(second.report)
        first_report.pop("outputs")
        second_report.pop("outputs")
        self.assertEqual(first_report, second_report)

    def test_output_independent_of_block_size(self):
        large = self.render(self.base_scene(), "large", block_size=4096)
        small = self.render(self.base_scene(), "small", block_size=257)
        self.assertEqual(
            Path(large.output_files["master"]).read_bytes(),
            Path(small.output_files["master"]).read_bytes(),
        )

    def test_wav_bytes_are_stable_across_processes(self):
        path = self.scene_path(self.base_scene())
        outputs = []
        for index in range(2):
            out_dir = self.dir / f"process_{index}"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "offline_audio_render",
                    str(path),
                    "-o",
                    str(out_dir),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            outputs.append((out_dir / "master.wav").read_bytes())
        self.assertEqual(outputs[0], outputs[1])

    def test_layout_channels_and_float32(self):
        result = self.render(self.base_scene(layout="5.1"), "surround")
        info = sf.info(result.output_files["master"])
        self.assertEqual(info.channels, 6)
        self.assertEqual(info.subtype, "FLOAT")
        data, sr = sf.read(result.output_files["master"], dtype="float32")
        self.assertEqual(sr, 48000)
        self.assertTrue(np.all(np.abs(data) <= 1.0))

    def test_error_recovery_keeps_silence_and_report(self):
        scene = self.base_scene(
            events=[
                {
                    "id": "good",
                    "time": 0.1,
                    "source": "a.wav",
                    "position": [1.0, 0.0, 0.0],
                },
                {
                    "id": "broken",
                    "time": 0.3,
                    "source": "broken.wav",
                    "position": [1.0, 0.0, 0.0],
                },
                {
                    "id": "wrong_rate",
                    "time": 0.4,
                    "source": "rate24.wav",
                    "position": [1.0, 0.0, 0.0],
                },
            ]
        )
        result = self.render(scene, "recovery")
        data, _ = sf.read(result.output_files["master"], dtype="float32")
        self.assertEqual(data.shape[1], 2)
        self.assertTrue(np.isfinite(data).all())
        skipped = set(result.report["skipped_events"])
        self.assertEqual(skipped, {"broken", "wrong_rate"})
        codes = {warning["code"] for warning in result.report["warnings"]}
        self.assertIn("W2001", codes)
        self.assertIn("W2002", codes)
        # Skipped event range remains silent in the output.
        broken = next(e for e in result.report["events"] if e["id"] == "broken")
        segment = data[broken["start_sample"] : broken["end_sample"]]
        np.testing.assert_array_equal(segment, np.zeros_like(segment))

    def test_downmix_warning_and_upmix(self):
        scene = self.base_scene(
            layout="mono",
            events=[
                {
                    "id": "stereo_to_mono",
                    "time": 0.1,
                    "source": "stereo.wav",
                    "position": [0.0, 0.0, 0.0],
                }
            ],
        )
        result = self.render(scene, "mono")
        info = sf.info(result.output_files["master"])
        self.assertEqual(info.channels, 1)
        codes = {warning["code"] for warning in result.report["warnings"]}
        self.assertIn("W2004", codes)

    def test_bus_chain_and_routed_output(self):
        scene = self.base_scene(
            buses=[
                {
                    "name": "fx",
                    "gain_db": -2.0,
                    "limiter": True,
                    "delay_seconds": 0.05,
                    "delay_feedback": 0.3,
                    "route_to": "master",
                },
                {"name": "master", "limiter": True},
            ],
            events=[
                {
                    "id": "routed",
                    "time": 0.1,
                    "source": "a.wav",
                    "position": [1.0, 0.0, 0.0],
                    "bus": "fx",
                }
            ],
        )
        result = self.render(scene, "bus")
        self.assertEqual(set(result.output_files), {"master"})
        data, _ = sf.read(result.output_files["master"], dtype="float32")
        self.assertTrue(np.isfinite(data).all())
        self.assertLessEqual(np.max(np.abs(data)), 1.0)

    def test_multiple_top_level_buses_produce_multiple_tracks(self):
        scene = self.base_scene(
            buses=[
                {"name": "bus_a", "limiter": True},
                {"name": "bus_b", "limiter": True},
            ],
            events=[
                {
                    "id": "a",
                    "time": 0.1,
                    "source": "a.wav",
                    "position": [1.0, 0.0, 0.0],
                    "bus": "bus_a",
                },
                {
                    "id": "b",
                    "time": 0.2,
                    "source": "b.wav",
                    "position": [-1.0, 0.0, 0.0],
                    "bus": "bus_b",
                },
            ],
        )
        result = self.render(scene, "multi_bus")
        self.assertEqual(
            set(result.output_files), {"bus_a", "bus_b"}
        )
        for path in result.output_files.values():
            info = sf.info(path)
            self.assertEqual(info.channels, 2)
            self.assertEqual(info.subtype, "FLOAT")

    def test_report_contains_event_sample_range_gain_and_seed(self):
        result = self.render(self.base_scene(), "report")
        event = result.report["events"][0]
        self.assertIn("start_sample", event)
        self.assertIn("end_sample", event)
        self.assertIn("applied_gain_db", event)
        self.assertIn("seed_value", event)
        self.assertGreater(event["end_sample"], event["start_sample"])

    def test_same_voice_duration_overlap_is_fatal(self):
        scene = self.base_scene(
            events=[
                {
                    "id": "first",
                    "time": 0.0,
                    "source": "a.wav",
                    "position": [0.0, 0.0, 0.0],
                    "voice": "shared",
                },
                {
                    "id": "second",
                    "time": 0.05,
                    "source": "b.wav",
                    "position": [0.0, 0.0, 0.0],
                    "voice": "shared",
                },
            ]
        )
        path = self.scene_path(scene)
        with self.assertRaises(Exception) as ctx:
            Renderer(load_scene(path), self.dir / "overlap").render()
        self.assertEqual(ctx.exception.code, E_EVENT_OVERLAP)

    def test_cli_runs_and_writes_report(self):
        path = self.scene_path(self.base_scene())
        out_dir = self.dir / "cli"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "offline_audio_render",
                str(path),
                "-o",
                str(out_dir),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads(proc.stdout)
        self.assertEqual(summary["status"], "ok")
        self.assertTrue((out_dir / "report.json").exists())

    def test_empty_scene_still_writes_valid_wav(self):
        scene = {
            "sample_rate": 48000,
            "layout": "stereo",
            "duration": 0.1,
            "events": [],
        }
        result = self.render(scene, "empty")
        info = sf.info(result.output_files["master"])
        self.assertEqual(info.frames, 4800)
        self.assertEqual(info.channels, 2)
        self.assertEqual(info.subtype, "FLOAT")


if __name__ == "__main__":
    unittest.main()
