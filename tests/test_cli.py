from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN = PROJECT_ROOT / "main.py"


def _run(args: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(MAIN), *args],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _write_scene(directory: Path, *, malformed: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rate = 48000
    sf.write(
        str(directory / "ambience.wav"),
        np.zeros((rate * 2, 1), dtype=np.float64),
        rate,
        subtype="PCM_16",
    )
    sf.write(
        str(directory / "music.wav"),
        np.zeros((rate, 2), dtype=np.float64),
        rate,
        subtype="PCM_16",
    )
    impulse = np.zeros((240, 1), dtype=np.float64)
    impulse[0, 0] = 0.5
    sf.write(
        str(directory / "hit.wav"),
        impulse,
        rate,
        subtype="PCM_16",
    )
    scene = {
        "scene_id": directory.name,
        "duration": 2.5,
        "ambience": {
            "file": "ambience.wav",
            "loop_start": 0.1,
            "loop_end": 1.9,
            "gain": 0.5,
        },
        "samples": {"hit": "hit.wav"},
        "events": [
            {"time": 0.333, "sample": "hit", "gain": 0.7},
            {"time": 1.75, "sample": "hit", "gain": 0.3},
        ],
        "music_layers": [
            {"file": "music.wav", "start": 0.4, "loops": 2, "gain": 0.2}
        ],
        "dialogue_intervals": [
            {"start": 0.5, "end": 1.2, "duck_depth_db": 9.0}
        ],
        "camera_cuts": [0.75, 2.0],
    }
    if malformed:
        del scene["duration"]
    (directory / "scene.json").write_text(
        json.dumps(scene), encoding="utf-8"
    )


class CliTests(unittest.TestCase):
    def test_no_arguments_prints_usage_and_exits_zero(self) -> None:
        result = _run([])
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage:", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_parameter_error_exits_one(self) -> None:
        result = _run(["only-one"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("expected exactly", result.stderr)

    def test_help_exits_zero(self) -> None:
        result = _run(["--help"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage:", result.stdout)

    def test_malformed_json_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene_dir = root / "scene"
            scene_dir.mkdir()
            (scene_dir / "scene.json").write_text("{not-json", encoding="utf-8")
            result = _run([str(scene_dir), str(root / "out")])
            self.assertEqual(result.returncode, 2)
            self.assertIn("invalid JSON", result.stderr)

    def test_invalid_scene_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene_dir = root / "scene"
            _write_scene(scene_dir, malformed=True)
            result = _run([str(scene_dir), str(root / "out")])
            self.assertEqual(result.returncode, 2)
            self.assertIn("missing required field", result.stderr)

    def test_single_scene_outputs_and_sample_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene_dir = root / "scene"
            out = root / "out"
            _write_scene(scene_dir)
            result = _run([str(scene_dir), str(out)])
            self.assertEqual(result.returncode, 0, result.stderr)

            info = sf.info(str(out / "mix.wav"))
            self.assertEqual(info.samplerate, 48000)
            self.assertEqual(info.channels, 2)
            self.assertEqual(info.subtype, "PCM_16")
            self.assertEqual(info.frames, 120000)

            audio, rate = sf.read(
                str(out / "mix.wav"), dtype="float64", always_2d=True
            )
            self.assertEqual(rate, 48000)
            expected_event_sample = 15984
            band = audio[
                expected_event_sample - 2 : expected_event_sample + 3,
                :,
            ]
            self.assertEqual(np.argmax(np.abs(band[:, 0])), 2)
            self.assertGreater(np.max(np.abs(band[:, 0])), 0.2)

            loudness = json.loads(
                (out / "loudness_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(loudness["sample_rate_hz"], 48000)
            self.assertIn("integrated_lufs", loudness)
            self.assertIn("true_peak_dbtp", loudness)
            self.assertEqual(len(loudness["dialogue_intervals"]), 1)
            self.assertIsInstance(
                loudness["dialogue_intervals"][0]["max_short_term_lufs"],
                (int, float),
            )
            self.assertEqual(
                loudness["dialogue_intervals"][0]["method"],
                "numpy_bs1770_3s_fallback",
            )

            switches = json.loads(
                (out / "switch_check.json").read_text(encoding="utf-8")
            )
            self.assertTrue(switches["all_cuts_passed"])
            self.assertTrue(
                switches["all_ducking_boundaries_continuous"]
            )
            for cut in switches["cuts"]:
                self.assertEqual(cut["fade_samples"], 960)
                self.assertTrue(cut["boundary_continuous"])

    def test_batch_partial_failure_exits_three_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            batch = root / "batch"
            _write_scene(batch / "good")
            _write_scene(batch / "bad", malformed=True)
            out = root / "out"
            result = _run([str(batch), str(out)])
            self.assertEqual(result.returncode, 3, result.stderr)
            self.assertTrue((out / "good" / "mix.wav").is_file())
            self.assertFalse((out / "bad" / "mix.wav").is_file())
            summary = json.loads(
                (out / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["total_scenes"], 2)
            self.assertEqual(summary["succeeded"], 1)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["exit_code"], 3)

    def test_batch_directory_without_scene_json_counts_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            batch = root / "batch"
            _write_scene(batch / "good")
            (batch / "missing").mkdir()
            (batch / "missing" / "asset.wav").write_bytes(b"not audio")
            out = root / "out"
            result = _run([str(batch), str(out)])
            self.assertEqual(result.returncode, 3)
            self.assertTrue((out / "good" / "mix.wav").is_file())
            summary = json.loads(
                (out / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["total_scenes"], 2)
            self.assertEqual(summary["succeeded"], 1)
            self.assertEqual(summary["failed"], 1)


if __name__ == "__main__":
    unittest.main()
