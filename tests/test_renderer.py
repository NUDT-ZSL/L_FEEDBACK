from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

from spatial_audio_renderer.dsp import distance_gain, occlusion_cutoff
from spatial_audio_renderer.paths import normalize_path
from spatial_audio_renderer.scene import AttenuationConfig


ROOT = Path(__file__).resolve().parents[1]


def scene_data(*, rt60: float = 0.5, asset_id: str = "noise_7") -> Dict[str, Any]:
    return {
        "sample_rate": 8000,
        "duration_seconds": 0.4,
        "total_samples": 3200,
        "bit_depth": 24,
        "frame_size": 80,
        "listener": {
            "position": [0.0, 0.0, 0.0],
            "forward": [0.0, 0.0, -1.0],
            "up": [0.0, 1.0, 0.0],
        },
        "master_gain_db": 0.0,
        "attenuation": {
            "model": "inverse_distance",
            "reference_distance": 1.0,
            "rolloff": 1.0,
            "min_distance": 0.1,
            "max_distance": 100.0,
            "max_gain": 1.0,
        },
        "occlusion": {
            "open_cutoff_hz": 3900.0,
            "closed_cutoff_hz": 500.0,
        },
        "reverb": {
            "rt60_seconds": rt60,
            "pre_delay_seconds": 0.012,
            "wet": 0.3,
            "dry": 1.0,
            "seed": 31,
            "damping_hz": 3000.0,
        },
        "assets": [
            {
                "id": "tone_440",
                "type": "sine",
                "frequency_hz": 440.0,
                "amplitude": 0.22,
                "phase": 0.1,
                "duration_seconds": 0.4,
            },
            {
                "id": "noise_7",
                "type": "noise",
                "seed": 7,
                "amplitude": 0.07,
                "duration_seconds": 0.4,
            },
        ],
        "regions": [
            {
                "id": "zone",
                "shape": "sphere",
                "center": [0.0, 0.0, -2.0],
                "radius": 1.5,
                "reverb": {
                    "rt60_seconds": rt60,
                    "pre_delay_seconds": 0.012,
                    "wet": 0.3,
                    "dry": 0.92,
                    "seed": 31,
                },
                "lowpass": {"cutoff_hz": 2800.0},
            }
        ],
        "emitters": [
            {
                "id": "inside",
                "asset": "tone_440",
                "position": [0.5, 0.0, -1.4],
                "gain_db": -2.0,
                "pan": "auto",
                "occlusion": {"amount": 0.35},
            },
            {
                "id": "outside",
                "asset": asset_id,
                "position": [-3.0, 0.0, -4.0],
                "gain_db": -8.0,
                "pan": -0.6,
                "occluded": False,
            },
        ],
    }


class RendererIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="sar-tests-")
        self.root = Path(self.temporary.name)
        self.scene_path = self.root / "scene.json"
        self.write_scene(scene_data())

    def tearDown(self) -> None:
        # Restore any deliberately permission-restricted directory.
        for directory in self.root.rglob("*"):
            try:
                if directory.is_dir():
                    directory.chmod(0o700)
            except OSError:
                pass
        self.temporary.cleanup()

    def write_scene(self, payload: Dict[str, Any], path: Optional[Path] = None) -> Path:
        target = path or self.scene_path
        target.write_text(json.dumps(payload), encoding="utf-8")
        return target

    def cli(self, scene: Path, output: Path | str, snapshots: Path | str,
            interval: int = 2, *extra: str) -> subprocess.CompletedProcess:
        command: List[str] = [
            sys.executable,
            "-m",
            "spatial_audio_renderer",
            "--scene",
            str(scene),
            "--output-dir",
            str(output),
            "--snapshot-dir",
            str(snapshots),
            "--snapshot-interval",
            str(interval),
            "--max-snapshots",
            "4",
        ]
        command.extend(extra)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT)
        return subprocess.run(
            command,
            cwd=str(ROOT),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )

    def assert_business_error(self, completed: subprocess.CompletedProcess,
                              code: str, exit_code: int) -> Dict[str, Any]:
        self.assertEqual(completed.returncode, exit_code, msg=completed.stderr)
        lines = [line for line in completed.stderr.splitlines() if line.strip()]
        self.assertTrue(lines, msg="missing JSON error line")
        payload = json.loads(lines[-1])
        self.assertEqual(payload["code"], code)
        self.assertEqual(payload["exit_code"], exit_code)
        return payload

    def success_payload(self, completed: subprocess.CompletedProcess) -> Dict[str, Any]:
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        self.assertTrue(lines)
        return json.loads(lines[-1])

    def assert_wav(self, path: Path, channels: int, rate: int, frames: int,
                   subtype: str) -> None:
        info = sf.info(str(path))
        self.assertEqual(info.channels, channels)
        self.assertEqual(info.samplerate, rate)
        self.assertEqual(info.frames, frames)
        self.assertEqual(info.subtype, subtype)
        with wave.open(str(path), "rb") as handle:
            self.assertEqual(handle.getnchannels(), channels)
            self.assertEqual(handle.getframerate(), rate)
            self.assertEqual(handle.getnframes(), frames)
            self.assertEqual(handle.getsampwidth(), 3)

    def test_normal_render_outputs_and_headers(self) -> None:
        output = self.root / "normal-out"
        snapshots = self.root / "normal-snap"
        completed = self.cli(self.scene_path, output, snapshots, 2)
        payload = self.success_payload(completed)
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(payload["exit_code"], 0)

        mix = output / "mix.wav"
        inside = output / "dry_inside.wav"
        outside = output / "dry_outside.wav"
        self.assert_wav(mix, 2, 8000, 3200, "PCM_24")
        self.assert_wav(inside, 1, 8000, 3200, "PCM_24")
        self.assert_wav(outside, 1, 8000, 3200, "PCM_24")

        manifest = json.loads((output / "render_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["sample_rate"], 8000)
        self.assertEqual(manifest["bit_depth"], 24)
        self.assertEqual(manifest["channels"], 2)
        self.assertEqual(len(manifest["dry_tracks"]), 2)
        self.assertTrue(manifest["reverb"]["precomputed"])

        snapshot_manifests = sorted(snapshots.glob("snapshot_*.json"))
        self.assertTrue(snapshot_manifests)
        latest = json.loads(snapshot_manifests[-1].read_text(encoding="utf-8"))
        self.assertEqual(latest["version"], 1)
        self.assertEqual(latest["format"], "spatial-audio-render-snapshot")
        self.assertIn("state_sha256", latest)
        self.assertTrue(latest["reverb"]["precomputed"])
        self.assertEqual(latest["progress"]["processed_samples"], 3200)
        self.assertEqual(len(latest["emitters"]), 2)

    def test_crash_resume_is_sample_identical(self) -> None:
        baseline_out = self.root / "baseline-out"
        baseline_snap = self.root / "baseline-snap"
        baseline = self.cli(self.scene_path, baseline_out, baseline_snap, 3)
        self.success_payload(baseline)

        crash_out = self.root / "crash-out"
        crash_snap = self.root / "crash-snap"
        crashed = self.cli(
            self.scene_path, crash_out, crash_snap, 3,
            "--simulate-crash-after-frames", "14",
        )
        self.assertEqual(crashed.returncode, 137, msg=crashed.stderr)

        resumed = self.cli(self.scene_path, crash_out, crash_snap, 3, "--resume")
        resumed_payload = self.success_payload(resumed)
        self.assertIsNotNone(resumed_payload["resumed_from"])
        self.assertGreater(resumed_payload["resumed_from_frame"], 0)

        filenames = ["mix.wav", "dry_inside.wav", "dry_outside.wav"]
        for filename in filenames:
            baseline_data, baseline_rate = sf.read(str(baseline_out / filename),
                                                   always_2d=True)
            resumed_data, resumed_rate = sf.read(str(crash_out / filename),
                                                 always_2d=True)
            self.assertEqual(baseline_rate, resumed_rate)
            self.assertLessEqual(
                float(np.max(np.abs(baseline_data - resumed_data))),
                1e-6,
                msg=filename,
            )

    def test_automatic_resume_without_flag(self) -> None:
        crash_out = self.root / "auto-out"
        crash_snap = self.root / "auto-snap"
        crashed = self.cli(
            self.scene_path, crash_out, crash_snap, 2,
            "--simulate-crash-after-frames", "10",
        )
        self.assertEqual(crashed.returncode, 137)
        resumed = self.cli(self.scene_path, crash_out, crash_snap, 2)
        payload = self.success_payload(resumed)
        self.assertIsNotNone(payload["resumed_from"])
        self.assertTrue((crash_out / "mix.wav").is_file())

    def test_snapshot_version_mismatch(self) -> None:
        output = self.root / "version-out"
        snapshots = self.root / "version-snap"
        self.success_payload(self.cli(self.scene_path, output, snapshots, 2))
        latest = sorted(snapshots.glob("snapshot_*.json"))[-1]
        manifest = json.loads(latest.read_text(encoding="utf-8"))
        manifest["version"] = 999
        latest.write_text(json.dumps(manifest), encoding="utf-8")

        completed = self.cli(self.scene_path, output, snapshots, 2, "--resume")
        payload = self.assert_business_error(completed, "SNAP_VERSION_MISMATCH", 1)
        self.assertEqual(payload["details"]["supported_version"], 1)

    def test_automatic_recovery_refuses_unusable_snapshot(self) -> None:
        output = self.root / "auto-mismatch-out"
        snapshots = self.root / "auto-mismatch-snap"
        self.success_payload(self.cli(self.scene_path, output, snapshots, 100))
        latest = sorted(snapshots.glob("snapshot_*.json"))[-1]
        manifest = json.loads(latest.read_text(encoding="utf-8"))
        manifest["version"] = 999
        latest.write_text(json.dumps(manifest), encoding="utf-8")

        completed = self.cli(self.scene_path, output, snapshots, 100)
        self.assert_business_error(completed, "SNAP_VERSION_MISMATCH", 1)

    def test_snapshot_checksum_corruption(self) -> None:
        output = self.root / "corrupt-out"
        snapshots = self.root / "corrupt-snap"
        self.success_payload(self.cli(self.scene_path, output, snapshots, 2))
        latest = sorted(snapshots.glob("snapshot_*.json"))[-1]
        manifest = json.loads(latest.read_text(encoding="utf-8"))
        for older in sorted(snapshots.glob("snapshot_*.json"))[:-1]:
            old_manifest = json.loads(older.read_text(encoding="utf-8"))
            old_state = snapshots / old_manifest["state_file"]
            if old_state.exists():
                old_state.unlink()
            older.unlink()
        state_path = snapshots / manifest["state_file"]
        state_bytes = bytearray(state_path.read_bytes())
        state_bytes[100:104] = b"\x00\x00\x00\x00"
        state_path.write_bytes(bytes(state_bytes))

        completed = self.cli(self.scene_path, output, snapshots, 2, "--resume")
        self.assert_business_error(completed, "SNAP_CORRUPTED", 1)

    def test_snapshot_not_found(self) -> None:
        completed = self.cli(
            self.scene_path,
            self.root / "missing-out",
            self.root / "missing-snap",
            2,
            "--resume",
        )
        self.assert_business_error(completed, "SNAP_NOT_FOUND", 1)

    def test_audio_asset_missing(self) -> None:
        payload = scene_data(asset_id="does_not_exist")
        scene_path = self.write_scene(payload, self.root / "missing-asset.json")
        completed = self.cli(
            scene_path, self.root / "asset-out", self.root / "asset-snap", 2
        )
        self.assert_business_error(completed, "AUDIO_ASSET_MISSING", 2)

    def test_invalid_scene(self) -> None:
        invalid = self.root / "invalid.json"
        invalid.write_text('{"sample_rate": 8000, "emitters": []}', encoding="utf-8")
        completed = self.cli(
            invalid, self.root / "invalid-out", self.root / "invalid-snap", 2
        )
        self.assert_business_error(completed, "INVALID_SCENE", 2)

    def test_invalid_output_directory(self) -> None:
        blocked = self.root / "not-a-directory"
        blocked.write_text("file", encoding="utf-8")
        completed = self.cli(
            self.scene_path, blocked, self.root / "blocked-snap", 2
        )
        self.assert_business_error(completed, "INVALID_OUTPUT_DIR", 2)

    def test_unwritable_output_directory(self) -> None:
        unwritable = self.root / "read-only-out"
        unwritable.mkdir()
        unwritable.chmod(0o500)
        completed = self.cli(
            self.scene_path, unwritable, self.root / "read-only-snap", 2
        )
        self.assert_business_error(completed, "INVALID_OUTPUT_DIR", 2)

    def test_unknown_cli_argument_uses_business_code(self) -> None:
        completed = self.cli(
            self.scene_path,
            self.root / "arg-out",
            self.root / "arg-snap",
            2,
            "--not-a-real-option",
        )
        self.assert_business_error(completed, "INVALID_ARGUMENT", 2)

    def test_reverb_change_does_not_touch_non_region_dry(self) -> None:
        first_out = self.root / "reverb-a-out"
        second_out = self.root / "reverb-b-out"
        self.success_payload(self.cli(self.scene_path, first_out,
                                     self.root / "reverb-a-snap", 2))
        changed_path = self.write_scene(
            scene_data(rt60=1.4), self.root / "reverb-b.json"
        )
        self.success_payload(self.cli(changed_path, second_out,
                                     self.root / "reverb-b-snap", 2))

        first, _ = sf.read(str(first_out / "dry_outside.wav"), always_2d=True)
        second, _ = sf.read(str(second_out / "dry_outside.wav"), always_2d=True)
        self.assertLessEqual(float(np.max(np.abs(first - second))), 1e-6)

        first_mix, _ = sf.read(str(first_out / "mix.wav"), always_2d=True)
        second_mix, _ = sf.read(str(second_out / "mix.wav"), always_2d=True)
        self.assertGreater(float(np.max(np.abs(first_mix - second_mix))), 0.0)

    def test_windows_style_relative_output_path(self) -> None:
        output_argument = str(self.root) + "\\window-style\\out"
        snapshots_argument = str(self.root) + "\\window-style\\snap"
        completed = self.cli(self.scene_path, output_argument,
                             snapshots_argument, 2)
        self.success_payload(completed)
        self.assertTrue((self.root / "window-style" / "out" / "mix.wav").is_file())


class DspUnitTest(unittest.TestCase):
    def test_distance_models_are_configurable(self) -> None:
        inverse = AttenuationConfig("inverse_distance", 1.0, 1.0, 0.0, 100.0, 1.0)
        self.assertAlmostEqual(distance_gain(1.0, inverse), 1.0)
        self.assertAlmostEqual(distance_gain(2.0, inverse), 0.5)
        self.assertAlmostEqual(distance_gain(101.0, inverse), 0.0)

        linear = AttenuationConfig("linear", 1.0, 1.0, 2.0, 10.0, 1.0)
        self.assertAlmostEqual(distance_gain(6.0, linear), 0.5)
        self.assertAlmostEqual(distance_gain(12.0, linear), 0.0)

        none = AttenuationConfig("none")
        self.assertEqual(distance_gain(100.0, none), 1.0)

    def test_occlusion_cutoff_is_linear(self) -> None:
        self.assertAlmostEqual(occlusion_cutoff(0.0, 12000, 900, 48000), 12000)
        self.assertAlmostEqual(occlusion_cutoff(0.5, 12000, 900, 48000), 6450)
        self.assertAlmostEqual(occlusion_cutoff(1.0, 12000, 900, 48000), 900)

    def test_windows_path_normalization_on_posix(self) -> None:
        if os.name == "nt":
            self.skipTest("POSIX separator normalization is not applicable")
        normalized = normalize_path("nested\\folder\\file.json")
        self.assertEqual(normalized.parts, ("nested", "folder", "file.json"))


if __name__ == "__main__":
    unittest.main()
