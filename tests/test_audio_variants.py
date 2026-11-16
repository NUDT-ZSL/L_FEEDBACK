from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_variants.loudness import (
    _high_shelf_coefficients,
    _highpass_coefficients,
    integrated_loudness,
    true_peak_dbtp,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_tool(config_path: Path, input_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "audio_variants",
            "--config",
            str(config_path),
            "--input",
            str(input_dir),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )


def write_tone(
    path: Path,
    sample_rate: int,
    seconds: float,
    channels: int,
    subtype: str,
    frequency: float = 220.0,
    amplitude: float = 0.2,
) -> None:
    frames = int(round(sample_rate * seconds))
    time = np.arange(frames, dtype=np.float64) / sample_rate
    envelope = np.minimum(1.0, time / 0.01) * np.minimum(
        1.0, np.maximum(0.0, (seconds - time) / 0.01)
    )
    base = (
        amplitude
        * envelope
        * (
            np.sin(2.0 * np.pi * frequency * time)
            + 0.2 * np.sin(2.0 * np.pi * frequency * 3.1 * time)
        )
    )
    if channels == 1:
        audio = base
    else:
        channels_data = [base]
        for channel_index in range(1, channels):
            channels_data.append((0.72**channel_index) * base)
        audio = np.stack(channels_data, axis=1)
    sf.write(str(path), audio, sample_rate, format="WAV", subtype=subtype)


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


class ToolTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_config(
        self,
        name: str,
        output_dir: str,
        *,
        global_seed: int = 123456,
        target_lufs: float = -23.0,
        default_variants: int = 2,
        sounds: object = None,
    ) -> Path:
        config = {
            "global_seed": global_seed,
            "target_lufs": target_lufs,
            "output_dir": output_dir,
            "default_variants": default_variants,
        }
        if sounds is not None:
            config["sounds"] = sounds
        path = self.root / name
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def load_json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))


class FormatAndReportTests(ToolTestCase):
    def test_formats_report_manifest_and_loudness(self) -> None:
        write_tone(self.input_dir / "mono16.wav", 44100, 1.3, 1, "PCM_16")
        write_tone(self.input_dir / "stereo24.wav", 48000, 1.2, 2, "PCM_24")
        write_tone(self.input_dir / "stereo32.wav", 32000, 1.1, 2, "PCM_32")
        write_tone(self.input_dir / "float32.wav", 48000, 1.0, 2, "FLOAT")
        config_path = self.write_config(
            "config.json",
            "out",
            default_variants=2,
            sounds={
                "mono16.wav": {"variants": 3, "seed": 42},
                "float32": {"variants": 1},
            },
        )

        completed = run_tool(config_path, self.input_dir)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output_dir = self.root / "out"
        report = self.load_json(output_dir / "report.json")
        manifest = self.load_json(output_dir / "manifest.json")

        self.assertEqual(report["schema_version"], 1)
        self.assertTrue(report["generated_at"].endswith("Z"))
        datetime.fromisoformat(report["generated_at"].replace("Z", "+00:00"))
        self.assertEqual(report["summary"]["sources_total"], 4)
        self.assertEqual(report["summary"]["sources_succeeded"], 4)
        self.assertEqual(report["summary"]["sources_failed"], 0)
        self.assertEqual(report["summary"]["variants_generated"], 8)
        self.assertEqual(manifest["variant_count"], 8)
        self.assertEqual(len(manifest["variants"]), 8)

        expected_formats = {
            "mono16": ("PCM_16", 44100, 1, 3),
            "stereo24": ("PCM_24", 48000, 2, 2),
            "stereo32": ("PCM_32", 32000, 2, 2),
            "float32": ("FLOAT", 48000, 2, 1),
        }
        for result in report["results"]:
            source = result["source_file"]
            stem = Path(source).stem
            subtype, sample_rate, channels, count = expected_formats[stem]
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["requested_variant_count"], count)
            self.assertEqual(result["generated_variant_count"], count)
            for variant in result["variants"]:
                index = variant["index"]
                output_path = output_dir / f"{stem}_{index}.wav"
                self.assertTrue(output_path.is_file())
                self.assertEqual(variant["output_file"], f"{stem}_{index}.wav")
                info = sf.info(str(output_path))
                self.assertEqual(info.subtype, subtype)
                self.assertEqual(info.samplerate, sample_rate)
                self.assertEqual(info.channels, channels)

                parameters = variant["parameters"]
                self.assertGreaterEqual(parameters["pitch_semitones"], -3.0)
                self.assertLessEqual(parameters["pitch_semitones"], 3.0)
                self.assertGreaterEqual(parameters["time_stretch_ratio"], 0.95)
                self.assertLessEqual(parameters["time_stretch_ratio"], 1.05)
                self.assertGreaterEqual(parameters["lowpass_cutoff_ratio"], 0.70)
                self.assertLessEqual(parameters["lowpass_cutoff_ratio"], 1.0)
                self.assertAlmostEqual(
                    parameters["lowpass_cutoff_hz"],
                    parameters["lowpass_cutoff_ratio"] * sample_rate / 2.0,
                    places=6,
                )

                written, written_rate = sf.read(
                    str(output_path), dtype="float64", always_2d=True
                )
                measured_lufs = integrated_loudness(written, written_rate)
                measured_peak = true_peak_dbtp(written)
                self.assertAlmostEqual(
                    variant["loudness_lufs"], measured_lufs, places=4
                )
                self.assertAlmostEqual(
                    variant["true_peak_dbtp"], measured_peak, places=4
                )
                if not variant["peak_limited"]:
                    self.assertAlmostEqual(variant["loudness_lufs"], -23.0, delta=0.03)
                self.assertLessEqual(measured_peak, -1.0 + 0.01)

        for entry in manifest["variants"]:
            self.assertFalse(Path(entry["path"]).is_absolute())
            self.assertFalse(Path(entry["source"]).is_absolute())
            self.assertTrue((output_dir / entry["path"]).is_file())

    def test_peak_limiter_lowers_actual_loudness(self) -> None:
        sample_rate = 48000
        audio = np.zeros(4800, dtype=np.float64)
        audio[1000] = 0.95
        audio[1001] = -0.4
        audio[1002] = 0.2
        sf.write(
            str(self.input_dir / "click.wav"),
            audio,
            sample_rate,
            subtype="PCM_16",
        )
        config_path = self.write_config(
            "config.json", "out", target_lufs=-5.0, default_variants=1
        )
        completed = run_tool(config_path, self.input_dir)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = self.load_json(self.root / "out" / "report.json")
        variant = report["results"][0]["variants"][0]
        self.assertTrue(variant["peak_limited"])
        self.assertLess(variant["loudness_lufs"], -5.0)
        self.assertLessEqual(variant["true_peak_dbtp"], -1.0)
        written, _ = sf.read(
            str(self.root / "out" / "click_1.wav"),
            dtype="float64",
            always_2d=True,
        )
        self.assertLessEqual(float(np.max(np.abs(written))), 1.0)

    def test_very_quiet_input_uses_ungated_fallback(self) -> None:
        write_tone(
            self.input_dir / "quiet.wav",
            48000,
            1.0,
            1,
            "PCM_24",
            amplitude=0.0001,
        )
        config_path = self.write_config("config.json", "out", default_variants=1)
        completed = run_tool(config_path, self.input_dir)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = self.load_json(self.root / "out" / "report.json")
        variant = report["results"][0]["variants"][0]
        self.assertEqual(variant["input_loudness_method"], "ungated_fallback")
        self.assertAlmostEqual(variant["loudness_lufs"], -23.0, delta=0.05)

    def test_seed_only_override_uses_default_variant_count(self) -> None:
        write_tone(self.input_dir / "hit.wav", 22050, 0.7, 1, "PCM_16")
        config_path = self.root / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "global_seed": 5,
                    "output_dir": "out",
                    "variants": {"hit.wav": {"seed": 77}},
                }
            ),
            encoding="utf-8",
        )
        completed = run_tool(config_path, self.input_dir)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = self.load_json(self.root / "out" / "report.json")
        result = report["results"][0]
        self.assertEqual(result["variant_count"], 3)
        self.assertEqual(result["generated_variant_count"], 3)
        self.assertEqual(result["seed"], 77)
        self.assertEqual(report["config"]["target_lufs"], -23.0)


class DeterminismTests(ToolTestCase):
    def test_same_seed_is_sample_identical(self) -> None:
        write_tone(self.input_dir / "hit.wav", 44100, 0.9, 2, "PCM_16")
        config_a = self.write_config("a.json", "out_a", default_variants=3)
        config_b = self.write_config("b.json", "out_b", default_variants=3)

        first = run_tool(config_a, self.input_dir)
        second = run_tool(config_b, self.input_dir)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        for index in range(1, 4):
            name = f"hit_{index}.wav"
            bytes_a = read_bytes(self.root / "out_a" / name)
            bytes_b = read_bytes(self.root / "out_b" / name)
            self.assertEqual(bytes_a, bytes_b)
            audio_a, rate_a = sf.read(
                str(self.root / "out_a" / name), dtype="float64", always_2d=True
            )
            audio_b, rate_b = sf.read(
                str(self.root / "out_b" / name), dtype="float64", always_2d=True
            )
            self.assertEqual(rate_a, rate_b)
            np.testing.assert_array_equal(audio_a, audio_b)

    def test_per_sound_seed_overrides_global_seed(self) -> None:
        write_tone(self.input_dir / "hit.wav", 48000, 0.8, 1, "FLOAT")
        config_a = self.write_config(
            "a.json",
            "out_a",
            global_seed=1,
            default_variants=1,
            sounds={"hit.wav": {"variants": 1, "seed": 999}},
        )
        config_b = self.write_config(
            "b.json",
            "out_b",
            global_seed=2,
            default_variants=1,
            sounds={"hit.wav": {"variants": 1, "seed": 999}},
        )
        self.assertEqual(run_tool(config_a, self.input_dir).returncode, 0)
        self.assertEqual(run_tool(config_b, self.input_dir).returncode, 0)
        self.assertEqual(
            read_bytes(self.root / "out_a" / "hit_1.wav"),
            read_bytes(self.root / "out_b" / "hit_1.wav"),
        )

    def test_different_seed_changes_output(self) -> None:
        write_tone(self.input_dir / "hit.wav", 48000, 0.8, 1, "PCM_16")
        config_a = self.write_config(
            "a.json", "out_a", global_seed=10, default_variants=1
        )
        config_b = self.write_config(
            "b.json", "out_b", global_seed=11, default_variants=1
        )
        self.assertEqual(run_tool(config_a, self.input_dir).returncode, 0)
        self.assertEqual(run_tool(config_b, self.input_dir).returncode, 0)
        self.assertNotEqual(
            read_bytes(self.root / "out_a" / "hit_1.wav"),
            read_bytes(self.root / "out_b" / "hit_1.wav"),
        )


class ErrorHandlingTests(ToolTestCase):
    def test_missing_arguments_exit_code_two(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "audio_variants"],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("usage:", completed.stderr)

    def test_missing_and_invalid_config_exit_nonzero(self) -> None:
        write_tone(self.input_dir / "hit.wav", 22050, 0.4, 1, "PCM_16")
        missing = run_tool(self.root / "missing.json", self.input_dir)
        self.assertEqual(missing.returncode, 1)
        self.assertIn("does not exist", missing.stderr)

        invalid_path = self.root / "invalid.json"
        invalid_path.write_text('{"global_seed": true}', encoding="utf-8")
        invalid = run_tool(invalid_path, self.input_dir)
        self.assertEqual(invalid.returncode, 1)
        self.assertIn("global_seed must be an integer", invalid.stderr)

        unknown_path = self.root / "unknown.json"
        unknown_path.write_text(
            json.dumps(
                {
                    "global_seed": 1,
                    "output_dir": "out",
                    "unexpected": 4,
                }
            ),
            encoding="utf-8",
        )
        unknown = run_tool(unknown_path, self.input_dir)
        self.assertEqual(unknown.returncode, 1)
        self.assertIn("unknown fields", unknown.stderr)

    def test_bad_files_are_reported_and_others_continue(self) -> None:
        write_tone(self.input_dir / "a_good.wav", 44100, 0.6, 1, "PCM_16")
        (self.input_dir / "b_corrupt.wav").write_bytes(b"not a RIFF WAV file")
        write_tone(self.input_dir / "c_three_channels.wav", 44100, 0.4, 3, "PCM_16")
        nested = self.input_dir / "nested"
        nested.mkdir()
        write_tone(nested / "ignored.wav", 44100, 0.4, 1, "PCM_16")
        config_path = self.write_config("config.json", "out", default_variants=2)

        completed = run_tool(config_path, self.input_dir)
        self.assertEqual(completed.returncode, 1)
        self.assertIn("b_corrupt.wav", completed.stderr)
        self.assertIn("c_three_channels.wav", completed.stderr)
        self.assertNotIn("ignored.wav", completed.stderr)
        report = self.load_json(self.root / "out" / "report.json")
        manifest = self.load_json(self.root / "out" / "manifest.json")
        self.assertEqual(report["summary"]["sources_total"], 3)
        self.assertEqual(report["summary"]["sources_succeeded"], 1)
        self.assertEqual(report["summary"]["sources_failed"], 2)
        self.assertEqual(report["summary"]["variants_generated"], 2)
        self.assertEqual(manifest["variant_count"], 2)
        self.assertEqual(
            [result["source_file"] for result in report["results"]],
            ["a_good.wav", "b_corrupt.wav", "c_three_channels.wav"],
        )
        self.assertTrue((self.root / "out" / "a_good_1.wav").is_file())
        self.assertFalse((self.root / "out" / "b_corrupt_1.wav").exists())
        self.assertFalse((self.root / "out" / "c_three_channels_1.wav").exists())

    def test_invalid_input_directory_exit_nonzero(self) -> None:
        config_path = self.write_config("config.json", "out")
        completed = run_tool(config_path, self.root / "missing-input")
        self.assertEqual(completed.returncode, 1)
        self.assertIn("does not exist", completed.stderr)

    def test_partial_variant_failure_is_reported_and_manifested(self) -> None:
        write_tone(self.input_dir / "hit.wav", 32000, 0.6, 1, "PCM_16")
        config_path = self.write_config("config.json", "out", default_variants=3)
        output_dir = self.root / "out"
        output_dir.mkdir()
        (output_dir / "hit_2.wav").mkdir()

        completed = run_tool(config_path, self.input_dir)
        self.assertEqual(completed.returncode, 1)
        self.assertIn("variant 2 failed", completed.stderr)
        report = self.load_json(output_dir / "report.json")
        manifest = self.load_json(output_dir / "manifest.json")
        result = report["results"][0]
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["generated_variant_count"], 1)
        self.assertEqual(len(result["variants"]), 1)
        self.assertEqual(manifest["variant_count"], 1)
        self.assertEqual(manifest["variants"][0]["path"], "hit_1.wav")
        self.assertTrue((output_dir / "hit_1.wav").is_file())
        self.assertFalse((output_dir / "hit_3.wav").exists())


class LoudnessUnitTests(unittest.TestCase):
    def test_48k_k_weighting_coefficients_match_bs1770(self) -> None:
        shelf_b, shelf_a = _high_shelf_coefficients(48000)
        highpass_b, highpass_a = _highpass_coefficients(48000)
        np.testing.assert_allclose(
            shelf_b,
            [1.53512485958697, -2.69169618940638, 1.19839281085285],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            shelf_a,
            [1.0, -1.69065929318241, 0.73248077421585],
            atol=1e-12,
        )
        np.testing.assert_allclose(highpass_b, [1.0, -2.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(
            highpass_a,
            [1.0, -1.99004745483398, 0.99007225036621],
            atol=1e-12,
        )

    def test_1khz_sine_loudness_is_approximately_minus_23_lufs(self) -> None:
        sample_rate = 48000
        time = np.arange(sample_rate * 2, dtype=np.float64) / sample_rate
        audio = (0.1 * np.sin(2.0 * np.pi * 1000.0 * time))[:, None]
        loudness = integrated_loudness(audio, sample_rate)
        self.assertTrue(math.isfinite(loudness))
        self.assertAlmostEqual(loudness, -23.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
