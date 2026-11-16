from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from skillsfx import engine
from skillsfx.dsp import clipping_sample_count, true_peak_stats
from skillsfx.errors import SkillSFXError
from skillsfx.recipe import load_recipe
from skillsfx.synth import synthesize_sources
from skillsfx.wavio import encode_wav, read_wav, verify_wav


ROOT = Path(__file__).resolve().parents[1]


def _base_recipe(source_dir: Path, *, duration_ms: float = 1600.0) -> dict:
    return {
        "schema_version": 1,
        "skill_name": "Unit Test Skill",
        "variant_count": 3,
        "seed": 424242,
        "output": {
            "sample_rate": 48000,
            "bit_depth": 24,
            "channels": 2,
            "duration_ms": duration_ms,
        },
        "events": [
            {
                "role": "slash",
                "source": str(source_dir / "slash_44k_mono.wav"),
                "offset_ms": 10,
                "duration_ms": 240,
                "gain": 0.8,
                "channel_map": "mono_to_stereo",
            },
            {
                "role": "impact",
                "source": str(source_dir / "impact_48k_stereo.wav"),
                "offset_ms": 120,
                "duration_ms": 250,
                "gain": 0.9,
                "channel_map": "identity",
            },
            {
                "role": "tail",
                "source": str(source_dir / "tail_48k_stereo.wav"),
                "offset_ms": 700,
                "duration_ms": 700,
                "gain": 0.7,
                "channel_map": {
                    "mode": "matrix",
                    "matrix": [[1.0, 0.0], [0.0, 1.0]],
                },
            },
        ],
        "randomization": {
            "gain_jitter_db": 0.5,
            "time_jitter_ms": 0.4,
            "tail_harmonics": {
                "enabled": True,
                "fundamental_hz": 1800.0,
                "ratios": [1.0, 1.47, 2.09, 2.71],
                "gain_db_min": -30.0,
                "gain_db_max": -18.0,
                "decay_ms": 600.0,
                "attack_ms": 2.0,
            },
        },
        "bus_by_level": {
            "1": {
                "compressor": {
                    "threshold_db": -20.0,
                    "ratio": 2.0,
                    "knee_db": 8.0,
                    "attack_ms": 6.0,
                    "release_ms": 120.0,
                },
                "tail_fade_ms": 100.0,
                "limiter": {
                    "ceiling_dbtp": -1.0,
                    "lookahead_ms": 5.0,
                    "attack_ms": 0.8,
                    "release_ms": 80.0,
                },
            },
            "2": {
                "compressor": {
                    "threshold_db": -30.0,
                    "ratio": 4.0,
                    "knee_db": 4.0,
                    "attack_ms": 3.0,
                    "release_ms": 70.0,
                },
                "tail_fade_ms": 400.0,
                "limiter": {
                    "ceiling_dbtp": -1.0,
                    "lookahead_ms": 6.0,
                    "attack_ms": 0.6,
                    "release_ms": 70.0,
                },
            },
        },
        "normalization": {
            "target_lufs": -12.0,
            "min_gain_db": -18.0,
            "max_gain_db": 18.0,
        },
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


class WavIoTest(unittest.TestCase):
    def test_integer_and_float_wav_roundtrip_and_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for sample_rate, channels, bit_depth in (
                (44100, 1, 16),
                (48000, 2, 24),
                (44100, 2, 32),
                (48000, 1, "float32"),
                (48000, 2, "float32"),
            ):
                with self.subTest(
                    sample_rate=sample_rate,
                    channels=channels,
                    bit_depth=bit_depth,
                ):
                    frames = 777
                    t = np.arange(frames) / sample_rate
                    mono = 0.41 * np.sin(2.0 * np.pi * 997.0 * t)
                    audio = np.repeat(mono[:, None], channels, axis=1)
                    path = root / f"test_{sample_rate}_{channels}_{bit_depth}.wav"
                    path.write_bytes(encode_wav(audio, sample_rate, bit_depth))
                    decoded, info = verify_wav(
                        path,
                        sample_rate=sample_rate,
                        channels=channels,
                        bit_depth=bit_depth,
                        frame_count=frames,
                    )
                    self.assertEqual(info.frame_count, frames)
                    self.assertEqual(
                        info.data_size,
                        frames * channels * (32 if bit_depth == "float32" else bit_depth) // 8,
                    )
                    tolerance = {
                        16: 1.0 / 32768.0,
                        24: 1.0 / 8388608.0,
                        32: 1.0 / 2147483648.0,
                        "float32": 1.0e-7,
                    }[bit_depth]
                    self.assertLess(
                        float(np.max(np.abs(decoded - audio))),
                        tolerance * 1.1,
                    )
                    self.assertEqual(
                        clipping_sample_count(decoded),
                        0,
                    )


class RecipeValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.sources = cls.root / "sources"
        synthesize_sources(cls.sources)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_missing_field_has_stable_code(self) -> None:
        recipe = _base_recipe(self.sources)
        del recipe["seed"]
        path = self.root / "missing_seed.json"
        _write_json(path, recipe)
        with self.assertRaises(SkillSFXError) as caught:
            load_recipe(path)
        self.assertEqual(caught.exception.code, "SFX-E1002")
        self.assertEqual(caught.exception.context["field"], "recipe.seed")

    def test_negative_offset_names_event_and_source(self) -> None:
        recipe = _base_recipe(self.sources)
        recipe["events"][1]["offset_ms"] = -3.0
        path = self.root / "negative_offset.json"
        _write_json(path, recipe)
        with self.assertRaises(SkillSFXError) as caught:
            load_recipe(path)
        error = caught.exception
        self.assertEqual(error.code, "SFX-E2003")
        self.assertEqual(error.context["event_index"], 1)
        self.assertIn("impact_48k_stereo.wav", error.context["source_path"])

    def test_out_of_timeline_offset_names_event_and_source(self) -> None:
        recipe = _base_recipe(self.sources, duration_ms=500.0)
        recipe["events"][2]["offset_ms"] = 500.0
        path = self.root / "late_offset.json"
        _write_json(path, recipe)
        with self.assertRaises(SkillSFXError) as caught:
            load_recipe(path)
        error = caught.exception
        self.assertEqual(error.code, "SFX-E2003")
        self.assertEqual(error.context["event_index"], 2)
        self.assertIn("tail_48k_stereo.wav", error.context["source_path"])

    def test_unknown_field_is_rejected(self) -> None:
        recipe = _base_recipe(self.sources)
        recipe["typo_field"] = True
        path = self.root / "unknown_field.json"
        _write_json(path, recipe)
        with self.assertRaises(SkillSFXError) as caught:
            load_recipe(path)
        self.assertEqual(caught.exception.code, "SFX-E1002")
        self.assertIn("typo_field", caught.exception.context["unknown_fields"])


class SynthesisTest(unittest.TestCase):
    def test_builtin_sources_are_deterministic_and_varied(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = synthesize_sources(root / "a")
            second = synthesize_sources(root / "b")
            self.assertEqual(
                [item["sha256"] for item in first],
                [item["sha256"] for item in second],
            )
            formats = set()
            for item in first:
                audio, info = read_wav(item["path"])
                formats.add(
                    (
                        info.sample_rate,
                        info.channels,
                        info.bits_per_sample,
                        info.sample_format,
                    )
                )
                self.assertGreater(info.frame_count, 0)
                self.assertLess(float(np.max(np.abs(audio))), 1.0)
            self.assertGreaterEqual(len(formats), 4)


class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.sources = cls.root / "sources"
        synthesize_sources(cls.sources)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_generation_replay_determinism_clipping_and_fade_warning(self) -> None:
        recipe = _base_recipe(self.sources)
        recipe_path = self.root / "e2e.json"
        _write_json(recipe_path, recipe)
        output_one = self.root / "workers_1"
        output_four = self.root / "workers_4"
        output_level_two = self.root / "level_2"

        result_one = engine.generate_batch(
            recipe_path,
            level=1,
            output_directory=output_one,
            workers=1,
        )
        engine.generate_batch(
            recipe_path,
            level=1,
            output_directory=output_four,
            workers=4,
        )
        engine.generate_batch(
            recipe_path,
            level=2,
            output_directory=output_level_two,
            workers=2,
        )
        manifest_one = json.loads(
            (output_one / "manifest.json").read_text(encoding="utf-8")
        )
        manifest_four = json.loads(
            (output_four / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest_one["run_id"], manifest_four["run_id"])
        self.assertEqual(
            [item["output"]["sha256"] for item in manifest_one["variants"]],
            [item["output"]["sha256"] for item in manifest_four["variants"]],
        )
        self.assertEqual(
            manifest_one["variants"],
            manifest_four["variants"],
        )
        hashes = [item["output"]["sha256"] for item in manifest_one["variants"]]
        self.assertEqual(len(hashes), len(set(hashes)))

        first_variant = manifest_one["variants"][0]
        timeline = first_variant["timeline"]
        self.assertGreater(
            timeline[0]["span_frames"],
            timeline[0]["source_resampled_frames"],
            "short source must be padded to the requested event duration",
        )
        self.assertLess(
            timeline[2]["span_frames"],
            timeline[2]["source_resampled_frames"],
            "long source must be trimmed to the requested event duration",
        )
        self.assertGreaterEqual(len(timeline[2]["tail_harmonics"]["partials"]), 1)
        self.assertLessEqual(len(timeline[2]["tail_harmonics"]["partials"]), 4)
        for index, variant in enumerate(manifest_one["variants"]):
            self.assertEqual(variant["index"], index)
            self.assertNotEqual(
                variant["timeline"][0]["gain_jitter_db"],
                manifest_one["variants"][(index + 1) % 3]["timeline"][0][
                    "gain_jitter_db"
                ],
            )
            output_path = output_one / variant["output"]["path"]
            decoded, info = read_wav(output_path)
            self.assertEqual(info.frame_count, recipe["output"]["duration_ms"] * 48)
            self.assertEqual(clipping_sample_count(decoded), 0)
            self.assertLessEqual(
                true_peak_stats(decoded, info.sample_rate)["true_peak_dbtp"],
                -1.0 + 1e-4,
            )
            self.assertEqual(
                variant["analysis"]["clipping_sample_count"],
                0,
            )
            self.assertGreaterEqual(
                variant["analysis"]["limiter_trigger_count"],
                1,
            )
            self.assertTrue(
                variant["analysis"]["limiter_trigger_intervals"],
            )
            self.assertIsNotNone(variant["analysis"]["normalization_gain_db"])

        manifest_two = json.loads(
            (output_level_two / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertTrue(
            manifest_two["variants"][0]["processing"]["tail_fade"]["truncated"]
        )
        self.assertIn(
            "fade_truncated",
            [
                warning["code"]
                for warning in manifest_two["variants"][0]["analysis"]["warnings"]
            ],
        )
        self.assertEqual(manifest_two["recipe"]["bus"]["tail_fade_ms"], 400.0)
        self.assertNotEqual(
            manifest_one["variants"][0]["output"]["sha256"],
            manifest_two["variants"][0]["output"]["sha256"],
        )

        # Replay needs only the manifest and referenced sources, never the recipe.
        recipe_path.unlink()
        verification = engine.verify_manifest(output_one / "manifest.json")
        self.assertEqual(verification["verified_variants"], 3)
        self.assertEqual(result_one["run_id"], manifest_one["run_id"])

    def test_all_output_encodings_end_to_end(self) -> None:
        cases = (
            (44100, 1, 16),
            (48000, 2, 24),
            (44100, 2, 32),
            (48000, 1, "float32"),
        )
        for index, (sample_rate, channels, bit_depth) in enumerate(cases):
            with self.subTest(
                sample_rate=sample_rate,
                channels=channels,
                bit_depth=bit_depth,
            ):
                recipe = _base_recipe(self.sources, duration_ms=300.0)
                recipe["skill_name"] = f"Encoding {index}"
                recipe["variant_count"] = 1
                recipe["output"] = {
                    "sample_rate": sample_rate,
                    "bit_depth": bit_depth,
                    "channels": channels,
                    "duration_ms": 300.0,
                }
                recipe["events"] = [
                    {
                        "role": "slash",
                        "source": str(self.sources / "slash_44k_mono.wav"),
                        "offset_ms": 10,
                        "duration_ms": 120,
                        "gain": 0.8,
                        "channel_map": (
                            "mono_to_stereo" if channels == 2 else "identity"
                        ),
                    }
                ]
                recipe["randomization"]["gain_jitter_db"] = 0.0
                recipe["randomization"]["time_jitter_ms"] = 0.0
                recipe["randomization"]["tail_harmonics"]["enabled"] = False
                recipe["bus_by_level"] = {"1": recipe["bus_by_level"]["1"]}
                recipe_path = self.root / f"encoding_{index}.json"
                _write_json(recipe_path, recipe)
                output = self.root / f"encoding_{index}"
                engine.generate_batch(
                    recipe_path,
                    level=1,
                    output_directory=output,
                    workers=1,
                )
                engine.verify_manifest(output / "manifest.json")
                manifest = json.loads(
                    (output / "manifest.json").read_text(encoding="utf-8")
                )
                variant = manifest["variants"][0]
                _, info = verify_wav(
                    output / variant["output"]["path"],
                    sample_rate=sample_rate,
                    channels=channels,
                    bit_depth=bit_depth,
                    frame_count=int(300 * sample_rate / 1000),
                )
                self.assertEqual(info.sample_rate, sample_rate)


class FailureHandlingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.sources = cls.root / "sources"
        synthesize_sources(cls.sources)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_worker_failure_removes_entire_stage_and_never_publishes(self) -> None:
        recipe = _base_recipe(self.sources)
        recipe_path = self.root / "worker_failure.json"
        _write_json(recipe_path, recipe)
        output = self.root / "worker_failure_output"
        original = engine._generate_one_variant

        def flaky(**kwargs):
            if kwargs["variant_index"] == 1:
                raise SkillSFXError(
                    "SFX-E3001",
                    "simulated worker failure",
                    {
                        "variant_index": 1,
                        "event_index": 2,
                        "source_path": str(
                            self.sources / "tail_48k_stereo.wav"
                        ),
                    },
                )
            return original(**kwargs)

        with mock.patch(
            "skillsfx.engine._generate_one_variant",
            side_effect=flaky,
        ):
            with self.assertRaises(SkillSFXError) as caught:
                engine.generate_batch(
                    recipe_path,
                    level=1,
                    output_directory=output,
                    workers=1,
                )
        self.assertEqual(caught.exception.code, "SFX-E3001")
        self.assertTrue(caught.exception.context["batch_failed"])
        self.assertFalse(output.exists())
        stages = list(self.root.glob(".worker_failure_output.stage-*"))
        self.assertEqual(stages, [])

    def test_cli_recipe_failure_emits_stderr_code_and_no_output(self) -> None:
        recipe = _base_recipe(self.sources)
        recipe["events"][0]["offset_ms"] = -5.0
        recipe_path = self.root / "cli_bad_recipe.json"
        _write_json(recipe_path, recipe)
        output = self.root / "cli_bad_output"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "skillsfx.py"),
                "generate",
                "--recipe",
                str(recipe_path),
                "--level",
                "1",
                "--output",
                str(output),
                "--workers",
                "2",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        error = json.loads(completed.stderr)
        self.assertEqual(error["error_code"], "SFX-E2003")
        self.assertEqual(error["context"]["event_index"], 0)
        self.assertIn("slash_44k_mono.wav", error["context"]["source_path"])
        self.assertFalse(output.exists())

    def test_manifest_tamper_detection(self) -> None:
        recipe = _base_recipe(self.sources, duration_ms=300.0)
        recipe["variant_count"] = 1
        recipe["events"] = [recipe["events"][0]]
        recipe_path = self.root / "tamper.json"
        _write_json(recipe_path, recipe)
        output = self.root / "tamper_output"
        engine.generate_batch(
            recipe_path,
            level=1,
            output_directory=output,
            workers=1,
        )
        manifest = json.loads(
            (output / "manifest.json").read_text(encoding="utf-8")
        )
        output_path = output / manifest["variants"][0]["output"]["path"]
        with output_path.open("r+b") as handle:
            handle.seek(60)
            original = handle.read(1)
            handle.seek(60)
            handle.write(bytes([original[0] ^ 0x01]))
        with self.assertRaises(SkillSFXError) as caught:
            engine.verify_manifest(output / "manifest.json")
        self.assertEqual(caught.exception.code, "SFX-E5001")
        self.assertGreater(
            caught.exception.context["failure_count"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
