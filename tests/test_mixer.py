from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from mixer.config import load_project_config, ms_to_samples
from mixer.envelope import crossfade_envelopes
from mixer.engine import mix_files
from mixer.errors import ErrorCode
from mixer.samples import generate_sample
from mixer.timeline import validate_and_order

ROOT = Path(__file__).resolve().parents[1]


class MixerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self._counter = 0

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _name(self, stem: str) -> Path:
        self._counter += 1
        return self.root / f"{self._counter:03d}_{stem}"

    def _json(self, payload: object, name: str = "data.json") -> Path:
        path = self._name(name)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _config(self, shots: list[dict], **extra: object) -> Path:
        payload: dict[str, object] = {
            "target_sample_rate": 48_000,
            "shots": shots,
        }
        payload.update(extra)
        return self._json(payload, "shots.json")

    def _choices(self, shot_ids: list[str]) -> Path:
        return self._json({"path": shot_ids}, "choices.json")

    def _run_cli(
        self,
        config: Path,
        choices: Path,
        output: Path,
        report: Path,
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mixer",
                "--config",
                str(config),
                "--path",
                str(choices),
                "--out",
                str(output),
                "--report",
                str(report),
            ],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if check and result.returncode != 0:
            self.fail(f"CLI failed: {result.stderr}")
        return result

    def test_resample_downmix_alignment_and_report_fields(self) -> None:
        media = self._name("stereo_44k.wav")
        generate_sample(
            media,
            duration_ms=1234,
            sample_rate=44_100,
            channels=2,
            amplitude=0.2,
            track_type="dialogue",
        )
        config = self._config(
            [
                {
                    "id": "s1",
                    "start_ms": 0,
                    "end_ms": 1234,
                    "tracks": [
                        {
                            "type": "dialogue",
                            "path": media.name,
                            "gain_db": 0,
                            "fade_in_ms": 0,
                            "fade_out_ms": 0,
                        }
                    ],
                }
            ]
        )
        choices = self._choices(["s1"])
        output = self._name("mix.wav")
        report_path = self._name("report.json")
        report = mix_files(config, choices, output, report_path)

        expected_samples = ms_to_samples(1234, 48_000)
        info = sf.info(str(output))
        self.assertEqual(info.frames, expected_samples)
        self.assertEqual(info.samplerate, 48_000)
        self.assertEqual(info.channels, 4)
        audio, _ = sf.read(str(output), dtype="float64", always_2d=True)
        np.testing.assert_allclose(audio[:, 0], audio[:, 3], atol=1e-4)
        np.testing.assert_allclose(audio[:, 1], 0.0, atol=1e-4)
        np.testing.assert_allclose(audio[:, 2], 0.0, atol=1e-4)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["target_sample_rate"], 48_000)
        self.assertEqual(report["total_samples"], expected_samples)
        shot = report["shots"][0]
        self.assertEqual(shot["shot_id"], "s1")
        self.assertEqual(shot["start_sample"], 0)
        self.assertEqual(shot["end_sample"], expected_samples)
        self.assertEqual(shot["duration_samples"], expected_samples)
        self.assertAlmostEqual(
            shot["peak_dbfs"],
            20.0 * math.log10(shot["raw_peak"]),
            places=12,
        )
        self.assertEqual(report["bus_weights"], {
            "dialogue": 1.0,
            "ambience": 0.7,
            "music": 0.5,
        })

    def test_overlap_uses_sample_exact_equal_power_crossfade(self) -> None:
        media = self._name("constant.wav")
        generate_sample(
            media,
            duration_ms=1000,
            sample_rate=48_000,
            channels=1,
            amplitude=0.4,
            waveform="constant",
        )
        track = {
            "type": "dialogue",
            "path": media.name,
            "gain_db": 0,
            "fade_in_ms": 0,
            "fade_out_ms": 0,
        }
        config_path = self._config(
            [
                {
                    "id": "s1",
                    "start_ms": 0,
                    "end_ms": 1000,
                    "tracks": [dict(track)],
                },
                {
                    "id": "s2",
                    "start_ms": 900,
                    "end_ms": 1900,
                    "allow_overlap": True,
                    "crossfade_ms": 100,
                    "tracks": [dict(track)],
                },
            ],
            output_subtype="FLOAT",
        )
        choices = self._choices(["s1", "s2"])
        output = self._name("overlap.wav")
        report_path = self._name("overlap.json")
        report = mix_files(config_path, choices, output, report_path)

        self.assertEqual(report["total_samples"], 91_200)
        self.assertEqual(report["shots"][0]["start_sample"], 0)
        self.assertEqual(report["shots"][0]["end_sample"], 48_000)
        self.assertEqual(report["shots"][1]["start_sample"], 43_200)
        self.assertEqual(report["shots"][1]["end_sample"], 91_200)
        self.assertAlmostEqual(report["shots"][0]["raw_peak"], 0.4, places=3)
        self.assertAlmostEqual(report["shots"][1]["raw_peak"], 0.4, places=3)

        project = load_project_config(config_path)
        ordered, overlaps = validate_and_order(
            [project.shots_by_id["s1"], project.shots_by_id["s2"]]
        )
        envelopes = crossfade_envelopes(ordered, overlaps)
        self.assertEqual(len(overlaps), 1)
        overlap = overlaps[0]
        self.assertEqual(overlap.start_sample, 43_200)
        self.assertEqual(overlap.duration_samples, 4_800)
        relative = overlap.start_sample - ordered[0].start_sample
        earlier = envelopes["s1"][relative : relative + 4_800]
        later = envelopes["s2"][:4_800]
        np.testing.assert_allclose(earlier**2 + later**2, 1.0, atol=1e-12)

        audio, rate = sf.read(str(output), dtype="float64", always_2d=True)
        self.assertEqual(rate, 48_000)
        self.assertEqual(audio.shape, (91_200, 4))
        np.testing.assert_allclose(audio[:, 0], audio[:, 3])
        self.assertAlmostEqual(
            float(np.max(np.abs(audio[0:40_000, 0]))), 0.4, places=3
        )
        self.assertAlmostEqual(
            float(np.max(np.abs(audio[50_000:, 0]))), 0.4, places=3
        )

    def test_gap_and_last_shot_determine_total_duration(self) -> None:
        media = self._name("tone.wav")
        generate_sample(
            media,
            duration_ms=500,
            sample_rate=48_000,
            channels=1,
            amplitude=0.1,
            track_type="ambience",
        )
        track = {"type": "ambience", "path": media.name, "gain_db": 0}
        config = self._config(
            [
                {"id": "a", "start_ms": 0, "end_ms": 500, "tracks": [dict(track)]},
                {"id": "b", "start_ms": 1000, "end_ms": 1800, "tracks": [dict(track)]},
            ]
        )
        choices = self._choices(["a", "b"])
        output = self._name("gap.wav")
        report_path = self._name("gap.json")
        report = mix_files(config, choices, output, report_path)

        self.assertEqual(report["total_samples"], 86_400)
        self.assertEqual([item["start_sample"] for item in report["shots"]], [0, 48_000])
        self.assertEqual([item["duration_samples"] for item in report["shots"]], [24_000, 38_400])
        with sf.SoundFile(str(output)) as handle:
            self.assertEqual(handle.frames, 86_400)

    def test_limiter_keeps_master_at_or_below_zero_dbfs(self) -> None:
        media = self._name("loud.wav")
        generate_sample(
            media,
            duration_ms=1000,
            sample_rate=48_000,
            channels=1,
            amplitude=0.75,
            waveform="constant",
        )
        tracks = [
            {"type": kind, "path": media.name, "gain_db": 0}
            for kind in ("dialogue", "ambience", "music")
        ]
        config = self._config(
            [{"id": "loud", "start_ms": 0, "end_ms": 1000, "tracks": tracks}],
            output_subtype="FLOAT",
        )
        choices = self._choices(["loud"])
        output = self._name("limited.wav")
        report_path = self._name("limited.json")
        report = mix_files(config, choices, output, report_path)

        self.assertTrue(report["limiter_active"])
        self.assertAlmostEqual(report["limiter_gain"], 1.0 / 1.65, places=12)
        self.assertAlmostEqual(report["master_peak_dbfs"], 0.0, places=12)
        self.assertAlmostEqual(report["shots"][0]["raw_peak"], 0.75, places=4)
        for key in ("dialogue", "ambience", "music"):
            self.assertAlmostEqual(
                report["bus_peak_dbfs"][key], -2.498774732935226, places=8
            )
        audio, _ = sf.read(str(output), dtype="float64", always_2d=True)
        self.assertLessEqual(float(np.max(np.abs(audio))), 1.0)
        np.testing.assert_allclose(audio[:, 0], 0.75, atol=1e-6)
        np.testing.assert_allclose(audio[:, 1], 0.75, atol=1e-6)
        np.testing.assert_allclose(audio[:, 2], 0.75, atol=1e-6)
        np.testing.assert_allclose(audio[:, 3], 1.0, atol=1e-6)

    def test_timeline_conflict_preserves_partial_report(self) -> None:
        media = self._name("conflict.wav")
        generate_sample(
            media,
            duration_ms=1000,
            sample_rate=48_000,
            channels=1,
            amplitude=0.1,
        )
        track = {"type": "dialogue", "path": media.name}
        config = self._config(
            [
                {"id": "one", "start_ms": 0, "end_ms": 1000, "tracks": [dict(track)]},
                {"id": "two", "start_ms": 800, "end_ms": 1600, "tracks": [dict(track)]},
            ]
        )
        choices = self._choices(["one", "two"])
        output = self._name("conflict.wav")
        report_path = self._name("conflict.json")
        result = self._run_cli(config, choices, output, report_path, check=False)

        self.assertEqual(result.returncode, 2)
        self.assertIn(ErrorCode.E_TIMELINE_CONFLICT.value, result.stderr)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["error_code"], ErrorCode.E_TIMELINE_CONFLICT.value)
        self.assertEqual([shot["shot_id"] for shot in report["shots"]], ["one"])

    def test_unsupported_media_rate_and_format_report_media_load(self) -> None:
        bad_rate = self._name("rate_96k.wav")
        generate_sample(
            bad_rate,
            duration_ms=200,
            sample_rate=96_000,
            channels=1,
            amplitude=0.1,
        )
        bad_format = self._name("not_audio.wav")
        bad_format.write_text("this is not an audio container", encoding="utf-8")

        for media in (bad_rate, bad_format):
            with self.subTest(media=media.name):
                config = self._config(
                    [
                        {
                            "id": "bad",
                            "start_ms": 0,
                            "end_ms": 200,
                            "tracks": [
                                {"type": "dialogue", "path": media.name}
                            ],
                        }
                    ]
                )
                choices = self._choices(["bad"])
                output = self._name("bad-media.wav")
                report_path = self._name("bad-media.json")
                result = self._run_cli(config, choices, output, report_path, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn(ErrorCode.E_MEDIA_LOAD.value, result.stderr)
                report = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertEqual(report["error_code"], ErrorCode.E_MEDIA_LOAD.value)

    def test_all_business_error_codes_and_error_reports(self) -> None:
        media = self._name("errors.wav")
        generate_sample(
            media,
            duration_ms=1000,
            sample_rate=48_000,
            channels=1,
            amplitude=0.1,
        )
        common_track = {"type": "dialogue", "path": media.name}

        scenarios = {
            "E_SHOT_DESC": (
                self._config(
                    [{"id": "bad", "start_ms": 400, "end_ms": 400, "tracks": [dict(common_track)]}]
                ),
                self._choices(["bad"]),
            ),
            "E_MEDIA_LOAD": (
                self._config(
                    [
                        {
                            "id": "missing",
                            "start_ms": 0,
                            "end_ms": 1000,
                            "tracks": [
                                {"type": "dialogue", "path": "does-not-exist.wav"}
                            ],
                        }
                    ]
                ),
                self._choices(["missing"]),
            ),
            "E_PATH_INVALID": (
                self._config(
                    [{"id": "ok", "start_ms": 0, "end_ms": 1000, "tracks": [dict(common_track)]}]
                ),
                self._choices(["unknown-shot"]),
            ),
            "E_FADE_OVERFLOW": (
                self._config(
                    [
                        {"id": "valid", "start_ms": 0, "end_ms": 1000, "tracks": [dict(common_track)]},
                        {
                            "id": "fade",
                            "start_ms": 1000,
                            "end_ms": 2000,
                            "tracks": [
                                {
                                    "type": "dialogue",
                                    "path": media.name,
                                    "fade_in_ms": 800,
                                    "fade_out_ms": 300,
                                }
                            ],
                        },
                    ]
                ),
                self._choices(["valid", "fade"]),
            ),
        }

        for expected_code, (config, choices) in scenarios.items():
            with self.subTest(code=expected_code):
                output = self._name(f"{expected_code}.wav")
                report_path = self._name(f"{expected_code}.json")
                result = self._run_cli(config, choices, output, report_path, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected_code, result.stderr)
                report = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "error")
                self.assertEqual(report["error_code"], expected_code)
                self.assertIn("error_message", report)
                if expected_code == "E_FADE_OVERFLOW":
                    self.assertEqual(
                        [shot["shot_id"] for shot in report["shots"]], ["valid"]
                    )

    def test_outputs_are_byte_deterministic_and_concurrency_safe(self) -> None:
        media = self._name("det.wav")
        generate_sample(
            media,
            duration_ms=750,
            sample_rate=48_000,
            channels=2,
            amplitude=0.15,
        )
        config = self._config(
            [
                {
                    "id": "det",
                    "start_ms": 0,
                    "end_ms": 750,
                    "tracks": [
                        {"type": "music", "path": media.name, "gain_db": -2.0}
                    ],
                }
            ]
        )
        choices = self._choices(["det"])

        first_output = self._name("first.wav")
        first_report = self._name("first.json")
        second_output = self._name("second.wav")
        second_report = self._name("second.json")
        self._run_cli(config, choices, first_output, first_report)
        self._run_cli(config, choices, second_output, second_report)

        self.assertEqual(
            hashlib.sha256(first_output.read_bytes()).digest(),
            hashlib.sha256(second_output.read_bytes()).digest(),
        )
        self.assertEqual(
            hashlib.sha256(first_report.read_bytes()).digest(),
            hashlib.sha256(second_report.read_bytes()).digest(),
        )

        shared_output = self._name("shared.wav")
        shared_report = self._name("shared.json")
        processes = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "mixer",
                    "--config",
                    str(config),
                    "--path",
                    str(choices),
                    "--out",
                    str(shared_output),
                    "--report",
                    str(shared_report),
                ],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(4)
        ]
        for process in processes:
            _, stderr = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 0, stderr)
        self.assertEqual(
            hashlib.sha256(shared_output.read_bytes()).digest(),
            hashlib.sha256(first_output.read_bytes()).digest(),
        )
        self.assertEqual(
            hashlib.sha256(shared_report.read_bytes()).digest(),
            hashlib.sha256(first_report.read_bytes()).digest(),
        )
        self.assertEqual(json.loads(shared_report.read_text())["status"], "ok")


if __name__ == "__main__":
    unittest.main()
