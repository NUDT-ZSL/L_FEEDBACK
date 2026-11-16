from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from offline_mixer.loudness import SignalAnalyzer
from offline_mixer.resample import SincResampler


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "mixer.py"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_sine(path, sample_rate, duration, frequency, amplitude, phase=0.0):
    time = np.arange(int(round(sample_rate * duration))) / sample_rate
    signal = amplitude * np.sin(2.0 * np.pi * frequency * time + phase)
    sf.write(str(path), np.stack((signal, signal), axis=1), sample_rate, subtype="FLOAT")


def run_cli(config_path, *extra):
    return subprocess.run(
        [sys.executable, str(CLI), str(config_path), *extra],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )


def parse_error(result):
    lines = [line for line in result.stderr.splitlines() if line.strip()]
    return json.loads(lines[-1])


class ResamplerTests(unittest.TestCase):
    def test_windows_sinc_snr_at_common_rates(self):
        for source_rate in (44100, 96000):
            with self.subTest(source_rate=source_rate):
                target_rate = 48000
                duration = 1.0
                frequency = 1000.0
                source = np.sin(
                    2.0
                    * np.pi
                    * frequency
                    * np.arange(int(source_rate * duration))
                    / source_rate
                )[:, None]
                output_frames = int(len(source) * target_rate / source_rate + 0.5)
                converter = SincResampler(
                    source_rate, target_rate, 1, output_frames
                )
                blocks = []
                for offset in range(0, len(source), 733):
                    blocks.append(converter.feed(source[offset:offset + 733]))
                blocks.append(converter.finish())
                output = np.concatenate(blocks)[:, 0]
                reference = np.sin(
                    2.0
                    * np.pi
                    * frequency
                    * np.arange(len(output))
                    / target_rate
                )
                interior = slice(128, -128)
                error = output[interior] - reference[interior]
                snr = 10.0 * np.log10(
                    np.sum(reference[interior] ** 2) / np.sum(error ** 2)
                )
                self.assertGreaterEqual(snr, 90.0, "SNR was {:.2f} dB".format(snr))


class LoudnessReferenceTests(unittest.TestCase):
    def test_integrated_loudness_matches_bs1770_reference(self):
        # Independent reference value produced by pyloudnorm for the exact
        # deterministic signal below: -12.289619170198348 LUFS.
        sample_rate = 48000
        time = np.arange(sample_rate * 7) / sample_rate
        signal = np.stack(
            (
                0.25 * np.sin(2.0 * np.pi * 1000.0 * time),
                0.25 * np.sin(2.0 * np.pi * 750.0 * time + 0.3),
            ),
            axis=1,
        )
        analyzer = SignalAnalyzer(sample_rate, 2)
        for offset in range(0, len(signal), 5555):
            analyzer.feed(signal[offset:offset + 5555])
        measured = analyzer.result()["integrated_lufs"]
        self.assertLessEqual(abs(measured - (-12.289619170198348)), 0.5)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mixer_test_")
        self.base = Path(self.temp.name)
        write_sine(self.base / "dialogue.wav", 44100, 2.0, 997.0, 0.45)
        write_sine(self.base / "sfx.wav", 48000, 1.5, 300.0, 0.35)
        write_sine(self.base / "music.wav", 96000, 2.2, 1200.0, 0.25, 0.4)

    def tearDown(self):
        self.temp.cleanup()

    def base_config(self, output_name):
        return {
            "output": {
                "path": output_name,
                "sample_rate": 48000,
                "channels": 2,
                "duration_seconds": 2.5,
                "limiter_placement": "bus",
            },
            "bus_limiter": {
                "ceiling_dbfs": -1.0,
                "lookahead_ms": 5.0,
                "release_ms": 80.0,
            },
            "processing": {"workers": 4, "chunk_frames": 8192},
            "tracks": [
                {
                    "id": "dialogue",
                    "input": "dialogue.wav",
                    "offset_ms": 0.0,
                    "gain_db": 0.0,
                    "compressor": {
                        "threshold_db": -24.0,
                        "ratio": 4.0,
                        "knee_db": 6.0,
                        "attack_ms": 8.0,
                        "release_ms": 120.0,
                    },
                },
                {
                    "id": "sfx",
                    "input": "sfx.wav",
                    "offset_ms": 123.4,
                    "gain_db": 10.0,
                },
                {
                    "id": "music",
                    "input": "music.wav",
                    "offset_ms": -50.0,
                    "gain_db": -1.0,
                },
            ],
        }

    def test_full_pipeline_limiter_parallel_equivalence_and_report(self):
        serial_config = self.base / "serial.json"
        parallel_config = self.base / "parallel.json"
        serial_config.write_text(json.dumps(self.base_config("serial.wav")), encoding="utf-8")
        parallel_config.write_text(
            json.dumps(self.base_config("parallel.wav")), encoding="utf-8"
        )
        serial = run_cli(serial_config, "--workers", "1")
        parallel = run_cli(parallel_config, "--workers", "4")
        self.assertEqual(serial.returncode, 0, serial.stderr)
        self.assertEqual(parallel.returncode, 0, parallel.stderr)
        serial_wav = self.base / "serial.wav"
        parallel_wav = self.base / "parallel.wav"
        self.assertEqual(sha256(serial_wav), sha256(parallel_wav))

        info = sf.info(str(parallel_wav))
        self.assertEqual(info.samplerate, 48000)
        self.assertEqual(info.channels, 2)
        self.assertEqual(info.frames, 120000)
        self.assertEqual(info.subtype, "PCM_32")

        report = json.loads(
            (self.base / "parallel.report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["track_count"], 3)
        self.assertEqual(
            {(track["id"], track["original_sample_rate"], track["target_sample_rate"])
             for track in report["tracks"]},
            {
                ("dialogue", 44100, 48000),
                ("sfx", 48000, 48000),
                ("music", 96000, 48000),
            },
        )
        self.assertTrue(
            any(
                warning["code"] == "W_OFFSET_NEGATIVE_TRIMMED"
                and warning.get("track") == "music"
                for warning in report["warnings"]
            )
        )
        self.assertLessEqual(
            report["loudness"]["true_peak_dbtp"],
            report["limiter"]["parameters"]["ceiling_dbfs"] + 1e-6,
        )
        self.assertGreater(report["limiter"]["gain_reduction"]["max_db"], 0.0)
        self.assertTrue(
            any(
                track["compressor"]
                and track["compressor"]["gain_reduction"]["max_db"] > 0.0
                for track in report["tracks"]
            )
        )
        self.assertGreater(report["memory"]["peak_working_set_bytes"], 0)
        self.assertGreater(report["memory"]["input_file_bytes"], 0)
        self.assertFalse(list(self.base.glob(".offline_mixer_*")))

        output, rate = sf.read(str(parallel_wav), dtype="float64", always_2d=True)
        analyzer = SignalAnalyzer(rate, output.shape[1])
        for offset in range(0, len(output), 4093):
            analyzer.feed(output[offset:offset + 4093])
        independent_file_measurement = analyzer.result()
        self.assertLessEqual(
            abs(
                independent_file_measurement["integrated_lufs"]
                - report["loudness"]["integrated_lufs_after_mix"]
            ),
            0.5,
        )

    def invalid_config(self, mutate):
        config = self.base_config("invalid.wav")
        mutate(config)
        path = self.base / "invalid.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def assert_business_error(self, config_path, code):
        result = run_cli(config_path, "--workers", "1")
        self.assertNotEqual(result.returncode, 0)
        payload = parse_error(result)
        self.assertEqual(payload["error"]["code"], code, payload)
        self.assertFalse((self.base / "invalid.wav").exists())
        self.assertFalse(list(self.base.glob(".offline_mixer_*")))

    def test_missing_input_error_and_cleanup(self):
        path = self.invalid_config(
            lambda cfg: cfg["tracks"][1].update({"input": "does_not_exist.wav"})
        )
        self.assert_business_error(path, "E_INPUT_MISSING")

    def test_invalid_compressor_ratio_has_specific_code(self):
        path = self.invalid_config(
            lambda cfg: cfg["tracks"][0]["compressor"].update({"ratio": 0.5})
        )
        self.assert_business_error(path, "E_INVALID_COMPRESSOR")

    def test_unsupported_input_rate(self):
        bad_rate = self.base / "bad_rate.wav"
        write_sine(bad_rate, 4000, 0.2, 500.0, 0.1)
        path = self.invalid_config(
            lambda cfg: cfg["tracks"][0].update({"input": "bad_rate.wav"})
        )
        self.assert_business_error(path, "E_UNSUPPORTED_RATE")

    def test_unsupported_format(self):
        (self.base / "not_audio.txt").write_text("not audio", encoding="utf-8")
        path = self.invalid_config(
            lambda cfg: cfg["tracks"][0].update({"input": "not_audio.txt"})
        )
        self.assert_business_error(path, "E_UNSUPPORTED_FORMAT")

    def test_missing_required_field(self):
        path = self.invalid_config(lambda cfg: cfg["output"].pop("sample_rate"))
        self.assert_business_error(path, "E_CONFIG_MISSING_FIELD")

    def test_inferred_duration_from_track_extents(self):
        config = self.base_config("inferred.wav")
        config["output"].pop("duration_seconds")
        config["tracks"] = [
            {
                "id": "one",
                "input": "sfx.wav",
                "offset_ms": 123.4,
                "gain_db": 0.0,
            }
        ]
        path = self.base / "inferred.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        result = run_cli(path, "--workers", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        info = sf.info(str(self.base / "inferred.wav"))
        self.assertEqual(info.frames, round(123.4 * 48) + 72000)

    def test_per_track_limiter_and_true_peak_ceiling(self):
        config = self.base_config("track_limited.wav")
        config["output"]["limiter_placement"] = "per_track"
        config["tracks"] = [
            {
                "id": "hot",
                "input": "sfx.wav",
                "offset_ms": 0.0,
                "gain_db": 12.0,
                "limiter": {
                    "ceiling_dbfs": -1.0,
                    "lookahead_ms": 5.0,
                    "release_ms": 80.0,
                },
            }
        ]
        path = self.base / "track_limited.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        result = run_cli(path, "--workers", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(
            (self.base / "track_limited.report.json").read_text(encoding="utf-8")
        )
        self.assertTrue(report["tracks"][0]["limiter"]["applied"])
        self.assertGreater(
            report["tracks"][0]["limiter"]["gain_reduction"]["max_db"], 0.0
        )
        self.assertFalse(report["limiter"]["applied"])
        self.assertLessEqual(report["loudness"]["true_peak_dbtp"], -1.0 + 1e-6)

    def test_malformed_json(self):
        path = self.base / "malformed.json"
        path.write_text('{"output": ', encoding="utf-8")
        result = run_cli(path)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(parse_error(result)["error"]["code"], "E_CONFIG_INVALID_JSON")


if __name__ == "__main__":
    unittest.main()
