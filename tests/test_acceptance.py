"""End-to-end acceptance tests for the deterministic post-production CLI."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from postaudio.automation import build_gain_curve
from postaudio.dsp import (
    apply_lookahead_limiter,
    detect_sync_marker,
    integrated_loudness,
    resample_polyphase,
)
from postaudio.generate import _track_signal
from postaudio.util import db_to_linear, linear_to_db, round_half_even, round_half_even_array
from postaudio.wavio import read_wav, read_wav_header, write_wav


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "postaudio.py"), *map(str, args)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def generate(scenario: str, root: Path) -> Path:
    if not root.exists():
        result = run_cli("generate", "--scenario", scenario, "--out", str(root))
        if result.returncode != 0:
            raise AssertionError(result.stderr)
    return root


def load_report(project: Path) -> dict:
    with open(project / "out" / "report.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def truncate(path: Path, count: int = 64) -> None:
    raw = path.read_bytes()
    path.write_bytes(raw[:-count])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="postaudio-test-")
        self.base = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_01_header_validation(self) -> None:
        project = generate("validate_sr_bad", self.base / "bad_sr")
        result = run_cli("validate", str(project))
        self.assertEqual(result.returncode, 1)
        self.assertIn("declared_48k_actual_44k1", result.stderr)
        self.assertIn("SAMPLE_RATE_MISMATCH", result.stderr)

        audio, _ = _track_signal(
            48000, 1, 1.0, amplitude_dbfs=-9.0, sync_seconds=0.010
        )
        write_wav(str(project / "sources" / "declared_48k_actual_44k1.wav"), audio, 48000, "pcm16")
        fixed = run_cli("validate", str(project))
        self.assertEqual(fixed.returncode, 0, fixed.stderr)
        summary = json.loads(fixed.stdout)
        self.assertEqual(summary["track_count"], 1)
        with open(project / "manifest.json", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertEqual(summary["track_count"], len(manifest["tracks"]))

        channels = generate("validate_channels_bad", self.base / "bad_channels")
        result = run_cli("validate", str(channels))
        self.assertEqual(result.returncode, 1)
        self.assertIn("declared_stereo_actual_mono", result.stderr)
        self.assertIn("CHANNEL_COUNT_MISMATCH", result.stderr)

    def test_02_resampling_and_fft(self) -> None:
        project = generate("resample", self.base / "resample")
        result = run_cli("render", str(project))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = load_report(project)
        by_name = {track["name"]: track for track in report["tracks"]}
        source = by_name["sine_44k1_mono"]
        bypass = by_name["sine_48k_stereo"]
        self.assertEqual(source["input_sr"], 44100)
        self.assertEqual(source["output_sr"], 48000)
        self.assertTrue(source["resampled"])
        self.assertEqual(source["resampled_samples"], 48000)
        self.assertEqual(source["rational_ratio"], "160/147")
        self.assertFalse(bypass["resampled"])
        self.assertEqual(bypass["input_sr"], 48000)
        output_header = read_wav_header(str(project / "out" / "output.wav"))
        self.assertEqual(output_header.sample_rate, 48000)
        self.assertEqual(output_header.frame_count, 48000)

        samples, header = read_wav(str(project / "sources" / "sine_44k1_mono.wav"))
        converted, _ = resample_polyphase(samples, 44100, 48000)
        segment = converted[2400:2400 + 1024, 0] * np.hanning(1024)
        spectrum = np.abs(np.fft.rfft(segment))
        frequencies = np.fft.rfftfreq(1024, 1.0 / 48000)
        main = spectrum[(frequencies > 900) & (frequencies < 1100)].max()
        image = spectrum[(frequencies > 4700) & (frequencies < 5100)].max()
        self.assertGreater(main, image)
        self.assertLess(20.0 * math.log10(image / main), -60.0)

    def test_03_sync_alignment(self) -> None:
        project = generate("sync", self.base / "sync")
        result = run_cli("render", str(project))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = load_report(project)
        shifted = next(track for track in report["tracks"] if track["name"] == "sync_shifted")
        self.assertEqual(shifted["offset_samples"], 37)
        self.assertEqual(shifted["offset_samples_output"], 37)
        output, _ = read_wav(str(project / "out" / "output.wav"))
        detected = detect_sync_marker(output, 48000, 480, {"amplitude": 0.1}, 96)
        self.assertTrue(detected["detected"])
        self.assertLessEqual(abs(detected["actual_position"] - 480), 1)

        multirate = generate("sync_multirate", self.base / "sync_multirate")
        result = run_cli("render", str(multirate))
        self.assertEqual(result.returncode, 0, result.stderr)
        multirate_report = load_report(multirate)
        shifted_multi = next(
            track for track in multirate_report["tracks"] if track["name"] == "sync_shifted"
        )
        self.assertEqual(shifted_multi["offset_samples"], 37)
        self.assertEqual(shifted_multi["offset_samples_output"], 40)
        self.assertTrue(shifted_multi["resampled"])
        output_multi, _ = read_wav(str(multirate / "out" / "output.wav"))
        marker_multi = detect_sync_marker(output_multi, 48000, 480, {"amplitude": 0.1}, 96)
        self.assertLessEqual(abs(marker_multi["actual_position"] - 480), 1)

        far = generate("sync_far", self.base / "sync_far")
        result = run_cli("render", str(far))
        self.assertEqual(result.returncode, 0)
        self.assertIn("ALIGNMENT_WARNING", result.stderr)
        far_report = load_report(far)
        self.assertEqual(far_report["tracks"][1]["offset_samples"], 200)

        missing = generate("sync_missing", self.base / "sync_missing")
        result = run_cli("render", str(missing))
        self.assertEqual(result.returncode, 0)
        self.assertIn("SYNC_MARKER_NOT_FOUND", result.stderr)
        self.assertEqual(load_report(missing)["tracks"][0]["offset_samples"], 0)

    def test_04_gain_automation_and_validation(self) -> None:
        self.assertEqual(round_half_even(0.5), 0)
        self.assertEqual(round_half_even(1.5), 2)
        self.assertEqual(round_half_even(2.5), 2)
        self.assertEqual(round_half_even(3.5), 4)
        self.assertEqual(
            round_half_even_array(np.array([-1.5, -0.5, -0.2, 0.5, 1.5, 2.5])).tolist(),
            [-2, 0, 0, 0, 2, 2],
        )
        project = generate("gain_ramp", self.base / "gain")
        result = run_cli("render", str(project))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = load_report(project)
        for track in report["tracks"]:
            nodes = track["gain_automation"]
            self.assertAlmostEqual(nodes["min_gain_db"], -12.0, places=2)
            self.assertAlmostEqual(nodes["max_gain_db"], 0.0, places=2)
            self.assertEqual(nodes["time_to_sample_method"].startswith("round-half-to-even"), True)
        up_nodes = [{"time": 0.0, "gain_db": -12.0}, {"time": 1.0, "gain_db": 0.0}]
        up_gain, _ = build_gain_curve(up_nodes, 48000, 48000, 10.0)
        self.assertAlmostEqual(linear_to_db(float(up_gain[4800])), -10.8, places=2)
        self.assertAlmostEqual(linear_to_db(float(up_gain[43200])), -1.2, places=2)
        down_nodes = [{"time": 0.0, "gain_db": 0.0}, {"time": 1.0, "gain_db": -12.0}]
        down_gain, _ = build_gain_curve(down_nodes, 48000, 48000, 10.0)
        self.assertAlmostEqual(linear_to_db(float(down_gain[4800])), -1.2, places=2)
        self.assertAlmostEqual(linear_to_db(float(down_gain[43200])), -10.8, places=2)
        mix, _ = read_wav(str(project / "out" / "output.wav"))
        middle = mix[24000 - 2400:24000 + 2400, 0]
        self.assertLess(float(np.max(np.abs(middle))), 0.14)
        self.assertGreater(float(np.max(np.abs(middle))), 0.08)

        bad_order = generate("gain_bad_order", self.base / "bad_order")
        result = run_cli("render", str(bad_order))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("AUTOMATION_TIME_ORDER", result.stderr)
        self.assertIn("node_index=2", result.stderr)

        out_of_range = generate("gain_out_of_range", self.base / "out_of_range")
        result = run_cli("render", str(out_of_range))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GAIN_OUT_OF_RANGE", result.stderr)
        self.assertIn("node_index=1", result.stderr)

    def test_05_smoothing(self) -> None:
        project = generate("smooth_step", self.base / "smooth_step")
        result = run_cli("render", str(project))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = load_report(project)
        self.assertEqual(report["tracks"][0]["smooth_steps"], 1)
        samples, _ = read_wav(str(project / "out" / "output.wav"))
        center = 24000
        local_diffs = np.abs(np.diff(samples[center - 300:center + 300, 0]))
        self.assertLess(float(np.max(local_diffs)), 0.05)

        gradient = generate("smooth_gradient", self.base / "smooth_gradient")
        result = run_cli("render", str(gradient))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(load_report(gradient)["tracks"][0]["smooth_steps"], 0)

        clamped = generate("smooth_clamped", self.base / "smooth_clamped")
        result = run_cli("render", str(clamped))
        self.assertEqual(result.returncode, 0)
        self.assertIn("SMOOTH_WINDOW_CLAMPED", result.stderr)

    def test_06_peak_limiter(self) -> None:
        project = generate("limiter_loud", self.base / "limiter_loud")
        result = run_cli("render", str(project))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = load_report(project)
        output, _ = read_wav(str(project / "out" / "output.wav"))
        output_peak = linear_to_db(float(np.max(np.abs(output))))
        self.assertLessEqual(output_peak, -1.0 + 0.01)
        self.assertGreater(report["peak_limiter"]["limited_samples"], 0)
        self.assertGreater(report["peak_limiter"]["max_attenuation_db"], 0.0)
        marker = detect_sync_marker(output, 48000, 480, {"amplitude": 0.1}, 96)
        self.assertLessEqual(abs(marker["actual_position"] - 480), 1)

        safe = generate("limiter_safe", self.base / "limiter_safe")
        result = run_cli("render", str(safe))
        self.assertEqual(result.returncode, 0)
        safe_report = load_report(safe)
        self.assertTrue(safe_report["peak_limiter"]["bypassed"])
        self.assertEqual(safe_report["peak_limiter"]["limited_samples"], 0)
        source, _ = read_wav(str(safe / "sources" / "hot_sine.wav"))
        signal = source[:48000] * db_to_linear(-10.0)
        limited, stats = apply_lookahead_limiter(signal, db_to_linear(-1.0), 240)
        self.assertEqual(signal.tobytes(), limited.tobytes())
        self.assertEqual(stats["limited_samples"], 0)

    def test_07_loudness(self) -> None:
        tone = np.sin(2.0 * np.pi * 1000.0 * np.arange(96000) / 48000.0) * 0.5
        reference = integrated_loudness(tone[:, None], 48000)
        self.assertIsNotNone(reference)
        self.assertLessEqual(abs(float(reference) - (-9.024)), 0.05)
        pink = generate("loudness_pink", self.base / "pink")
        result = run_cli("render", str(pink))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = load_report(pink)
        self.assertEqual(abs(report["integrated_loudness_lufs"] - (-16.0)) <= 0.5, True)
        expected_gain = -16.0 - report["loudness_normalization"]["input_loudness_lufs"]
        self.assertAlmostEqual(report["normalization_gain_db"], expected_gain, places=4)
        self.assertLess(report["normalization_gain_db"], 4.0)
        self.assertGreater(report["normalization_gain_db"], 0.0)

        conflict = generate("loudness_conflict", self.base / "conflict")
        result = run_cli("render", str(conflict))
        self.assertEqual(result.returncode, 0, result.stderr)
        conflict_report = load_report(conflict)
        self.assertIn("NORMALIZATION_PEAK_CONFLICT", result.stderr)
        self.assertTrue(conflict_report["loudness_normalization"]["peak_conflict"])
        output, _ = read_wav(str(conflict / "out" / "output.wav"))
        self.assertLessEqual(linear_to_db(float(np.max(np.abs(output)))), -1.0 + 0.01)

        equal = generate("loudness_equal", self.base / "equal")
        result = run_cli("render", str(equal))
        self.assertEqual(result.returncode, 0, result.stderr)
        equal_report = load_report(equal)
        self.assertEqual(equal_report["normalization_gain_db"], 0.0)
        self.assertFalse(equal_report["loudness_normalization"]["peak_conflict"])
        self.assertEqual(equal_report["warnings"], [])

    def test_08_output_formats_and_downmix(self) -> None:
        project = generate("resample", self.base / "formats")
        base_config = json.loads((project / "output_config.json").read_text(encoding="utf-8"))
        paths = {}
        reports = {}
        for encoding in ("float32", "pcm16", "pcm24"):
            config = dict(base_config)
            config["encoding"] = encoding
            config_path = self.base / f"{encoding}.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = self.base / f"{encoding}.wav"
            report = self.base / f"{encoding}.report.json"
            result = run_cli(
                "render",
                str(project),
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--report",
                str(report),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            paths[encoding] = output
            reports[encoding] = json.loads(report.read_text(encoding="utf-8"))
        float_header = read_wav_header(str(paths["float32"]))
        pcm16_header = read_wav_header(str(paths["pcm16"]))
        pcm24_header = read_wav_header(str(paths["pcm24"]))
        self.assertEqual(
            (float_header.sample_rate, float_header.frame_count),
            (pcm16_header.sample_rate, pcm16_header.frame_count),
        )
        self.assertEqual(
            (float_header.sample_rate, float_header.frame_count),
            (pcm24_header.sample_rate, pcm24_header.frame_count),
        )
        float_samples, _ = read_wav(str(paths["float32"]))
        with wave.open(str(paths["pcm16"]), "rb") as handle:
            raw = handle.readframes(handle.getnframes())
            integer_samples = np.frombuffer(raw, dtype="<i2").astype(np.int64)
        expected = np.rint(float_samples[:, 0] * 32767.0).astype(np.int64)
        self.assertTrue(np.array_equal(integer_samples, expected))
        self.assertEqual(paths["pcm24"].stat().st_size, 44 + pcm24_header.frame_count * 3)
        raw24 = paths["pcm24"].read_bytes()[44:]
        packed = np.frombuffer(raw24, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        decoded24 = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        decoded24 = np.where(decoded24 & 0x800000, decoded24 - 0x1000000, decoded24)
        expected24 = np.rint(float_samples[:, 0] * 8388607.0).astype(np.int64)
        expected24 = np.where(expected24 < -8388608, -8388608, expected24)
        expected24 = np.where(expected24 > 8388607, 8388607, expected24)
        self.assertLessEqual(
            int(np.max(np.abs(decoded24.astype(np.int64) - expected24))), 1
        )
        self.assertTrue(reports["pcm16"]["tracks"][1]["downmix_applied"])
        self.assertEqual(reports["pcm16"]["output"]["channels"], 1)
        self.assertEqual(reports["pcm24"]["output"]["bit_depth"], 24)
        self.assertEqual(reports["float32"]["output"]["encoding"], "float32")

    def test_09_determinism_and_manifest_order(self) -> None:
        project = generate("resample", self.base / "determinism")
        result = run_cli("render", str(project))
        self.assertEqual(result.returncode, 0)
        first_wav = sha256(project / "out" / "output.wav")
        first_report = sha256(project / "out" / "report.json")
        result = run_cli("render", str(project))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(first_wav, sha256(project / "out" / "output.wav"))
        self.assertEqual(first_report, sha256(project / "out" / "report.json"))

        config = json.loads((project / "output_config.json").read_text(encoding="utf-8"))
        config["comment"] = "ignored metadata"
        (project / "output_config.json").write_text(json.dumps(config), encoding="utf-8")
        result = run_cli("render", str(project))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(first_wav, sha256(project / "out" / "output.wav"))
        self.assertEqual(first_report, sha256(project / "out" / "report.json"))

        manifest = json.loads((project / "manifest.json").read_text(encoding="utf-8"))
        original_manifest = (project / "manifest.json").read_bytes()
        manifest["tracks"] = list(reversed(manifest["tracks"]))
        (project / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = run_cli(
            "render",
            str(project),
            "--output",
            str(project / "out" / "reordered.wav"),
            "--report",
            str(project / "out" / "reordered.json"),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(first_wav, sha256(project / "out" / "reordered.wav"))
        (project / "manifest.json").write_bytes(original_manifest)

    def test_10_partial_and_total_failure(self) -> None:
        partial = generate("failure_partial", self.base / "partial")
        truncate(partial / "sources" / "mid_track.wav")
        result = run_cli("render", str(partial))
        self.assertEqual(result.returncode, 2, result.stderr)
        partial_report = load_report(partial)
        self.assertEqual(
            partial_report["errors"],
            [
                {
                    "track": "mid_track",
                    "code": "SOURCE_READ_ERROR",
                    "detail": "truncated data chunk: declares 96000 bytes, only 95936 available",
                }
            ],
        )
        self.assertTrue((partial / "out" / "output.wav").exists())
        self.assertEqual(len(partial_report["tracks"]), 2)

        all_fail = generate("failure_partial", self.base / "all_fail")
        for name in ("alpha_track", "mid_track", "zeta_track"):
            truncate(all_fail / "sources" / f"{name}.wav")
        result = run_cli("render", str(all_fail))
        self.assertEqual(result.returncode, 1)
        self.assertFalse((all_fail / "out" / "output.wav").exists())
        all_report = load_report(all_fail)
        self.assertEqual(list(all_report.keys()), ["errors"])
        self.assertEqual([error["track"] for error in all_report["errors"]], ["alpha_track", "mid_track", "zeta_track"])

        two_fail = generate("failure_partial", self.base / "two_fail")
        truncate(two_fail / "sources" / "zeta_track.wav")
        truncate(two_fail / "sources" / "alpha_track.wav")
        result = run_cli("render", str(two_fail))
        self.assertEqual(result.returncode, 2)
        two_report = load_report(two_fail)
        names = [error["track"] for error in two_report["errors"]]
        self.assertEqual(names, sorted(names))


if __name__ == "__main__":
    unittest.main()
