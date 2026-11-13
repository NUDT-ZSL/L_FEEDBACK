from __future__ import annotations

import array
import hashlib
import math
import struct
import subprocess
import sys
import unittest
import wave

from replay_mixer.util import db_to_lin
from tests.helpers import (
    PROJECT_ROOT,
    make_project,
    parse_success,
    run_cli,
    sine_samples,
    temporary_workspace,
    zero_samples,
)


def _anchors(duration_ms: float) -> list[dict[str, float]]:
    return [
        {"replay_ms": 0.0, "source_ms": 0.0},
        {"replay_ms": duration_ms, "source_ms": duration_ms},
    ]


class FullExportTests(unittest.TestCase):
    def test_bundled_sample_generator_is_deterministic(self) -> None:
        with temporary_workspace("sample-") as root:
            first = root / "first"
            second = root / "second"
            for target in (first, second):
                result = subprocess.run(
                    [
                        sys.executable,
                        "tools/generate_sample.py",
                        "--output",
                        str(target),
                        "--sample-rate",
                        "16000",
                        "--duration-ms",
                        "1500",
                        "--force",
                    ],
                    cwd=PROJECT_ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            for relative in (
                "replay.json",
                "inputs/commentary_lead.wav",
                "inputs/commentary_analyst.wav",
                "inputs/engine.wav",
                "inputs/ambience.wav",
            ):
                self.assertEqual(
                    (first / relative).read_bytes(),
                    (second / relative).read_bytes(),
                )
            stats = parse_success(run_cli(first / "replay.json"))
            self.assertEqual(stats["output_frames"], 24000)
            self.assertEqual(stats["channels"], 2)
            self.assertTrue(all(row["trimmed_intervals"] >= 1 for row in stats["commentary"]))
            self.assertAlmostEqual(
                stats["normalization"]["final_rms_dbfs"],
                -20.0,
                delta=0.05,
            )

    def test_stats_and_wav_are_consistent(self) -> None:
        sample_rate = 8000
        duration_ms = 500.0
        frames = int(round(duration_ms * sample_rate / 1000.0))
        commentary = sine_samples(
            frames,
            sample_rate,
            frequency=220.0,
            amplitude=0.12,
            channels=1,
        )
        with temporary_workspace("full-") as root:
            config_path = make_project(
                root,
                sample_rate=sample_rate,
                duration_ms=duration_ms,
                output_channels=2,
                block_frames=256,
                target_rms_dbfs=-20.0,
                commentary=[
                    {
                        "name": "c",
                        "filename": "c.wav",
                        "channels": 1,
                        "samples": commentary,
                        "anchors": _anchors(duration_ms),
                        "gain_db": -2.0,
                        "pan": 0.0,
                        "threshold_dbfs": -50.0,
                        "min_duration_ms": 1000.0,
                    }
                ],
            )
            result = run_cli(config_path)
            stats = parse_success(result)
            self.assertTrue(stats["ok"])
            self.assertEqual(stats["output_frames"], frames)
            self.assertEqual(stats["channels"], 2)
            self.assertEqual(stats["bit_depth"], 16)
            self.assertEqual(stats["resumed_blocks"], 0)
            self.assertEqual(stats["commentary"][0]["trimmed_intervals"], 0)
            self.assertEqual(stats["commentary"][0]["effective_frames"], frames)
            self.assertEqual(stats["commentary"][0]["effective_samples"], frames * 2)

            output = root / "out" / "mix.wav"
            raw = output.read_bytes()
            self.assertEqual(raw[:4], b"RIFF")
            self.assertEqual(raw[8:12], b"WAVE")
            self.assertEqual(struct.unpack("<I", raw[4:8])[0], len(raw) - 8)
            self.assertEqual(raw[36:40], b"data")
            data_size = struct.unpack("<I", raw[40:44])[0]
            self.assertEqual(data_size, frames * 2 * 2)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), stats["output_sha256"])

            with wave.open(str(output), "rb") as wav:
                self.assertEqual(wav.getnchannels(), 2)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertEqual(wav.getframerate(), sample_rate)
                self.assertEqual(wav.getnframes(), frames)
                wav_channels = wav.getnchannels()
                wav_sample_width = wav.getsampwidth()
                wav_frames = wav.getnframes()
                pcm_bytes = wav.readframes(wav.getnframes())
            self.assertEqual(
                len(pcm_bytes),
                wav_frames * wav_channels * wav_sample_width,
            )
            pcm = array.array("h")
            pcm.frombytes(pcm_bytes)
            if sys.byteorder == "big":
                pcm.byteswap()
            left_peak = max(abs(value) for value in pcm[0::2])
            right_peak = max(abs(value) for value in pcm[1::2])
            self.assertGreater(left_peak, 0)
            self.assertAlmostEqual(left_peak, right_peak, delta=1)
            sum_squares = sum((value / 32767.0) ** 2 for value in pcm)
            rms = math.sqrt(sum_squares / len(pcm))
            peak = max(abs(value / 32767.0) for value in pcm)
            rms_db = 20.0 * math.log10(rms)
            peak_db = 20.0 * math.log10(peak)
            self.assertAlmostEqual(rms_db, -20.0, delta=0.03)
            self.assertAlmostEqual(
                stats["normalization"]["final_rms_dbfs"],
                rms_db,
                delta=1.0e-9,
            )
            self.assertAlmostEqual(
                stats["normalization"]["final_peak_dbfs"],
                peak_db,
                delta=1.0e-9,
            )
            self.assertLessEqual(peak, db_to_lin(-1.0) + 1.0 / 32767.0)
            self.assertFalse(stats["peak_limit_triggered"])

    def test_sparse_transient_triggers_peak_limited_normalization(self) -> None:
        sample_rate = 8000
        duration_ms = 500.0
        frames = int(round(duration_ms * sample_rate / 1000.0))
        impulse = zero_samples(frames)
        impulse[100] = 32767
        with temporary_workspace("peak-") as root:
            config_path = make_project(
                root,
                sample_rate=sample_rate,
                duration_ms=duration_ms,
                output_channels=1,
                block_frames=256,
                target_rms_dbfs=-20.0,
                max_gain_db=60.0,
                peak_limit_dbfs=-1.0,
                commentary=[
                    {
                        "name": "impulse",
                        "filename": "impulse.wav",
                        "channels": 1,
                        "samples": impulse,
                        "anchors": _anchors(duration_ms),
                        "gain_db": 0.0,
                        "pan": 0.0,
                        "threshold_dbfs": -160.0,
                        "min_duration_ms": 5000.0,
                        "window_ms": 10.0,
                    }
                ],
            )
            stats = parse_success(run_cli(config_path))
            self.assertTrue(stats["peak_limit_triggered"])
            self.assertTrue(stats["normalization"]["peak_limit_triggered"])
            self.assertLessEqual(
                stats["normalization"]["final_peak_dbfs"],
                -1.0 + 0.001,
            )
            self.assertGreater(
                stats["normalization"]["final_peak_dbfs"],
                stats["normalization"]["measured_peak_dbfs"],
            )
