from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cinemix.processor import process_project
from cinemix.wavio import (
    read_pcm_wav,
    riff_info,
    wav_sha256,
    write_pcm16_wav,
    write_pcm_wav,
)


def _write_fixture(directory: Path) -> Path:
    audio = directory / "audio"
    audio.mkdir()
    t44 = np.arange(44100 * 2) / 44100
    dialogue = (
        np.sin(2.0 * math.pi * 180.0 * t44)
        + 0.2 * np.sin(2.0 * math.pi * 360.0 * t44)
    )[:, None] * 0.25
    write_pcm16_wav(audio / "dialogue.wav", dialogue, 44100)

    t48 = np.arange(48000 * 2) / 48000
    music_a = np.stack(
        (
            np.sin(2.0 * math.pi * 220.0 * t48) * 0.2,
            np.sin(2.0 * math.pi * 330.0 * t48) * 0.2,
        ),
        axis=1,
    )
    music_b = np.stack(
        (
            np.sin(2.0 * math.pi * 440.0 * t48) * 0.18,
            np.sin(2.0 * math.pi * 550.0 * t48) * 0.18,
        ),
        axis=1,
    )
    write_pcm_wav(audio / "music_a.wav", music_a, 48000)
    write_pcm_wav(audio / "music_b.wav", music_b, 48000)

    t96 = np.arange(96000) / 96000
    effect = (np.sin(2.0 * math.pi * 1000.0 * t96) * 0.15)[:, None]
    write_pcm_wav(audio / "effect.wav", effect, 96000)

    project = {
        "project": {
            "sample_rate": 48000,
            "output": "out/mix.wav",
            "target_lufs": -23.0,
            "true_peak_dbtp": -1.0,
        },
        "tracks": [
            {
                "id": "dialogue",
                "path": "audio/dialogue.wav",
                "start": 0.1,
                "gain_db": -2.0,
                "resample_to": 48000,
                "fade_in": {"start": 0.1, "end": 0.3},
            },
            {
                "id": "music_a",
                "path": "audio/music_a.wav",
                "gain_db": -8.0,
            },
            {
                "id": "music_b",
                "path": "audio/music_b.wav",
                "start": 0.8,
                "gain_db": -8.0,
            },
            {
                "id": "effect",
                "path": "audio/effect.wav",
                "start": 1.0,
                "gain_db": -3.0,
                "resample_to": 48000,
            },
        ],
        "crossfades": [
            {
                "from": "music_a",
                "to": "music_b",
                "start": 0.8,
                "end": 1.2,
                "curve": "equal_power",
            }
        ],
        "limiter": {
            "attack_ms": 4.0,
            "release_ms": 80.0,
            "ratio": 12.0,
            "lookahead_ms": 6.0,
        },
    }
    path = directory / "project.json"
    path.write_text(json.dumps(project), encoding="utf-8")
    return path


class IntegrationTests(unittest.TestCase):
    def test_end_to_end_metrics_format_and_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = _write_fixture(directory)
            first = process_project(config)
            report = first.report
            self.assertLessEqual(
                abs(report["measurement"]["loudness_error_lu"]), 0.5
            )
            self.assertLessEqual(
                report["measurement"]["true_peak_dbtp"],
                report["target"]["true_peak_dbtp"],
            )
            self.assertLessEqual(
                report["maximum_crossfade_boundary_error_ms"], 1.0
            )
            metadata = riff_info(first.output_path)
            self.assertEqual(metadata["bit_depth"], 24)
            self.assertEqual(metadata["sample_rate"], 48000)
            samples, info = read_pcm_wav(first.output_path)
            self.assertEqual(info.channels, 2)
            self.assertGreater(samples.shape[0], 0)

            repeat = process_project(
                config, output_override=directory / "out" / "repeat.wav"
            )
            self.assertEqual(
                wav_sha256(first.output_path),
                wav_sha256(repeat.output_path),
            )


if __name__ == "__main__":
    unittest.main()
