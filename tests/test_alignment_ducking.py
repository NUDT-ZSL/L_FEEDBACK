from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.support import (
    goertzel_amplitude,
    manifest,
    read_audio,
    run_cli,
    write_audio,
    write_manifest,
)


class AlignmentAndDuckingTests(unittest.TestCase):
    def test_resampled_impulse_alignment_within_one_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_rate = 44_100
            source_impulse = 12_345
            start_ms = 137.3
            write_audio(
                root / "assets" / "impulse.wav",
                source_rate,
                1,
                1.2,
                frequency=200.0,
                amplitude=0.02,
                impulse_at=source_impulse,
            )
            write_audio(
                root / "assets" / "bed.wav",
                48_000,
                2,
                2.0,
                frequency=90.0,
                amplitude=0.1,
            )
            value = manifest(
                root,
                [
                    {
                        "id": "impulse",
                        "path": "assets/impulse.wav",
                        "sample_rate": source_rate,
                        "channels": 1,
                        "start_ms": start_ms,
                        "gain_db": 0.0,
                        "role": "sfx",
                    },
                    {
                        "id": "bed",
                        "path": "assets/bed.wav",
                        "sample_rate": 48_000,
                        "channels": 2,
                        "start_ms": 0.0,
                        "gain_db": -6.0,
                        "role": "ambience",
                    },
                ],
                duration_ms=2_000.0,
                target_lufs=-23.0,
            )
            manifest_path = write_manifest(root / "manifest.json", value)
            completed = run_cli(manifest_path)
            self.assertEqual(completed.returncode, 0, completed.stderr)

            output, output_rate = read_audio(root / "out" / "master.wav")
            report = json.loads((root / "out" / "report.json").read_text())
            start_frame = int(round(start_ms * output_rate / 1000.0))
            expected = start_frame + int(
                round(source_impulse * output_rate / source_rate)
            )
            search = output[max(0, expected - 8) : expected + 9, 0]
            peak = int(np.argmax(np.abs(search))) + max(0, expected - 8)
            self.assertLessEqual(abs(peak - expected), 1)
            track = next(item for item in report["tracks"] if item["track_id"] == "impulse")
            self.assertEqual(track["first_valid_sample"], start_frame)
            self.assertLessEqual(report["timeline"]["max_start_rounding_error_samples"], 1.0)

    def test_ducking_interval_and_non_dialogue_gain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_audio(
                root / "assets" / "lead.wav", 48_000, 1, 3.0, frequency=3_000.0, amplitude=0.16
            )
            write_audio(
                root / "assets" / "support.wav", 48_000, 1, 3.0, frequency=700.0, amplitude=0.12
            )
            write_audio(
                root / "assets" / "amb.wav", 48_000, 2, 4.0, frequency=120.0, amplitude=0.07
            )
            value = manifest(
                root,
                [
                    {
                        "id": "lead",
                        "path": "assets/lead.wav",
                        "sample_rate": 48_000,
                        "channels": 1,
                        "start_ms": 0.0,
                        "gain_db": -1.0,
                        "role": "dialogue",
                        "priority": 20,
                    },
                    {
                        "id": "support",
                        "path": "assets/support.wav",
                        "sample_rate": 48_000,
                        "channels": 1,
                        "start_ms": 1_000.0,
                        "gain_db": -1.0,
                        "role": "dialogue",
                        "priority": 10,
                    },
                    {
                        "id": "amb",
                        "path": "assets/amb.wav",
                        "sample_rate": 48_000,
                        "channels": 2,
                        "start_ms": 0.0,
                        "gain_db": -8.0,
                        "role": "ambience",
                    },
                ],
                duration_ms=4_000.0,
            )
            manifest_path = write_manifest(root / "manifest.json", value)
            completed = run_cli(manifest_path)
            self.assertEqual(completed.returncode, 0, completed.stderr)

            output, sample_rate = read_audio(root / "out" / "master.wav")
            report = json.loads((root / "out" / "report.json").read_text())
            overlap = output[1_200 * 48 : 1_800 * 48, 0]
            after = output[3_200 * 48 : 3_800 * 48, 0]
            support_overlap = goertzel_amplitude(overlap, sample_rate, 700.0)
            support_after = goertzel_amplitude(after, sample_rate, 700.0)
            amb_overlap = goertzel_amplitude(overlap, sample_rate, 120.0)
            amb_after = goertzel_amplitude(after, sample_rate, 120.0)
            self.assertAlmostEqual(support_overlap / support_after, 10 ** (-12.0 / 20.0), delta=0.035)
            self.assertAlmostEqual(amb_overlap / amb_after, 1.0, delta=0.03)

            support = next(item for item in report["tracks"] if item["track_id"] == "support")
            ambience = next(item for item in report["tracks"] if item["track_id"] == "amb")
            self.assertEqual(
                [(item["start_frame"], item["end_frame"]) for item in support["duck_intervals"]],
                [(48_000, 144_000)],
            )
            self.assertEqual(ambience["duck_intervals"], [])
            self.assertEqual(report["dialogue_ducking"]["non_dialogue_tracks_affected"], False)
            for track in report["tracks"]:
                segments = track["gain_segments"]
                self.assertEqual(segments[0]["start_frame"], track["timeline_start_frame"])
                self.assertEqual(segments[-1]["end_frame"], track["last_valid_sample_exclusive"])
                for left, right in zip(segments, segments[1:]):
                    self.assertEqual(left["end_frame"], right["start_frame"])

    def test_same_priority_overlap_is_structured_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_audio(root / "assets" / "a.wav", 48_000, 1, 2.0, amplitude=0.1)
            write_audio(root / "assets" / "b.wav", 48_000, 1, 2.0, frequency=330.0, amplitude=0.1)
            value = manifest(
                root,
                [
                    {
                        "id": "a",
                        "path": "assets/a.wav",
                        "sample_rate": 48_000,
                        "channels": 1,
                        "start_ms": 0.0,
                        "gain_db": 0.0,
                        "role": "dialogue",
                        "priority": 10,
                    },
                    {
                        "id": "b",
                        "path": "assets/b.wav",
                        "sample_rate": 48_000,
                        "channels": 1,
                        "start_ms": 500.0,
                        "gain_db": 0.0,
                        "role": "dialogue",
                        "priority": 10,
                    },
                ],
                duration_ms=2_500.0,
            )
            manifest_path = write_manifest(root / "manifest.json", value)
            completed = run_cli(manifest_path)
            self.assertNotEqual(completed.returncode, 0)
            error = json.loads(completed.stderr)
            self.assertEqual(error["error"]["code"], "E_DUCK_UNSATISFIABLE")
            self.assertFalse((root / "out" / "master.wav").exists())


if __name__ == "__main__":
    unittest.main()
