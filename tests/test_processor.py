from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audio_event_mixer.errors import BusinessError, ErrorCode
from audio_event_mixer.generator import generate_demo_dataset
from audio_event_mixer.processor import process_events
from audio_event_mixer.wavio import read_wav, write_pcm_wav


class ProcessorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="aem_processor_test_")
        cls.root = Path(cls.temp.name)
        cls.manifest = generate_demo_dataset(cls.root)
        cls.events_path = Path(cls.manifest["events_path"])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def _json(self, name: str, payload: dict) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_end_to_end_multi_format_timeline_report(self) -> None:
        output = self.root / "out" / "mix.wav"
        report_path = self.root / "out" / "report.json"
        report = process_events(self.events_path, output, report_path)
        parsed = json.loads(report_path.read_text(encoding="utf-8"))
        decoded = read_wav(output)

        self.assertEqual(parsed["status"], "ok")
        self.assertEqual(parsed["event_count"], 4)
        self.assertEqual(decoded.info.sample_rate, 48000)
        self.assertEqual(decoded.info.channels, 2)
        self.assertEqual(decoded.info.bits_per_sample, 16)
        self.assertEqual(decoded.samples.shape[0], report["timeline_frames"])
        self.assertTrue(np.all(np.isfinite(decoded.samples)))
        self.assertLessEqual(float(np.max(np.abs(decoded.samples))), 1.0)

        expected_formats = {
            "pcm_s16le",
            "pcm_s24le",
            "pcm_s32le",
            "float_s32le",
        }
        self.assertEqual(
            {event["source_encoding"] for event in parsed["events"]},
            expected_formats,
        )
        for event in parsed["events"]:
            expected_trigger = int(
                math.floor(event["trigger_seconds"] * 48000 + 0.5)
            )
            self.assertEqual(event["trigger_sample"], expected_trigger)
            self.assertLessEqual(abs(event["timing_error_samples"]), 0.5)
            self.assertLessEqual(
                abs(
                    event["duration_seconds"]
                    - event["source_duration_seconds"]
                ),
                0.001,
            )
            self.assertEqual(
                event["output_frames"], event["resampled_frames"]
            )
            self.assertEqual(event["output_path"], str(output.resolve()))
            if event["loop_cleanup"]:
                loop = event["loop_cleanup"]
                fade = min(
                    math.floor(
                        (loop["end_sample"] - loop["start_sample"]) * 0.10
                    ),
                    math.floor(48000 * 0.005),
                )
                self.assertEqual(loop["crossfade_samples"], fade)
                self.assertLessEqual(loop["boundary_jump_after"], 1e-9)

    def test_trigger_rounding_and_impulse_onset(self) -> None:
        impulse = np.zeros(16, dtype=np.float64)
        impulse[0] = 0.5
        source = self.root / "impulse.wav"
        write_pcm_wav(source, impulse, 48000, 16)
        trigger_sample = 100
        events_path = self._json(
            "impulse_events.json",
            {
                "events": [
                    {
                        "file": str(source),
                        "time": (trigger_sample + 0.4) / 48000.0,
                        "pan_frequency_hz": 0.0,
                        "pan_depth": 0.0,
                    }
                ]
            },
        )
        output = self.root / "impulse_mix.wav"
        report = process_events(events_path, output, self.root / "impulse.report.json")
        event = report["events"][0]
        self.assertEqual(event["trigger_sample"], trigger_sample)
        decoded = read_wav(output)
        nonzero = np.flatnonzero(np.max(np.abs(decoded.samples), axis=1) > 1e-6)
        self.assertEqual(int(nonzero[0]), trigger_sample)
        self.assertAlmostEqual(
            float(decoded.samples[trigger_sample, 0]),
            math.tanh(0.5 * math.sqrt(0.5)),
            places=3,
        )
        self.assertAlmostEqual(
            float(decoded.samples[trigger_sample, 1]),
            math.tanh(0.5 * math.sqrt(0.5)),
            places=3,
        )

    def test_overlap_uses_sample_sum_then_tanh(self) -> None:
        impulse = np.zeros(8, dtype=np.float64)
        impulse[0] = 0.45
        source = self.root / "overlap_impulse.wav"
        write_pcm_wav(source, impulse, 48000, 16)
        event = {
            "file": str(source),
            "time": 0.0,
            "pan_frequency_hz": 0.0,
            "pan_depth": 0.0,
        }
        events_path = self._json(
            "overlap_events.json", {"events": [event, dict(event)]}
        )
        output = self.root / "overlap_mix.wav"
        process_events(events_path, output, self.root / "overlap.report.json")
        decoded = read_wav(output)
        expected = math.tanh(2.0 * 0.45 * math.sqrt(0.5))
        self.assertAlmostEqual(
            float(decoded.samples[0, 0]), expected, places=3
        )

    def test_final_wav_loop_boundary_matches_in_both_channels(self) -> None:
        sr = 48000
        frames = 2400
        t = np.arange(frames) / float(sr)
        source = (
            0.2 * np.sin(2.0 * math.pi * 317.0 * t)
            + 0.05 * np.sin(2.0 * math.pi * 71.0 * t + 0.2)
        )
        source_path = self.root / "seamless_source.wav"
        write_pcm_wav(source_path, source, sr, 16)
        loop_start = 120
        loop_end = 2000
        trigger = 321
        events_path = self._json(
            "seamless_events.json",
            {
                "events": [
                    {
                        "file": str(source_path),
                        "time": trigger / sr,
                        "pan_frequency_hz": 1.1,
                        "pan_depth": 0.65,
                        "loop": {
                            "start": loop_start / sr,
                            "end": loop_end / sr,
                        },
                    }
                ]
            },
        )
        output = self.root / "seamless.wav"
        report = process_events(
            events_path, output, self.root / "seamless.report.json"
        )
        decoded = read_wav(output).samples
        first = decoded[trigger + loop_start]
        last = decoded[trigger + loop_end - 1]
        np.testing.assert_array_equal(first, last)
        self.assertEqual(
            report["events"][0]["loop_cleanup"]["crossfade_samples"], 188
        )

    def test_invalid_loop_range_reports_failed_event_index(self) -> None:
        source = self.root / "samples" / "source_44k_s16_mono.wav"
        events_path = self._json(
            "bad_loop_events.json",
            {
                "events": [
                    {
                        "file": str(source),
                        "time": 0.0,
                        "pan_frequency_hz": 1.0,
                        "pan_depth": 0.5,
                        "loop": {"start": 0.2, "end": 1.2},
                    }
                ]
            },
        )
        output = self.root / "bad_loop.wav"
        report_path = self.root / "bad_loop.report.json"
        with self.assertRaises(BusinessError) as caught:
            process_events(events_path, output, report_path)
        self.assertEqual(caught.exception.code, ErrorCode.INVALID_EVENT)
        self.assertEqual(caught.exception.event_index, 0)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["error_code"], ErrorCode.INVALID_EVENT)
        self.assertEqual(report["failure_event_index"], 0)

    def test_missing_and_unsupported_source_codes(self) -> None:
        cases = [
            (
                self.root / "missing_source.wav",
                ErrorCode.FILE_NOT_FOUND,
            ),
            (
                self.root / "samples" / "source_unsupported_mulaw.wav",
                ErrorCode.UNSUPPORTED_FORMAT,
            ),
        ]
        for index, (source, expected_code) in enumerate(cases):
            with self.subTest(code=expected_code):
                events_path = self._json(
                    f"error_source_{index}.json",
                    {
                        "events": [
                            {
                                "file": str(source),
                                "time": 0.0,
                                "pan_frequency_hz": 1.0,
                                "pan_depth": 0.5,
                            }
                        ]
                    },
                )
                with self.assertRaises(BusinessError) as caught:
                    process_events(
                        events_path,
                        self.root / f"error_source_{index}.wav",
                        self.root / f"error_source_{index}.report.json",
                    )
                self.assertEqual(caught.exception.code, expected_code)
                self.assertEqual(caught.exception.event_index, 0)

    def test_atomic_replace_preserves_existing_on_processing_error(self) -> None:
        output = self.root / "sentinel.wav"
        sentinel = b"existing-good-output"
        output.write_bytes(sentinel)
        events_path = self._json(
            "atomic_error.json",
            {
                "events": [
                    {
                        "file": str(self.root / "not-there.wav"),
                        "time": 0.0,
                        "pan_frequency_hz": 1.0,
                        "pan_depth": 0.5,
                    }
                ]
            },
        )
        with self.assertRaises(BusinessError):
            process_events(events_path, output, self.root / "atomic.report.json")
        self.assertEqual(output.read_bytes(), sentinel)

    def test_output_denied_code(self) -> None:
        parent = self.root / "not_a_dir"
        parent.write_text("x", encoding="utf-8")
        events_path = self._json(
            "denied_events.json",
            {
                "events": [
                    {
                        "file": str(
                            self.root / "samples" / "source_44k_s16_mono.wav"
                        ),
                        "time": 0.0,
                        "pan_frequency_hz": 1.0,
                        "pan_depth": 0.2,
                    }
                ]
            },
        )
        report_path = self.root / "denied.report.json"
        with self.assertRaises(BusinessError) as caught:
            process_events(events_path, parent / "denied.wav", report_path)
        self.assertEqual(caught.exception.code, ErrorCode.OUTPUT_DENIED)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["error_code"], ErrorCode.OUTPUT_DENIED)

    def test_memmap_spill_for_long_timeline(self) -> None:
        frames = 5 * 48000
        t = np.arange(frames) / 48000.0
        source = self.root / "long_source.wav"
        write_pcm_wav(
            source,
            0.1 * np.sin(2.0 * math.pi * 220.0 * t),
            48000,
            16,
        )
        events_path = self._json(
            "long_events.json",
            {
                "events": [
                    {
                        "file": str(source),
                        "time": 0.0,
                        "pan_frequency_hz": 0.5,
                        "pan_depth": 0.4,
                    }
                ]
            },
        )
        report = process_events(
            events_path,
            self.root / "long.wav",
            self.root / "long.report.json",
            memory_limit_bytes=8 * 1024 * 1024,
        )
        self.assertEqual(
            report["processing"]["mix_storage"], "memmap"
        )
        self.assertEqual(report["timeline_frames"], frames)


if __name__ == "__main__":
    unittest.main()
