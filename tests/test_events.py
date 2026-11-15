from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audio_event_mixer.errors import BusinessError, ErrorCode
from audio_event_mixer.events import load_events


class EventParsingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="aem_event_test_")
        self.root = Path(self.temp.name)
        (self.root / "samples").mkdir()
        (self.root / "samples" / "one.wav").write_bytes(b"placeholder")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write(self, payload: object) -> Path:
        path = self.root / "events.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_relative_path_and_alias_fields(self) -> None:
        path = self._write(
            {
                "events": [
                    {
                        "file_path": "samples/one.wav",
                        "trigger_time": 1.25,
                        "lfo_frequency_hz": 2.0,
                        "panning_depth": 0.5,
                        "loop_region": {
                            "start_time": 0.1,
                            "end_time": 0.4,
                        },
                    }
                ]
            }
        )
        events = load_events(path)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.source_path, (self.root / "samples" / "one.wav").resolve())
        self.assertEqual(event.trigger_seconds, 1.25)
        self.assertEqual(event.pan_frequency_hz, 2.0)
        self.assertEqual(event.pan_depth, 0.5)
        self.assertEqual(event.loop.start_seconds, 0.1)
        self.assertEqual(event.loop.end_seconds, 0.4)

    def test_top_level_loop_fields(self) -> None:
        path = self._write(
            [
                {
                    "file": "samples/one.wav",
                    "time": 0.0,
                    "pan_frequency_hz": 0.0,
                    "pan_depth": 0.0,
                    "loop_start": 0.0,
                    "loop_end": 0.2,
                }
            ]
        )
        event = load_events(path)[0]
        self.assertEqual(event.loop.start_seconds, 0.0)
        self.assertEqual(event.loop.end_seconds, 0.2)

    def test_nested_panning_uses_top_level_fallback(self) -> None:
        path = self._write(
            [
                {
                    "audio_file": "samples/one.wav",
                    "trigger_time_seconds": 0.3,
                    "pan_depth": 0.7,
                    "panning": {"frequency_hz": 1.5},
                }
            ]
        )
        event = load_events(path)[0]
        self.assertEqual(event.pan_frequency_hz, 1.5)
        self.assertEqual(event.pan_depth, 0.7)

    def test_missing_required_field(self) -> None:
        path = self._write(
            [
                {
                    "file": "samples/one.wav",
                    "time": 0.0,
                    "pan_depth": 0.5,
                }
            ]
        )
        with self.assertRaises(BusinessError) as caught:
            load_events(path)
        self.assertEqual(caught.exception.code, ErrorCode.INVALID_EVENT)
        self.assertEqual(caught.exception.event_index, 0)

    def test_depth_out_of_range(self) -> None:
        for depth in (-0.1, 1.1):
            with self.subTest(depth=depth):
                path = self._write(
                    [
                        {
                            "file": "samples/one.wav",
                            "time": 0.0,
                            "pan_frequency_hz": 1.0,
                            "pan_depth": depth,
                        }
                    ]
                )
                with self.assertRaises(BusinessError) as caught:
                    load_events(path)
                self.assertEqual(caught.exception.code, ErrorCode.INVALID_EVENT)

    def test_invalid_loop(self) -> None:
        path = self._write(
            [
                {
                    "file": "samples/one.wav",
                    "time": 0.0,
                    "pan_frequency_hz": 1.0,
                    "pan_depth": 0.5,
                    "loop": {"start": 0.4, "end": 0.2},
                }
            ]
        )
        with self.assertRaises(BusinessError) as caught:
            load_events(path)
        self.assertEqual(caught.exception.code, ErrorCode.INVALID_EVENT)
        self.assertEqual(caught.exception.event_index, 0)

    def test_missing_json_file(self) -> None:
        with self.assertRaises(BusinessError) as caught:
            load_events(self.root / "missing.json")
        self.assertEqual(caught.exception.code, ErrorCode.FILE_NOT_FOUND)

    def test_malformed_json(self) -> None:
        path = self.root / "bad.json"
        path.write_text("{not-json", encoding="utf-8")
        with self.assertRaises(BusinessError) as caught:
            load_events(path)
        self.assertEqual(caught.exception.code, ErrorCode.INVALID_EVENT)


if __name__ == "__main__":
    unittest.main()
