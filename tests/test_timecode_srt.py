from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from dialogue_sync.errors import TimecodeError
from dialogue_sync.model import load_project
from dialogue_sync.srt import find_overlaps, parse_srt_text
from dialogue_sync.timecode import (
    format_timecode,
    parse_srt_timecode,
    parse_timecode,
)


class TimecodeTests(unittest.TestCase):
    def test_round_trip_is_frame_rate_independent(self) -> None:
        cases = {
            "00:00:00.000": 0,
            "00:00:01.001": 1_001,
            "01:02:03.456": 3_723_456,
            "99:59:59.999": 359_999_999,
        }
        for text, milliseconds in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_timecode(text), milliseconds)
                self.assertEqual(format_timecode(milliseconds), text)

    def test_invalid_timecodes_are_rejected(self) -> None:
        invalid = [
            "00:00:01",
            "0:00:01.000",
            "00:60:01.000",
            "00:00:60.000",
            "00:00:01.00",
            "00:00:-1.000",
        ]
        for text in invalid:
            with self.subTest(text=text):
                with self.assertRaises(TimecodeError):
                    parse_timecode(text)

    def test_srt_accepts_comma_and_dot(self) -> None:
        self.assertEqual(parse_srt_timecode("01:02:03,456"), 3_723_456)
        self.assertEqual(parse_srt_timecode("01:02:03.456"), 3_723_456)


class SrtTests(unittest.TestCase):
    SAMPLE = (
        "\ufeff1\r\n"
        "00:00:00,000 --> 00:00:01,500\r\n"
        "First line\r\n\r\n"
        "2\r\n"
        "00:00:01,400 --> 00:00:02,500\r\n"
        "Second line\r\n"
        "with a second row\r\n"
    )

    def test_parse_render_and_overlap(self) -> None:
        document = parse_srt_text(self.SAMPLE)
        self.assertEqual([cue.index for cue in document.cues], [1, 2])
        self.assertEqual(document.cues[1].text, "Second line\nwith a second row")
        rendered = document.render()
        self.assertIn("00:00:01,400 --> 00:00:02,500", rendered)
        overlaps = find_overlaps(document.cues)
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0]["overlap_ms"], 100)
        self.assertEqual(overlaps[0]["code"], "W_SUBTITLE_OVERLAP")

    def test_zero_duration_is_rejected(self) -> None:
        invalid = self.SAMPLE.replace(
            "00:00:01,400 --> 00:00:02,500",
            "00:00:01,400 --> 00:00:01,400",
        )
        with self.assertRaises(Exception):
            parse_srt_text(invalid)


class EntryValidationTests(unittest.TestCase):
    def test_invalid_entry_timecode_becomes_reportable_error(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dialogue-sync-timecode-") as temp:
            root = Path(temp)
            timeline = {
                "srt": "dialogue.srt",
                "total_duration": "00:00:10.000",
                "dialogue_entries": [
                    {
                        "source": "line.wav",
                        "source_start": "not-a-timecode",
                        "target_start": "00:00:01.000",
                        "subtitle_index": 1,
                    }
                ],
            }
            path = root / "timeline.json"
            path.write_text(json.dumps(timeline), encoding="utf-8")
            project = load_project(path)
            self.assertEqual(project.entries[0].field_errors[0]["code"], "E_TIMECODE")
            self.assertEqual(project.entries[0].field_errors[0]["field"], "entry 1 source_start")


if __name__ == "__main__":
    unittest.main()

