from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from bcsubts.alignment import estimate_offset
from bcsubts.dvb import (
    build_clear_segments,
    build_show_segments,
    build_subtitle_pes,
    parse_subtitle_pes,
    quantize_rgba,
)
from bcsubts.errors import BusinessError, EXIT_CODES
from bcsubts.resample import sinc_resample
from bcsubts.samplegen import SampleSpec, generate_sample
from bcsubts.scte35 import (
    ScteEvent,
    SegmentationDescriptor,
    build_splice_info,
    parse_command_table,
    parse_splice_info,
)
from bcsubts.subtitles import (
    Cue,
    correct_timeline,
    parse_srt_text,
    timeline_metrics,
)
from bcsubts.wavio import read_wav16


class SubtitleTests(unittest.TestCase):
    def test_parse_and_correct_overlap_gap(self) -> None:
        text = """1
00:00:01,000 --> 00:00:03,000
First line

2
00:00:02,500 --> 00:00:04,000
Second line

3
00:00:10,000 --> 00:00:11,000
Third line
"""
        cues = parse_srt_text(text)
        self.assertEqual(len(cues), 3)
        metrics = timeline_metrics(cues)
        self.assertEqual(metrics.overlap_count, 1)
        self.assertEqual(metrics.gap_count, 1)
        corrected, log = correct_timeline(cues, "truncate")
        self.assertEqual(len(corrected), 3)
        self.assertEqual(corrected[0].end_ms, 2500)
        self.assertTrue(any(item.reason == "overlap_clip_previous" for item in log))

    def test_invalid_timecode(self) -> None:
        text = """1
00:00:75,000 --> 00:00:78,000
Bad time
"""
        with self.assertRaises(BusinessError) as context:
            parse_srt_text(text)
        self.assertEqual(context.exception.code, "E_TIMECODE")
        self.assertEqual(context.exception.exit_code, EXIT_CODES["E_TIMECODE"])

    def test_negative_duration_skip_policy(self) -> None:
        cues = [Cue(0, 2000, 1000, "negative", 0), Cue(1, 3000, 4000, "valid", 1)]
        corrected, log = correct_timeline(cues, "skip")
        self.assertEqual([cue.text for cue in corrected], ["valid"])
        self.assertEqual(log[0].reason, "negative_duration")


class AudioTests(unittest.TestCase):
    def test_resample_length_and_rms(self) -> None:
        rate = 44100
        time = np.arange(rate, dtype=np.float64) / rate
        signal = np.sin(2 * np.pi * 1000 * time)
        output = sinc_resample(signal, rate, 48000)
        self.assertEqual(output.shape[0], 48000)
        self.assertAlmostEqual(float(np.sqrt(np.mean(output * output))), 1 / np.sqrt(2), delta=0.01)

    def test_downsample_antialias(self) -> None:
        rate = 96000
        time = np.arange(rate, dtype=np.float64) / rate
        signal = np.sin(2 * np.pi * 30000 * time)
        output = sinc_resample(signal, rate, 48000)
        self.assertLess(float(np.sqrt(np.mean(output * output))), 0.05)

    def test_24bit_wav_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "bad.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(3)
                wav.setframerate(48000)
                wav.writeframes(b"\x00\x00\x00" * 100)
            with self.assertRaises(BusinessError) as context:
                read_wav16(path)
            self.assertEqual(context.exception.code, "E_FORMAT")


class ScteTests(unittest.TestCase):
    def test_section_round_trip(self) -> None:
        event = ScteEvent(
            event_id=0x10203040,
            event_type="splice_insert",
            offset_ms=6000,
            duration_ms=4500,
            segmentation_descriptor=SegmentationDescriptor(
                segmentation_type_id=0x34,
                segment_num=1,
                segments_expected=1,
                upid_type=8,
                upid="id-1",
            ),
            target_pts=0x123456789,
        )
        section = build_splice_info(event)
        parsed = parse_splice_info(section)
        self.assertEqual(parsed["event_id"], event.event_id)
        self.assertEqual(parsed["pts_adjustment"], event.target_pts & ((1 << 33) - 1))
        self.assertEqual(parsed["break_duration"], 4500 * 90)
        self.assertEqual(parsed["crc32"], int.from_bytes(section[-4:], "big"))

    def test_overflow_and_duplicate_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "events.json"
            path.write_text(
                json.dumps(
                    {
                        "events": [
                            {"event_id": 1, "type": "splice_null", "offset_ms": 30000}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(BusinessError) as context:
                parse_command_table(path, 20000, 48000)
            self.assertEqual(context.exception.code, "E_OVERFLOW")
            path.write_text(
                json.dumps(
                    {
                        "events": [
                            {"event_id": 1, "type": "splice_null", "offset_ms": 1000},
                            {"event_id": 1, "type": "splice_null", "offset_ms": 2000},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(BusinessError) as duplicate:
                parse_command_table(path, 20000, 48000)
            self.assertEqual(duplicate.exception.code, "E_FORMAT")


class DvbTests(unittest.TestCase):
    def test_segments_and_pes(self) -> None:
        rgba = np.zeros((576, 720, 4), dtype=np.uint8)
        rgba[500:540, 100:620] = (255, 255, 255, 255)
        region = quantize_rgba(rgba, "test")
        show = build_show_segments(region, 1)
        clear = build_clear_segments(2)
        show_parsed = parse_subtitle_pes(b"\x20\x00" + show)
        clear_parsed = parse_subtitle_pes(b"\x20\x00" + clear)
        self.assertTrue(show_parsed["has_object"])
        self.assertEqual(show_parsed["region_count"], 1)
        self.assertFalse(clear_parsed["has_object"])
        self.assertEqual(clear_parsed["region_count"], 0)
        pes = build_subtitle_pes(show, 90000)
        self.assertEqual(pes[:4], b"\x00\x00\x01\xbd")
        self.assertGreater(len(pes), len(show))


class AlignmentTests(unittest.TestCase):
    def test_known_delay_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            sample = generate_sample(
                Path(temp_name),
                SampleSpec("unit", 48000, 1, -850, "srt", 1234),
            )
            wav = read_wav16(sample["audio"])
            internal = sinc_resample(wav.samples, wav.sample_rate, 48000)
            cues = parse_srt_text(Path(sample["subtitles"]).read_text(encoding="utf-8"))
            corrected, _ = correct_timeline(cues)
            result = estimate_offset(internal, corrected)
            self.assertLess(abs(result.offset_ms - (-850)), 10)
            self.assertGreater(result.confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
