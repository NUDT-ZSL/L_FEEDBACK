from __future__ import annotations

import json
import os

from replay_exporter import ErrorCode, ExporterError
from replay_exporter.planner import (
    build_plan,
    load_segment_inputs,
    load_sources,
    verify_length_invariant,
)

from .base import ExporterTestCase


class PlannerTests(ExporterTestCase):
    def test_length_invariant_and_disjoint_intervals(self):
        inputs, timeline = load_segment_inputs(self.timeline_path)
        plan = build_plan(inputs, timeline)
        verify_length_invariant(plan)
        segment_frames = sum(segment.frame_count for segment in plan.segments)
        gap_frames = sum(region.gap_frames for region in plan.crossfades)
        self.assertEqual(segment_frames + gap_frames, plan.total_frames)
        for left, right in zip(plan.segments, plan.segments[1:]):
            self.assertLessEqual(left.end_frame, right.start_frame)

    def test_negative_duration_returns_business_code(self):
        metadata = self.segment_metadata(1)
        metadata["duration_ms"] = -5
        self.write_segment_metadata(1, metadata)
        with self.assertRaises(ExporterError) as caught:
            load_segment_inputs(self.timeline_path)
        self.assertEqual(caught.exception.code, ErrorCode.NEGATIVE_DURATION)
        self.assertEqual(caught.exception.segment_index, 1)

    def test_fade_longer_than_segment(self):
        metadata = self.segment_metadata(0)
        metadata["fade_ms"] = 600
        self.write_segment_metadata(0, metadata)
        inputs, timeline = load_segment_inputs(self.timeline_path)
        with self.assertRaises(ExporterError) as caught:
            build_plan(inputs, timeline)
        self.assertEqual(caught.exception.code, ErrorCode.FADE_LONGER_THAN_SEGMENT)
        self.assertEqual(caught.exception.segment_index, 0)

    def test_missing_dialogue_range(self):
        metadata = self.segment_metadata(2)
        del metadata["dialogue_start_ms"]
        self.write_segment_metadata(2, metadata)
        with self.assertRaises(ExporterError) as caught:
            load_segment_inputs(self.timeline_path)
        self.assertEqual(caught.exception.code, ErrorCode.MISSING_DIALOGUE_RANGE)
        self.assertEqual(caught.exception.segment_index, 2)

    def test_invalid_dialogue_range_with_offset(self):
        metadata = self.segment_metadata(0)
        metadata["dialogue_end_ms"] = metadata["duration_ms"] + 50
        self.write_segment_metadata(0, metadata)
        with self.assertRaises(ExporterError) as caught:
            load_segment_inputs(self.timeline_path)
        self.assertEqual(caught.exception.code, ErrorCode.INVALID_DIALOGUE_RANGE)
        self.assertIsNotNone(caught.exception.offset_ms)

    def test_negative_gap(self):
        with open(self.timeline_path, "r", encoding="utf-8") as handle:
            timeline = json.load(handle)
        timeline["gaps_ms"][0] = -25
        with open(self.timeline_path, "w", encoding="utf-8") as handle:
            json.dump(timeline, handle)
        with self.assertRaises(ExporterError) as caught:
            load_segment_inputs(self.timeline_path)
        self.assertEqual(caught.exception.code, ErrorCode.NEGATIVE_GAP)

    def test_invalid_json_returns_code(self):
        with open(self.timeline_path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        with self.assertRaises(ExporterError) as caught:
            load_segment_inputs(self.timeline_path)
        self.assertEqual(caught.exception.code, ErrorCode.INVALID_JSON)

    def test_sample_rate_mismatch(self):
        from replay_exporter.audioio import write_mono_wav

        bad_path = os.path.join(self.project_dir, "segment_0", "bad_rate.wav")
        write_mono_wav(bad_path, [1, 2, 3, 4], sample_rate=44100)
        metadata = self.segment_metadata(0)
        metadata["voice_wav"] = os.path.join("segment_0", "bad_rate.wav")
        self.write_segment_metadata(0, metadata)
        inputs, _ = load_segment_inputs(self.timeline_path)
        with self.assertRaises(ExporterError) as caught:
            load_sources(inputs)
        self.assertEqual(caught.exception.code, ErrorCode.SAMPLE_RATE_MISMATCH)

    def test_non_integer_frame_data_chunk(self):
        target = os.path.join(self.project_dir, "segment_0", "ambience.wav")
        with open(target, "rb") as handle:
            blob = bytearray(handle.read())
        # Grow the data chunk size by one odd byte without a matching frame.
        struct_index = blob.index(b"data") + 4
        import struct

        current = struct.unpack_from("<I", blob, struct_index)[0]
        struct.pack_into("<I", blob, struct_index, current + 1)
        blob.extend(b"\x00")
        with open(target, "wb") as handle:
            handle.write(blob)
        inputs, _ = load_segment_inputs(self.timeline_path)
        with self.assertRaises(ExporterError) as caught:
            load_sources(inputs)
        self.assertEqual(caught.exception.code, ErrorCode.NON_INTEGER_FRAME)
        self.assertEqual(caught.exception.segment_index, 0)
