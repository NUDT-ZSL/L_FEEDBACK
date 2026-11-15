from __future__ import annotations

import struct
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from hls_audio_builder.timeline import (
    align_pcm,
    plan_timeline,
    relative_shift_frames,
)


class TimelineTest(unittest.TestCase):
    def test_plan_uses_uniform_sample_boundaries(self) -> None:
        segments = plan_timeline(
            total_samples=10001,
            sample_rate=1000,
            segment_duration=Decimal("2.5"),
        )
        self.assertEqual(
            [segment.end_sample for segment in segments],
            [2500, 5000, 7500, 10000, 10001],
        )
        self.assertTrue(
            all(segment.sample_count == 2500 for segment in segments[:-1])
        )
        self.assertEqual(segments[-1].sample_count, 1)

    def test_positive_shift_trims_and_pads_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.pcm"
            destination = Path(temp) / "aligned.pcm"
            source.write_bytes(struct.pack("<5h", 1, 2, 3, 4, 5))
            align_pcm(
                source,
                destination,
                target_frames=5,
                frame_bytes=2,
                shift_frames=2,
            )
            self.assertEqual(
                struct.unpack("<5h", destination.read_bytes()),
                (3, 4, 5, 0, 0),
            )

    def test_negative_shift_prepends_silence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.pcm"
            destination = Path(temp) / "aligned.pcm"
            source.write_bytes(struct.pack("<5h", 1, 2, 3, 4, 5))
            align_pcm(
                source,
                destination,
                target_frames=5,
                frame_bytes=2,
                shift_frames=-2,
            )
            self.assertEqual(
                struct.unpack("<5h", destination.read_bytes()),
                (0, 0, 1, 2, 3),
            )

    def test_shift_is_relative_to_reference(self) -> None:
        shift = relative_shift_frames(
            variant_offset_ms=Decimal("25"),
            variant_start_time=Decimal("0.000000"),
            reference_offset_ms=Decimal("0"),
            reference_start_time=Decimal("0.000000"),
            sample_rate=44100,
        )
        self.assertEqual(shift, 1103)


if __name__ == "__main__":
    unittest.main()

