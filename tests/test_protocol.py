import io
import os
import struct
import tempfile
import unittest

from audio_sync.constants import FRAME_BYTES, HEADER_SIZE
from audio_sync.protocol import (
    build_packet,
    parse_packet_file,
    check_sequence_continuity,
)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "stream_0.pkt")

    def _write(self, data):
        with open(self.path, "wb") as handle:
            handle.write(data)

    def test_round_trip_valid_packets(self):
        blob = b"".join(
            build_packet(seq, seq * 960, b"\x01\x02" * (FRAME_BYTES // 2))
            for seq in range(5)
        )
        self._write(blob)
        packets = parse_packet_file(self.path, "stream_0")
        self.assertEqual(len(packets), 5)
        self.assertEqual([p.seq for p in packets], [0, 1, 2, 3, 4])
        self.assertEqual(packets[3].frame_index, 3)
        self.assertEqual(len(packets[0].payload), FRAME_BYTES)

    def test_bad_length_record_skipped_but_parsing_continues(self):
        good_a = build_packet(0, 0, b"\x00" * FRAME_BYTES)
        bad = struct.pack("<HIH", 9, 960, 12) + b"\x00" * 12
        good_b = build_packet(2, 2 * 960, b"\x00" * FRAME_BYTES)
        self._write(good_a + bad + good_b)
        sink = io.StringIO()
        packets = parse_packet_file(self.path, "stream_0", sink)
        self.assertEqual([p.seq for p in packets], [0, 2])
        self.assertIn("E_BAD_LENGTH", sink.getvalue())

    def test_truncated_payload_stops_parse(self):
        good = build_packet(0, 0, b"\x00" * FRAME_BYTES)
        truncated = struct.pack("<HIH", 1, 960, FRAME_BYTES) + b"\x00" * 5
        self._write(good + truncated)
        sink = io.StringIO()
        packets = parse_packet_file(self.path, "stream_0", sink)
        self.assertEqual(len(packets), 1)
        self.assertIn("E_TRUNCATED", sink.getvalue())

    def test_truncated_header_reported(self):
        self._write(b"\x00" * (HEADER_SIZE - 2))
        sink = io.StringIO()
        packets = parse_packet_file(self.path, "stream_0", sink)
        self.assertEqual(packets, [])
        self.assertIn("E_TRUNCATED", sink.getvalue())

    def test_sequence_gap_warning(self):
        blob = build_packet(0, 0, b"\x00" * FRAME_BYTES) + build_packet(
            3, 3 * 960, b"\x00" * FRAME_BYTES
        )
        self._write(blob)
        sink = io.StringIO()
        parse_packet_file(self.path, "stream_0", sink)
        self.assertIn("W_SEQ_GAP", sink.getvalue())

    def test_sequence_wrap_continuity(self):
        packets = []
        # Build lightweight pseudo packets only for continuity logic.
        from audio_sync.protocol import Packet

        for seq in (65534, 65535, 0, 1):
            packets.append(
                Packet(seq=seq, timestamp=0, payload_length=0, payload=b"", offset=0)
            )
        sink = io.StringIO()
        gaps = check_sequence_continuity(packets, "s", sink)
        self.assertEqual(gaps, 0)


if __name__ == "__main__":
    unittest.main()
