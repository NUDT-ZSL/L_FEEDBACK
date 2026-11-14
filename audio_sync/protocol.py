"""Custom RTP-style binary packet format and parser.

Packet layout (little endian)::

    offset  size  field
    0       2     sequence number  (uint16, wraps at 65536)
    2       4     timestamp       (uint32, 48 kHz sample clock)
    6       2     payload length  (uint16, bytes of PCM payload)
    8       N     payload         (48 kHz mono 16-bit little-endian PCM)

A valid audio frame carries exactly :data:`FRAME_BYTES` payload bytes
(20 ms at 48 kHz / 16-bit). Malformed records are reported with business
error codes on stderr and skipped; parsing continues whenever framing can
still be recovered.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, TextIO

from .constants import (
    ERR_BAD_LENGTH,
    ERR_SEQ_GAP,
    ERR_SEQ_REORDER,
    ERR_TRUNCATED,
    FRAME_BYTES,
    FRAME_SAMPLES,
    HEADER_FORMAT,
    HEADER_SIZE,
    SEQ_MOD,
)


@dataclass(frozen=True)
class Packet:
    """One parsed audio packet."""

    seq: int
    timestamp: int
    payload_length: int
    payload: bytes
    offset: int  # byte offset inside the source file, for diagnostics

    @property
    def frame_index(self) -> int:
        """Global frame index derived from the 32-bit timestamp."""
        return self.timestamp // FRAME_SAMPLES


def _log_error(
    sink: Optional[TextIO],
    code: str,
    stream: str,
    offset: int,
    detail: str,
) -> None:
    if sink is not None:
        sink.write(
            f"[{code}] stream={stream} offset={offset} {detail}\n"
        )
        sink.flush()


def check_sequence_continuity(
    packets: Sequence[Packet],
    stream: str,
    sink: Optional[TextIO] = None,
) -> int:
    """Check 16-bit sequence number continuity.

    Gaps are reported with :data:`ERR_SEQ_GAP` on stderr but never raise,
    because a gap is normally healed by the jitter buffer / PLC layer.
    Returns the number of gaps observed.
    """
    gaps = 0
    expected: Optional[int] = None
    for packet in packets:
        if expected is not None and packet.seq != expected:
            forward_gap = (packet.seq - expected) % SEQ_MOD
            if forward_gap == 0:
                continue
            if forward_gap < 0x8000:
                gaps += 1
                _log_error(
                    sink,
                    ERR_SEQ_GAP,
                    stream,
                    packet.offset,
                    f"expected_seq={expected} actual_seq={packet.seq} "
                    f"missing={forward_gap}",
                )
            else:
                # Backward jump means reordering / duplicate, not packet loss.
                _log_error(
                    sink,
                    ERR_SEQ_REORDER,
                    stream,
                    packet.offset,
                    f"expected_seq={expected} actual_seq={packet.seq}",
                )
        expected = (packet.seq + 1) % SEQ_MOD
    return gaps


def parse_packet_file(
    path: str,
    stream: str,
    sink: Optional[TextIO] = None,
) -> List[Packet]:
    """Parse a custom binary packet file into :class:`Packet` objects.

    Structural errors are logged and the offending record skipped. A record
    whose header is truncated at end of file terminates the walk. When a
    payload is truncated mid-record framing cannot be recovered, so parsing
    stops after reporting the error.
    """
    with open(path, "rb") as handle:
        data = handle.read()

    packets: List[Packet] = []
    offset = 0
    total = len(data)

    while offset < total:
        remaining = total - offset
        if remaining < HEADER_SIZE:
            _log_error(
                sink,
                ERR_TRUNCATED,
                stream,
                offset,
                f"remaining={remaining} < header={HEADER_SIZE}",
            )
            break

        seq, timestamp, payload_length = struct.unpack_from(
            HEADER_FORMAT, data, offset
        )
        record_end = offset + HEADER_SIZE + payload_length

        if record_end > total:
            _log_error(
                sink,
                ERR_TRUNCATED,
                stream,
                offset,
                f"declared_payload={payload_length} "
                f"available={total - offset - HEADER_SIZE}",
            )
            break

        if payload_length != FRAME_BYTES:
            _log_error(
                sink,
                ERR_BAD_LENGTH,
                stream,
                offset,
                f"payload_length={payload_length} expected={FRAME_BYTES}",
            )
            # Framing is still intact: skip the record and keep going.
            offset = record_end
            continue

        payload = data[offset + HEADER_SIZE:record_end]
        packets.append(
            Packet(
                seq=seq,
                timestamp=timestamp,
                payload_length=payload_length,
                payload=payload,
                offset=offset,
            )
        )
        offset = record_end

    check_sequence_continuity(packets, stream, sink)
    return packets


def build_packet(seq: int, timestamp: int, pcm: bytes) -> bytes:
    """Serialize one packet (used by the deterministic simulator).

    The payload-length field always stores the byte count of ``pcm``.
    """
    return struct.pack(
        HEADER_FORMAT,
        seq & 0xFFFF,
        timestamp & 0xFFFFFFFF,
        len(pcm),
    ) + pcm


def iter_packet_files(directory: str) -> Iterator[str]:
    """Yield ``stream_<id>.pkt`` paths ordered by numeric stream id."""
    names = []
    for name in os.listdir(directory):
        if name.startswith("stream_") and name.endswith(".pkt"):
            stem = name[len("stream_"):-len(".pkt")]
            if stem.isdigit():
                names.append((int(stem), name))
    names.sort()
    for _, name in names:
        yield os.path.join(directory, name)


def stream_id_from_path(path: str) -> int:
    """Extract the numeric stream id from a ``stream_<id>.pkt`` path."""
    name = os.path.basename(path)
    stem = name[len("stream_"):-len(".pkt")]
    return int(stem)
