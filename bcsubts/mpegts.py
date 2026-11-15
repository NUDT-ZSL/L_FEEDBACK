from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .scte35 import crc32_mpeg


TS_PACKET_SIZE = 188


@dataclass
class StreamInfo:
    stream_type: int
    pid: int
    descriptors: bytes = b""


@dataclass
class PmtInfo:
    pmt_pid: int
    program_number: int
    version: int
    pcr_pid: int
    program_info: bytes
    streams: list[StreamInfo]


def iter_ts_packets(data: bytes) -> Iterable[bytes]:
    if len(data) % TS_PACKET_SIZE:
        data = data[: len(data) - (len(data) % TS_PACKET_SIZE)]
    for offset in range(0, len(data), TS_PACKET_SIZE):
        packet = data[offset : offset + TS_PACKET_SIZE]
        if len(packet) == TS_PACKET_SIZE and packet[0] == 0x47:
            yield packet


def packet_pid(packet: bytes) -> int:
    return ((packet[1] & 0x1F) << 8) | packet[2]


def packet_cc(packet: bytes) -> int:
    return packet[3] & 0x0F


def packet_pusi(packet: bytes) -> bool:
    return bool(packet[1] & 0x40)


def packet_payload(packet: bytes) -> bytes:
    afc = (packet[3] >> 4) & 0x03
    offset = 4
    if afc & 0x02:
        adaptation_length = packet[4]
        offset = 5 + adaptation_length
    if afc & 0x01 and offset < TS_PACKET_SIZE:
        return packet[offset:]
    return b""


def _section_from_psi_payload(payload: bytes) -> bytes:
    if not payload:
        return b""
    pointer = payload[0]
    start = 1 + pointer
    if start + 3 > len(payload):
        return b""
    length = ((payload[start + 1] & 0x0F) << 8) | payload[start + 2]
    total = 3 + length
    if start + total > len(payload):
        return b""
    return payload[start : start + total]


def parse_pat(data: bytes) -> tuple[int, int] | None:
    for packet in iter_ts_packets(data):
        if packet_pid(packet) != 0 or not packet_pusi(packet):
            continue
        section = _section_from_psi_payload(packet_payload(packet))
        if len(section) < 12 or section[0] != 0x00:
            continue
        program_number = int.from_bytes(section[3:5], "big")
        pmt_pid = ((section[10] & 0x1F) << 8) | section[11]
        if program_number:
            return program_number, pmt_pid
    return None


def parse_pmt_section(section: bytes, pmt_pid: int) -> PmtInfo:
    if len(section) < 16 or section[0] != 0x02:
        raise ValueError("invalid PMT section")
    program_number = int.from_bytes(section[3:5], "big")
    version = (section[5] >> 1) & 0x1F
    pcr_pid = ((section[8] & 0x1F) << 8) | section[9]
    program_info_length = ((section[10] & 0x0F) << 8) | section[11]
    program_info = section[12 : 12 + program_info_length]
    position = 12 + program_info_length
    end = len(section) - 4
    streams: list[StreamInfo] = []
    while position + 5 <= end:
        stream_type = section[position]
        pid = ((section[position + 1] & 0x1F) << 8) | section[position + 2]
        info_length = ((section[position + 3] & 0x0F) << 8) | section[position + 4]
        descriptors = section[position + 5 : position + 5 + info_length]
        streams.append(StreamInfo(stream_type, pid, descriptors))
        position += 5 + info_length
    return PmtInfo(pmt_pid, program_number, version, pcr_pid, program_info, streams)


def find_pmt(data: bytes) -> PmtInfo:
    pat = parse_pat(data)
    if pat is None:
        raise ValueError("PAT not found")
    _program, pmt_pid = pat
    for packet in iter_ts_packets(data):
        if packet_pid(packet) == pmt_pid and packet_pusi(packet):
            section = _section_from_psi_payload(packet_payload(packet))
            if section and section[0] == 0x02:
                return parse_pmt_section(section, pmt_pid)
    raise ValueError("PMT not found")


def _build_psi_packet(pid: int, section: bytes, cc: int, pusi: bool = True) -> bytes:
    payload = (b"\x00" if pusi else b"") + section
    if len(payload) > 184:
        raise ValueError("PSI section does not fit one packet")
    payload = payload + b"\xFF" * (184 - len(payload))
    header = bytes(
        [
            0x47,
            (0x40 if pusi else 0x00) | ((pid >> 8) & 0x1F),
            pid & 0xFF,
            0x10 | (cc & 0x0F),
        ]
    )
    return header + payload


def build_pmt_section(
    original: PmtInfo,
    extra_streams: list[StreamInfo],
    program_descriptors: bytes = b"\x05\x04CUEI",
) -> bytes:
    program_info = original.program_info
    if b"CUEI" not in program_info:
        program_info += program_descriptors
    streams = list(original.streams)
    used_pids = {stream.pid for stream in streams}
    for extra in extra_streams:
        if extra.pid not in used_pids:
            streams.append(extra)
            used_pids.add(extra.pid)
    body = bytearray()
    body.extend(original.program_number.to_bytes(2, "big"))
    version = (original.version + 1) & 0x1F
    body.append(0xC1 | (version << 1))
    body.append(0)
    body.append(0)
    body.extend((0xE000 | (original.pcr_pid & 0x1FFF)).to_bytes(2, "big"))
    body.extend((0xF000 | (len(program_info) & 0x0FFF)).to_bytes(2, "big"))
    body.extend(program_info)
    for stream in streams:
        body.append(stream.stream_type)
        body.extend((0xE000 | (stream.pid & 0x1FFF)).to_bytes(2, "big"))
        body.extend((0xF000 | (len(stream.descriptors) & 0x0FFF)).to_bytes(2, "big"))
        body.extend(stream.descriptors)
    section_length = len(body) + 4
    header = bytes([0x02, 0xB0 | ((section_length >> 8) & 0x0F), section_length & 0xFF])
    section = header + bytes(body)
    return section + crc32_mpeg(section).to_bytes(4, "big")


def rewrite_pmt_packet(packet: bytes, section: bytes) -> bytes:
    if len(section) + 1 > 184:
        raise ValueError("new PMT does not fit one TS packet")
    cc = packet_cc(packet)
    pid = packet_pid(packet)
    return _build_psi_packet(pid, section, cc)


def make_ts_packet(pid: int, payload: bytes, pusi: bool, cc: int) -> bytes:
    if len(payload) > 184:
        raise ValueError("payload exceeds one TS packet")
    header = bytearray(
        [
            0x47,
            (0x40 if pusi else 0x00) | ((pid >> 8) & 0x1F),
            pid & 0xFF,
            0,
        ]
    )
    if len(payload) == 184:
        header[3] = 0x10 | (cc & 0x0F)
        packet = bytes(header) + payload
    else:
        header[3] = 0x30 | (cc & 0x0F)
        adaptation_length = 183 - len(payload)
        adaptation = bytearray([adaptation_length])
        if adaptation_length:
            adaptation.append(0)
            adaptation.extend(b"\xFF" * (adaptation_length - 1))
        packet = bytes(header) + bytes(adaptation) + payload
    if len(packet) != 188:
        raise AssertionError(f"TS packet length is {len(packet)}")
    return packet


def packetize_pes(pid: int, pes: bytes, cc: int) -> tuple[list[bytes], int]:
    packets: list[bytes] = []
    position = 0
    first = True
    while position < len(pes):
        chunk = pes[position : position + 184]
        packets.append(make_ts_packet(pid, chunk, first, cc))
        first = False
        position += len(chunk)
        cc = (cc + 1) & 0x0F
    return packets, cc


def packetize_section(pid: int, section: bytes, cc: int) -> tuple[list[bytes], int]:
    packets: list[bytes] = []
    first_payload = b"\x00" + section
    position = 0
    first = True
    while position < len(first_payload):
        capacity = 184
        chunk = first_payload[position : position + capacity]
        packets.append(make_ts_packet(pid, chunk, first, cc))
        first = False
        position += len(chunk)
        cc = (cc + 1) & 0x0F
    return packets, cc


def decode_pts(data: bytes) -> int:
    return (
        ((data[0] >> 1) & 0x07) << 30
        | data[1] << 22
        | ((data[2] >> 1) & 0x7F) << 15
        | data[3] << 7
        | ((data[4] >> 1) & 0x7F)
    )


def parse_pes_header(pes: bytes) -> dict[str, int | bytes]:
    if len(pes) < 9 or pes[:3] != b"\x00\x00\x01":
        raise ValueError("invalid PES header")
    packet_length = int.from_bytes(pes[4:6], "big")
    flags = pes[7]
    header_length = pes[8]
    payload_start = 9 + header_length
    pts = None
    if flags & 0x80 and header_length >= 5 and len(pes) >= 14:
        pts = decode_pts(pes[9:14])
    return {
        "stream_id": pes[3],
        "packet_length": packet_length,
        "pts": pts,
        "header_length": header_length,
        "payload_start": payload_start,
        "payload": pes[payload_start:],
    }


def audio_pes_points(data: bytes, audio_pid: int) -> list[dict[str, int]]:
    points: list[dict[str, int]] = []
    for index, packet in enumerate(iter_ts_packets(data)):
        if packet_pid(packet) != audio_pid or not packet_pusi(packet):
            continue
        payload = packet_payload(packet)
        if len(payload) < 14 or payload[:3] != b"\x00\x00\x01":
            continue
        try:
            header = parse_pes_header(payload)
        except ValueError:
            continue
        pts = header.get("pts")
        if isinstance(pts, int):
            points.append({"packet_index": index, "pts": pts})
    return points


def extract_pes_streams(data: bytes, pid: int) -> list[dict[str, object]]:
    streams: list[dict[str, object]] = []
    buffer = bytearray()
    start_packet = 0
    expected_length: int | None = None
    for index, packet in enumerate(iter_ts_packets(data)):
        if packet_pid(packet) != pid:
            continue
        payload = packet_payload(packet)
        if not payload:
            continue
        if packet_pusi(packet):
            if buffer:
                _finalize_pes(buffer, start_packet, streams)
            buffer = bytearray(payload)
            start_packet = index
            expected_length = None
            if len(buffer) >= 6:
                packet_length = int.from_bytes(buffer[4:6], "big")
                expected_length = 6 + packet_length if packet_length else None
        else:
            buffer.extend(payload)
        if expected_length is not None and len(buffer) >= expected_length:
            _finalize_pes(buffer[:expected_length], start_packet, streams)
            buffer = bytearray()
            expected_length = None
    if buffer:
        _finalize_pes(buffer, start_packet, streams)
    return streams


def _finalize_pes(buffer: bytearray, start_packet: int, streams: list[dict[str, object]]) -> None:
    try:
        header = parse_pes_header(bytes(buffer))
    except ValueError:
        return
    streams.append(
        {
            "start_packet": start_packet,
            "pts": header.get("pts"),
            "payload": header.get("payload", b""),
            "stream_id": header.get("stream_id"),
        }
    )


def extract_private_sections(data: bytes, pid: int) -> list[bytes]:
    sections: list[bytes] = []
    buffer = bytearray()
    for packet in iter_ts_packets(data):
        if packet_pid(packet) != pid:
            continue
        payload = packet_payload(packet)
        if not payload:
            continue
        if packet_pusi(packet):
            pointer = payload[0]
            prefix = payload[1 : 1 + pointer]
            if buffer:
                buffer.extend(prefix)
                _drain_sections(buffer, sections)
            buffer.extend(payload[1 + pointer :])
        else:
            buffer.extend(payload)
        _drain_sections(buffer, sections)
    return sections


def _drain_sections(buffer: bytearray, sections: list[bytes]) -> None:
    while len(buffer) >= 3:
        if buffer[0] == 0xFF:
            buffer.clear()
            return
        total = 3 + (((buffer[1] & 0x0F) << 8) | buffer[2])
        if total > 4096:
            buffer.clear()
            return
        if len(buffer) < total:
            return
        sections.append(bytes(buffer[:total]))
        del buffer[:total]


def choose_free_pid(existing: set[int], preferred: int, start: int = 0x1F0) -> int:
    if preferred not in existing:
        return preferred
    for candidate in range(start, 0x1FF0):
        if candidate not in existing:
            return candidate
    raise ValueError("no free MPEG-TS PID available")
