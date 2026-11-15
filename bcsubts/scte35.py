from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import BusinessError


PTS_MASK = (1 << 33) - 1
SCTE_STREAM_TYPE = 0x86
SPLICE_INSERT = 0x05
SPLICE_NULL = 0x00


class BitWriter:
    def __init__(self) -> None:
        self._bits: list[int] = []

    def write(self, value: int, width: int) -> None:
        if value < 0 or value >= (1 << width):
            raise ValueError(f"value {value} does not fit in {width} bits")
        for shift in range(width - 1, -1, -1):
            self._bits.append((value >> shift) & 1)

    def bytes(self) -> bytes:
        while len(self._bits) % 8:
            self._bits.append(0)
        output = bytearray()
        for start in range(0, len(self._bits), 8):
            byte = 0
            for bit in self._bits[start : start + 8]:
                byte = (byte << 1) | bit
            output.append(byte)
        return bytes(output)


class BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.bit = 0

    def read(self, width: int) -> int:
        if self.bit + width > len(self.data) * 8:
            raise ValueError("truncated bitstream")
        value = 0
        for _ in range(width):
            byte = self.data[self.bit // 8]
            bit = (byte >> (7 - self.bit % 8)) & 1
            value = (value << 1) | bit
            self.bit += 1
        return value

    def align(self) -> None:
        self.bit = (self.bit + 7) // 8 * 8


def _crc_table() -> list[int]:
    table: list[int] = []
    for value in range(256):
        crc = value << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) if crc & 0x80000000 else (crc << 1)
        table.append(crc & 0xFFFFFFFF)
    return table


_CRC_TABLE = _crc_table()


def crc32_mpeg(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _CRC_TABLE[((crc >> 24) ^ byte) & 0xFF]
    return crc & 0xFFFFFFFF


@dataclass
class SegmentationDescriptor:
    segmentation_type_id: int = 0x34
    segment_num: int = 0
    segments_expected: int = 1
    sub_segment_num: int | None = None
    sub_segments_expected: int | None = None
    upid_type: int = 0x00
    upid: str = ""


@dataclass
class ScteEvent:
    event_id: int
    event_type: str
    offset_ms: float
    duration_ms: float
    segmentation_descriptor: SegmentationDescriptor | None = None
    offset_sample: int = 0
    target_pts: int = 0
    section: bytes | None = None


def _encode_segmentation_descriptor(event: ScteEvent) -> bytes:
    descriptor = event.segmentation_descriptor
    if descriptor is None:
        return b""
    writer = BitWriter()
    writer.write(0x43554549, 32)  # "CUEI"
    writer.write(event.event_id, 32)
    writer.write(0, 1)
    writer.write(0x7F, 7)
    duration_flag = event.duration_ms > 0
    writer.write(1, 1)  # program_segmentation_flag
    writer.write(1 if duration_flag else 0, 1)
    writer.write(0x3F, 6)
    if duration_flag:
        writer.write(int(round(event.duration_ms * 90)) & ((1 << 40) - 1), 40)
    upid = descriptor.upid.encode("utf-8")
    if len(upid) > 255:
        raise BusinessError(
            "E_FORMAT",
            "scte35_build",
            "Segmentation UPID exceeds 255 bytes",
            {"event_id": event.event_id, "upid_length": len(upid)},
        )
    writer.write(descriptor.upid_type, 8)
    writer.write(len(upid), 8)
    for byte in upid:
        writer.write(byte, 8)
    writer.write(descriptor.segmentation_type_id, 8)
    writer.write(descriptor.segment_num, 8)
    writer.write(descriptor.segments_expected, 8)
    if 0x30 <= descriptor.segmentation_type_id <= 0x3F:
        writer.write(descriptor.sub_segment_num or 0, 8)
        writer.write(descriptor.sub_segments_expected or 1, 8)
    payload = writer.bytes()
    return bytes([0x02, len(payload)]) + payload


def build_splice_info(event: ScteEvent) -> bytes:
    command = BitWriter()
    if event.event_type == "splice_insert":
        command.write(command_type := SPLICE_INSERT, 8)
        command.write(event.event_id, 32)
        command.write(0, 1)
        command.write(0x7F, 7)
        command.write(1, 1)  # out_of_network_indicator
        command.write(1, 1)  # program_splice_flag
        command.write(1 if event.duration_ms > 0 else 0, 1)
        command.write(0, 1)  # splice_immediate_flag
        command.write(0x0F, 4)
        command.write(1, 1)  # time_specified_flag
        command.write(0x3F, 6)
        command.write(0, 33)  # pts_time; pts_adjustment carries the audio PTS
        if event.duration_ms > 0:
            command.write(0, 1)  # auto_return
            command.write(0x3F, 6)
            command.write(int(round(event.duration_ms * 90)) & PTS_MASK, 33)
        command.write(0, 16)  # unique_program_id
        command.write(0, 8)  # avail_num
        command.write(0, 8)  # avails_expected
        command_bytes = command.bytes()
        command_length = len(command_bytes)
        command_type = command_type
    elif event.event_type == "splice_null":
        command_bytes = b""
        command_length = 0
        command_type = SPLICE_NULL
    else:
        raise BusinessError(
            "E_FORMAT",
            "scte35_build",
            "Unsupported SCTE-35 event type",
            {"event_id": event.event_id, "type": event.event_type},
        )

    descriptors = _encode_segmentation_descriptor(event)
    body = BitWriter()
    body.write(0, 1)  # section_syntax_indicator
    body.write(0, 1)  # private_indicator
    body.write(3, 2)  # sap_type
    body.write(0, 12)  # section_length placeholder
    body.write(0, 8)  # protocol_version
    body.write(0, 1)  # encrypted_packet
    body.write(0, 6)  # encryption_algorithm
    body.write(event.target_pts & PTS_MASK, 33)
    body.write(0, 8)  # cw_index
    body.write(0xFFF, 12)  # tier
    body.write(command_length, 12)
    body.write(command_type, 8)
    for byte in command_bytes[1:]:  # command type already written above
        body.write(byte, 8)
    body.write(len(descriptors), 16)
    for byte in descriptors:
        body.write(byte, 8)
    preliminary = body.bytes()
    # section_length counts bytes after the section_length field through CRC.
    section_length = len(preliminary) + 2
    if section_length > 0xFFF:
        raise BusinessError(
            "E_FORMAT",
            "scte35_build",
            "SCTE-35 section is too large",
            {"event_id": event.event_id, "section_length": section_length},
        )
    header = bytearray(preliminary)
    header[0] = (header[0] & 0xF0) | ((section_length >> 8) & 0x0F)
    header[1] = section_length & 0xFF
    with_crc = bytes([0xFC]) + bytes(header) + b"\x00\x00\x00\x00"
    crc = crc32_mpeg(with_crc[:-4])
    return with_crc[:-4] + crc.to_bytes(4, "big")


def parse_splice_info(section: bytes, verify_crc: bool = True) -> dict[str, Any]:
    if len(section) < 20 or section[0] != 0xFC:
        raise ValueError("not an SCTE-35 splice_info_section")
    section_length = ((section[1] & 0x0F) << 8) | section[2]
    total = 3 + section_length
    if total > len(section):
        raise ValueError("truncated SCTE-35 section")
    section = section[:total]
    if verify_crc:
        expected = int.from_bytes(section[-4:], "big")
        actual = crc32_mpeg(section[:-4])
        if expected != actual:
            raise ValueError("SCTE-35 CRC mismatch")
    reader = BitReader(section)
    reader.read(8)  # table_id
    reader.read(1)
    reader.read(1)
    sap_type = reader.read(2)
    reader.read(12)
    protocol_version = reader.read(8)
    encrypted = reader.read(1)
    encryption_algorithm = reader.read(6)
    pts_adjustment = reader.read(33)
    cw_index = reader.read(8)
    tier = reader.read(12)
    command_length = reader.read(12)
    command_type = reader.read(8)
    command_start = reader.bit
    result: dict[str, Any] = {
        "table_id": 0xFC,
        "section_length": section_length,
        "sap_type": sap_type,
        "protocol_version": protocol_version,
        "encrypted_packet": encrypted,
        "encryption_algorithm": encryption_algorithm,
        "pts_adjustment": pts_adjustment,
        "cw_index": cw_index,
        "tier": tier,
        "command_length": command_length,
        "splice_command_type": command_type,
        "crc32": int.from_bytes(section[-4:], "big"),
    }
    if command_type == SPLICE_INSERT:
        event_id = reader.read(32)
        cancel = reader.read(1)
        reader.read(7)
        out_of_network = reader.read(1)
        program_splice = reader.read(1)
        duration_flag = reader.read(1)
        immediate = reader.read(1)
        reader.read(4)
        splice_time = None
        if program_splice and not immediate:
            time_specified = reader.read(1)
            reader.read(6)
            splice_time = reader.read(33) if time_specified else None
        break_duration = None
        auto_return = None
        if duration_flag:
            auto_return = reader.read(1)
            reader.read(6)
            break_duration = reader.read(33)
        result.update(
            event_id=event_id,
            splice_event_cancel=cancel,
            out_of_network_indicator=out_of_network,
            program_splice_flag=program_splice,
            duration_flag=duration_flag,
            splice_immediate_flag=immediate,
            splice_time=splice_time,
            auto_return=auto_return,
            break_duration=break_duration,
            effective_splice_pts=(pts_adjustment + (splice_time or 0)) & PTS_MASK,
        )
    elif command_type != SPLICE_NULL:
        result.update(event_id=None)
    else:
        result.update(event_id=None, splice_time=None)
    reader.bit = command_start + max(0, command_length - 1) * 8
    descriptor_length = reader.read(16)
    descriptor_end = reader.bit + descriptor_length * 8
    descriptors = []
    while reader.bit + 16 <= descriptor_end:
        tag = reader.read(8)
        length = reader.read(8)
        payload = bytes(reader.read(8) for _ in range(length))
        descriptors.append({"tag": tag, "length": length, "payload": payload})
    result["descriptors"] = descriptors
    result["descriptor_loop_length"] = descriptor_length
    return result


def _parse_segmentation_descriptor(event_id: int, payload: bytes) -> SegmentationDescriptor | None:
    if len(payload) < 8 or payload[:4] != b"CUEI":
        return None
    reader = BitReader(payload)
    reader.read(32)
    seg_event_id = reader.read(32)
    if seg_event_id != event_id:
        return None
    reader.read(1)
    reader.read(7)
    program_flag = reader.read(1)
    duration_flag = reader.read(1)
    reader.read(6)
    if duration_flag:
        reader.read(40)
    if not program_flag:
        component_count = reader.read(8)
        for _ in range(component_count):
            reader.read(8)
            reader.read(1)
            reader.read(7)
            reader.read(33)
    upid_type = reader.read(8)
    upid_len = reader.read(8)
    upid = bytes(reader.read(8) for _ in range(upid_len)).decode("utf-8", errors="replace")
    seg_type = reader.read(8)
    segment_num = reader.read(8)
    segments_expected = reader.read(8)
    sub_num = sub_expected = None
    if 0x30 <= seg_type <= 0x3F and reader.bit + 16 <= len(payload) * 8:
        sub_num = reader.read(8)
        sub_expected = reader.read(8)
    return SegmentationDescriptor(
        segmentation_type_id=seg_type,
        segment_num=segment_num,
        segments_expected=segments_expected,
        sub_segment_num=sub_num,
        sub_segments_expected=sub_expected,
        upid_type=upid_type,
        upid=upid,
    )


def parse_command_table(path: str | Path, audio_duration_ms: float, sample_rate: int = 48000) -> list[ScteEvent]:
    path = Path(path)
    if not path.is_file():
        raise BusinessError("E_INPUT", "scte35_parse", "SCTE-35 JSON not found", {"path": str(path)})
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise BusinessError(
            "E_FORMAT",
            "scte35_parse",
            "Invalid SCTE-35 JSON",
            {"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    rows = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise BusinessError(
            "E_FORMAT",
            "scte35_parse",
            "SCTE-35 JSON must be a list or an object with an events list",
        )
    events: list[ScteEvent] = []
    seen: set[int] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BusinessError("E_FORMAT", "scte35_parse", "Event entry must be an object", {"index": index})
        try:
            raw_id = row["event_id"]
            event_id = int(raw_id)
            event_type = str(row["type"])
            offset_ms = float(row["offset_ms"])
            duration_ms = float(row.get("duration_ms", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise BusinessError(
                "E_FORMAT",
                "scte35_parse",
                "Event is missing or has invalid fields",
                {"index": index, "error": str(exc)},
            ) from exc
        if event_id < 0 or event_id > 0xFFFFFFFF:
            raise BusinessError(
                "E_FORMAT",
                "scte35_parse",
                "event_id must be in the 32-bit unsigned range",
                {"index": index, "event_id": event_id},
            )
        if event_id in seen:
            raise BusinessError(
                "E_FORMAT",
                "scte35_parse",
                "Duplicate SCTE-35 event_id",
                {"index": index, "event_id": event_id},
            )
        seen.add(event_id)
        if event_type not in {"splice_insert", "splice_null"}:
            raise BusinessError(
                "E_FORMAT",
                "scte35_parse",
                "type must be splice_insert or splice_null",
                {"index": index, "type": event_type},
            )
        if offset_ms < 0 or offset_ms > audio_duration_ms:
            raise BusinessError(
                "E_OVERFLOW",
                "scte35_parse",
                "SCTE-35 offset is outside the audio duration",
                {
                    "index": index,
                    "event_id": event_id,
                    "offset_ms": offset_ms,
                    "audio_duration_ms": round(audio_duration_ms, 3),
                },
            )
        if duration_ms < 0:
            raise BusinessError(
                "E_OVERFLOW",
                "scte35_parse",
                "SCTE-35 duration_ms must be non-negative",
                {"index": index, "event_id": event_id, "duration_ms": duration_ms},
            )
        seg = None
        raw_seg = row.get("segmentation_descriptor")
        if raw_seg is not None:
            if not isinstance(raw_seg, dict):
                raise BusinessError(
                    "E_FORMAT",
                    "scte35_parse",
                    "segmentation_descriptor must be an object",
                    {"index": index},
                )
            try:
                seg = SegmentationDescriptor(
                    segmentation_type_id=int(raw_seg.get("segmentation_type_id", 0x34)),
                    segment_num=int(raw_seg.get("segment_num", 0)),
                    segments_expected=int(raw_seg.get("segments_expected", 1)),
                    sub_segment_num=(
                        int(raw_seg["sub_segment_num"]) if "sub_segment_num" in raw_seg else None
                    ),
                    sub_segments_expected=(
                        int(raw_seg["sub_segments_expected"])
                        if "sub_segments_expected" in raw_seg
                        else None
                    ),
                    upid_type=int(raw_seg.get("upid_type", 0)),
                    upid=str(raw_seg.get("upid", "")),
                )
            except (TypeError, ValueError) as exc:
                raise BusinessError(
                    "E_FORMAT",
                    "scte35_parse",
                    "Invalid segmentation_descriptor",
                    {"index": index, "error": str(exc)},
                ) from exc
        event = ScteEvent(
            event_id=event_id,
            event_type=event_type,
            offset_ms=offset_ms,
            duration_ms=duration_ms,
            segmentation_descriptor=seg,
        )
        events.append(event)
    events.sort(key=lambda item: (item.offset_ms, item.event_id))
    for event in events:
        event.offset_sample = int(round(event.offset_ms * sample_rate / 1000))
    return events


def build_event_sections(events: list[ScteEvent], origin_pts: int = 0) -> list[ScteEvent]:
    for event in events:
        target = (origin_pts + int(round(event.offset_sample * 90000 / 48000))) & PTS_MASK
        event.target_pts = target
        event.section = build_splice_info(event)
        parsed = parse_splice_info(event.section)
        if event.event_type == "splice_insert" and parsed.get("event_id") != event.event_id:
            raise BusinessError(
                "E_INTERNAL",
                "scte35_build",
                "Internal SCTE-35 round-trip check failed",
                {"event_id": event.event_id},
            )
    return events


def descriptor_summary(event: ScteEvent) -> dict[str, Any] | None:
    descriptor = event.segmentation_descriptor
    if descriptor is None:
        return None
    return {
        "segmentation_type_id": descriptor.segmentation_type_id,
        "segment_num": descriptor.segment_num,
        "segments_expected": descriptor.segments_expected,
        "sub_segment_num": descriptor.sub_segment_num,
        "sub_segments_expected": descriptor.sub_segments_expected,
        "upid_type": descriptor.upid_type,
        "upid": descriptor.upid,
    }
