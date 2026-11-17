"""Minimal MPEG-TS structure validator.

Parses 188-byte TS packets, extracts the PMT to identify audio/video PIDs
and validates PES_packet_length semantics: audio PES must carry a non-zero
length that matches the actual payload extent; video PES may use length 0
(unbounded) per ISO/IEC 13818-1.
"""
from pathlib import Path

TS_PACKET_SIZE = 188
SYNC_BYTE = 0x47
STREAM_TYPE_H264 = 0x1B
STREAM_TYPE_AAC = 0x0F


class TsFormatError(RuntimeError):
    pass


def _parse_pmt(data, pmt_pid):
    video_pid = audio_pid = None
    offset = 0
    while offset + TS_PACKET_SIZE <= len(data):
        pkt = data[offset:offset + TS_PACKET_SIZE]
        offset += TS_PACKET_SIZE
        pid = ((pkt[1] & 0x1F) << 8) | pkt[2]
        if pid != pmt_pid or not (pkt[1] & 0x40):
            continue
        afc = (pkt[3] >> 4) & 0x03
        pos = 4
        if afc in (2, 3):
            pos += 1 + pkt[4]
        if afc not in (1, 3) or pos >= TS_PACKET_SIZE:
            continue
        pos += pkt[pos] + 1
        section = pkt[pos:]
        if len(section) < 12:
            continue
        section_length = ((section[1] & 0x0F) << 8) | section[2]
        program_info_length = ((section[10] & 0x0F) << 8) | section[11]
        es = 12 + program_info_length
        end = 3 + section_length - 4
        while es + 5 <= min(end, len(section)):
            stream_type = section[es]
            elem_pid = ((section[es + 1] & 0x1F) << 8) | section[es + 2]
            es_info_length = ((section[es + 3] & 0x0F) << 8) | section[es + 4]
            if stream_type == STREAM_TYPE_H264:
                video_pid = elem_pid
            elif stream_type == STREAM_TYPE_AAC:
                audio_pid = elem_pid
            es += 5 + es_info_length
        break
    return video_pid, audio_pid


def _find_pmt_pid(data):
    offset = 0
    while offset + TS_PACKET_SIZE <= len(data):
        pkt = data[offset:offset + TS_PACKET_SIZE]
        offset += TS_PACKET_SIZE
        pid = ((pkt[1] & 0x1F) << 8) | pkt[2]
        if pid != 0 or not (pkt[1] & 0x40):
            continue
        afc = (pkt[3] >> 4) & 0x03
        pos = 4
        if afc in (2, 3):
            pos += 1 + pkt[4]
        pos += pkt[pos] + 1
        section = pkt[pos:]
        if len(section) < 12:
            continue
        return ((section[10] & 0x1F) << 8) | section[11]
    raise TsFormatError("PAT not found")


def validate(path):
    data = Path(path).read_bytes()
    if len(data) < TS_PACKET_SIZE or len(data) % TS_PACKET_SIZE != 0:
        raise TsFormatError(f"{path}: size {len(data)} is not a multiple of 188")
    pmt_pid = _find_pmt_pid(data)
    video_pid, audio_pid = _parse_pmt(data, pmt_pid)
    if video_pid is None or audio_pid is None:
        raise TsFormatError("PMT does not declare both H.264 video and AAC audio")

    pes_violations = []
    pes_counts = {"video": 0, "audio": 0}
    remaining = {}
    sync_errors = 0
    offset = 0
    packet_index = 0
    while offset + TS_PACKET_SIZE <= len(data):
        pkt = data[offset:offset + TS_PACKET_SIZE]
        if pkt[0] != SYNC_BYTE:
            sync_errors += 1
        pid = ((pkt[1] & 0x1F) << 8) | pkt[2]
        pusi = bool(pkt[1] & 0x40)
        afc = (pkt[3] >> 4) & 0x03
        pos = 4
        if afc in (2, 3):
            pos += 1 + pkt[4]
        payload = pkt[pos:] if afc in (1, 3) and pos < TS_PACKET_SIZE else b""
        for label, av_pid in (("video", video_pid), ("audio", audio_pid)):
            if pid != av_pid:
                continue
            if pusi and payload[:3] == b"\x00\x00\x01":
                pes_len = (payload[4] << 8) | payload[5]
                if label == "audio":
                    if pes_len == 0:
                        pes_violations.append(
                            f"packet {packet_index}: audio PES length must be non-zero")
                    else:
                        remaining[pid] = pes_len + 6 - len(payload)
                pes_counts[label] += 1
            elif pid in remaining:
                remaining[pid] -= len(payload)
                if remaining[pid] <= 0:
                    del remaining[pid]
        offset += TS_PACKET_SIZE
        packet_index += 1
    if sync_errors:
        raise TsFormatError(f"{sync_errors} TS packets lost sync")
    return {
        "pmt_pid": pmt_pid,
        "video_pid": video_pid,
        "audio_pid": audio_pid,
        "video_pes_count": pes_counts["video"],
        "audio_pes_count": pes_counts["audio"],
        "pes_length_violations": pes_violations,
        "ts_packets": packet_index,
    }
