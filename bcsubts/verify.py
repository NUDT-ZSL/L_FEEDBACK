from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .dvb import parse_subtitle_pes
from .errors import BusinessError
from .ffmpeg_tools import probe_json
from .mpegts import audio_pes_points, extract_pes_streams, extract_private_sections, find_pmt
from .scte35 import SCTE_STREAM_TYPE, parse_splice_info


def _has_subtitling_descriptor(descriptors: bytes) -> bool:
    position = 0
    while position + 2 <= len(descriptors):
        tag = descriptors[position]
        length = descriptors[position + 1]
        if tag == 0x59 and length >= 8:
            return True
        position += 2 + length
    return False


def verify_output(
    ts_path: str | Path,
    ffprobe: str,
    ffmpeg: str,
    scte_log: dict[str, Any],
    expected_subtitle_starts_ms: list[float],
    expected_clear_count: int,
) -> dict[str, Any]:
    ts_path = Path(ts_path)
    if not ts_path.is_file() or ts_path.stat().st_size == 0:
        raise BusinessError("E_VERIFY", "verify", "Output TS is missing or empty", {"path": str(ts_path)})
    data = ts_path.read_bytes()
    if len(data) % 188:
        raise BusinessError(
            "E_VERIFY",
            "verify",
            "Output TS length is not a multiple of 188 bytes",
            {"size": len(data)},
        )
    if any(data[index] != 0x47 for index in range(0, len(data), 188)):
        raise BusinessError("E_VERIFY", "verify", "Output TS contains invalid sync bytes")
    try:
        pmt = find_pmt(data)
    except ValueError as exc:
        raise BusinessError("E_VERIFY", "verify", f"Unable to parse output PMT: {exc}") from exc

    scte_streams = [stream for stream in pmt.streams if stream.stream_type == SCTE_STREAM_TYPE]
    subtitle_streams = [
        stream
        for stream in pmt.streams
        if stream.stream_type == 0x06 and _has_subtitling_descriptor(stream.descriptors)
    ]
    if not scte_streams:
        raise BusinessError("E_VERIFY", "verify", "No SCTE-35 stream (type 0x86) in PMT")
    if not subtitle_streams:
        raise BusinessError("E_VERIFY", "verify", "No DVB subtitle stream in PMT")
    if b"CUEI" not in pmt.program_info:
        raise BusinessError("E_VERIFY", "verify", "PMT program registration descriptor CUEI is missing")

    scte_pid = scte_streams[0].pid
    subtitle_pid = subtitle_streams[0].pid
    sections = extract_private_sections(data, scte_pid)
    parsed_sections: list[dict[str, Any]] = []
    for section in sections:
        try:
            parsed_sections.append(parse_splice_info(section))
        except ValueError as exc:
            raise BusinessError(
                "E_VERIFY",
                "verify",
                f"Invalid SCTE-35 section: {exc}",
                {"pid": scte_pid},
            ) from exc

    expected_events = scte_log.get("events", [])
    expected_by_id = {
        int(item["event_id"]): item
        for item in expected_events
        if item.get("type") == "splice_insert"
    }
    expected_types = [
        0x00 if item.get("type") == "splice_null" else 0x05 for item in expected_events
    ]
    actual_types = [int(item["splice_command_type"]) for item in parsed_sections]
    if actual_types != expected_types:
        raise BusinessError(
            "E_VERIFY",
            "verify",
            "SCTE-35 command sequence in the TS does not match the packaging log",
            {"expected": expected_types, "actual": actual_types},
        )
    actual_ids = [
        item.get("event_id")
        for item in parsed_sections
        if item.get("splice_command_type") == 0x05
    ]
    actual_set = {int(item) for item in actual_ids if item is not None}
    if len(actual_ids) != len(actual_set):
        raise BusinessError(
            "E_VERIFY",
            "verify",
            "Duplicate SCTE-35 event IDs were found in the TS",
            {"event_ids": actual_ids},
        )
    if actual_set != set(expected_by_id):
        raise BusinessError(
            "E_VERIFY",
            "verify",
            "SCTE-35 event IDs in the TS do not match the packaging log",
            {"expected": sorted(expected_by_id), "actual": sorted(actual_set)},
        )
    for section, expected in zip(parsed_sections, expected_events, strict=False):
        if section["pts_adjustment"] != expected["target_pts_90k"]:
            raise BusinessError(
                "E_VERIFY",
                "verify",
                "SCTE-35 pts_adjustment does not match the audio PTS mapping",
                {
                    "event_id": expected.get("event_id"),
                    "expected_pts": expected["target_pts_90k"],
                    "actual_pts_adjustment": section["pts_adjustment"],
                },
            )
        insertion_pts = expected.get("transport_insertion_pts_90k")
        if not isinstance(insertion_pts, int):
            raise BusinessError(
                "E_VERIFY",
                "verify",
                "SCTE-35 transport insertion point is missing",
                {"event_id": expected.get("event_id")},
            )
        insertion_delta = (insertion_pts - expected["target_pts_90k"]) & ((1 << 33) - 1)
        if insertion_delta > int(0.030 * 90000):
            raise BusinessError(
                "E_VERIFY",
                "verify",
                "SCTE-35 transport insertion is too far from its target PTS",
                {
                    "event_id": expected.get("event_id"),
                    "target_pts": expected["target_pts_90k"],
                    "insertion_pts": insertion_pts,
                    "delta_ticks": insertion_delta,
                },
            )

    subtitle_pes = extract_pes_streams(data, subtitle_pid)
    show_pts: list[int] = []
    clear_pts: list[int] = []
    for item in subtitle_pes:
        payload = item.get("payload")
        if not isinstance(payload, bytes):
            continue
        try:
            parsed = parse_subtitle_pes(payload)
        except ValueError as exc:
            raise BusinessError("E_VERIFY", "verify", f"Invalid DVB subtitle PES: {exc}") from exc
        if not parsed["has_end"]:
            raise BusinessError("E_VERIFY", "verify", "DVB subtitle display set has no end marker")
        pts = item.get("pts")
        if not isinstance(pts, int):
            continue
        if parsed["has_object"]:
            show_pts.append(pts)
        elif parsed["region_count"] == 0:
            clear_pts.append(pts)
    if len(show_pts) != len(expected_subtitle_starts_ms):
        raise BusinessError(
            "E_VERIFY",
            "verify",
            "DVB subtitle display-set count does not match corrected subtitles",
            {"expected": len(expected_subtitle_starts_ms), "actual": len(show_pts)},
        )
    if len(clear_pts) != expected_clear_count:
        raise BusinessError(
            "E_VERIFY",
            "verify",
            "DVB subtitle clear-set count does not match corrected subtitles",
            {"expected": expected_clear_count, "actual": len(clear_pts)},
        )
    for actual_pts, expected_ms in zip(sorted(show_pts), sorted(expected_subtitle_starts_ms), strict=False):
        delta_ms = abs(actual_pts / 90.0 - expected_ms)
        if delta_ms > 50.0:
            raise BusinessError(
                "E_VERIFY",
                "verify",
                "DVB subtitle PTS is not aligned to the corrected timeline",
                {"actual_pts_ms": actual_pts / 90.0, "expected_ms": expected_ms, "delta_ms": delta_ms},
            )

    probe = probe_json(
        ffprobe,
        ts_path,
        ["-show_programs", "-show_streams", "-show_format"],
    )
    streams = probe.get("streams", [])
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    subtitle_probe = [
        stream for stream in streams if stream.get("codec_type") == "subtitle"
    ]
    data_probe = [stream for stream in streams if stream.get("codec_type") == "data"]
    if not audio_streams:
        raise BusinessError("E_VERIFY", "verify", "ffprobe found no audio stream")
    audio = audio_streams[0]
    if audio.get("codec_name") != "aac" or str(audio.get("sample_rate")) != "48000":
        raise BusinessError(
            "E_VERIFY",
            "verify",
            "ffprobe did not identify the expected AAC 48 kHz audio stream",
            {"audio": audio},
        )
    if not subtitle_probe:
        raise BusinessError("E_VERIFY", "verify", "ffprobe did not identify an embedded subtitle stream")
    if not data_probe:
        raise BusinessError("E_VERIFY", "verify", "ffprobe did not identify an SCTE-35 data stream")

    audio_pid = int(str(audio["id"]), 0)
    audio_points = audio_pes_points(data, audio_pid)
    audio_pts_values = [int(item["pts"]) for item in audio_points]
    if len(audio_pts_values) < 2 or any(
        later <= earlier for earlier, later in zip(audio_pts_values, audio_pts_values[1:])
    ):
        raise BusinessError(
            "E_VERIFY",
            "verify",
            "Audio PTS values are missing or not strictly increasing",
        )
    audio_steps = [
        later - earlier for earlier, later in zip(audio_pts_values, audio_pts_values[1:])
    ]

    decode_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(ts_path),
        "-map",
        "0:a:0",
        "-map",
        "0:s:0",
        "-c:a",
        "pcm_s16le",
        "-c:s",
        "dvbsub",
        "-f",
        "null",
        "-",
    ]
    try:
        decoded = subprocess.run(
            decode_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BusinessError("E_VERIFY", "verify", f"Unable to decode output TS: {exc}") from exc
    if decoded.returncode != 0:
        raise BusinessError(
            "E_VERIFY",
            "verify",
            "FFmpeg could not decode the audio/subtitle streams",
            {
                "returncode": decoded.returncode,
                "stderr": decoded.stderr.decode("utf-8", errors="replace")[-2000:],
            },
        )

    overlay_max = None
    if expected_subtitle_starts_ms:
        overlay_time_ms = max(0.0, expected_subtitle_starts_ms[0] + 50.0)
        overlay_command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{overlay_time_ms / 1000:.3f}",
            "-i",
            str(ts_path),
            "-filter_complex",
            "[0:v][0:s]overlay",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        try:
            overlay = subprocess.run(
                overlay_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BusinessError("E_VERIFY", "verify", f"Unable to overlay subtitles: {exc}") from exc
        if overlay.returncode != 0 or len(overlay.stdout) != 720 * 576 * 3:
            raise BusinessError(
                "E_VERIFY",
                "verify",
                "Unable to render a decoded subtitle frame",
                {
                    "returncode": overlay.returncode,
                    "stderr": overlay.stderr.decode("utf-8", errors="replace")[-1000:],
                },
            )
        overlay_max = max(overlay.stdout)
        if overlay_max <= 40:
            raise BusinessError(
                "E_VERIFY",
                "verify",
                "Decoded subtitle overlay has no visible pixels",
                {"max_channel_value": overlay_max},
            )

    return {
        "status": "ok",
        "ts_path": str(ts_path),
        "ts_size": len(data),
        "pmt_pid": pmt.pmt_pid,
        "audio_pid": audio["id"],
        "scte35_pid": scte_pid,
        "subtitle_pid": subtitle_pid,
        "scte35_event_count": len(parsed_sections),
        "scte35_event_ids": [int(item["event_id"]) for item in expected_events],
        "dvb_display_sets": len(show_pts),
        "dvb_clear_sets": len(clear_pts),
        "ffprobe_audio_codec": audio.get("codec_name"),
        "ffprobe_audio_sample_rate": audio.get("sample_rate"),
        "ffprobe_subtitle_codec": subtitle_probe[0].get("codec_name"),
        "ffprobe_data_codec": data_probe[0].get("codec_name"),
        "audio_pts_packet_count": len(audio_pts_values),
        "audio_pts_continuous": True,
        "audio_pts_step_90k_values": sorted(set(audio_steps)),
        "ffmpeg_decode_check": "ok",
        "subtitle_overlay_max_channel": overlay_max,
        "subtitle_visual_check": "ok" if overlay_max is not None else "no_subtitles",
    }
