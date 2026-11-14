from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import JitterSettings, PlcSettings
from .jitter import FrameDecision, JitterBuffer
from .packets import ParsedArrival, WirePacket


@dataclass
class ReceiverResult:
    frames: list[FrameDecision]
    event_logs: list[dict[str, Any]]
    accepted_packets: list[WirePacket]
    diagnostics: list[dict[str, Any]]
    jitter_log: dict[str, Any]
    invalid_arrivals: int
    duplicate_arrivals: int


def _arrival_sort_key(item: ParsedArrival) -> tuple[float, int, int]:
    if item.packet is not None:
        return (item.packet.arrival_ms, item.packet.sequence, item.packet.line_number)
    assert item.issue is not None
    arrival = item.issue.arrival_ms
    if arrival is None:
        arrival = item.issue.timestamp_ms if item.issue.timestamp_ms is not None else 0.0
    sequence = item.issue.sequence if item.issue.sequence is not None else 2**31 - 1
    return (float(arrival), sequence, item.issue.line_number)


def receive_packets(
    parsed: Iterable[ParsedArrival],
    total_packets: int,
    jitter_config: JitterSettings,
    plc_config: PlcSettings,
    frame_ms: int = 20,
) -> ReceiverResult:
    arrivals = sorted(parsed, key=_arrival_sort_key)
    jitter = JitterBuffer(total_packets, jitter_config, plc_config)
    jitter.frame_ms = frame_ms
    jitter.buffer_depth_frames = max(
        1, int(math.ceil(jitter_config.max_delay_ms / frame_ms))
    )
    accepted: dict[int, WirePacket] = {}
    valid_seen: set[int] = set()
    event_logs: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    invalid_arrivals = 0
    duplicate_arrivals = 0

    for event_index, arrival in enumerate(arrivals):
        packet = arrival.packet
        issue = arrival.issue
        if packet is None:
            assert issue is not None
            if issue.arrival_ms is not None:
                jitter.advance_to(issue.arrival_ms)
            invalid_arrivals += 1
            diagnostic = {
                "sequence": issue.sequence,
                "timestamp_ms": issue.timestamp_ms,
                "arrival_ms": issue.arrival_ms,
                "line_number": issue.line_number,
                "diagnostic": issue.diagnostic,
                "raw_prefix": issue.raw[:240],
            }
            diagnostics.append(diagnostic)
            event_logs.append(
                {
                    "event_index": event_index,
                    "line_number": issue.line_number,
                    "sequence": issue.sequence,
                    "timestamp_ms": issue.timestamp_ms,
                    "arrival_ms": issue.arrival_ms,
                    "status": "invalid",
                    "reason": issue.diagnostic,
                    "buffer_water_ms": jitter.water_level_ms(),
                    "fast_consume": False,
                    "plc_triggered": False,
                    "plc_strategy": None,
                    "fade_regions": [],
                    "diagnostic": issue.diagnostic,
                }
            )
            continue

        jitter.advance_to(packet.arrival_ms)
        if packet.sequence >= total_packets:
            invalid_arrivals += 1
            reason = (
                f"sequence {packet.sequence} is outside source timeline "
                f"0..{total_packets - 1}"
            )
            diagnostics.append(
                {
                    "sequence": packet.sequence,
                    "timestamp_ms": packet.timestamp_ms,
                    "arrival_ms": packet.arrival_ms,
                    "line_number": packet.line_number,
                    "diagnostic": reason,
                    "raw_prefix": "",
                }
            )
            event_logs.append(
                {
                    "event_index": event_index,
                    "line_number": packet.line_number,
                    "sequence": packet.sequence,
                    "timestamp_ms": packet.timestamp_ms,
                    "arrival_ms": packet.arrival_ms,
                    "status": "invalid",
                    "reason": reason,
                    "buffer_water_ms": jitter.water_level_ms(),
                    "fast_consume": False,
                    "plc_triggered": False,
                    "plc_strategy": None,
                    "fade_regions": [],
                    "diagnostic": reason,
                }
            )
            continue

        if packet.sequence in valid_seen:
            duplicate_arrivals += 1
            event_logs.append(
                {
                    "event_index": event_index,
                    "line_number": packet.line_number,
                    "sequence": packet.sequence,
                    "timestamp_ms": packet.timestamp_ms,
                    "arrival_ms": packet.arrival_ms,
                    "status": "duplicate",
                    "reason": "first valid packet retained",
                    "buffer_water_ms": jitter.water_level_ms(),
                    "fast_consume": False,
                    "plc_triggered": False,
                    "plc_strategy": None,
                    "fade_regions": [],
                    "diagnostic": None,
                }
            )
            continue

        valid_seen.add(packet.sequence)
        is_late = (
            packet.sequence < jitter.next_sequence
            or packet.arrival_ms > packet.timestamp_ms + jitter_config.timeout_ms
        )
        if is_late:
            jitter.mark_timeout(packet.sequence)
            reason = (
                "arrived after its playout slot"
                if packet.sequence < jitter.next_sequence
                else (
                    f"arrival exceeds timestamp window of "
                    f"{jitter_config.timeout_ms} ms"
                )
            )
            event_logs.append(
                {
                    "event_index": event_index,
                    "line_number": packet.line_number,
                    "sequence": packet.sequence,
                    "timestamp_ms": packet.timestamp_ms,
                    "arrival_ms": packet.arrival_ms,
                    "status": "timeout",
                    "reason": reason,
                    "buffer_water_ms": jitter.water_level_ms(),
                    "fast_consume": False,
                    "plc_triggered": False,
                    "plc_strategy": None,
                    "fade_regions": [],
                    "diagnostic": None,
                }
            )
            continue

        accepted[packet.sequence] = packet
        insert_info = jitter.insert(packet)
        jitter.advance_to(packet.arrival_ms)
        event_logs.append(
            {
                "event_index": event_index,
                "line_number": packet.line_number,
                "sequence": packet.sequence,
                "timestamp_ms": packet.timestamp_ms,
                "arrival_ms": packet.arrival_ms,
                "status": "received",
                "reason": "accepted into jitter buffer",
                "buffer_water_ms": insert_info["water_ms"],
                "fast_consume": insert_info["fast_consume"],
                "fast_consume_event": insert_info["fast_consume_event"],
                "plc_triggered": False,
                "plc_strategy": None,
                "fade_regions": [],
                "diagnostic": None,
            }
        )

    jitter.flush()
    for sequence in range(total_packets):
        if sequence in valid_seen:
            continue
        frame = jitter.frames[sequence]
        event_logs.append(
            {
                "event_index": len(event_logs),
                "line_number": None,
                "sequence": sequence,
                "timestamp_ms": sequence * frame_ms,
                "arrival_ms": None,
                "status": "lost",
                "reason": "no valid packet arrived in the timestamp window",
                "buffer_water_ms": frame.buffer_water_ms,
                "fast_consume": False,
                "plc_triggered": False,
                "plc_strategy": None,
                "fade_regions": [],
                "diagnostic": None,
            }
        )

    accepted_packets = [accepted[key] for key in sorted(accepted)]
    return ReceiverResult(
        frames=jitter.frames,
        event_logs=event_logs,
        accepted_packets=accepted_packets,
        diagnostics=diagnostics,
        jitter_log=jitter.to_log(),
        invalid_arrivals=invalid_arrivals,
        duplicate_arrivals=duplicate_arrivals,
    )


def enrich_events_with_frames(
    event_logs: list[dict[str, Any]], frames: list[FrameDecision]
) -> None:
    by_sequence = {frame.sequence: frame for frame in frames}
    for event in event_logs:
        sequence = event.get("sequence")
        if not isinstance(sequence, int) or sequence not in by_sequence:
            continue
        frame = by_sequence[sequence]
        event["plc_triggered"] = frame.plc_triggered
        event["plc_strategy"] = frame.plc_strategy
        event["fade_regions"] = frame.fade_regions
        if event["status"] == "lost":
            event["buffer_water_ms"] = frame.buffer_water_ms


def write_accepted_packets(path: Path, packets: list[WirePacket]) -> None:
    lines: list[str] = []
    for packet in packets:
        lines.append(
            json.dumps(
                {
                    "sequence": packet.sequence,
                    "timestamp_ms": packet.timestamp_ms,
                    "arrival_ms": packet.arrival_ms,
                    "payload_bytes": packet.payload_size,
                    "checksum": packet.checksum,
                    "line_number": packet.line_number,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    data = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(data, encoding="utf-8")
