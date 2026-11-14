from __future__ import annotations

import base64
import binascii
import hashlib
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .audio import pcm_bytes
from .config import NetworkSettings


PACKET_SCHEMA = "radio-opus-pcm-packet-v1"
INVALID_BASE64_TOKEN = "!!CORRUPTED_BASE64!!"


@dataclass(frozen=True)
class WirePacket:
    sequence: int
    timestamp_ms: int
    arrival_ms: float
    payload: bytes
    checksum: str
    line_number: int

    @property
    def payload_size(self) -> int:
        return len(self.payload)


@dataclass(frozen=True)
class ParseIssue:
    sequence: int | None
    timestamp_ms: float | None
    arrival_ms: float | None
    line_number: int
    diagnostic: str
    raw: str


@dataclass(frozen=True)
class ParsedArrival:
    packet: WirePacket | None
    issue: ParseIssue | None


def _sample_jitter(
    rng: np.random.Generator, distribution: str, scale_ms: float, max_abs_ms: float
) -> float:
    if distribution == "none" or scale_ms <= 0.0:
        value = 0.0
    elif distribution == "uniform":
        value = float(rng.uniform(-scale_ms, scale_ms))
    elif distribution == "exponential":
        value = float(rng.exponential(scale_ms))
    else:
        value = float(rng.normal(0.0, scale_ms))
    return float(np.clip(value, -max_abs_ms, max_abs_ms))


def generate_network_events(
    source_frames: np.ndarray, config: NetworkSettings, seed: int
) -> list[dict[str, Any]]:
    """Build a deterministic JSONL-ready arrival sequence.

    The payload is a decoded 20 ms PCM frame. It intentionally behaves like
    the output of an Opus decoder so the receive/jitter/PLC stages can be
    exercised without external media or a live Opus stream.
    """
    total_packets = int(source_frames.shape[0])
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    forced_loss: set[int] = set()
    for burst in config.forced_bursts:
        for sequence in range(burst.start_sequence, burst.start_sequence + burst.length):
            if 0 <= sequence < total_packets:
                forced_loss.add(sequence)

    random_loss: set[int] = set()
    remaining_burst = 0
    for sequence in range(total_packets):
        if sequence in forced_loss:
            random_loss.add(sequence)
            continue
        if remaining_burst > 0:
            random_loss.add(sequence)
            remaining_burst -= 1
            continue
        if rng.random() < config.loss_rate:
            random_loss.add(sequence)
            if rng.random() < config.burst_probability:
                remaining_burst = int(rng.integers(1, config.burst_length + 1))

    forced_late = set(config.forced_late_sequences)
    forced_corrupt = set(config.forced_corrupt_sequences)
    events: list[dict[str, Any]] = []
    event_counter = 0

    for sequence in range(total_packets):
        if sequence in random_loss and sequence not in forced_late:
            continue
        timestamp_ms = sequence * 20
        jitter = _sample_jitter(
            rng,
            config.arrival_jitter.distribution,
            config.arrival_jitter.scale_ms,
            config.arrival_jitter.max_abs_ms,
        )
        arrival_ms = max(0.0, timestamp_ms + config.base_latency_ms + jitter)
        if config.ordering == "reordered" and rng.random() < config.reorder_probability:
            arrival_ms += float(
                rng.uniform(config.reorder_min_extra_ms, config.reorder_max_extra_ms)
            )
        if sequence in forced_late:
            arrival_ms = timestamp_ms + config.base_latency_ms + 360.0

        payload = pcm_bytes(source_frames[sequence])
        checksum = f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"

        should_corrupt = sequence in forced_corrupt or rng.random() < config.corrupt_rate
        if should_corrupt:
            events.append(
                {
                    "schema": PACKET_SCHEMA,
                    "event_id": event_counter,
                    "sequence": sequence,
                    "timestamp_ms": timestamp_ms,
                    "arrival_ms": round(max(0.0, arrival_ms - 0.0001), 6),
                    "payload_b64": INVALID_BASE64_TOKEN,
                    "payload_bytes": len(payload),
                    "checksum": "00000000",
                }
            )
            event_counter += 1

        events.append(
            {
                "schema": PACKET_SCHEMA,
                "event_id": event_counter,
                "sequence": sequence,
                "timestamp_ms": timestamp_ms,
                "arrival_ms": round(max(0.0, arrival_ms), 6),
                "payload_b64": base64.b64encode(payload).decode("ascii"),
                "payload_bytes": len(payload),
                "checksum": checksum,
            }
        )
        event_counter += 1

        if sequence not in forced_late and rng.random() < config.duplicate_rate:
            duplicate_arrival = arrival_ms + float(
                np.clip(rng.uniform(12.0, 95.0), 0.0, 300.0)
            )
            events.append(
                {
                    "schema": PACKET_SCHEMA,
                    "event_id": event_counter,
                    "sequence": sequence,
                    "timestamp_ms": timestamp_ms,
                    "arrival_ms": round(duplicate_arrival, 6),
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                    "payload_bytes": len(payload),
                    "checksum": checksum,
                }
            )
            event_counter += 1

    events.sort(
        key=lambda item: (
            float(item["arrival_ms"]),
            int(item["sequence"]),
            int(item["event_id"]),
        )
    )
    return events


def write_packet_jsonl(path: Path, events: Iterable[dict[str, Any]]) -> str:
    data = "".join(
        json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for event in events
    )
    path.write_text(data, encoding="utf-8", newline="\n")
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def read_packet_jsonl(path: Path, frame_bytes: int) -> list[ParsedArrival]:
    parsed: list[ParsedArrival] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read packet sequence {path}: {exc}") from exc

    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        parsed.append(parse_packet_line(raw, line_number, frame_bytes))
    return parsed


def _safe_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def parse_packet_line(raw: str, line_number: int, frame_bytes: int) -> ParsedArrival:
    sequence_value: Any = None
    timestamp_value: Any = None
    arrival_value: Any = None
    try:
        item = json.loads(raw)
        if not isinstance(item, dict):
            raise ValueError("packet line must contain a JSON object")
        sequence_value = item.get("sequence")
        timestamp_value = item.get("timestamp_ms")
        arrival_value = item.get("arrival_ms")
        sequence = (
            int(sequence_value)
            if isinstance(sequence_value, int) and not isinstance(sequence_value, bool)
            else None
        )
        timestamp = _safe_number(timestamp_value)
        arrival = _safe_number(arrival_value)
        issue_base = {
            "sequence": sequence,
            "timestamp_ms": timestamp,
            "arrival_ms": arrival,
            "line_number": line_number,
            "raw": raw,
        }
        if sequence is None or sequence < 0:
            return ParsedArrival(
                packet=None,
                issue=ParseIssue(diagnostic="invalid or missing sequence", **issue_base),
            )
        if timestamp is None or timestamp < 0:
            return ParsedArrival(
                packet=None,
                issue=ParseIssue(diagnostic="invalid or missing timestamp_ms", **issue_base),
            )
        if arrival is None or arrival < 0:
            return ParsedArrival(
                packet=None,
                issue=ParseIssue(diagnostic="invalid or missing arrival_ms", **issue_base),
            )
        if item.get("schema") not in {None, PACKET_SCHEMA}:
            return ParsedArrival(
                packet=None,
                issue=ParseIssue(diagnostic="unsupported packet schema", **issue_base),
            )
        payload_token = item.get("payload_b64")
        if not isinstance(payload_token, str):
            return ParsedArrival(
                packet=None,
                issue=ParseIssue(diagnostic="missing base64 payload", **issue_base),
            )
        try:
            payload = base64.b64decode(payload_token, validate=True)
        except (binascii.Error, ValueError) as exc:
            return ParsedArrival(
                packet=None,
                issue=ParseIssue(
                    diagnostic=f"payload base64 decode failed: {exc}", **issue_base
                ),
            )
        if len(payload) != frame_bytes:
            return ParsedArrival(
                packet=None,
                issue=ParseIssue(
                    diagnostic=(
                        f"payload size {len(payload)} does not match expected {frame_bytes}"
                    ),
                    **issue_base,
                ),
            )
        checksum = item.get("checksum")
        actual_checksum = f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
        if checksum is not None and checksum != actual_checksum:
            return ParsedArrival(
                packet=None,
                issue=ParseIssue(diagnostic="payload checksum mismatch", **issue_base),
            )
        return ParsedArrival(
            packet=WirePacket(
                sequence=sequence,
                timestamp_ms=int(round(timestamp)),
                arrival_ms=float(arrival),
                payload=payload,
                checksum=actual_checksum,
                line_number=line_number,
            ),
            issue=None,
        )
    except json.JSONDecodeError as exc:
        return ParsedArrival(
            packet=None,
            issue=ParseIssue(
                sequence=None,
                timestamp_ms=None,
                arrival_ms=None,
                line_number=line_number,
                diagnostic=f"JSON decode failed: {exc.msg}",
                raw=raw,
            ),
        )
    except (TypeError, ValueError) as exc:
        sequence = (
            int(sequence_value)
            if isinstance(sequence_value, int) and not isinstance(sequence_value, bool)
            else None
        )
        return ParsedArrival(
            packet=None,
            issue=ParseIssue(
                sequence=sequence,
                timestamp_ms=_safe_number(timestamp_value),
                arrival_ms=_safe_number(arrival_value),
                line_number=line_number,
                diagnostic=str(exc),
                raw=raw,
            ),
        )
