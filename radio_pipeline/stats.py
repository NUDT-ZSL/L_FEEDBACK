from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .audio import loudness_lufs_approx
from .errors import VerificationFailure


REPORT_SCHEMA = "radio-pipeline-report-v1"
LOG_SCHEMA = "radio-pipeline-processing-log-v1"


def build_report(log: dict[str, Any]) -> dict[str, Any]:
    if log.get("schema") != LOG_SCHEMA:
        raise VerificationFailure(
            f"unsupported processing log schema: {log.get('schema')!r}"
        )
    try:
        source = log["source"]
        frames = log["frames"]
        events = log["events"]
        output = log["output"]
        jitter = log["jitter_buffer"]
        config = log["config"]
    except KeyError as exc:
        raise VerificationFailure(f"processing log is missing {exc.args[0]}") from exc

    total_packets = _integer(source, "packets")
    frame_samples = _integer(source, "frame_samples")
    sample_rate = _integer(source, "sample_rate")
    channels = _integer(source, "channels")
    if not isinstance(frames, list) or not isinstance(events, list):
        raise VerificationFailure("processing log frames/events must be lists")
    if len(frames) != total_packets:
        raise VerificationFailure(
            f"frame count {len(frames)} does not equal source packet count {total_packets}"
        )

    timeline_monotonic = True
    last_timestamp = -1
    received_packets = 0
    lost_packets = 0
    timeout_packets = 0
    plc_frames = 0
    pitch_frames = 0
    comfort_frames = 0
    muted_frames = 0
    max_frame_water = 0
    max_event_water = 0
    square_sum = 0
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise VerificationFailure(f"frame {index} is not an object")
        sequence = _integer(frame, "sequence")
        timestamp = _integer(frame, "timestamp_ms")
        if sequence != index:
            raise VerificationFailure(
                f"frame sequence {sequence} is not monotonic at index {index}"
            )
        if timestamp < last_timestamp:
            timeline_monotonic = False
        last_timestamp = timestamp
        outcome = frame.get("source")
        if outcome == "received":
            received_packets += 1
        elif outcome == "lost":
            lost_packets += 1
        elif outcome == "timeout":
            timeout_packets += 1
        else:
            raise VerificationFailure(
                f"frames[{index}].source has invalid value {outcome!r}"
            )
        is_plc = bool(frame.get("plc_triggered"))
        if is_plc:
            plc_frames += 1
            strategy = frame.get("plc_strategy")
            if strategy == "pitch":
                pitch_frames += 1
            elif strategy == "comfort":
                comfort_frames += 1
            elif strategy == "mute":
                muted_frames += 1
            else:
                raise VerificationFailure(
                    f"frames[{index}] has unknown PLC strategy {strategy!r}"
                )
        if outcome != "received" and not is_plc:
            raise VerificationFailure(
                f"missing frame {sequence} was not marked PLC-triggered"
            )
        if outcome == "received" and is_plc:
            raise VerificationFailure(
                f"received frame {sequence} was unexpectedly marked PLC-triggered"
            )
        max_frame_water = max(
            max_frame_water, _integer(frame, "buffer_water_ms")
        )
        square_sum += _integer(frame, "pcm_square_sum")

    if received_packets + lost_packets + timeout_packets != total_packets:
        raise VerificationFailure("frame source counts do not sum to total packet count")

    event_counts = {"received": 0, "duplicate": 0, "lost": 0, "timeout": 0, "invalid": 0}
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise VerificationFailure(f"event {index} is not an object")
        status = event.get("status")
        if status not in event_counts:
            raise VerificationFailure(f"events[{index}] has invalid status {status!r}")
        event_counts[status] += 1
        event_water = event.get("buffer_water_ms")
        if isinstance(event_water, int) and not isinstance(event_water, bool):
            max_event_water = max(max_event_water, event_water)
    if event_counts["received"] != received_packets:
        raise VerificationFailure("received event count does not match frame count")
    if event_counts["lost"] != lost_packets:
        raise VerificationFailure("lost event count does not match frame count")
    if event_counts["timeout"] != timeout_packets:
        raise VerificationFailure("timeout event count does not match frame count")

    expected_samples = total_packets * frame_samples
    output_samples = _integer(output, "sample_count")
    if output_samples != expected_samples:
        raise VerificationFailure(
            f"output sample count {output_samples} != expected {expected_samples}"
        )
    output_sum_squares = _integer(output, "sum_squares")
    if output_sum_squares != square_sum:
        raise VerificationFailure(
            "output square sum does not match the per-frame PCM square sums"
        )
    loudness = loudness_lufs_approx(square_sum, output_samples)
    logged_loudness = _number(output, "loudness_lufs_approx")
    if abs(loudness - logged_loudness) > 0.0005:
        raise VerificationFailure(
            f"logged loudness {logged_loudness} != recomputed {loudness}"
        )
    output_duration_ms = _integer(output, "duration_ms")
    expected_duration_ms = int(round(total_packets * _integer(source, "frame_ms")))
    if output_duration_ms != expected_duration_ms:
        raise VerificationFailure(
            f"output duration {output_duration_ms} != expected {expected_duration_ms}"
        )

    jitter_max = _integer(jitter, "max_buffer_delay_ms")
    observed_max_water = max(max_frame_water, max_event_water)
    if jitter_max != observed_max_water:
        raise VerificationFailure(
            f"jitter maximum {jitter_max} != logged frame/event maximum "
            f"{observed_max_water}"
        )
    degradation_events = jitter.get("degradation_events", [])
    fast_events = jitter.get("fast_consume_events", [])
    if not isinstance(degradation_events, list) or not isinstance(fast_events, list):
        raise VerificationFailure("jitter event logs must be lists")

    counts = {
        "total_packets": total_packets,
        "received_packets": received_packets,
        "lost_packets": lost_packets,
        "timeout_packets": timeout_packets,
        "duplicate_arrivals": event_counts["duplicate"],
        "invalid_arrivals": event_counts["invalid"],
        "plc_frames": plc_frames,
        "pitch_plc_frames": pitch_frames,
        "comfort_plc_frames": comfort_frames,
        "muted_plc_frames": muted_frames,
    }
    rates = {
        "lost_rate": _round_rate((lost_packets + timeout_packets) / total_packets),
        "hard_loss_rate": _round_rate(lost_packets / total_packets),
        "timeout_rate": _round_rate(timeout_packets / total_packets),
        "duplicate_rate": _round_rate(event_counts["duplicate"] / total_packets),
        "invalid_arrival_rate": _round_rate(event_counts["invalid"] / total_packets),
        "plc_ratio": _round_rate(plc_frames / total_packets),
    }
    if not timeline_monotonic:
        raise VerificationFailure("frame timestamps are not monotonic")

    output_config = config.get("output", {}) if isinstance(config, dict) else {}
    source_packets = source.get("packets") if isinstance(source, dict) else None
    if source_packets is not None and source_packets != total_packets:
        raise VerificationFailure("source packet list length does not match total packets")

    return {
        "schema": REPORT_SCHEMA,
        "config_sha256": log.get("config_sha256"),
        "input_sha256": log.get("input_sha256"),
        "source": {
            "sample_rate": sample_rate,
            "channels": channels,
            "frame_ms": _integer(source, "frame_ms"),
            "frame_samples": frame_samples,
            "duration_ms": expected_duration_ms,
            "total_samples": expected_samples,
        },
        "counts": counts,
        "rates": rates,
        "buffer": {
            "min_delay_ms": _integer(jitter, "min_delay_ms"),
            "max_delay_ms": _integer(jitter, "max_delay_ms"),
            "effective_max_delay_ms": _integer(jitter, "effective_max_delay_ms"),
            "timeout_ms": _integer(jitter, "timeout_ms"),
            "max_buffer_delay_ms": jitter_max,
            "fast_consume_events": len(fast_events),
            "trim_total_ms": _integer(jitter, "trim_total_ms"),
            "conservative_mode_activated": bool(
                jitter.get("conservative_mode_activated")
            ),
        },
        "plc": {
            "strategy": _string(config.get("plc", {}), "strategy", fallback="unknown"),
            "fade_ms": _number(config.get("plc", {}), "fade_ms", fallback=0.0),
            "plc_frames": plc_frames,
            "plc_ratio": rates["plc_ratio"],
        },
        "degradation": {
            "triggered": bool(degradation_events),
            "event_count": len(degradation_events),
            "events": degradation_events,
        },
        "diagnostics": {
            "invalid_payload_count": event_counts["invalid"],
            "items": log.get("diagnostics", []),
        },
        "output": {
            "sample_rate": sample_rate,
            "channels": channels,
            "duration_ms": output_duration_ms,
            "sample_count": output_samples,
            "loudness_lufs_approx": loudness,
            "loudness_formula": (
                "-0.691 + 10*log10(sum(x^2)/(N*32768^2)) over the full PCM timeline"
            ),
            "sum_squares": square_sum,
            "wav_file": output_config.get("wav_name"),
            "ogg_file": output_config.get("ogg_name"),
            "wav_sha256": output.get("wav_sha256"),
            "ogg_sha256": output.get("ogg_sha256"),
            "pcm_sha256": output.get("pcm_sha256"),
            "wav_stream": output.get("wav_stream"),
            "ogg_stream": output.get("ogg_stream"),
            "embedded_metadata": output.get("embedded_metadata"),
        },
        "validation": {
            "timeline_monotonic": timeline_monotonic,
            "packet_count_consistent": True,
            "plc_count_consistent": True,
            "loudness_consistent_with_frames": True,
            "frame_square_sum_matches_output": True,
        },
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_processing_log(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            log = json.load(handle)
    except FileNotFoundError as exc:
        raise VerificationFailure(f"processing log not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise VerificationFailure(
            f"processing log is not valid JSON: line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(log, dict):
        raise VerificationFailure("processing log root must be a JSON object")
    return log


def _integer(mapping: Any, label: str) -> int:
    current: Any = mapping
    if not isinstance(current, dict) or label not in current:
        raise VerificationFailure(f"{label} is missing")
    current = current[label]
    if isinstance(current, bool) or not isinstance(current, int):
        raise VerificationFailure(f"{label} must be an integer")
    return current


def _number(mapping: Any, label: str, fallback: float | None = None) -> float:
    current: Any = mapping
    if not isinstance(current, dict) or label not in current:
        if fallback is not None:
            return fallback
        raise VerificationFailure(f"{label} is missing")
    current = current[label]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        if fallback is not None:
            return fallback
        raise VerificationFailure(f"{label} must be a number")
    value = float(current)
    if not math.isfinite(value):
        raise VerificationFailure(f"{label} must be finite")
    return value


def _string(mapping: Any, label: str, fallback: str) -> str:
    current: Any = mapping
    if not isinstance(current, dict) or label not in current:
        return fallback
    current = current[label]
    return current if isinstance(current, str) else fallback


def _round_rate(value: float) -> float:
    return round(value, 6)
