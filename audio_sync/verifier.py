"""Cross-check pipeline outputs against the deterministic ground truth."""

from __future__ import annotations

import json
from typing import Dict, List

from .constants import FRAME_DURATION_MS
from .wavio import read_wav_duration


def _load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def verify_run(truth_path: str, log_path: str, wav_path: str) -> Dict[str, object]:
    """Return a structured verification report (``ok`` is the overall verdict)."""
    truth = _load_jsonl(truth_path)
    logs = _load_jsonl(log_path)
    failures: List[str] = []

    if len(logs) != len(truth):
        failures.append(
            f"frame count mismatch log={len(logs)} truth={len(truth)}"
        )

    # Strictly monotonic frame index / timestamp order.
    previous_index = None
    for record in logs:
        index = record["frame_index"]
        if previous_index is not None and index != previous_index + 1:
            failures.append(
                f"non-monotonic frame index at {index} (prev={previous_index})"
            )
            break
        if record["timestamp"] != index * 48000 * FRAME_DURATION_MS // 1000:
            failures.append(f"timestamp mismatch at frame {index}")
            break
        previous_index = index

    # Every ground-truth loss must be reflected (PLC or silence), and a present
    # truth frame must never be flagged as lost on an otherwise healthy stream.
    plc_total = 0
    recovery_total = 0
    mismatches = 0
    for truth_record, log_record in zip(truth, logs):
        frame_index = truth_record["frame_index"]
        for stream_key, stream_truth in truth_record["streams"].items():
            log_stream = log_record["streams"].get(f"stream_{stream_key}")
            if log_stream is None:
                mismatches += 1
                continue
            was_lost = bool(stream_truth["lost"])
            handled = (
                bool(log_stream["plc"])
                or log_stream.get("plc_method") == "silence"
            )
            if was_lost and not handled:
                mismatches += 1
            if not was_lost and not log_stream["frame_present"]:
                # A clean frame can be missing at playout only when the buffer
                # previously overflowed (oldest non-critical frame dropped).
                overflow_key = f"stream_{stream_key}:buffer_overflow"
                underflow_key = f"stream_{stream_key}:buffer_underflow"
                if not any(
                    overflow_key == event or underflow_key == event
                    for event in log_record.get("events", [])
                ):
                    mismatches += 1
            plc_total += int(bool(log_stream["plc"]))
            recovery_total += int(bool(log_stream.get("recovery")))

    if mismatches:
        failures.append(f"{mismatches} PLC/loss alignment mismatches")

    expected_duration = len(truth) * FRAME_DURATION_MS / 1000.0
    actual_duration = read_wav_duration(wav_path)
    if abs(actual_duration - expected_duration) > 0.005:
        failures.append(
            f"wav duration {actual_duration:.3f}s != expected "
            f"{expected_duration:.3f}s"
        )

    backpressure = sum(
        1
        for record in logs
        for event in record.get("events", [])
        if event.endswith("backpressure")
    )
    peak_values = [record["mix_peak"] for record in logs]

    return {
        "ok": not failures,
        "failures": failures,
        "frames": len(logs),
        "plc_frames": plc_total,
        "recovery_points": recovery_total,
        "backpressure_events": backpressure,
        "wav_duration_s": round(actual_duration, 4),
        "expected_duration_s": round(expected_duration, 4),
        "max_mix_peak": max(peak_values, default=0),
    }
