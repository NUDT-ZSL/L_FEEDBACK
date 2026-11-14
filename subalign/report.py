"""JSON correction report with millisecond precision and ISO 8601 timestamps."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from .drift import AlignmentResult, DriftModel
from .matching import Match


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ms(value: float) -> int:
    return int(round(value))


def iso_duration(value_ms: float) -> str:
    """Render a millisecond duration as an ISO 8601 ``PT`` string."""

    total = max(int(round(value_ms)), 0)
    minutes, milliseconds = divmod(total, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    hours, minutes = divmod(minutes, 60)
    parts = ["PT"]
    if hours:
        parts.append(f"{hours}H")
    if minutes:
        parts.append(f"{minutes}M")
    if seconds or milliseconds or (not hours and not minutes):
        if milliseconds:
            parts.append(f"{seconds}.{milliseconds:03d}S")
        else:
            parts.append(f"{seconds}S")
    return "".join(parts)


def segment_payload(model: DriftModel) -> list[dict]:
    return [
        {
            "index": segment.index,
            "target_time_range": {
                "start_ms": ms(segment.start_time_ms),
                "end_ms": ms(segment.end_time_ms),
            },
            "scale": round(segment.scale, 6),
            "offset_ms": ms(segment.offset_ms),
            "match_count": segment.match_count,
            "rms_residual_ms": ms(segment.rms_residual_ms),
        }
        for segment in model.segments
    ]


def match_payload(matches: list[Match]) -> list[dict]:
    return [
        {
            "reference_index": match.reference_index,
            "target_index": match.target_index,
            "reference_start_ms": ms(match.reference_start),
            "target_start_ms": ms(match.target_start),
            "text_similarity": round(match.text_similarity, 6),
            "time_distance_ms": ms(match.time_distance_ms),
            "confidence": round(match.confidence, 6),
        }
        for match in matches
    ]


def build_file_report(
    input_path: str,
    output_path: str,
    result: AlignmentResult,
    elapsed_ms: float,
    translated: bool,
    extra_warnings: list[dict] | None = None,
) -> dict:
    warnings = list(result.warnings)
    if extra_warnings:
        warnings.extend(extra_warnings)
    payload = {
        "input_path": input_path,
        "output_path": output_path,
        "status": "aligned",
        "translated_target": translated,
        "drift_model": {
            "type": result.model.kind,
            "scale": round(result.model.scale, 6),
            "offset_ms": ms(result.model.offset_ms),
            "rms_residual_ms": ms(result.model.rms_residual_ms),
            "segments": segment_payload(result.model),
        },
        "match_statistics": result.stats,
        "matches": match_payload(result.matches),
        "warnings": warnings,
        "processing_time": iso_duration(elapsed_ms),
        "processing_time_ms": ms(elapsed_ms),
    }
    return payload


def write_report(path: str, report: dict) -> None:
    if directory := os.path.dirname(path):
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
