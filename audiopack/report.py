"""Slicing report construction and deterministic JSON serialization."""

from __future__ import annotations

import json

from .errors import ERROR_CODES


def build_report(
    sample_rate: int,
    output_frames: int,
    event_reports: list[dict],
    overlap_records: list[dict],
    loudness: dict,
    warnings: list[str],
    warning_details: list[dict],
    errors: list[dict],
) -> dict:
    """Assemble the final report object (no timestamps, fully deterministic)."""
    warning_codes = sorted({item["code"] for item in warning_details} | set(warnings))
    error_codes = sorted({item["code"] for item in errors if item.get("code") in ERROR_CODES})
    return {
        "sample_rate": int(sample_rate),
        "output_frames": int(output_frames),
        "output_channels": 2,
        "events": event_reports,
        "overlaps": overlap_records,
        "loudness": loudness,
        "warnings": warning_codes,
        "warning_details": warning_details,
        "errors": errors,
        "error_codes": error_codes,
        "status": "error" if error_codes else "ok",
    }


def write_report(path: str, report: dict) -> None:
    """Write pretty, key-sorted JSON so identical runs are byte-identical."""
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
