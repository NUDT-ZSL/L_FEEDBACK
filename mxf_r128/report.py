"""QC document creation and atomic report replacement."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from .constants import (
    GOP_SIZE,
    LUFS_TOLERANCE,
    MAX_START_DELTA_MS,
    REPORT_VERSION,
    TARGET_LUFS,
    TOOL_NAME,
    TRUE_PEAK_LIMIT_DBTP,
)


def empty_file_record(source_file: str) -> dict[str, Any]:
    return {
        "source_file": source_file,
        "status": "failed",
        "output_path": None,
        "video_codec": None,
        "video_profile": None,
        "video_profile_level": None,
        "video_level": None,
        "pix_fmt": None,
        "frame_rate": None,
        "width": None,
        "height": None,
        "gop_size": None,
        "audio_codec": None,
        "loudness_lufs": None,
        "loudness_lra": None,
        "true_peak_dbtp": None,
        "channel_layout": None,
        "audio_channels": None,
        "source_video_start_time": None,
        "source_audio_start_time": None,
        "source_audio_delay_sec": None,
        "output_video_start_time": None,
        "output_audio_start_time": None,
        "output_start_delta_ms": None,
        "warnings": [],
        "error_code": None,
    }


def build_report(
    *,
    input_dir: Path,
    output_dir: Path,
    max_workers: int,
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    sorted_records = sorted(records, key=lambda item: str(item["source_file"]))
    error_codes = sorted(
        {
            str(record["error_code"])
            for record in sorted_records
            if record.get("error_code")
        }
    )
    success_count = sum(
        1 for record in sorted_records if record.get("status") == "success"
    )
    failed_count = len(sorted_records) - success_count
    return {
        "report_version": REPORT_VERSION,
        "tool": TOOL_NAME,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "max_workers": max_workers,
        "thresholds": {
            "target_lufs": TARGET_LUFS,
            "loudness_tolerance_lu": LUFS_TOLERANCE,
            "true_peak_limit_dbtp": TRUE_PEAK_LIMIT_DBTP,
            "max_start_delta_ms": MAX_START_DELTA_MS,
            "gop_size": GOP_SIZE,
        },
        "summary": {
            "total": len(sorted_records),
            "success": success_count,
            "failed": failed_count,
            "error_codes": error_codes,
        },
        "files": sorted_records,
    }


def write_report_atomic(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
