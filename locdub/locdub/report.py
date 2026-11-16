"""Per-language grouped JSON processing report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .constants import (
    EXIT_LANG_FAILED,
    EXIT_OK,
    EXIT_PARTIAL,
    STATUS_FAILED,
    STATUS_PARTIAL,
    STATUS_SUCCESS,
)
from .model import LanguageResult


def build_report(results: List[LanguageResult], *, timeline: Path,
                 video: Path, gain_db: float, workers: int,
                 invalid_segments: List[Dict[str, str]]) -> Dict[str, Any]:
    statuses = [result.status for result in results]
    if statuses and all(status == STATUS_SUCCESS for status in statuses):
        overall = STATUS_SUCCESS
    elif any(status == STATUS_FAILED for status in statuses):
        overall = STATUS_FAILED
    else:
        overall = STATUS_PARTIAL

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "timeline": str(timeline),
        "video": str(video),
        "gain_db": gain_db,
        "workers": workers,
        "language_count": len(results),
        "overall_status": overall,
        "status_values": {
            "success": STATUS_SUCCESS,
            "partial_success": STATUS_PARTIAL,
            "failed": STATUS_FAILED,
        },
        "invalid_timeline_segments": invalid_segments,
    }
    return {
        "summary": summary,
        "languages": [result.to_report_dict() for result in results],
    }


def write_report(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def exit_code_for(report: Dict[str, Any]) -> int:
    overall = report["summary"]["overall_status"]
    if overall == STATUS_SUCCESS:
        return EXIT_OK
    if overall == STATUS_FAILED:
        return EXIT_LANG_FAILED
    return EXIT_PARTIAL
