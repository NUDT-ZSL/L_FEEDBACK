"""Processing report assembly and serialization."""
from __future__ import annotations

import json
from datetime import datetime, timezone

PROCESSOR_VERSION = "1.0.0"


def build_report(batch_result: dict, input_dir: str, output_dir: str,
                 parallelism: int) -> dict:
    tracks = batch_result["tracks"]
    error_list = []
    for track in tracks:
        for message in track.get("errors", []):
            error_list.append({
                "file_name": track["file_name"],
                "message": message,
            })
        if track.get("temp_dir"):
            error_list.append({
                "file_name": track["file_name"],
                "retained_temp_dir": track["temp_dir"],
            })

    if not tracks:
        overall_status = "failed"
    elif all(track["status"] == "success" for track in tracks):
        overall_status = "success"
    else:
        overall_status = "partial"

    return {
        "processing_timestamp": datetime.now(timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "processor_version": PROCESSOR_VERSION,
        "overall_status": overall_status,
        "input_directory": input_dir,
        "output_directory": output_dir,
        "parallelism": parallelism,
        "total_tracks": len(tracks),
        "successful_tracks": sum(
            1 for track in tracks if track["status"] == "success"
        ),
        "failed_tracks": sum(
            1 for track in tracks if track["status"] == "error"
        ),
        "retained_temp_dirs": batch_result.get("retained_temp_dirs", []),
        "tracks": tracks,
        "errors": error_list,
    }


def write_report(report: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
