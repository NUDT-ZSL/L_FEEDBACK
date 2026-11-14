"""Deterministic end-to-end acceptance run without external downloads."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

from subalign.cli import main
from subalign.subtitles import load_subtitle
from tests.samples import (
    ARABIC_LINES,
    drop_event,
    linear_drift,
    reference_events,
    segmented_drift,
    swap_events,
    translated_events,
    write_bytes,
    write_subtitle,
)


def main_demo() -> int:
    workspace = tempfile.mkdtemp(prefix="subalign_acceptance_")
    print(f"[setup] deterministic workspace: {workspace}")
    events = reference_events(count=24)
    reference_path = os.path.join(workspace, "reference.srt")
    write_subtitle(reference_path, events)

    targets = {
        "linear_en.srt": linear_drift(events, 1.0, 1250),
        "scaled_en.srt": linear_drift(events, 1.04, 1200),
        "segmented_en.srt": segmented_drift(
            events, events[12][0], 1.0, 600, 1.02, -1500
        ),
        "missing_event.srt": drop_event(linear_drift(events, 1.0, 900), 7),
        "reordered.srt": swap_events(linear_drift(events, 1.0, 700), 5),
        "chinese.srt": linear_drift(translated_events(events), 1.0, 1100),
        "arabic.srt": linear_drift(translated_events(events, ARABIC_LINES), 1.0, 450),
        "russian_utf16.srt": linear_drift(events, 1.0, 800),
        "japanese.ass": linear_drift(translated_events(events), 1.0, 350),
    }
    encodings = {"russian_utf16.srt": "utf-16-le"}
    target_paths: list[str] = []
    for name, target_events in targets.items():
        path = os.path.join(workspace, name)
        fmt = "ass" if name.endswith(".ass") else "srt"
        write_subtitle(path, target_events, fmt, encodings.get(name, "utf-8"))
        target_paths.append(path)

    broken = os.path.join(workspace, "broken.srt")
    write_bytes(broken, b"\x80\x81 invalid byte sequence")
    target_paths.append(broken)

    output_dir = os.path.join(workspace, "aligned")
    report_path = os.path.join(workspace, "report.json")
    exit_code = main(
        [
            "--reference",
            reference_path,
            "--targets",
            *target_paths,
            "--output-dir",
            output_dir,
            "--report",
            report_path,
        ]
    )

    report = json.load(open(report_path, encoding="utf-8"))
    print("\n[results]")
    for entry in report["files"]:
        aligned = load_subtitle(entry["output_path"])
        match_pairs = {
            item["target_index"]: item["reference_index"] for item in entry["matches"]
        }
        errors = [
            abs(aligned.events[target_index].start - events[ref_index][0])
            for target_index, ref_index in match_pairs.items()
        ]
        if entry["input_path"].endswith("reordered.srt"):
            # The swapped cues intentionally stay in file order by design.
            errors = [
                abs(aligned.events[target_index].start - events[ref_index][0])
                for target_index, ref_index in match_pairs.items()
                if target_index not in (5, 6)
            ]
        max_error = max(errors) if errors else float("nan")
        model = entry["drift_model"]
        print(
            f"  {os.path.basename(entry['input_path']):>20} "
            f"model={model['type']:<9} max_err={max_error:6.1f}ms "
            f"matched={entry['match_statistics']['matched_event_count']:>2} "
            f"warnings={len(entry['warnings'])}"
        )

    print("\n[errors]")
    for error in report["errors"]:
        print(f"  {os.path.basename(error['path'])}: {error['error_type']}")

    print(f"\n[report] {report_path}")
    print(f"[exit code] {exit_code} (non-zero expected because one file is invalid)")
    shutil.rmtree(workspace, ignore_errors=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main_demo())
