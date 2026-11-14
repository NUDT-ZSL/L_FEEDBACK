"""End-to-end alignment orchestration used by the command line entry point."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .drift import align_events, fit_segmented
from .matching import match_events
from .report import build_file_report, utc_now_iso
from .subtitles import SubtitleError, SubtitleEvent, SubtitleFile, load_subtitle, output_path_for, save_subtitle
from .video import FFmpegUnavailable, extract_reference_events

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts"}


@dataclass
class RunResult:
    report: dict
    output_paths: list[str] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)


def load_reference(reference_path: str) -> tuple[list[SubtitleEvent], list[dict], str]:
    """Load a reference subtitle or derive events from a video file."""

    warnings: list[dict] = []
    lower = reference_path.lower()
    if any(lower.endswith(ext) for ext in VIDEO_EXTENSIONS):
        try:
            events = extract_reference_events(reference_path)
        except FFmpegUnavailable as exc:
            warnings.append(
                {
                    "code": "FFMPEG_UNAVAILABLE",
                    "message": f"{exc}; attempting to read the path as a subtitle file",
                }
            )
            try:
                reference = load_subtitle(reference_path)
            except SubtitleError as error:
                error.warnings = warnings  # type: ignore[attr-defined]
                raise
            return reference.events, warnings, f"subtitle:{reference_path}"
        source = f"video:{reference_path}"
        return events, warnings, source

    reference = load_subtitle(reference_path)
    return reference.events, warnings, f"subtitle:{reference_path}"


def align_one_file(
    reference_events: list[SubtitleEvent],
    target_path: str,
    output_dir: str | None,
) -> tuple[dict, str]:
    target: SubtitleFile = load_subtitle(target_path)
    started = time.perf_counter()
    matches, translated = match_events(reference_events, target.events)
    if len(matches) < 2:
        raise SubtitleError(
            target_path,
            "InsufficientMatches",
            f"only {len(matches)} events could be matched against the reference",
        )
    model = fit_segmented(matches)
    result = align_events(reference_events, target.events, matches, model)
    output_path = output_path_for(target_path, output_dir)
    save_subtitle(target, result.events, output_path)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    report_entry = build_file_report(
        input_path=target_path,
        output_path=output_path,
        result=result,
        elapsed_ms=elapsed_ms,
        translated=translated,
    )
    return report_entry, output_path


def run(
    reference_path: str,
    target_paths: list[str],
    output_dir: str | None,
    report_path: str,
) -> RunResult:
    reference_events: list[SubtitleEvent]
    reference_warnings: list[dict] = []
    reference_source = ""
    report: dict = {
        "tool": "subalign",
        "schema_version": "1.0",
        "generated_at": utc_now_iso(),
        "reference": {
            "path": reference_path,
            "source": "",
            "event_count": 0,
            "warnings": [],
        },
        "files": [],
        "errors": [],
    }
    run_result = RunResult(report=report)
    run_result.report = report

    try:
        reference_events, reference_warnings, reference_source = load_reference(reference_path)
    except SubtitleError as error:
        run_result.errors.append(error.to_dict())
        report["reference"].update(
            {
                "source": f"failed:{reference_path}",
                "warnings": getattr(error, "warnings", []),
            }
        )
        report["errors"] = run_result.errors
        from .report import write_report

        write_report(report_path, report)
        return run_result

    report["reference"].update(
        {
            "source": reference_source,
            "event_count": len(reference_events),
            "warnings": reference_warnings,
        }
    )

    for target_path in sorted(dict.fromkeys(target_paths)):
        try:
            entry, output_path = align_one_file(reference_events, target_path, output_dir)
            if reference_warnings:
                entry["warnings"] = list(reference_warnings) + entry["warnings"]
            report["files"].append(entry)
            run_result.output_paths.append(output_path)
        except SubtitleError as error:
            run_result.errors.append(error.to_dict())

    report["errors"] = run_result.errors
    from .report import write_report

    write_report(report_path, report)
    return run_result
