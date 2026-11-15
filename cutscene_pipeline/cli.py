"""Command line interface: orchestrates the full offline pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__, mux, pipeline, wavfile
from .project import ProjectError, load_project

EXIT_OK = 0
EXIT_PARTIAL = 1
EXIT_FAILED = 2


def build_report(project, timer, alignment, mix_info, loudness_report,
                 files, mode, mux_info):
    return {
        "report_version": 1,
        "tool": {"name": "cutscene-pipeline", "version": __version__},
        "mode": mode,
        "project": project.name,
        "input_event_count": len(project.events) + len(project.skipped_events),
        "processed_event_count": len(project.events),
        "skipped_event_count": len(project.skipped_events),
        "skipped_events": project.skipped_events,
        "validation": project.validation,
        "timeline": {
            "sample_rate": pipeline.TIMELINE_RATE,
            "duration_s": round(project.timeline_duration, 6),
            "alignment": alignment,
        },
        "mix": mix_info,
        "loudness": loudness_report,
        "chapters": project.chapters,
        "mux": mux_info,
        "stage_timings_s": timer.timings,
        "warnings": project.warnings,
        "files": files,
    }


def run(project_path: str, out_dir: str, dry_run: bool) -> int:
    timer = pipeline.StageTimer()
    t0 = time.perf_counter()
    try:
        with timer("parse_validate"):
            project = load_project(project_path)
    except ProjectError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED

    out = Path(out_dir)
    mode = "dry-run" if dry_run else "normal"
    work = None if dry_run else out
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)

    tracks, alignment = pipeline.align_events(project, timer, work)
    mix, mix_info = pipeline.mix_tracks(project, tracks, timer, work)
    normalized, loudness_report = pipeline.normalize_loudness(
        mix, project.loudness["target_lufs"],
        project.loudness["true_peak_dbtp"], timer)

    files = {"aligned_dir": None, "mix_wav": None, "normalized_wav": None,
             "mp4": None, "report": str(out / "report.json")}
    mux_info = None
    if not dry_run:
        norm_wav = out / "normalized_48k_24bit_stereo.wav"
        wavfile.write_wav24(norm_wav, normalized, pipeline.TIMELINE_RATE)
        files["aligned_dir"] = str(out / "aligned")
        files["mix_wav"] = str(out / "mix_48k_24bit_stereo.wav")
        files["normalized_wav"] = str(norm_wav)
        try:
            with timer("mux"):
                mux_info = mux.mux_mp4(norm_wav, out / "cutscene.mp4",
                                       project.video, project.timeline_duration,
                                       project.chapters, out)
        except mux.MuxError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_FAILED
        files["mp4"] = str(out / "cutscene.mp4")

    timer.timings["total"] = round(time.perf_counter() - t0, 6)
    report = build_report(project, timer, alignment, mix_info, loudness_report,
                          files, mode, mux_info)
    if dry_run:
        # Dry-run writes no media files; the report goes to stdout only.
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"report: {report_path}")

    if project.skipped_events or project.warnings:
        print(f"completed with {len(project.warnings)} warning(s), "
              f"{len(project.skipped_events)} skipped event(s)", file=sys.stderr)
        return EXIT_PARTIAL
    return EXIT_OK


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="cutscene-pipeline",
        description="Offline cutscene audio pipeline: deterministic synth, "
                    "timeline alignment, mixing, EBU R128 normalization, MP4 mux.")
    parser.add_argument("project", help="path to project JSON")
    parser.add_argument("-o", "--out-dir", required=True, help="output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and estimate loudness without writing "
                             "any audio/video files (report goes to stdout)")
    args = parser.parse_args(argv)
    return run(args.project, args.out_dir, args.dry_run)
