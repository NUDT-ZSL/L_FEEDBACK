"""Command line interface for the offline dialogue mixer."""

from __future__ import annotations

import argparse
import json
import sys

from . import ERR_INTERNAL, LUFS_TOLERANCE
from .audioio import ffmpeg_available
from .mixer import run_mix
from .samplegen import DEFAULT_SEED, generate_project
from .timeline import TimelineValidationError, load_timeline


def _emit(diagnostic: dict, code: int = 1) -> int:
    print(json.dumps(diagnostic, indent=2, ensure_ascii=False), file=sys.stderr)
    return code


def cmd_generate(args) -> int:
    metadata = generate_project(args.output_dir, seed=args.seed)
    print(json.dumps({"status": "ok", **metadata}, indent=2, ensure_ascii=False))
    return 0


def cmd_mix(args) -> int:
    try:
        timeline = load_timeline(args.timeline)
        report = run_mix(timeline, args.output_wav, args.report)
    except TimelineValidationError as exc:
        return _emit(exc.diagnostics, 2)
    except Exception as exc:  # defensive: never crash without diagnostics
        return _emit({"status": "error",
                      "error": {"code": ERR_INTERNAL, "message": str(exc)}}, 3)

    print(json.dumps({
        "status": "ok",
        "mix_wav": args.output_wav,
        "report": args.report,
        "segments": report["summary"]["segment_count"],
        "aligned": report["summary"]["aligned_count"],
        "unalignable": report["summary"]["unalignable_count"],
        "limiter_engaged": report["summary"]["limiter"]["limiter_engaged"],
        "processing_seconds": report["summary"]["total_processing_seconds"],
    }, indent=2, ensure_ascii=False))
    return 0


def cmd_verify(args) -> int:
    with open(args.report, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    checks = []

    for segment in report["segments"]:
        if not segment["aligned"]:
            checks.append(_check(segment["timeline_index"], "alignment_exception_recorded",
                                  True, segment["alignment_error"] is not None))
            continue
        duration_ok = abs(segment["aligned_duration_seconds"]
                          - segment["target_duration_seconds"]) <= 1.0 / 48000.0 + 1e-9
        checks.append(_check(segment["timeline_index"], "aligned_duration_matches_target",
                             segment["target_duration_seconds"], duration_ok))
        if not segment["is_silence"] and segment["loudness_error_lu"] is not None:
            loudness_ok = abs(segment["loudness_error_lu"]) <= LUFS_TOLERANCE
            checks.append(_check(segment["timeline_index"], "loudness_within_tolerance",
                                 segment["loudness_error_lu"], loudness_ok))

    ducking = report["summary"]["ducking"]
    recovery = ducking["recovery_time_seconds"]
    recovery_ok = recovery is None or recovery <= ducking["recovery_limit_seconds"] + 1e-9
    checks.append(_check(None, "ducking_recovery_within_200ms", recovery, recovery_ok))

    limiter = report["summary"]["limiter"]
    peak_ok = limiter["post_limit_peak_dbfs"] <= limiter["ceiling_dbfs"] + 1e-9
    checks.append(_check(None, "post_limit_peak_at_or_below_ceiling",
                         limiter["post_limit_peak_dbfs"], peak_ok))

    required_fields = {"original_duration_seconds", "aligned_duration_seconds",
                       "gain_adjustment_db", "alignment_error",
                       "speed_ratio", "resampling"}
    fields_ok = all(required_fields.issubset(segment.keys())
                    for segment in report["segments"])
    checks.append(_check(None, "report_fields_complete", sorted(required_fields), fields_ok))

    passed = all(item["passed"] for item in checks)
    print(json.dumps({"status": "ok" if passed else "failed",
                      "passed": passed, "checks": checks}, indent=2, ensure_ascii=False))
    return 0 if passed else 4


def _check(index, name, observed, passed) -> dict:
    return {"timeline_index": index, "check": name,
            "observed": observed, "passed": bool(passed)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="offline-mixer",
        description="Offline multilingual dialogue alignment/loudness/ducking mixer.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate a deterministic test project")
    gen.add_argument("output_dir")
    gen.add_argument("--seed", type=int, default=DEFAULT_SEED)
    gen.set_defaults(func=cmd_generate)

    mix = sub.add_parser("mix", help="run the full mixing pipeline")
    mix.add_argument("timeline")
    mix.add_argument("-o", "--output-wav", default="mix_output.wav")
    mix.add_argument("-r", "--report", default="mix_report.json")
    mix.set_defaults(func=cmd_mix)

    verify = sub.add_parser("verify", help="validate a mix report against acceptance rules")
    verify.add_argument("report")
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    if ffmpeg_available() is None:
        print("warning: ffmpeg not found on PATH", file=sys.stderr)
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
