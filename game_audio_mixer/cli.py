from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .engine import (
    MixOptions,
    MixProcessingError,
    render_timeline,
    save_outputs,
)
from .timeline import TimelineFatalError, load_timeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="game-audio-mixer",
        description=(
            "Mix timestamped game-audio events to a 48 kHz / 24-bit stereo WAV "
            "and write a fixed-schema EBU R128 loudness report."
        ),
    )
    parser.add_argument(
        "--input-dir",
        default=".",
        help="base directory for relative asset paths (default: current directory)",
    )
    parser.add_argument(
        "--timeline",
        "--events",
        dest="timeline",
        required=True,
        help="event timeline JSON file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory for mix.wav and loudness_report.json",
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=-23.0,
        help="integrated loudness target in LUFS (default: -23.0)",
    )
    parser.add_argument(
        "--mute",
        action="append",
        metavar="TRACK[,TRACK...]",
        help=(
            "override initial mute states; listed tracks become muted and all "
            "others unmuted. Repeatable. Pass an empty value to clear all mutes."
        ),
    )
    parser.add_argument(
        "--solo",
        action="append",
        metavar="TRACK[,TRACK...]",
        help=(
            "override initial solo states; listed tracks become soloed and all "
            "others non-soloed. Repeatable. Pass an empty value to clear all solos."
        ),
    )
    parser.add_argument(
        "--compressor-threshold-db", type=float, default=-24.0
    )
    parser.add_argument("--compressor-ratio", type=float, default=2.5)
    parser.add_argument("--compressor-knee-db", type=float, default=6.0)
    parser.add_argument("--compressor-attack-ms", type=float, default=10.0)
    parser.add_argument("--compressor-release-ms", type=float, default=180.0)
    parser.add_argument("--limiter-threshold-db", type=float, default=-1.0)
    parser.add_argument("--limiter-ceiling-db", type=float, default=-0.2)
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def _validate_options(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not math.isfinite(args.target_lufs) or not (-70.0 <= args.target_lufs <= -1.0):
        parser.error("--target-lufs must be a finite value between -70 and -1")
    if args.compressor_ratio <= 1.0 or not math.isfinite(args.compressor_ratio):
        parser.error("--compressor-ratio must be finite and greater than 1")
    if args.compressor_knee_db < 0.0:
        parser.error("--compressor-knee-db must be non-negative")
    if args.compressor_attack_ms <= 0.0 or args.compressor_release_ms <= 0.0:
        parser.error("compressor attack and release must be positive")
    if args.limiter_threshold_db >= 0.0:
        parser.error("--limiter-threshold-db must be below 0 dBFS")
    if not (args.limiter_threshold_db < args.limiter_ceiling_db < 0.0):
        parser.error(
            "--limiter-ceiling-db must be above the limiter threshold and below 0 dBFS"
        )


def _print_diagnostics(report: dict) -> None:
    for warning in report.get("warnings", []):
        print(f"warning: {warning}", file=sys.stderr)
    for error in report.get("errors", []):
        scope = error.get("scope", "event")
        event_id = error.get("event_id")
        prefix = f"{scope} {event_id}" if event_id else scope
        print(f"error: {prefix}: {error.get('reason', 'unknown error')}", file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_options(args, parser)

    input_dir = Path(args.input_dir).expanduser().resolve()
    timeline_path = Path(args.timeline).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        print(f"fatal: input directory not found: {input_dir}", file=sys.stderr)
        return 3

    try:
        timeline = load_timeline(timeline_path, input_dir)
    except TimelineFatalError as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        return 3

    options = MixOptions(
        target_lufs=args.target_lufs,
        mute_override=args.mute,
        solo_override=args.solo,
        compressor_threshold_db=args.compressor_threshold_db,
        compressor_ratio=args.compressor_ratio,
        compressor_knee_db=args.compressor_knee_db,
        compressor_attack_ms=args.compressor_attack_ms,
        compressor_release_ms=args.compressor_release_ms,
        limiter_threshold_db=args.limiter_threshold_db,
        limiter_ceiling_db=args.limiter_ceiling_db,
    )
    try:
        final_audio, report = render_timeline(timeline, input_dir, options)
    except MixProcessingError as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        return 5

    try:
        wav_path, report_path = save_outputs(output_dir, final_audio, report)
    except MixProcessingError as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        return 4

    _print_diagnostics(report)
    loudness = report["loudness"]["integrated_lufs"]
    loudness_text = "silent" if loudness is None else f"{loudness:.2f} LUFS"
    print(
        f"wrote {wav_path} ({report['output']['duration_s']:.3f}s), "
        f"{loudness_text}; report {report_path}"
    )
    if report["status"] == "constraint_failed":
        print(
            "fatal: final mix failed loudness/true-peak delivery constraints; "
            "see report for measured values",
            file=sys.stderr,
        )
        return 5
    return 0
