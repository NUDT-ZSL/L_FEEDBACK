from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from . import __version__
from .core import RunConfig, run_pipeline, run_regression
from .errors import BusinessError, EXIT_INTERNAL, EXIT_INTERRUPTED, EXIT_VERIFY
from .samplegen import generate_samples


def _add_tool_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ffmpeg", help="Explicit ffmpeg executable path or command name")
    parser.add_argument("--ffprobe", help="Explicit ffprobe executable path or command name")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bcsubts",
        description="Align broadcast subtitles and insert SCTE-35 markers into MPEG-TS",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full alignment/insertion pipeline")
    run_parser.add_argument("--audio", required=True, help="16-bit PCM WAV (44.1/48/96 kHz)")
    run_parser.add_argument("--subtitles", required=True, help="Input SRT or CSV file")
    run_parser.add_argument("--scte35", required=True, help="SCTE-35 JSON event table")
    run_parser.add_argument("--output-dir", required=True, help="Output directory")
    run_parser.add_argument(
        "--overlap-policy",
        choices=("truncate", "skip"),
        default="truncate",
        help="Subtitle overlap/negative-duration correction policy",
    )
    run_parser.add_argument("--force", action="store_true", help="Discard recovery state and restart")
    _add_tool_args(run_parser)

    generate_parser = subparsers.add_parser("generate-samples", help="Generate the deterministic sample matrix")
    generate_parser.add_argument("--out", required=True, help="Sample root directory")

    regress_parser = subparsers.add_parser("regress", help="Generate samples, run all cases and verify TS output")
    regress_parser.add_argument("--out", required=True, help="Regression output directory")
    _add_tool_args(regress_parser)
    return parser


def _print_run_summary(result: dict) -> None:
    report = result["report"]
    alignment = report["alignment"]
    print(json.dumps(
        {
            "status": result["status"],
            "estimated_offset_ms": alignment["estimated_offset_ms"],
            "confidence": alignment["confidence"],
            "corrected_subtitles": report["subtitles"]["corrected_srt"],
            "alignment_report": report["outputs"]["alignment_report"],
            "mpeg_ts": report["outputs"]["mpeg_ts"],
            "scte35_events": report["scte35"]["event_count"],
            "scte35_event_ids": result["verification"]["scte35_event_ids"],
            "embedded_subtitle_sets": result["verification"]["dvb_display_sets"],
        },
        ensure_ascii=False,
        indent=2,
    ))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate-samples":
            manifest = generate_samples(args.out)
            print(json.dumps({"status": "ok", "samples": manifest}, indent=2, ensure_ascii=False))
            return 0
        if args.command == "regress":
            summary = run_regression(args.out, args.ffmpeg, args.ffprobe)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0 if summary["status"] == "ok" else EXIT_VERIFY
        result = run_pipeline(
            RunConfig(
                audio=Path(args.audio),
                subtitles=Path(args.subtitles),
                scte35=Path(args.scte35),
                output_dir=Path(args.output_dir),
                overlap_policy=args.overlap_policy,
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
                force=args.force,
            )
        )
        _print_run_summary(result)
        return 0
    except BusinessError as exc:
        sys.stderr.write(exc.as_json() + "\n")
        return exc.exit_code
    except KeyboardInterrupt:
        interrupted = BusinessError(
            "E_INTERRUPTED",
            "interrupt",
            "Interrupted; recovery state and partial outputs were retained",
        )
        sys.stderr.write(interrupted.as_json() + "\n")
        return EXIT_INTERRUPTED
    except Exception as exc:  # pragma: no cover - last-resort CLI guard
        traceback.print_exc(file=sys.stderr)
        internal = BusinessError("E_INTERNAL", "internal", str(exc))
        sys.stderr.write(internal.as_json() + "\n")
        return EXIT_INTERNAL


if __name__ == "__main__":
    raise SystemExit(main())
