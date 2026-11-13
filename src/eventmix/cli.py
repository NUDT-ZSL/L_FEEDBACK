"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .batch import BatchOptions, run_batch
from .demo import generate_demo
from .errors import ConfigError, EventMixError
from .util import atomic_write_bytes, pretty_json_bytes
from .version import REPORT_SCHEMA, TOOL_VERSION


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eventmix",
        description="Align multitrack esports commentary and package broadcast WAV programs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {TOOL_VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build one timeline or a batch manifest")
    build.add_argument("input", type=Path, help="timeline JSON or batch manifest JSON")
    build.add_argument(
        "--output-dir", "--output", dest="output_dir", type=Path, required=True
    )
    build.add_argument("--checkpoint", type=Path)
    checkpoint_mode = build.add_mutually_exclusive_group()
    checkpoint_mode.add_argument(
        "--resume",
        action="store_true",
        help="reuse valid checkpoints and partially completed batches (default)",
    )
    checkpoint_mode.add_argument(
        "--fresh", action="store_true", help="ignore and replace checkpoint state"
    )
    checkpoint_mode.add_argument(
        "--no-resume", action="store_true", help="alias for --fresh"
    )
    build.add_argument("--jobs", type=_positive_int, default=1)
    build.add_argument("--max-open-wav", type=_positive_int, default=8)
    build.add_argument("--max-memory-mb", type=_positive_int, default=256)
    build.add_argument("--chunk-frames", type=_positive_int, default=8192)
    build.add_argument("--sinc-half-width", type=_positive_int, default=16)
    build.add_argument("--sinc-max-half-width", type=_positive_int, default=64)
    build.add_argument("--on-track-error", choices=("silence", "fail"))
    build.add_argument("--bit-depth", type=int, choices=(16, 24, 32))
    build.add_argument("--target-rate", type=_positive_int)
    build.add_argument("--silence-threshold-dbfs", type=float)
    build.add_argument("--min-silence-seconds", type=float)
    build.add_argument("--wav-name", help="output WAV name when input is one timeline")
    build.add_argument(
        "--timeline-output-name",
        help="playback timeline output name when input is one timeline",
    )
    build.add_argument("--report", type=Path, help="also write the JSON report here")
    build.add_argument("--no-progress", action="store_true")

    demo = subparsers.add_parser("demo", help="generate deterministic WAV/timeline fixtures")
    demo.add_argument("--output-dir", type=Path, required=True)
    return parser


def _fatal_report(command: str, exc: BaseException) -> dict[str, Any]:
    code = exc.code if isinstance(exc, EventMixError) else "internal_error"
    return {
        "schema_version": REPORT_SCHEMA,
        "tool_version": TOOL_VERSION,
        "status": "fatal",
        "exit_code": 2,
        "invocation": {"command": command},
        "summary": {
            "total": 0,
            "completed": 0,
            "skipped": 0,
            "failed": 0,
            "warnings": 0,
            "degraded": False,
        },
        "tasks": [],
        "errors": [
            {
                "code": code,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "fatal": True,
            }
        ],
    }


def _write_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is not None:
        atomic_write_bytes(path, pretty_json_bytes(report))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        try:
            paths = generate_demo(args.output_dir)
        except BaseException as exc:
            report = _fatal_report("demo", exc)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            print(f"eventmix: fatal: {exc}", file=sys.stderr)
            return 2
        result = {
            "schema_version": "eventmix-demo/v1",
            "status": "ok",
            "exit_code": 0,
            "outputs": {key: str(value) for key, value in paths.items()},
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    checkpoint = args.checkpoint or (args.output_dir / ".eventmix-checkpoint.json")
    progress = None if args.no_progress else (lambda message: print(message, file=sys.stderr))
    options = BatchOptions(
        output_dir=args.output_dir,
        checkpoint_path=checkpoint,
        resume=not (args.fresh or args.no_resume),
        jobs=args.jobs,
        max_open_wav=args.max_open_wav,
        max_memory_mb=args.max_memory_mb,
        chunk_frames=args.chunk_frames,
        sinc_half_width=args.sinc_half_width,
        sinc_max_half_width=args.sinc_max_half_width,
        on_track_error=args.on_track_error,
        bit_depth=args.bit_depth,
        target_sample_rate=args.target_rate,
        silence_threshold_dbfs=args.silence_threshold_dbfs,
        min_silence_seconds=args.min_silence_seconds,
        wav_name=args.wav_name,
        timeline_output_name=args.timeline_output_name,
    )
    try:
        report = run_batch(args.input, options, progress=progress)
    except KeyboardInterrupt:
        report = _fatal_report("build", ConfigError("interrupted"))
        report["errors"][0]["code"] = "interrupted"
    except BaseException as exc:
        report = _fatal_report("build", exc)
        print(f"eventmix: fatal: {exc}", file=sys.stderr)
    try:
        _write_report(args.report, report)
    except Exception as exc:
        report = _fatal_report("build", exc)
        print(f"eventmix: fatal: cannot write report: {exc}", file=sys.stderr)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(report.get("exit_code", 2))


if __name__ == "__main__":
    raise SystemExit(main())

