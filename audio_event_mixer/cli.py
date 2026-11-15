"""Command-line interface for generation, processing, and self-test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .errors import BusinessError, ErrorCode
from .generator import generate_demo_dataset
from .processor import (
    DEFAULT_CHUNK_FRAMES,
    DEFAULT_MEMORY_LIMIT_BYTES,
    DEFAULT_OUTPUT_BIT_DEPTH,
    DEFAULT_TARGET_SAMPLE_RATE,
    process_events,
)

_EXIT_CODES = {
    ErrorCode.INVALID_EVENT: 2,
    ErrorCode.FILE_NOT_FOUND: 3,
    ErrorCode.UNSUPPORTED_FORMAT: 4,
    ErrorCode.OUTPUT_DENIED: 5,
}


def _add_process_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--events", required=True, help="event-description JSON")
    parser.add_argument("--output", required=True, help="destination stereo WAV")
    parser.add_argument("--report", default=None, help="destination JSON report")
    parser.add_argument(
        "--sample-rate",
        "--target-sample-rate",
        dest="sample_rate",
        type=int,
        default=DEFAULT_TARGET_SAMPLE_RATE,
        help="target sample rate in Hz (default: 48000)",
    )
    parser.add_argument(
        "--bit-depth",
        dest="bit_depth",
        type=int,
        default=DEFAULT_OUTPUT_BIT_DEPTH,
        help="output PCM bit depth (default: 16)",
    )
    parser.add_argument(
        "--memory-limit-mb",
        type=int,
        default=DEFAULT_MEMORY_LIMIT_BYTES // (1024 * 1024),
        help="working-set budget in MiB (default: 2048)",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=DEFAULT_CHUNK_FRAMES,
        help="streaming I/O block size in frames",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Deterministic offline game-audio event mixer with panning LFOs, "
            "loop crossfades, and JSON reports."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="write deterministic multi-format sample data"
    )
    generate.add_argument(
        "--out-dir",
        default="demo_data",
        help="dataset directory (default: demo_data)",
    )

    process = subparsers.add_parser(
        "process", help="render events to a stereo WAV mix"
    )
    _add_process_arguments(process)

    selftest = subparsers.add_parser(
        "selftest", help="run end-to-end acceptance checks"
    )
    selftest.add_argument(
        "--out-dir",
        default="validation",
        help="validation output directory (default: validation)",
    )
    return parser


def _print_business_error(error: BusinessError) -> None:
    sys.stderr.write(f"{error}\n")


def _run_process(args: argparse.Namespace) -> int:
    try:
        report = process_events(
            args.events,
            args.output,
            args.report,
            target_sample_rate=args.sample_rate,
            output_bit_depth=args.bit_depth,
            memory_limit_bytes=args.memory_limit_mb * 1024 * 1024,
            chunk_frames=args.chunk_frames,
        )
    except BusinessError as exc:
        _print_business_error(exc)
        return _EXIT_CODES.get(exc.code, 1)
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"input error: {exc}\n")
        return 2

    sys.stdout.write(
        "status=ok "
        f"output={report['output_path']} "
        f"report={report['report_path']} "
        f"events={report['event_count']} "
        f"frames={report['timeline_frames']} "
        f"sample_rate={report['output_sample_rate']} "
        f"bit_depth={report['output_bit_depth']}\n"
    )
    return 0


def _run_generate(args: argparse.Namespace) -> int:
    try:
        manifest = generate_demo_dataset(args.out_dir)
    except OSError as exc:
        sys.stderr.write(f"{ErrorCode.OUTPUT_DENIED}: {exc}\n")
        return _EXIT_CODES[ErrorCode.OUTPUT_DENIED]
    sys.stdout.write(
        f"dataset={Path(manifest['root'])} "
        f"events={manifest['events_path']} "
        f"files={len(manifest['files'])}\n"
    )
    return 0


def _run_selftest(args: argparse.Namespace) -> int:
    from .selftest import run_self_test

    result = run_self_test(args.out_dir)
    report_path = result["report_path"]
    if result["status"] == "passed":
        sys.stdout.write(
            f"status=passed checks={len(result['checks'])} "
            f"report={report_path}\n"
        )
        return 0
    failed = [check["name"] for check in result["checks"] if not check["passed"]]
    sys.stderr.write(
        f"status=failed failed_checks={','.join(failed)} report={report_path}\n"
    )
    return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] not in {
        "generate",
        "process",
        "selftest",
        "-h",
        "--help",
    }:
        raw_argv.insert(0, "process")
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    if args.command == "generate":
        return _run_generate(args)
    if args.command == "process":
        return _run_process(args)
    if args.command == "selftest":
        return _run_selftest(args)
    parser.error(f"unknown command: {args.command}")
    return 2
