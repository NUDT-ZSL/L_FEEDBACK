"""Command-line entry point for the audio packaging tool."""

from __future__ import annotations

import argparse
import json
import sys

from .errors import ERROR_CODES, PipelineError
from .memoryutil import DEFAULT_MEMORY_LIMIT_BYTES
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiopack",
        description=(
            "Render WAV clips and a timeline.json into one normalized, "
            "24-bit stereo WAV plus a slicing report."
        ),
    )
    parser.add_argument("input_dir", help="directory containing timeline.json and .wav clips")
    parser.add_argument("output_wav", help="output .wav path")
    parser.add_argument(
        "--report",
        dest="report_path",
        default=None,
        help="report JSON path (defaults to <output_wav>_report.json)",
    )
    parser.add_argument(
        "--memory-limit-mb",
        type=int,
        default=DEFAULT_MEMORY_LIMIT_BYTES // (1024 * 1024),
        help="audio buffer memory cap in MiB (default: 512)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        exit_code, _report, _errors = run_pipeline(
            input_dir=args.input_dir,
            output_wav=args.output_wav,
            report_path=args.report_path,
            memory_limit_bytes=args.memory_limit_mb * 1024 * 1024,
        )
    except PipelineError as exc:
        _emit_error(exc)
        return 1

    # Per-clip corrupt errors still produce outputs but force exit code 1.
    if _errors:
        for error in _errors:
            if error.get("code") in ERROR_CODES:
                json.dump({"error": error}, sys.stderr, sort_keys=True)
                sys.stderr.write("\n")
    return exit_code


def _emit_error(error: PipelineError) -> None:
    payload = {"error": {"code": error.code, "details": error.details}}
    json.dump(payload, sys.stderr, sort_keys=True)
    sys.stderr.write("\n")


if __name__ == "__main__":
    sys.exit(main())
