"""Command-line interface for the offline loudness pipeline."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .errors import PipelineError
from .pipeline import (
    DEFAULT_SEGMENT_SECONDS,
    DEFAULT_TARGET_LUFS,
    MAX_TARGET_LUFS,
    MIN_TARGET_LUFS,
    process_file,
)


def _target_lufs(value: str) -> float:
    try:
        numeric = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number")
    if not (MIN_TARGET_LUFS <= numeric <= MAX_TARGET_LUFS):
        raise argparse.ArgumentTypeError(
            "must be between {0:.0f} and {1:.0f} LUFS".format(
                MIN_TARGET_LUFS, MAX_TARGET_LUFS
            )
        )
    return numeric


def _positive_seconds(value: str) -> float:
    try:
        numeric = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number")
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return numeric


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loudness-normalize",
        description=(
            "Normalize multichannel WAV loudness using ITU-R BS.1770-4 "
            "measurement and EBU R128 gating."
        ),
    )
    parser.add_argument("input", type=Path, help="input WAV file")
    parser.add_argument("output", type=Path, help="output WAV file")
    parser.add_argument(
        "-t",
        "--target",
        "--target-lufs",
        dest="target_lufs",
        type=_target_lufs,
        default=DEFAULT_TARGET_LUFS,
        metavar="LUFS",
        help="target integrated loudness, -35 to -5 (default: -23)",
    )
    parser.add_argument(
        "--limiter",
        dest="limiter_enabled",
        action="store_true",
        default=True,
        help="enable look-ahead peak limiting (default)",
    )
    parser.add_argument(
        "--no-limiter",
        dest="limiter_enabled",
        action="store_false",
        help="disable dynamic limiting; use peak-safe static gain instead",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path (default: <output>.json)",
    )
    parser.add_argument(
        "--segment-seconds",
        type=_positive_seconds,
        default=DEFAULT_SEGMENT_SECONDS,
        metavar="SECONDS",
        help="report segment duration (default: 3)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {0}".format(__version__),
    )
    return parser


def format_summary(result: object) -> str:
    lines = [
        "OK",
        "input_lufs: {0:.2f} LUFS".format(result.input_loudness_lufs),
        "output_lufs: {0:.2f} LUFS".format(result.output_loudness_lufs),
        "gain_db: {0:.2f} dB".format(result.requested_gain_db),
        "applied_base_gain_db: {0:.2f} dB".format(result.applied_base_gain_db),
        "limiter_applied: {0}".format(
            "true" if result.limiter_applied else "false"
        ),
        "max_limiting_db: {0:.2f} dB".format(result.max_limiting_db),
        "elapsed_seconds: {0:.3f}".format(result.elapsed_seconds),
        "output: {0}".format(result.output_path),
        "report: {0}".format(result.report_path),
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = process_file(
            input_path=args.input,
            output_path=args.output,
            target_lufs=args.target_lufs,
            limiter_enabled=args.limiter_enabled,
            report_path=args.report,
            segment_seconds=args.segment_seconds,
        )
    except PipelineError as error:
        print("{0}: {1}".format(error.code, error.message), file=sys.stderr)
        return error.exit_code
    except KeyboardInterrupt:
        print("E_INTERNAL: interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print(
            "E_INTERNAL: unexpected failure: {0}".format(error),
            file=sys.stderr,
        )
        return PipelineError.EXIT_CODES["E_INTERNAL"]

    print(format_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

