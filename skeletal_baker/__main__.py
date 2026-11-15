"""CLI entry point.

Usage: python -m skeletal_baker --output-dir DIR [--frames N] [--seed S]
       [--fps F]
Invalid arguments exit with code 2 and a usage message on stderr
(argparse default). Business errors never change the exit code.
"""

from __future__ import annotations

import argparse
import sys

from .pipeline import bake


def _positive_int(value: str) -> int:
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid integer value: {value!r}"
        ) from None
    if ivalue < 1:
        raise argparse.ArgumentTypeError(
            f"value must be a positive integer: {value!r}"
        )
    return ivalue


def _non_negative_int(value: str) -> int:
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid integer value: {value!r}"
        ) from None
    if ivalue < 0:
        raise argparse.ArgumentTypeError(
            f"value must be a non-negative integer: {value!r}"
        )
    return ivalue


def _positive_float(value: str) -> float:
    try:
        fvalue = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid float value: {value!r}"
        ) from None
    if fvalue <= 0.0:
        raise argparse.ArgumentTypeError(
            f"value must be a positive number: {value!r}"
        )
    return fvalue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skeletal_baker",
        description="Bake a two-bone skinned capsule animation into "
                    "per-frame vertex/normal buffers and PNG previews.",
    )
    parser.add_argument("--output-dir", required=True,
                        help="directory for frame buffers, PNGs, overview "
                             "and errors.log")
    parser.add_argument("--frames", type=_positive_int, default=48,
                        help="number of frames to sample (default: 48)")
    parser.add_argument("--seed", type=_non_negative_int, default=0,
                        help="deterministic seed (default: 0)")
    parser.add_argument("--fps", type=_positive_float, default=24.0,
                        help="fixed sampling frame rate (default: 24.0)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return bake(args.output_dir, args.frames, args.seed, args.fps)


if __name__ == "__main__":
    sys.exit(main())
