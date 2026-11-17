"""Command line entry point.

Usage:
    python -m audiotl <manifest.json> <output_dir> [--workers N] [--state-dir DIR]

Exit codes:
    0  success
    1  internal error
    2  manifest invalid
    3  input missing
    4  render business error (fade/overlap/format)
    5  stage version conflict
"""

import argparse
import json
import sys

from .pipeline import Pipeline


def build_parser():
    parser = argparse.ArgumentParser(
        prog="audiotl",
        description="Deterministic audio timeline builder",
    )
    parser.add_argument("manifest", help="path to the timeline manifest JSON")
    parser.add_argument("output_dir", help="directory for mix.wav/mapping/summary/report")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="segment render workers (clamped to logical CPU count)",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="shared cache/state directory (default: <output_dir>/_state)",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    pipeline = Pipeline(
        args.manifest,
        args.output_dir,
        workers=args.workers,
        state_dir=args.state_dir,
    )
    report, exit_code = pipeline.run()
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
