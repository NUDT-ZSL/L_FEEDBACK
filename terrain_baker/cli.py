"""Command line entry point.

Usage:
    python -m terrain_baker --config config.json --output out_dir
                            [--resume] [--jobs N]

Exit codes:
    0  success
    1  configuration / usage error
    2  bake finished but consistency errors were detected
    3  bake finished with failed tasks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .errors import BakeError, ErrorCode
from .jobs import run_bake
from .report import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="terrain_baker",
        description="Offline terrain baker: LOD meshes + world-space normal maps.")
    parser.add_argument("--config", required=True, help="path to JSON config")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--resume", action="store_true",
                        help="resume from the state file, skipping done tasks")
    parser.add_argument("--jobs", type=int, default=4,
                        help="number of worker threads (default: 4)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs < 1:
        print("error: --jobs must be >= 1", file=sys.stderr)
        return 1
    try:
        cfg = load_config(args.config)
    except BakeError as exc:
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 1

    output_dir = Path(args.output)
    try:
        exit_code, state = run_bake(cfg, output_dir, jobs=args.jobs,
                                    resume=args.resume)
    except BakeError as exc:
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level guard
        print(f"error [{ErrorCode.E_TASK_EXECUTION}]: {exc}", file=sys.stderr)
        return 1

    report_path = write_report(cfg, state, output_dir)
    print(f"report written to {report_path}")
    if exit_code == 0:
        print("bake completed successfully")
    elif exit_code == 2:
        print("bake completed with consistency errors", file=sys.stderr)
    else:
        print("bake completed with failed tasks", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
