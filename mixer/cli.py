"""Command-line entry point for the offline mixer."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .atomic_io import write_json_atomic
from .engine import mix_files
from .errors import MixerError
from .report import build_error_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mixer",
        description="Render a deterministic choice-path audio mix.",
    )
    parser.add_argument("--config", required=True, help="shot description JSON")
    parser.add_argument("--path", required=True, help="selected shot path JSON")
    parser.add_argument("--out", required=True, help="output multichannel WAV")
    parser.add_argument("--report", required=True, help="output alignment report JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        mix_files(args.config, args.path, args.out, args.report)
    except MixerError as error:
        report = build_error_report(error)
        try:
            write_json_atomic(args.report, report)
        except OSError as write_error:
            print(
                f"cannot write error report {args.report}: {write_error}",
                file=sys.stderr,
            )
        print(f"{error.error_code}: {error.message}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
