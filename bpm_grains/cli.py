"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys

from .errors import AudioToolError
from .pipeline import render


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bpm-grain",
        description=(
            "BPM-align and granular-synthesize a PCM WAV file with optional "
            "percussion mixing."
        ),
    )
    parser.add_argument("--input", required=True, help="input 44.1 kHz PCM WAV")
    parser.add_argument(
        "--percussion", default=None, help="optional percussion PCM WAV"
    )
    parser.add_argument("--output", required=True, help="final stereo WAV path")
    parser.add_argument(
        "--seed", type=int, default=0, help="deterministic grain scheduling seed"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="DEBUG, INFO, WARNING, ERROR, or CRITICAL (logs go to stderr)",
    )
    return parser


def _configure_logging(level_name: str) -> int:
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"invalid log level: {level_name}")
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(levelname)s %(message)s",
        force=True,
    )
    return level


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        _configure_logging(args.log_level)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        render(
            input_path=args.input,
            percussion_path=args.percussion,
            output_path=args.output,
            seed=args.seed,
        )
    except AudioToolError as exc:
        # Business error codes must remain visible even at CRITICAL log level.
        sys.stderr.write(f"ERROR {exc.code}: {exc.detail}\n")
        sys.stderr.flush()
        return exc.exit_code
    except Exception:
        logging.exception("ERR_INTERNAL: unexpected processing failure")
        return 1
    return 0

