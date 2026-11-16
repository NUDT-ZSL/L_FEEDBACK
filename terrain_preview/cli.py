"""Command-line entry point with stable error reporting."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import load_config
from .errors import PreviewError
from .renderer import render


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="render_terrain.py",
        description="Offline deterministic terrain preview renderer",
    )
    parser.add_argument(
        "--config",
        default="sample_config.json",
        help="path to the local JSON configuration (default: sample_config.json)",
    )
    parser.add_argument(
        "--force-full",
        action="store_true",
        help="ignore the tile cache and redraw every screen tile",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        render(config, force_full=args.force_full)
    except PreviewError as exc:
        print(f"ERROR[{exc.code}]: {exc.message}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("ERROR[E_INTERRUPTED]: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # The CLI must never leak an uncaught traceback.
        print(f"ERROR[E_UNEXPECTED]: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

