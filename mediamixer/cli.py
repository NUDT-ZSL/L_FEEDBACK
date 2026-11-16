"""Command line entry point."""

from __future__ import annotations

import argparse

from .errors import emit_error
from .errors import MixerError
from .pipeline import execute
from .project import load_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mediamixer",
        description="Mix multiple audio tracks onto a video using FFmpeg and NumPy.",
    )
    parser.add_argument("project", help="path to the project description JSON file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        project = load_project(args.project)
        execute(project)
    except KeyboardInterrupt:
        return emit_error(MixerError("interrupted by user", "INTERRUPTED"))
    except Exception as exc:  # noqa: BLE001 - all failures become JSON errors
        return emit_error(exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
