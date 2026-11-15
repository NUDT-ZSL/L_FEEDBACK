"""Command line entry point.

Usage::

    python -m seamless_loop path/to/project.json

The project JSON path is the only positional argument.  All documented error
conditions terminate with a non-zero integer exit code and a message on
stderr.
"""

import argparse
import logging
import sys

from .config import load_project
from .errors import LoopProcessorError
from .processor import process_project


def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="seamless_loop",
        description=(
            "Build a seamless, equal-power crossfaded multi-track audio loop "
            "from a JSON project description."
        ),
    )
    parser.add_argument(
        "project_json",
        help="Path to the JSON project description (the only argument).",
    )
    return parser


def main(argv=None):
    _configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        project = load_project(args.project_json)
        output_path = process_project(project)
    except LoopProcessorError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:  # pragma: no cover - interactive only.
        print("ERROR: interrupted by user", file=sys.stderr)
        return 130

    print("OK: wrote %s" % output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
