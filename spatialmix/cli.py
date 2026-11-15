"""Command-line interface for the offline spatial mixer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .engine import process_project
from .errors import ToolError


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ToolError("arguments: {}".format(message))


def _build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="spatial_mixer.py",
        description=(
            "Build a deterministic 48 kHz stereo spatial mix from WAV files "
            "and a JSON event timeline."
        ),
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        help="directory containing the source WAV files",
    )
    parser.add_argument(
        "events_json",
        nargs="?",
        help="JSON event description file",
    )
    parser.add_argument(
        "mix_output",
        nargs="?",
        help="output stereo 16-bit PCM WAV path",
    )
    parser.add_argument(
        "report_output",
        nargs="?",
        help="output JSON report path",
    )
    parser.add_argument(
        "--source-dir",
        "--sources-dir",
        "--sources",
        dest="source_dir_option",
        metavar="DIR",
        help="alternative to the first positional argument",
    )
    parser.add_argument(
        "--events",
        "--events-file",
        "--event-json",
        dest="events_option",
        metavar="JSON",
        help="alternative to the second positional argument",
    )
    parser.add_argument(
        "--mix-output",
        "--output-mix",
        "--output",
        "-o",
        dest="mix_option",
        metavar="WAV",
        help="alternative to the third positional argument",
    )
    parser.add_argument(
        "--report-output",
        "--output-report",
        "--report-file",
        "--report",
        "-r",
        dest="report_option",
        metavar="JSON",
        help="alternative to the fourth positional argument",
    )
    parser.add_argument(
        "--global-conflict-policy",
        "--conflict-policy",
        choices=("mix", "drop", "delay"),
        dest="global_policy",
        help="fallback policy for events that omit conflict_policy",
    )
    return parser


def _resolve_argument(
    positional: Optional[str],
    option: Optional[str],
    field: str,
) -> str:
    if positional is not None and option is not None:
        raise ToolError(
            "{}: specified as both positional and option".format(field)
        )
    value = option if option is not None else positional
    if value is None or not str(value):
        raise ToolError("{}: missing required argument".format(field))
    return str(value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        source_dir = _resolve_argument(
            args.source_dir,
            args.source_dir_option,
            "sources_dir",
        )
        events_json = _resolve_argument(
            args.events_json,
            args.events_option,
            "events_file",
        )
        mix_output = _resolve_argument(
            args.mix_output,
            args.mix_option,
            "mix_output",
        )
        report_output = _resolve_argument(
            args.report_output,
            args.report_option,
            "report_output",
        )
        process_project(
            Path(source_dir),
            Path(events_json),
            Path(mix_output),
            Path(report_output),
            global_policy=args.global_policy,
        )
        return 0
    except ToolError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("ERROR: execution interrupted", file=sys.stderr)
        return 1
    except Exception as exc:
        print("ERROR: internal processing failure: {}".format(exc), file=sys.stderr)
        return 1
