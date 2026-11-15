"""Command line interface for the cutscene audio mixer."""

import argparse
import json
import sys

from .engine import run
from .errors import SCHEMA_INVALID, SchemaError


def build_parser():
    parser = argparse.ArgumentParser(
        prog="cutscene-mixer",
        description="Align multi-track audio to video frame events and "
                    "render a loudness-normalized mix WAV plus sync report.",
    )
    parser.add_argument("--input", required=True,
                        help="Path to the JSON descriptor (relative or absolute).")
    parser.add_argument("--output", required=True,
                        help="Path of the mixed WAV to write.")
    parser.add_argument("--report", required=True,
                        help="Path of the sync report JSON to write.")
    parser.add_argument("--target-lufs", type=float, default=-16.0,
                        help="Integrated loudness target for the mix "
                             "(default: -16 LUFS).")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return run(args.input, args.output, args.report, args.target_lufs)
    except SchemaError as exc:
        payload = {"error": SCHEMA_INVALID, "fields": exc.paths}
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print("INPUT_NOT_FOUND: %s" % exc, file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        payload = {"error": SCHEMA_INVALID, "fields": ["$"], "detail": str(exc)}
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
