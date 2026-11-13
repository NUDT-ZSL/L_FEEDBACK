"""Command line entry point: generate-samples / export / validate."""

from __future__ import annotations

import argparse
import json
import sys

from .errors import ExporterError
from .exporter import export, generate_sample_project, validate


def _print(payload) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2, default=str))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="replay-exporter",
        description="Deterministic chunked replay master exporter",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-samples", help="create deterministic input project")
    gen.add_argument("root")
    gen.add_argument("--segments", type=int, default=4)
    gen.add_argument("--segment-ms", type=int, default=1200)

    exp = sub.add_parser("export", help="render or resume a master export")
    exp.add_argument("timeline")
    exp.add_argument("output_dir")
    exp.add_argument("--checkpoint-every", type=int, default=5)
    exp.add_argument("--stop-after-blocks", type=int, default=None)

    val = sub.add_parser("validate", help="produce the replay validation report")
    val.add_argument("output_dir")
    val.add_argument("--timeline", default=None)

    args = parser.parse_args(argv)
    try:
        if args.command == "generate-samples":
            _print(
                generate_sample_project(
                    args.root,
                    segment_count=args.segments,
                    segment_ms=args.segment_ms,
                )
            )
        elif args.command == "export":
            _print(
                export(
                    args.timeline,
                    args.output_dir,
                    checkpoint_every=args.checkpoint_every,
                    stop_after_blocks=args.stop_after_blocks,
                )
            )
        elif args.command == "validate":
            _print(validate(args.output_dir, timeline_path=args.timeline))
    except ExporterError as error:
        _print(error.to_dict())
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
