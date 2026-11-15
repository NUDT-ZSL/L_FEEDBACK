from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .builder import build
from .errors import BuilderError


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hls-audio-builder",
        description=(
            "Build sample-aligned multi-bitrate AAC HLS audio segments and playlists."
        ),
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        help="path to the JSON build configuration",
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_option",
        help="path to the JSON build configuration",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    if bool(args.config_path) == bool(args.config_option):
        parser.error("specify exactly one config path (positional or --config)")
    config_path = Path(args.config_option or args.config_path)

    try:
        result = build(config_path)
    except BuilderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:
        print(f"error: unexpected failure: {exc}", file=sys.stderr)
        return 1

    print(
        f"built {len(result.variants)} variants x {result.segment_count} segments "
        f"at {result.sample_rate} Hz into {result.output_dir}"
    )
    print(f"report: {result.report_path}")
    return 0

