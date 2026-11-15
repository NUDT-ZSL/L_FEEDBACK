"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .errors import BusinessError
from .pipeline import process_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="postproduce",
        description=(
            "Render a deterministic multitrack podcast WAV, chapter metadata, "
            "and processing report."
        ),
    )
    parser.add_argument(
        "-m",
        "--manifest",
        required=True,
        type=Path,
        help="path to the clip manifest JSON",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="output directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = process_manifest(args.manifest, args.output)
    except BusinessError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # Keep the CLI business-facing and non-zero.
        print(f"E_PROCESSING: {exc}", file=sys.stderr)
        return 8

    loudness = result.report["loudness"]
    print(
        f"wrote {args.output / 'episode.wav'} "
        f"({result.manifest.total_samples} samples, "
        f"{loudness['integrated_lufs']:.3f} LUFS, "
        f"{loudness['true_peak_dbtp']:.3f} dBTP)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

