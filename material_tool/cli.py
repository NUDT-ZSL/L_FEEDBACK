"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .graph import CHANNELS
from .pipeline import RunSettings, run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="material-graph-renderer",
        description=(
            "Compile deterministic material graphs to GLSL and render semantic "
            "channels with an OpenGL 3.3 offscreen framebuffer."
        ),
    )
    parser.add_argument(
        "material_dir",
        nargs="?",
        type=Path,
        help="Directory containing material graph JSON files; omit to use built-ins.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("render_output"),
        help="Output directory (default: render_output).",
    )
    parser.add_argument(
        "--channels",
        default=",".join(CHANNELS),
        help="Comma-separated channels (default: all five).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Square framebuffer size (default: 512).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Cross-channel absolute tolerance for shared parameters (default: 1e-4).",
    )
    parser.add_argument(
        "--no-reload-check",
        action="store_true",
        help="Disable the per-graph JSON hot-reload compile/render/resource probe.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.size <= 0:
        parser.error("--size must be positive")
    if args.tolerance < 0:
        parser.error("--tolerance must be non-negative")
    channels = [item.strip() for item in args.channels.split(",") if item.strip()]
    if not channels:
        parser.error("--channels must contain at least one channel")
    unknown = [channel for channel in channels if channel not in CHANNELS]
    if unknown:
        parser.error(f"unknown channel(s): {', '.join(unknown)}")
    if len(set(channels)) != len(channels):
        parser.error("--channels contains duplicates")
    settings = RunSettings(
        output_dir=args.output_dir,
        channels=channels,
        size=args.size,
        tolerance=args.tolerance,
        reload_check=not args.no_reload_check,
        input_dir=args.material_dir,
    )
    summary, exit_code = run(settings)
    counts = summary["summary"]
    print(
        "material-graph-renderer: "
        f"{counts['rendered']} rendered, {counts['invalid']} invalid, "
        f"{counts['failed']} failed, {counts['error_count']} errors; "
        f"summary={args.output_dir / 'summary.json'}"
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

