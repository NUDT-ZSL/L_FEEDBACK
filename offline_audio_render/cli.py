"""Command-line entry point for the deterministic offline audio renderer."""

from __future__ import annotations

import argparse
import json
import sys

from .errors import SceneError
from .renderer import Renderer
from .scene import load_scene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="offline-audio-render",
        description=(
            "Render a YAML game-audio scene into deterministic multi-track "
            "float WAV files plus a JSON report."
        ),
    )
    parser.add_argument("scene", help="path to the YAML scene description")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="render_out",
        help="directory for WAV outputs and report.json (default: render_out)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="override the streaming block size in frames (default from scene)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scene = load_scene(args.scene)
        if args.block_size is not None:
            if args.block_size <= 0:
                raise SceneError(
                    "E1004", "--block-size must be a positive integer"
                )
            scene.block_size = args.block_size
        result = Renderer(scene, args.output_dir).render()
    except SceneError as error:
        json.dump(error.to_dict(), sys.stderr, ensure_ascii=False)
        sys.stderr.write("\n")
        return 2
    summary = {
        "status": "ok",
        "outputs": result.output_files,
        "report": result.report_path,
        "warnings": len(result.report["warnings"]),
        "skipped_events": result.report["skipped_events"],
    }
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
