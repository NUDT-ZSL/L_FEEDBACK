"""Command-line interface with structured stdout/stderr JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

from .errors import InputValidationError, PipelineError
from .pipeline import run_pipeline


class StructuredArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise InputValidationError(message, field="cli")


def build_parser() -> argparse.ArgumentParser:
    parser = StructuredArgumentParser(
        prog="terrain_pbr",
        description=(
            "Deterministically rasterize layered terrain vectors into RGBA albedo, "
            "height, and tangent-space normal PNGs."
        ),
    )
    parser.add_argument("--input", "-i", required=True, help="Input scene JSON.")
    parser.add_argument(
        "--output-dir", "-o", required=True, help="Directory for generated outputs."
    )
    parser.add_argument("--width", required=True, help="Canvas width in pixels.")
    parser.add_argument("--height", required=True, help="Canvas height in pixels.")
    parser.add_argument(
        "--lod",
        required=True,
        help="Geometry LOD: high, medium, or low.",
    )
    parser.add_argument(
        "--grid-size",
        required=True,
        help="Rasterization tile size in pixels.",
    )
    return parser


def _json_line(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        result = run_pipeline(
            input_path=args.input,
            output_directory=args.output_dir,
            width=args.width,
            height=args.height,
            lod=args.lod,
            grid_size=args.grid_size,
        )
    except PipelineError as exc:
        print(_json_line(exc.to_payload()), file=sys.stderr, flush=True)
        return exc.exit_code
    except Exception as exc:  # Keep CLI failures machine-readable.
        if os.environ.get("TERRAIN_PBR_DEBUG"):
            raise
        payload = {
            "ok": False,
            "error": {
                "code": "E_INTERNAL_ERROR",
                "message": "Unexpected pipeline failure.",
                "details": {"reason": type(exc).__name__},
            },
        }
        print(_json_line(payload), file=sys.stderr, flush=True)
        return 1

    print(_json_line(result.summary), flush=True)
    return 0
