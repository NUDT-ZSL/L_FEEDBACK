#!/usr/bin/env python3
"""Offline deterministic PBR material baker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from material_baker import __version__
from material_baker.errors import MaterialError, SimulatedInterrupt
from material_baker.pipeline import run_pipeline
from material_baker.util import atomic_write_json
from material_baker.validation import error_report


def _resolution(text: str) -> tuple[int, int]:
    normalized = text.lower().replace("×", "x")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution values must be integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution values must be positive")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bake deterministic PBR material textures and validation previews."
    )
    parser.add_argument("--scene", type=Path, required=True, help="scene JSON path")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--resolution",
        "--preview-resolution",
        type=_resolution,
        help="preview resolution WIDTHxHEIGHT (overrides scene JSON)",
    )
    parser.add_argument(
        "--bit-depth",
        "--output-bit-depth",
        type=int,
        choices=(8, 16),
        help="PNG bit depth (overrides scene JSON)",
    )
    parser.add_argument(
        "--name-pattern",
        "--naming-pattern",
        help="PNG filename pattern containing one {kind} placeholder",
    )
    parser.add_argument(
        "--chunk-rows",
        "--texture-block-rows",
        type=int,
        help="bake/checkpoint row-block height (overrides scene JSON)",
    )
    parser.add_argument(
        "--seed",
        "--sample-seed",
        type=int,
        help="validation sampling seed (overrides scene JSON)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="discard checkpoints for the output directory before starting",
    )
    parser.add_argument(
        "--fail-after-chunks",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--version", action="version", version=f"bake_material {__version__}"
    )
    return parser


def _write_failure(out_dir: Path, error: MaterialError) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_dir / "validation_report.json", error_report(error))
    except OSError:
        pass
    print(f"{error.code} {error.message}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cv2.setNumThreads(1)
    cv2.setUseOptimized(False)
    if args.chunk_rows is not None and args.chunk_rows <= 0:
        error = MaterialError("E011", "--chunk-rows must be positive")
        _write_failure(args.out, error)
        return error.exit_code
    if args.seed is not None and args.seed < 0:
        error = MaterialError("E011", "--seed must be non-negative")
        _write_failure(args.out, error)
        return error.exit_code
    if args.resolution is not None and (
        args.resolution[0] > 32768 or args.resolution[1] > 32768
    ):
        error = MaterialError("E011", "preview resolution is unreasonably large")
        _write_failure(args.out, error)
        return error.exit_code
    try:
        return run_pipeline(
            args.scene,
            args.out,
            chunk_rows_override=args.chunk_rows,
            preview_resolution_override=args.resolution,
            bit_depth_override=args.bit_depth,
            name_pattern_override=args.name_pattern,
            seed_override=args.seed,
            force=args.force,
            fail_after_chunks=args.fail_after_chunks,
        )
    except SimulatedInterrupt as exc:
        print(f"[interrupted] {exc}; checkpoints retained", file=sys.stderr, flush=True)
        return 70
    except MaterialError as exc:
        _write_failure(args.out, exc)
        return exc.exit_code
    except KeyboardInterrupt:
        print("[interrupted] keyboard interrupt; checkpoints retained", file=sys.stderr)
        return 130
    except OSError as exc:
        error = MaterialError("E008", f"I/O failure: {exc}")
        _write_failure(args.out, error)
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
