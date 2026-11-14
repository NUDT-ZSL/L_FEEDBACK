from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .aperture import KernelSpec, build_aperture_kernel
from .config import load_config
from .errors import PipelineError
from .pipeline import run_pipeline, verify_slice_consistency
from .sample import generate_sample


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dof-pipeline",
        description="Deterministic offline depth-of-field rendering pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="generate a deterministic sample sequence and config"
    )
    generate.add_argument("--output", required=True, help="sample input directory")
    generate.add_argument("--width", type=int, default=240)
    generate.add_argument("--height", type=int, default=135)
    generate.add_argument("--frames", type=int, default=8)
    generate.add_argument("--fps", type=float, default=12.0)
    generate.add_argument("--seed", type=int, default=12345)

    render = subparsers.add_parser("render", help="render a complete DOF sequence")
    render.add_argument("--config", required=True, help="pipeline JSON config")
    render.add_argument("--input", required=True, help="input sequence directory")
    render.add_argument("--output", required=True, help="output directory")
    render.add_argument("--slices", type=int, default=None, help="override depth slices")
    render.add_argument(
        "--gather-rounds", type=int, default=None, help="override gather rounds"
    )
    render.add_argument("--quiet", action="store_true")

    verify = subparsers.add_parser(
        "verify-slices", help="compare full scatter with configured depth slices"
    )
    verify.add_argument("--config", required=True)
    verify.add_argument("--input", required=True)
    verify.add_argument("--json-output", default=None)

    kernel = subparsers.add_parser(
        "export-kernel", help="export the normalized aperture kernel matrix"
    )
    kernel.add_argument("--output", required=True, help=".npy or .json output")
    kernel.add_argument("--blades", type=int, default=0, help="0=circle, >0 polygon")
    kernel.add_argument("--radius", type=float, required=True, help="radius in pixels")
    kernel.add_argument("--rotation-deg", type=float, default=0.0)
    kernel.add_argument("--supersample", type=int, default=8)

    return parser


def _cmd_generate(args: argparse.Namespace) -> int:
    manifest = generate_sample(
        args.output,
        width=args.width,
        height=args.height,
        frame_count=args.frames,
        fps=args.fps,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    progress = None if args.quiet else (lambda message: print(message, file=sys.stderr))
    summary = run_pipeline(
        config,
        args.input,
        args.output,
        depth_slices=args.slices,
        gather_rounds=args.gather_rounds,
        progress=progress,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "report": str(summary.report_path),
                "total_frames": summary.report["total_frames"],
                "warnings": len(summary.report["warnings"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = verify_slice_consistency(config, args.input)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0 if result["status"] == "passed" else 1


def _cmd_export_kernel(args: argparse.Namespace) -> int:
    if not math.isfinite(args.radius) or args.radius < 0:
        raise PipelineError("kernel radius must be finite and >= 0")
    kernel = build_aperture_kernel(
        KernelSpec(
            blades=args.blades,
            radius_px=args.radius,
            rotation_deg=args.rotation_deg,
            supersample=args.supersample,
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = kernel.metadata()
    if output.suffix.lower() == ".npy":
        np.save(output, kernel.weights, allow_pickle=False)
    elif output.suffix.lower() == ".json":
        payload: dict[str, Any] = dict(metadata)
        payload["weights"] = kernel.weights.tolist()
        output.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    else:
        raise PipelineError("kernel output must end in .npy or .json")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            return _cmd_generate(args)
        if args.command == "render":
            return _cmd_render(args)
        if args.command == "verify-slices":
            return _cmd_verify(args)
        if args.command == "export-kernel":
            return _cmd_export_kernel(args)
        parser.error(f"unknown command {args.command}")
        return 2
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
