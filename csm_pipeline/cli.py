"""Configuration-driven command line entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from .camera_path import save_path
from .depth_archive import read_depth_archive
from .errors import ERR_UNKNOWN, PipelineError
from .metrics import array_checksum
from .pipeline import REPORT_FILENAME, run_pipeline
from .scene import generate_scene


def _write_error_report(output_dir: Optional[str], code: str, message: str) -> str:
    target_dir = output_dir if output_dir else os.getcwd()
    try:
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, REPORT_FILENAME)
        document = {"status": "error", "error_code": code, "error": message}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path
    except OSError:
        fallback = os.path.join(os.getcwd(), REPORT_FILENAME)
        document = {"status": "error", "error_code": code, "error": message}
        with open(fallback, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return fallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="csm-pipeline",
        description="Deterministic pure-software cascaded shadow-map pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the full offline pipeline")
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--camera-path", required=True)
    run.add_argument("--cascades", type=int, required=True)
    run.add_argument("--shadow-resolution", type=int, required=True)
    run.add_argument("--width", type=int, default=320)
    run.add_argument("--height", type=int, default=240)
    run.add_argument("--camera-near", type=float, default=0.1)
    run.add_argument("--camera-far", type=float, default=40.0)
    run.add_argument("--frame-time", type=float, default=1.0)
    run.add_argument("--pcf-kernel", type=int, default=3)
    run.add_argument("--blend-fraction", type=float, default=0.15)
    run.add_argument("--normal-bias-scale", type=float, default=0.02)
    run.add_argument("--depth-bias", type=float, default=2.0e-4)
    run.add_argument("--output-dir", required=True)

    make_path = sub.add_parser(
        "make-path", help="write the deterministic keyframe path for a seed"
    )
    make_path.add_argument("--seed", type=int, required=True)
    make_path.add_argument("--output", required=True)

    read = sub.add_parser(
        "read-archive", help="validate a depth archive and print its summary"
    )
    read.add_argument("--archive", required=True)
    read.add_argument("--output-dir", default=None)
    return parser


def _run_command(args: argparse.Namespace) -> dict:
    config = {
        "seed": args.seed,
        "camera_path": args.camera_path,
        "cascade_count": args.cascades,
        "shadow_resolution": args.shadow_resolution,
        "width": args.width,
        "height": args.height,
        "camera_near": args.camera_near,
        "camera_far": args.camera_far,
        "frame_time": args.frame_time,
        "pcf_kernel": args.pcf_kernel,
        "blend_fraction": args.blend_fraction,
        "normal_bias_scale": args.normal_bias_scale,
        "depth_bias": args.depth_bias,
    }
    return run_pipeline(config, args.output_dir)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = getattr(args, "output_dir", None)
    try:
        if args.command == "run":
            report = _run_command(args)
            print(
                json.dumps(
                    {
                        "status": report["status"],
                        "report": os.path.join(
                            os.path.abspath(args.output_dir), REPORT_FILENAME
                        ),
                    },
                    indent=2,
                )
            )
        elif args.command == "make-path":
            scene = generate_scene(int(args.seed))
            save_path(args.output, scene.keyframes)
            print(json.dumps({"status": "ok", "path": args.output}, indent=2))
        elif args.command == "read-archive":
            stack = read_depth_archive(args.archive)
            summary = {
                "status": "ok",
                "archive": args.archive,
                "cascade_count": int(stack.shape[0]),
                "width": int(stack.shape[2]),
                "height": int(stack.shape[1]),
                "pixel_dtype": "float32",
                "checksum_sha256": array_checksum(stack),
            }
            print(json.dumps(summary, indent=2))
        else:  # pragma: no cover - argparse enforces this
            parser.error("unknown command")
        return 0
    except PipelineError as exc:
        report_path = _write_error_report(output_dir, exc.code, exc.message)
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": exc.code,
                    "error": exc.message,
                    "report": os.path.abspath(report_path),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return exc.exit_code
    except (ValueError, TypeError, OSError) as exc:
        message = f"unexpected input/output failure: {exc}"
        report_path = _write_error_report(output_dir, ERR_UNKNOWN, message)
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": ERR_UNKNOWN,
                    "error": message,
                    "report": os.path.abspath(report_path),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

