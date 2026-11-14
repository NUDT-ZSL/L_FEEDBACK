from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .errors import BakeError
from .pipeline import BakeConfig, run_bake, write_error_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bake_tool",
        description="Deterministic offline PBR texture baker with a NumPy preview renderer.",
    )
    parser.add_argument("--out", default="out", help="Output directory (default: out).")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--max-resolution", type=int, default=512)
    parser.add_argument("--dilation-radius", type=int, default=2)
    parser.add_argument("--miss-threshold", type=float, default=0.02)
    parser.add_argument("--normal-dot-threshold", type=float, default=0.98)
    parser.add_argument(
        "--normal-convention",
        choices=("opengl", "directx"),
        default="opengl",
    )
    parser.add_argument(
        "--scenario",
        choices=("default", "uv_out_of_range", "missing_high", "miss_rate"),
        default="default",
    )
    parser.add_argument(
        "--no-backface-culling",
        action="store_true",
        help="Disable geometric backface culling during texture-space rasterization.",
    )
    parser.add_argument(
        "--application-traceback",
        action="store_true",
        help="Include a traceback in the error report for unexpected internal failures.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = BakeConfig(
        resolution=args.resolution,
        max_resolution=args.max_resolution,
        dilation_radius=args.dilation_radius,
        miss_threshold=args.miss_threshold,
        normal_convention=args.normal_convention,
        normal_dot_threshold=args.normal_dot_threshold,
        scenario=args.scenario,
        backface_culling=not args.no_backface_culling,
        output_dir=args.out,
    )
    try:
        result = run_bake(config, args.out)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "output_dir": result.output_dir,
                    "resolution": config.resolution,
                    "normal_convention": config.normal_convention,
                    "actual_elapsed_ms": result.actual_elapsed_ms,
                    "deterministic_elapsed_ms": result.report["timing"]["total_elapsed_ms"],
                    "miss_ratio": result.report["raycast"]["fallback_ratio"],
                    "normal_mismatch_ratio": result.report["normal_consistency"]["mismatch_ratio"],
                },
                sort_keys=True,
            )
        )
        return 0
    except BakeError as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "business_code": error.business_code,
                    "exit_code": error.exit_code,
                    "report": str(Path(args.out) / "error_report.json"),
                },
                sort_keys=True,
            )
        )
        return int(error.exit_code)
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        context = {"exception_type": type(exc).__name__, "message": str(exc)}
        if args.application_traceback:
            context["traceback"] = traceback.format_exc()
        error = BakeError(
            "E_INTERNAL_ERROR",
            "Unexpected internal failure; no textures were committed.",
            context,
            ["Re-run with --application-traceback to collect diagnostics."],
            exit_code=1,
        )
        try:
            write_error_report(args.out, error, config.snapshot())
        except Exception:
            pass
        print(
            json.dumps(
                {
                    "status": "error",
                    "business_code": error.business_code,
                    "exit_code": error.exit_code,
                    "report": str(Path(args.out) / "error_report.json"),
                },
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

