"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Sequence

from . import __version__
from .errors import ValidationError
from .perf import PerformanceMonitor
from .runner import RenderValidator, RunConfig, finalize


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATERIALS = PROJECT_ROOT / "assets" / "materials.json"


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValidationError("E_ARG_001", 2, message, {"argument_error": message})


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="python -m render_check",
        description=(
            "Headless OpenGL 4.1 core ModernGL material batch validator. "
            "Renders Color/Normal/Depth/ObjectID RGBA32F channels."
        ),
    )
    parser.add_argument("--instances", type=int, default=3, help="instances to render (1..64)")
    parser.add_argument("--seed", type=int, default=1337, help="deterministic batch seed")
    parser.add_argument("--out", default="render_output", help="log and channel output directory")
    parser.add_argument(
        "--materials",
        default=str(DEFAULT_MATERIALS),
        help="material description JSON (default: bundled assets/materials.json)",
    )
    parser.add_argument("--width", type=int, default=256, help="channel width in pixels")
    parser.add_argument("--height", type=int, default=256, help="channel height in pixels")
    parser.add_argument(
        "--backend",
        choices=("auto", "wgl", "egl", "glx", "cgl"),
        default="auto",
        help="standalone context backend (auto selects WGL on Windows)",
    )
    parser.add_argument(
        "--fault-inject",
        choices=("none", "state_residue", "diff"),
        default="none",
        help="diagnostic-only runtime fault injection",
    )
    parser.add_argument(
        "--memory-limit-mb",
        type=float,
        default=512.0,
        help="process peak-memory limit (default 512)",
    )
    parser.add_argument(
        "--time-limit-sec",
        type=float,
        default=30.0,
        help="total elapsed-time limit (default 30)",
    )
    parser.add_argument("--debug", action="store_true", help="print tracebacks for internal errors")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.instances <= 64:
        raise ValidationError(
            "E_ARG_001",
            2,
            f"--instances must be in [1, 64], got {args.instances}",
            {"instances": args.instances},
        )
    if not 16 <= args.width <= 2048 or not 16 <= args.height <= 2048:
        raise ValidationError(
            "E_ARG_001",
            2,
            f"--width/--height must be in [16, 2048], got {args.width}x{args.height}",
            {"width": args.width, "height": args.height},
        )
    if not args.memory_limit_mb > 0:
        raise ValidationError(
            "E_ARG_001",
            2,
            "--memory-limit-mb must be > 0",
            {"memory_limit_mb": args.memory_limit_mb},
        )
    if not args.time_limit_sec > 0:
        raise ValidationError(
            "E_ARG_001",
            2,
            "--time-limit-sec must be > 0",
            {"time_limit_sec": args.time_limit_sec},
        )


def main(argv: Sequence[str] | None = None) -> int:
    started_at = time.perf_counter()
    parser = build_parser()
    args: argparse.Namespace | None = None
    tracker: PerformanceMonitor | None = None
    validator: RenderValidator | None = None
    try:
        args = parser.parse_args(argv)
        _validate_args(args)
        config = RunConfig(
            materials_path=Path(args.materials),
            output_dir=Path(args.out),
            instances=args.instances,
            seed=args.seed,
            width=args.width,
            height=args.height,
            backend=args.backend,
            fault_inject=args.fault_inject,
            memory_limit_mb=args.memory_limit_mb,
            time_limit_sec=args.time_limit_sec,
        )
        tracker = PerformanceMonitor(
            memory_limit_mb=args.memory_limit_mb,
            time_limit_sec=args.time_limit_sec,
            start=started_at,
        )
        validator = RenderValidator(config, tracker)
        state = validator.run()
        finalize(state, None)
        return 0
    except ValidationError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        if validator is not None and tracker is not None:
            try:
                finalize(validator.state, exc)
            except ValidationError as finalize_error:
                print(
                    f"{finalize_error.code}: {finalize_error.message}",
                    file=sys.stderr,
                )
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - intentionally converted to a stable code
        if args is not None and args.debug:
            traceback.print_exc()
        print(f"E_INTERNAL_001: {type(exc).__name__}: {exc}", file=sys.stderr)
        internal = ValidationError(
            "E_INTERNAL_001",
            1,
            f"{type(exc).__name__}: {exc}",
            {"exception_type": type(exc).__name__},
        )
        if validator is not None and tracker is not None:
            try:
                finalize(validator.state, internal)
            except ValidationError:
                pass
        return 1

