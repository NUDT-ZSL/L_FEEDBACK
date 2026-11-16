from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .gl_context import create_headless_context
from .pipeline import run_validation
from .samples import generate_sample


def _validate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="material-validator",
        description="Validate GLSL materials and compare deferred/forward rendering deterministically.",
    )
    parser.add_argument("--manifest", required=True, help="Path to the JSON material manifest")
    parser.add_argument("--out", required=True, help="Directory for reports and float buffers")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel shader compile workers")
    parser.add_argument("--render-parallel", type=int, default=1, help="Parallel scene render workers")
    parser.add_argument("--abs-tol", type=float, default=None, help="Override absolute comparison threshold")
    parser.add_argument("--rel-tol", type=float, default=None, help="Override relative comparison threshold")
    parser.add_argument(
        "--unused-uniform-policy",
        choices=("manifest", "warn", "error", "ignore"),
        default="manifest",
        help="Policy for manifest uniforms optimized out of the linked program",
    )
    parser.add_argument("--no-render", action="store_true", help="Compile and validate without rendering")
    parser.add_argument("--write-raw", action="store_true", help="Also write little-endian float32 raw buffers")
    parser.add_argument("--fail-on-warning", action="store_true", help="Return a non-zero exit code on warnings")
    parser.add_argument("--quiet", action="store_true", help="Only print failures")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def _check_gl() -> int:
    try:
        context, info = create_headless_context()
    except Exception as exc:
        print(f"OpenGL context check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    try:
        print(json.dumps(info.to_dict(), indent=2, sort_keys=True))
    finally:
        context.release()
    return 0


def _generate(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="material-validator generate-sample")
    parser.add_argument("--out", required=True, help="Target sample directory")
    parser.add_argument("--kind", choices=("valid", "invalid", "blend"), default="valid")
    args = parser.parse_args(argv)
    try:
        manifest = generate_sample(args.out, args.kind)
    except Exception as exc:
        print(f"Sample generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(str(manifest))
    return 0


def _write_failure_report(out: str, exc: Exception) -> None:
    try:
        output = Path(out).resolve()
        output.mkdir(parents=True, exist_ok=True)
        report = {
            "tool": "material-validator",
            "tool_version": __version__,
            "status": "failed",
            "diagnostics": [
                {
                    "severity": "error",
                    "code": "unhandled_setup_error",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            ],
        }
        with (output / "report.json").open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        with (output / "summary.txt").open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                f"Material Validator {__version__}\n"
                f"Status: failed\n"
                f"Error: {type(exc).__name__}: {exc}\n"
            )
    except Exception:
        pass


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "check-gl":
        return _check_gl()
    if arguments and arguments[0] == "generate-sample":
        return _generate(arguments[1:])
    parser = _validate_parser()
    args = parser.parse_args(arguments)
    if args.parallel < 1 or args.render_parallel < 1:
        parser.error("--parallel and --render-parallel must be at least 1")
    if args.abs_tol is not None and args.abs_tol < 0.0:
        parser.error("--abs-tol must be non-negative")
    if args.rel_tol is not None and args.rel_tol < 0.0:
        parser.error("--rel-tol must be non-negative")
    policy = None if args.unused_uniform_policy == "manifest" else args.unused_uniform_policy
    try:
        run = run_validation(
            manifest_path=args.manifest,
            output_dir=args.out,
            parallel_compile=args.parallel,
            parallel_render=args.render_parallel,
            absolute_threshold=args.abs_tol,
            relative_threshold=args.rel_tol,
            unused_uniform_policy=policy,
            render=not args.no_render,
            write_raw=args.write_raw,
            fail_on_warning=args.fail_on_warning,
        )
    except Exception as exc:
        _write_failure_report(args.out, exc)
        print(f"Validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    summary = run.report["summary"]
    if not args.quiet or run.exit_code != 0:
        print(
            f"{run.status.upper()}: {summary['materials_compiled']}/{summary['materials_total']} materials compiled, "
            f"{summary['comparison']['pixels_over_threshold']} pixels over threshold"
        )
        print(f"Report: {run.report_path}")
        print(f"Summary: {run.summary_path}")
    if run.exit_code != 0:
        for diagnostic in run.report["diagnostics"]:
            if diagnostic["severity"] == "error":
                location = diagnostic.get("path") or diagnostic.get("material") or ""
                print(f"ERROR {diagnostic['code']}: {diagnostic['message']} {location}".rstrip(), file=sys.stderr)
    return run.exit_code
