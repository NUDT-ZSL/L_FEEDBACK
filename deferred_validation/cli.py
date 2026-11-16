"""Command-line interface for rendering and validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .errors import AppError, ConfigError, RenderError
from .gl_context import EGLContext
from .renderer import DeferredRenderer
from .report import write_outputs
from .scene import load_scene
from .verify import compare_reports, load_reference_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deferred-validation",
        description="Offline OpenGL 4.5 deferred-render consistency validator.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="write a self-contained example scene")
    generate.add_argument("-o", "--output", default="examples/default_scene.json")

    render = subparsers.add_parser("render", help="render a scene and write PNG channels plus report.json")
    render.add_argument("--scene", required=True, help="input scene JSON")
    render.add_argument("-o", "--output", required=True, help="output directory")

    verify = subparsers.add_parser(
        "verify", help="re-render and compare against a reference report"
    )
    verify.add_argument("--scene", required=True, help="current scene JSON")
    verify.add_argument("--report", required=True, help="reference report.json")
    verify.add_argument("-o", "--output", required=True, help="current render output directory")
    verify.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="normalized sRGB per-pixel tolerance (default: 0.01)",
    )
    verify.add_argument(
        "--json-out",
        help="optional path for the verification summary JSON",
    )
    return parser


def _run_render(scene_path: str, output: str) -> tuple[dict, Path]:
    scene = load_scene(scene_path)
    output_dir = Path(output)
    with EGLContext() as context:
        assert context.info is not None
        renderer = DeferredRenderer(context.info)
        try:
            result = renderer.render(scene)
        finally:
            renderer.close()
    report = write_outputs(scene, result, output_dir)
    return report, output_dir


def _generate(path: str) -> int:
    from .default_scene import build_scene

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_scene(), indent=2), encoding="utf-8")
    print(output.resolve())
    return 0


def _render_command(args: argparse.Namespace) -> int:
    report, output_dir = _run_render(args.scene, args.output)
    summary = {
        "status": report["validation"]["status"],
        "output": str(output_dir.resolve()),
        "report": str((output_dir / "report.json").resolve()),
        "artifacts": len(report["artifacts"]),
        "coverage_fraction": report["coverage"]["pixel_fraction"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["validation"]["status"] == "pass" else 4


def _verify_command(args: argparse.Namespace) -> int:
    if args.threshold < 0.0 or args.threshold > 1.0:
        raise ConfigError("--threshold must be in [0, 1]")
    reference, reference_dir = load_reference_report(args.report)
    current, current_dir = _run_render(args.scene, args.output)
    summary = compare_reports(
        reference,
        reference_dir,
        current,
        current_dir,
        threshold=args.threshold,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    if summary["status"] != "pass":
        print("verification failed: substantive differences detected", file=sys.stderr)
        for reason in summary["substantive_reasons"]:
            print(f"- {json.dumps(reason, sort_keys=True)}", file=sys.stderr)
        return 3
    if summary["classification"] == "floating_point_noise":
        print(
            "verification passed: image hashes changed only within floating-point noise tolerance",
            file=sys.stderr,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "generate":
            return _generate(args.output)
        if args.command == "render":
            return _render_command(args)
        if args.command == "verify":
            return _verify_command(args)
        parser.error(f"unknown command: {args.command}")
        return 2
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except RenderError as exc:
        print(f"render error: {exc}", file=sys.stderr)
        return 4
    except AppError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
