"""Command line entry points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bake import bake_scene
from .errors import BusinessError, EXIT_INTERNAL


def _print_error(error: BusinessError) -> int:
    print(f"skin-bake: {error.code}: {error.message}", file=sys.stderr)
    return error.exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skin-bake")
    parser.add_argument("--version", action="store_true", help="print tool version")
    sub = parser.add_subparsers(dest="command")

    bake = sub.add_parser("bake", help="bake a scene into an RG16F curve atlas")
    bake.add_argument("scene", type=Path)
    bake.add_argument("-o", "--output", type=Path, required=True)
    bake.add_argument("--quiet", action="store_true")

    view = sub.add_parser("view", help="open the OpenGL 3.3 viewer")
    view.add_argument("scene", type=Path)
    view.add_argument("bake_dir", type=Path)
    view.add_argument("--frame", type=int, default=0)
    view.add_argument("--headless-check", type=Path, default=None)
    view.add_argument("--width", type=int, default=900)
    view.add_argument("--height", type=int, default=640)

    check = sub.add_parser("check", help="validate baked assets and render a software preview")
    check.add_argument("scene", type=Path)
    check.add_argument("bake_dir", type=Path)
    check.add_argument("--frame", type=int, default=0)
    check.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from . import __version__

        print(__version__)
        return 0
    if args.command is None:
        parser.print_help(sys.stderr)
        return 2
    try:
        if args.command == "bake":
            result = bake_scene(args.scene, args.output)
            if not args.quiet:
                print(json.dumps(result.report, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "view":
            from .viewer import run_viewer

            return run_viewer(
                args.scene,
                args.bake_dir,
                start_frame=args.frame,
                headless_output=args.headless_check,
                width=args.width,
                height=args.height,
            )
        if args.command == "check":
            from .validate import run_check

            return run_check(args.scene, args.bake_dir, frame=args.frame, output=args.output)
    except BusinessError as exc:
        return _print_error(exc)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"skin-bake: E_INTERNAL: {exc}", file=sys.stderr)
        return EXIT_INTERNAL
    return EXIT_INTERNAL

