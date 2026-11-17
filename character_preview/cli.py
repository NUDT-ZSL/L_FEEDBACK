from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from .config import RenderConfig, config_from_dict, load_config
from .errors import AppError, ConfigError
from .model import load_model
from .sequence import render_sequence


def _configure_logging() -> None:
    pass


def _warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def _apply_overrides(config: RenderConfig, overrides: dict[str, Any]) -> tuple[RenderConfig, list[str]]:
    raw = dataclasses.asdict(config)
    raw.update({key: value for key, value in overrides.items() if value is not None})
    return config_from_dict(raw)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="character-preview",
        description="Offscreen OpenGL character preview renderer",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render", help="render an animated PNG sequence")
    render.add_argument("--model", required=True, help="self-describing character JSON")
    render.add_argument("--config", help="render configuration JSON")
    render.add_argument("--output", help="output directory (overrides config)")
    render.add_argument("--frames", type=int, help="frame count override")
    render.add_argument("--time-step", type=float, help="seconds between frames")

    validate = subparsers.add_parser("validate", help="validate model and config JSON")
    validate.add_argument("--model", required=True, help="self-describing character JSON")
    validate.add_argument("--config", help="render configuration JSON")
    return parser


def _run_render(args: argparse.Namespace) -> int:
    config, warnings = load_config(args.config)
    config, override_warnings = _apply_overrides(
        config, {"frame_count": args.frames, "time_step": args.time_step}
    )
    for warning in [*warnings, *override_warnings]:
        _warn(warning)
    model = load_model(args.model)
    output = args.output or config.output_dir
    manifest = render_sequence(model, config, output)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "frames": len(manifest["frames"]),
                "output": str(Path(output)),
            }
        )
    )
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    config, warnings = load_config(args.config)
    for warning in warnings:
        _warn(warning)
    model = load_model(args.model)
    print(
        json.dumps(
            {
                "valid": True,
                "nodes": len(model.nodes),
                "materials": len(model.materials),
                "point_lights": len(model.point_lights),
                "frame_count": config.frame_count,
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "render":
            return _run_render(args)
        if args.command == "validate":
            return _run_validate(args)
        raise ConfigError("E_ARGS", f"unknown command {args.command}")
    except AppError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[E_INTERRUPTED] render interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[E_UNEXPECTED] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

