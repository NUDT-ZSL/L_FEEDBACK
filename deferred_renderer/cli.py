from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from .errors import RenderError
from .renderer import DeferredRenderer, open_gl_context
from .scene import (
    build_assets,
    default_scene_dict,
    load_scene_file,
    parse_scene,
    write_default_scene,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deferred_renderer",
        description="Deterministic tiled deferred renderer with G-Buffer, tile culling, DOF and PNG output.",
    )
    parser.add_argument("--scene", type=Path, help="scene JSON path (default: built-in scene)")
    parser.add_argument(
        "--write-default-scene",
        type=Path,
        metavar="PATH",
        help="write the deterministic built-in scene JSON and exit",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/render"),
        help="output directory for frame_0000.png and stats.json",
    )
    parser.add_argument("--stats", type=Path, help="override statistics JSON path")
    parser.add_argument(
        "--mode",
        choices=("headless", "window"),
        default="headless",
        help="headless FBO or GLFW window",
    )
    parser.add_argument("--width", type=int, help="override render.width")
    parser.add_argument("--height", type=int, help="override render.height")
    parser.add_argument("--tile-size", type=int, help="override render.tile_size")
    parser.add_argument(
        "--window-hidden",
        action="store_true",
        help="create a hidden GLFW window (useful for automated parity tests)",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=0.0,
        help="keep an interactive window alive for N seconds after the first frame",
    )
    parser.add_argument(
        "--no-dof",
        action="store_true",
        help="disable depth-of-field for this invocation",
    )
    parser.add_argument(
        "--no-reference-check",
        action="store_true",
        help="skip the CPU full-screen BRDF reference comparison",
    )
    return parser


def _apply_overrides(scene_data: dict[str, Any], args: argparse.Namespace) -> None:
    render = scene_data.setdefault("render", {})
    if args.width is not None:
        render["width"] = args.width
    if args.height is not None:
        render["height"] = args.height
    if args.tile_size is not None:
        render["tile_size"] = args.tile_size
    if args.no_dof:
        dof = render.setdefault("dof", {})
        dof["enabled"] = False
    if args.no_reference_check:
        reference = render.setdefault("reference_check", {})
        reference["enabled"] = False


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_error(
    error: RenderError,
    stats_path: Path,
    *,
    mode: str,
) -> None:
    document = error.as_dict()
    document["mode"] = mode
    try:
        _write_json(stats_path, document)
    except OSError as write_error:
        print(
            f"ERROR [{error.code}] unable to write error statistics: {write_error}",
            file=sys.stderr,
        )
    print(f"ERROR {error}", file=sys.stderr)
    if error.details:
        print(
            "DETAILS " + json.dumps(error.details, sort_keys=True, ensure_ascii=False),
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    stats_path = args.stats if args.stats is not None else args.out / "stats.json"

    if args.write_default_scene is not None:
        try:
            path = write_default_scene(args.write_default_scene)
        except OSError as exc:
            error = RenderError(
                "E_SCENE_IO",
                "unable to write the default scene",
                {"path": str(args.write_default_scene), "reason": str(exc)},
            )
            _write_error(error, stats_path, mode=args.mode)
            return error.exit_code
        print(str(path))
        return 0

    try:
        if args.scene is not None:
            raw, scene = load_scene_file(args.scene)
        else:
            raw = default_scene_dict()
            scene = parse_scene(raw)
        _apply_overrides(raw, args)
        if args.scene is not None or any(
            value is not None
            for value in (args.width, args.height, args.tile_size)
        ) or args.no_dof or args.no_reference_check:
            scene = parse_scene(raw)

        assets = build_assets(scene)
        args.out.mkdir(parents=True, exist_ok=True)
        with open_gl_context(
            args.mode,
            scene.render.width,
            scene.render.height,
            hidden=args.window_hidden,
        ) as context_bundle:
            renderer = DeferredRenderer(scene, assets, context_bundle.context)
            renderer.attach_window(context_bundle)
            result = renderer.render_frame(
                mode=args.mode,
                gl_info=context_bundle.gl_info,
            )

            frame_path = args.out / "frame_0000.png"
            try:
                Image.fromarray(result.image_rgba, mode="RGBA").save(
                    frame_path,
                    format="PNG",
                    optimize=False,
                    compress_level=6,
                )
                png_bytes = frame_path.read_bytes()
            except OSError as exc:
                raise RenderError(
                    "E_OUTPUT_IO",
                    "unable to write the PNG frame",
                    {"path": str(frame_path), "reason": str(exc)},
                    exit_code=5,
                ) from exc

            result.stats["output"]["png_file"] = str(frame_path)
            result.stats["output"]["png_sha256"] = hashlib.sha256(png_bytes).hexdigest()
            result.stats["output"]["png_size_bytes"] = len(png_bytes)
            result.stats["scene"]["source"] = (
                str(args.scene) if args.scene is not None else "<builtin>"
            )
            if context_bundle.window is not None:
                result.stats["window_backend"] = context_bundle.window.backend.name
            result.stats["scene"]["render_config"] = {
                "width": scene.render.width,
                "height": scene.render.height,
                "tile_size": scene.render.tile_size,
                "background": list(scene.render.background),
                "ambient_strength": scene.render.ambient_strength,
                "g_buffer_attachments": list(scene.render.attachments),
            }
            _write_json(stats_path, result.stats)
            if args.mode == "window":
                renderer.run_window_loop(args.window_seconds)

        print(
            "Rendered "
            f"{scene.render.width}x{scene.render.height} "
            f"mode={args.mode} "
            f"pixel_sha256={result.stats['output']['pixel_sha256']}"
        )
        print(f"PNG {frame_path}")
        print(f"STATS {stats_path}")
        return 0
    except RenderError as error:
        _write_error(error, stats_path, mode=args.mode)
        return error.exit_code
    except Exception as exc:
        print(traceback.format_exc(), file=sys.stderr)
        error = RenderError(
            "E_INTERNAL",
            "unexpected internal renderer failure",
            {"exception": type(exc).__name__, "message": str(exc)},
            exit_code=1,
        )
        _write_error(error, stats_path, mode=args.mode)
        return error.exit_code

