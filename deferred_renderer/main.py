"""Command-line entry point for the teaching deferred renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gbuffer import ATTACHMENT_SPECS
from .gl_context import OffscreenContext
from .renderer import DeferredRenderer, RenderError
from .scene import build_scene, write_scene


REBUILD_AFTER_FRAME = 10
ATTACHMENT_NAMES = [spec["name"] for spec in ATTACHMENT_SPECS]


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_error_log(path: str, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def parse_force_formats(values: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                f"--force-format expects ATTACHMENT=FORMAT, got '{value}'"
            )
        attachment, token = value.split("=", 1)
        attachment = attachment.strip()
        if attachment not in ATTACHMENT_NAMES:
            raise argparse.ArgumentTypeError(
                f"Unknown attachment '{attachment}'. "
                f"Valid: {', '.join(ATTACHMENT_NAMES)}"
            )
        parsed[attachment] = token.strip().upper()
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offscreen MRT deferred renderer (PyOpenGL + hidden GLFW window)."
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Deterministic RNG seed (default: 42).")
    parser.add_argument("--frames", type=int, default=21,
                        help="Number of frames to render (default: 21).")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: out/seed_<seed>_<mode>).")
    parser.add_argument("--fast-approx", action="store_true",
                        help="Emulate half-precision BRDF intermediates.")
    parser.add_argument("--force-format", action="append", default=[],
                        metavar="ATTACHMENT=FORMAT",
                        help=("Force an attachment internal format, e.g. "
                              "'world_position=UNSUPPORTED'. Repeatable."))
    parser.add_argument("--no-rebuild", action="store_true",
                        help="Disable the frame-10 G-Buffer rebuild (control run).")
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        force_formats = parse_force_formats(args.force_format)
    except argparse.ArgumentTypeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    mode = "fast_approx" if args.fast_approx else "float32"
    output_dir = args.out or os.path.join("out", f"seed_{args.seed}_{mode}")
    os.makedirs(output_dir, exist_ok=True)
    error_log_path = os.path.join(output_dir, "errors.jsonl")

    scene = build_scene(args.seed)
    scene_path = os.path.join(output_dir, "scene.json")
    write_scene(scene, scene_path)

    manifest: dict[str, Any] = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "frame_size": {"width": args.width, "height": args.height},
        "precision_mode": mode,
        "rebuild_enabled": not args.no_rebuild,
        "rebuild_after_frame": None if args.no_rebuild else REBUILD_AFTER_FRAME,
        "requested_formats": {},
        "gpu": {},
        "scene_file": "scene.json",
        "scene_sha256": file_sha256(scene_path),
        "frames": [],
        "events": [],
        "warnings": [],
        "errors": [],
    }

    def fail(code: str, message: str, details: dict[str, Any]) -> int:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "code": code,
            "message": message,
            "details": details,
        }
        manifest["status"] = "failed"
        manifest["errors"].append(record)
        append_error_log(error_log_path, record)
        write_json(os.path.join(output_dir, "manifest.json"), manifest)
        print(f"[ERROR] {code}: {message}", file=sys.stderr)
        return 1

    try:
        with OffscreenContext(args.width, args.height) as ctx:
            assert ctx.info is not None
            manifest["gpu"] = {
                "vendor": ctx.info.vendor,
                "renderer": ctx.info.renderer,
                "opengl_version": ctx.info.version,
                "glsl_version": ctx.info.glsl_version,
                "glfw_version": ctx.info.glfw_version,
            }

            renderer = DeferredRenderer(
                scene=scene,
                width=args.width,
                height=args.height,
                fast_approx=args.fast_approx,
                output_dir=output_dir,
                force_formats=force_formats,
            )
            manifest["requested_formats"] = dict(renderer.requested_formats)
            manifest["active_formats"] = dict(renderer.gbuffer.active_formats)

            try:
                for frame_index in range(args.frames):
                    result = renderer.render_frame(frame_index)
                    manifest["frames"].append({
                        "frame": frame_index,
                        "png": result.png_path,
                        "png_sha256": result.png_sha256,
                        "gbuffer_write_sha256": result.gbuffer_write_hash,
                        "gbuffer_verify_sha256": result.gbuffer_verify_hash,
                    })

                    if (
                        not args.no_rebuild
                        and frame_index == REBUILD_AFTER_FRAME
                        and frame_index + 1 < args.frames
                    ):
                        renderer.rebuild_gbuffer(frame_index + 1)
            except RenderError as render_error:
                renderer.close()
                return fail(
                    render_error.code, str(render_error), render_error.details
                )

            manifest["warnings"] = list(renderer.warnings)
            manifest["events"] = list(renderer.events)
            for warning in renderer.warnings:
                append_error_log(error_log_path, warning)

            renderer.close()
    except RenderError as render_error:
        return fail(render_error.code, str(render_error), render_error.details)
    except Exception as exc:  # noqa: BLE001 - top-level safety net, must not hang
        return fail(
            "RENDER_FATAL",
            f"{type(exc).__name__}: {exc}",
            {"exception": type(exc).__name__},
        )

    write_json(os.path.join(output_dir, "manifest.json"), manifest)
    print(
        f"[OK] mode={mode} seed={args.seed} frames={args.frames} "
        f"warnings={len(manifest['warnings'])} out={output_dir}"
    )
    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
