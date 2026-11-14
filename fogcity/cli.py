"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from .errors import (
    EXIT_INTERNAL,
    EXIT_OUTPUT_WRITE,
    INTERNAL,
    OUTPUT_WRITE,
    BusinessError,
    scene_field_error,
)
from .output import write_frame_files, write_manifest
from .reporting import make_manifest
from .runner import iter_frames
from .sample import build_sample_document
from .scene import load_scene, validate_scene_object, scene_public_metadata
from .util import pretty_json_bytes
from .verify import verify_existing_output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fogcity",
        description="Deterministic offline OpenGL 4.1 fog-city keyframe preview renderer",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render", help="render a scene JSON to PNG files")
    render.add_argument("--scene", required=True, help="path to scene JSON")
    render.add_argument("--out", required=True, help="output directory")
    render.add_argument(
        "--parallel-frames",
        type=int,
        default=1,
        help="maximum simultaneously active frames/contexts (default: 1)",
    )
    render.add_argument(
        "--backend",
        choices=("auto", "default", "egl", "glx", "wgl"),
        default="auto",
        help="standalone OpenGL context backend (default: auto)",
    )
    render.add_argument(
        "--validate-only",
        "--verify",
        dest="validate_only",
        action="store_true",
        help="render in memory and verify existing outputs without writing files",
    )

    sample = subparsers.add_parser(
        "generate-sample", help="write the deterministic sample scene JSON"
    )
    sample.add_argument("--path", required=True, help="destination scene JSON path")
    sample.add_argument("--width", type=int, default=480)
    sample.add_argument("--height", type=int, default=270)
    sample.add_argument("--fps", type=float, default=1.0)
    sample.add_argument("--start-time", type=float, default=0.0)
    sample.add_argument("--end-time", type=float, default=6.0)
    sample.add_argument("--steps", type=int, default=32)
    sample.add_argument("--shadow-resolution", type=int, default=768)
    return parser


def _error_json(error: BusinessError) -> str:
    return json.dumps(error.as_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _success_json(**values: Any) -> str:
    return json.dumps(
        {"ok": True, **values},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _write_sample(args: argparse.Namespace) -> int:
    document = build_sample_document(
        width=args.width,
        height=args.height,
        fps=args.fps,
        start_time=args.start_time,
        end_time=args.end_time,
        fog_steps=args.steps,
        shadow_resolution=args.shadow_resolution,
    )
    # Validate before touching the destination.
    validate_scene_object(document, args.path, "")
    path = Path(args.path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pretty_json_bytes(document))
    except OSError as exc:
        raise BusinessError(
            OUTPUT_WRITE,
            f"could not write sample scene: {exc}",
            EXIT_OUTPUT_WRITE,
            {"path": str(path)},
        ) from exc
    print(_success_json(mode="generate-sample", path=str(path.resolve())))
    return 0


def _render_scene(args: argparse.Namespace) -> int:
    if args.parallel_frames < 1 or args.parallel_frames > 32:
        raise scene_field_error("parallel_frames", "must be in [1, 32]")
    scene = load_scene(args.scene)
    if args.validate_only:
        result = verify_existing_output(
            scene,
            args.out,
            backend=args.backend,
            parallel_frames=args.parallel_frames,
        )
        print(_success_json(mode="validate-only", **result))
        return 0

    frame_stats: list[dict[str, Any]] = [{} for _ in scene.frame_times]
    gl_info: dict[str, str] | None = None
    completed = 0
    iterator = iter_frames(
        scene,
        backend=args.backend,
        parallel_frames=args.parallel_frames,
    )
    try:
        for artifact in iterator:
            gl_info = artifact.gl_info
            frame_stats[artifact.frame_index] = artifact.frame_stats
            write_frame_files(
                args.out,
                artifact.frame_index,
                artifact.png_bytes,
                artifact.frame_stats,
            )
            completed += 1
            print(f"frame {artifact.frame_index + 1}/{len(scene.frame_times)} written", flush=True)
    finally:
        iterator.close()

    if gl_info is None or completed != len(scene.frame_times):
        raise BusinessError(
            INTERNAL,
            "render loop ended before all frames were produced",
            EXIT_INTERNAL,
            {"expected_frames": len(scene.frame_times), "completed_frames": completed},
        )
    manifest = make_manifest(scene, frame_stats, gl_info)
    stats_path = write_manifest(args.out, manifest)
    print(
        _success_json(
            mode="render",
            output_dir=str(Path(args.out).resolve()),
            frames=len(scene.frame_times),
            stats_path=str(stats_path.resolve()),
            scene=scene_public_metadata(scene),
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate-sample":
            return _write_sample(args)
        if args.command == "render":
            return _render_scene(args)
    except BusinessError as error:
        print(_error_json(error), file=sys.stderr)
        return error.exit_code
    except KeyboardInterrupt:
        print(
            _error_json(
                BusinessError(
                    "E_INTERRUPTED",
                    "render interrupted by user",
                    130,
                    {},
                )
            ),
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        if os.environ.get("FOGCITY_DEBUG") == "1":
            traceback.print_exc(file=sys.stderr)
        print(
            _error_json(
                BusinessError(
                    INTERNAL,
                    f"unexpected internal error: {type(exc).__name__}: {exc}",
                    EXIT_INTERNAL,
                    {"exception_type": type(exc).__name__},
                )
            ),
            file=sys.stderr,
        )
        return EXIT_INTERNAL
    return 0

