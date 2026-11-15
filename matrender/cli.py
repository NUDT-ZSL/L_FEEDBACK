"""Command-line interface for deterministic offline rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
from PIL import Image

from . import __version__
from .color import ACES_A, ACES_B, ACES_C, ACES_D, ACES_E, linear_to_srgb8
from .renderer import RenderOptions, RenderPlan, RenderResult, make_render_plan, render
from .scene import Scene, SceneValidationError, load_scene


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="matrender",
        description=(
            "Deterministic CPU renderer for metallic/roughness material-sphere "
            "calibration scenes."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser(
        "render", help="render a validated scene JSON"
    )
    render_parser.add_argument("scene", type=Path, help="scene JSON path")
    render_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("render_output"),
        help="output directory (default: render_output)",
    )
    render_parser.add_argument(
        "--samples",
        type=int,
        default=8,
        help="anti-aliasing samples per pixel, 1..64 (default: 8)",
    )
    render_parser.add_argument(
        "--threads",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="worker threads (clamped to logical core count)",
    )
    render_parser.add_argument(
        "--tile-size",
        type=int,
        default=64,
        help="square tile edge in pixels (default: 64)",
    )
    render_parser.add_argument(
        "--max-memory-mb",
        type=float,
        default=2048.0,
        help="preflight memory ceiling in MiB (default: 2048)",
    )
    render_parser.add_argument(
        "--ssao",
        choices=("auto", "on", "off"),
        default="auto",
        help="override the scene SSAO setting",
    )
    render_parser.add_argument("--width", type=int, help="override output width")
    render_parser.add_argument("--height", type=int, help="override output height")
    render_parser.add_argument(
        "--quiet", action="store_true", help="suppress the successful stdout summary"
    )

    validate_parser = subparsers.add_parser(
        "validate", help="validate a scene JSON without rendering"
    )
    validate_parser.add_argument("scene", type=Path)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _build_log(
    scene: Scene,
    result: RenderResult,
    output_dir: Path,
    paths: dict[str, Path],
) -> dict[str, Any]:
    plan = result.plan
    return {
        "schema_version": "1.0",
        "status": "ok",
        "scene": {
            "name": scene.name,
            "source": scene.source,
            "sphere_count": len(scene.spheres),
        },
        "render": {
            "requested_resolution": [
                plan.requested_width,
                plan.requested_height,
            ],
            "effective_resolution": [plan.width, plan.height],
            "requested_samples": plan.requested_samples,
            "effective_samples": plan.samples,
            "tile_size": plan.tile_size,
            "tile_order": "row-major",
            "deterministic_aa": "stratified-jitter-from-integer-hash",
        },
        "threading": {
            "requested_threads": plan.requested_threads,
            "effective_threads": plan.threads,
            "logical_cores": plan.logical_cores,
        },
        "resource": {
            "limit_bytes": plan.max_memory_bytes,
            "initial_estimated_bytes": plan.initial_estimated_bytes,
            "effective_estimated_bytes": plan.estimated_bytes,
            "degraded": plan.degraded,
            "degradation_events": list(plan.degradation_events),
            "estimate_model": {
                "per_sample_rgb_bytes": 24,
                "accumulation_rgb_bytes": 24,
                "gbuffer_bytes_per_pixel": 60,
                "display_buffer_bytes_per_pixel": 24,
            },
        },
        "warnings": list(plan.warnings),
        "ssao": {
            "enabled": plan.ssao_enabled,
            "radius": scene.ssao_radius,
            "intensity": scene.ssao_intensity,
            "screen_samples": scene.ssao_samples,
            "inputs": ["depth", "world_normal"],
        },
        "color": {
            "render_space": "linear Rec.709 RGB, float32",
            "display_transform": "Narkowicz ACES filmic approximation",
            "aces_coefficients": {
                "a": ACES_A,
                "b": ACES_B,
                "c": ACES_C,
                "d": ACES_D,
                "e": ACES_E,
            },
            "transfer_function": "sRGB IEC 61966-2-1",
            "quantization": "round(encoded * 255)",
        },
        "linear_hdr": {
            "path": paths["linear_hdr"].name,
            "dtype": "float32",
            "shape": list(result.linear_hdr.shape),
            "sha256": _sha256(paths["linear_hdr"]),
            "global": result.global_stats,
        },
        "regions": [
            {
                "index": sphere.index,
                "name": sphere.name,
                **stats,
            }
            for sphere, stats in zip(
                scene.spheres, result.region_stats, strict=True
            )
        ],
        "outputs": {
            "linear_hdr": str(paths["linear_hdr"]),
            "preview_png": str(paths["preview_png"]),
            "render_log": str(paths["render_log"]),
            "material_report": str(paths["material_report"]),
            "sphere_masks": str(paths["sphere_masks"]),
            "output_dir": str(output_dir),
        },
    }


def _write_outputs(
    scene: Scene,
    result: RenderResult,
    output_dir: Path,
) -> dict[str, Path]:
    paths = {
        "linear_hdr": output_dir / "linear_hdr.npy",
        "preview_png": output_dir / "preview_aces_srgb.png",
        "render_log": output_dir / "render_log.json",
        "material_report": output_dir / "material_report.json",
        "sphere_masks": output_dir / "sphere_masks.npz",
    }
    np.save(paths["linear_hdr"], result.linear_hdr, allow_pickle=False)
    np.savez_compressed(
        paths["sphere_masks"],
        **{
            f"sphere_{index:03d}": mask
            for index, mask in enumerate(result.sphere_masks)
        },
    )
    rgb8 = linear_to_srgb8(result.linear_hdr)
    Image.fromarray(rgb8, mode="RGB").save(
        paths["preview_png"], format="PNG", compress_level=9
    )

    report = {
        "schema_version": "1.0",
        "scene_name": scene.name,
        "source": scene.source,
        "linear_space": "Rec.709 RGB",
        "luminance_weights": {"r": 0.2126, "g": 0.7152, "b": 0.0722},
        "dominant_hue_method": "HSV hue of mean linear RGB, normalized by max channel",
        "regions": list(result.material_regions),
    }
    _write_json(paths["material_report"], report)
    _write_json(paths["render_log"], _build_log(scene, result, output_dir, paths))
    return paths


def _print_warnings(plan: RenderPlan) -> None:
    for warning in plan.warnings:
        print(
            f"{warning['code']}: {warning['message']}",
            file=sys.stderr,
            flush=True,
        )


def _run_render(args: argparse.Namespace) -> int:
    try:
        scene = load_scene(args.scene)
    except SceneValidationError as exc:
        print(f"E_INVALID_SCENE: {exc}", file=sys.stderr)
        return 0

    if args.samples < 1 or args.samples > 64:
        print("error: --samples must be in [1, 64]", file=sys.stderr)
        return 2
    if args.threads < 1:
        print("error: --threads must be at least 1", file=sys.stderr)
        return 2
    if args.tile_size < 1:
        print("error: --tile-size must be at least 1", file=sys.stderr)
        return 2
    if args.max_memory_mb <= 0.0:
        print("error: --max-memory-mb must be positive", file=sys.stderr)
        return 2

    ssao_override = None
    if args.ssao == "on":
        ssao_override = True
    elif args.ssao == "off":
        ssao_override = False

    options = RenderOptions(
        samples=args.samples,
        threads=args.threads,
        tile_size=args.tile_size,
        max_memory_bytes=max(1, int(args.max_memory_mb * 1024.0 * 1024.0)),
        ssao=ssao_override,
        width=args.width,
        height=args.height,
    )
    try:
        plan = make_render_plan(scene, options)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _print_warnings(plan)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = render(scene, plan)
    paths = _write_outputs(scene, result, args.output_dir)

    if not args.quiet:
        summary = {
            "status": "ok",
            "linear_hdr": str(paths["linear_hdr"]),
            "preview_png": str(paths["preview_png"]),
            "render_log": str(paths["render_log"]),
            "material_report": str(paths["material_report"]),
            "sphere_masks": str(paths["sphere_masks"]),
            "resolution": [plan.width, plan.height],
            "samples": plan.samples,
            "threads": plan.threads,
            "ssao": plan.ssao_enabled,
            "degraded": plan.degraded,
        }
        print(json.dumps(summary, sort_keys=True))
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    try:
        scene = load_scene(args.scene)
    except SceneValidationError as exc:
        print(f"E_INVALID_SCENE: {exc}", file=sys.stderr)
        return 0
    print(
        json.dumps(
            {
                "status": "valid",
                "scene_name": scene.name,
                "resolution": [scene.width, scene.height],
                "spheres": len(scene.spheres),
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] not in {
        "render",
        "validate",
        "-h",
        "--help",
        "--version",
    }:
        arguments.insert(0, "render")
    args = _parser().parse_args(arguments)
    if args.command == "render":
        return _run_render(args)
    if args.command == "validate":
        return _run_validate(args)
    return 2

