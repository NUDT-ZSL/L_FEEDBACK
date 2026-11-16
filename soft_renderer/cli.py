from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .errors import RenderError, UsageError
from .imageio import AtomicOutputSet, pixel_result_hash, write_pfm, write_png
from .layers import add_layers_to_output_set, composite_layers_from_directory
from .renderer import RenderResult, render_scene
from .sample_scene import build_indoor_scene
from .scene import load_scene


def _json_bytes(value: Dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json_target(path: str, value: Dict[str, Any]) -> None:
    payload = _json_bytes(value)
    with open(path, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _resolve_output(parser: argparse.ArgumentParser, positional: Optional[str], option: Optional[str]) -> str:
    if positional is not None and option is not None:
        parser.error("specify the output path either positionally or with --output, not both")
    output = option if option is not None else positional
    if output is None:
        parser.error("an output PFM path is required")
    return output


def _resolve_render_paths(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace
) -> tuple:
    if arguments.scene_option is not None:
        scene = arguments.scene_option
        if arguments.output_option is not None:
            if arguments.scene is not None:
                parser.error("unexpected positional path when --scene and --output are used")
            output = arguments.output_option
        else:
            if arguments.output is not None:
                parser.error("too many positional paths with --scene")
            output = arguments.scene
    else:
        if arguments.scene is None:
            parser.error("a scene JSON path is required")
        scene = arguments.scene
        output = _resolve_output(parser, arguments.output, arguments.output_option)
    if output is None:
        parser.error("an output PFM path is required")
    return scene, output


def _render_report(
    result: RenderResult,
    scene_sha256: str,
    output_path: str,
    png_path: Optional[str],
    layers_dir: Optional[str],
) -> Dict[str, Any]:
    return {
        "format": "offline-soft-renderer-report",
        "version": 1,
        "renderer_version": __version__,
        "preset": result.preset.name,
        "parameters": {
            "shadow_resolution": result.preset.shadow_resolution,
            "pcf_samples": result.preset.pcf_samples,
            "supersample": result.preset.supersample,
            "block_size": result.block_size,
            "no_tiling": result.no_tiling,
            "material_id": result.material_id,
        },
        "scene_sha256": scene_sha256,
        "light_shadow_resolutions": result.light_shadow_resolutions,
        "image": {
            "width": result.width,
            "height": result.height,
            "render_width": result.render_width,
            "render_height": result.render_height,
        },
        "outputs": {
            "pfm": os.path.abspath(output_path),
            "png": None if png_path is None else os.path.abspath(png_path),
            "layers_dir": None if layers_dir is None else os.path.abspath(layers_dir),
        },
        "pixel_result_sha256": result.pixel_hash,
        "shadow_statistics": result.shadow_statistics,
        "timings_ms": result.timings_ms,
    }


def _command_render(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    scene_path, output_path = _resolve_render_paths(parser, arguments)
    if arguments.exposure <= 0.0:
        parser.error("--exposure must be positive")
    started = time.perf_counter()
    load_started = time.perf_counter()
    scene = load_scene(scene_path)
    load_ms = (time.perf_counter() - load_started) * 1000.0
    result = render_scene(
        scene,
        block_size=arguments.block_size,
        preset_name=arguments.preset,
        material_id=arguments.material,
        no_tiling=arguments.no_tiling,
    )
    result.timings_ms["load_scene"] = load_ms
    result.timings_ms["total_with_load"] = (time.perf_counter() - started) * 1000.0

    output_set = AtomicOutputSet()
    output_set.add(output_path, lambda path: write_pfm(path, result.frame))
    if arguments.png is not None:
        output_set.add(
            arguments.png,
            lambda path: write_png(path, result.frame, arguments.exposure),
        )
    if arguments.layers_dir is not None:
        add_layers_to_output_set(
            output_set,
            result,
            arguments.layers_dir,
            {identifier: material.name for identifier, material in scene.materials.items()},
        )
    if arguments.report is not None:
        report = _render_report(
            result,
            scene.source_sha256,
            output_path,
            arguments.png,
            arguments.layers_dir,
        )
        output_set.add(arguments.report, lambda path, data=report: _write_json_target(path, data))
    output_set.commit()

    if not arguments.quiet:
        print(
            "rendered %dx%d preset=%s block=%d hash=%s total=%.1fms"
            % (
                result.width,
                result.height,
                result.preset.name,
                result.block_size,
                result.pixel_hash,
                result.timings_ms["total"],
            )
        )
    return 0


def _composite_report(
    frame: np.ndarray,
    loaded: Any,
    timings: Dict[str, float],
    output_path: str,
    png_path: Optional[str],
) -> Dict[str, Any]:
    return {
        "format": "offline-soft-renderer-report",
        "version": 1,
        "operation": "composite",
        "renderer_version": __version__,
        "preset": loaded.preset,
        "parameters": {
            "supersample": loaded.supersample,
            "block_size": loaded.block_size,
            "no_tiling": loaded.no_tiling,
            "layer_count": len(loaded.layers),
        },
        "image": {
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "render_width": loaded.render_width,
            "render_height": loaded.render_height,
        },
        "outputs": {
            "pfm": os.path.abspath(output_path),
            "png": None if png_path is None else os.path.abspath(png_path),
        },
        "pixel_result_sha256": pixel_result_hash(frame),
        "timings_ms": timings,
    }


def _command_composite(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    output_path = _resolve_output(parser, arguments.output, arguments.output_option)
    if arguments.exposure <= 0.0:
        parser.error("--exposure must be positive")
    frame, loaded, timings = composite_layers_from_directory(arguments.layers_dir)
    output_set = AtomicOutputSet()
    output_set.add(output_path, lambda path: write_pfm(path, frame))
    if arguments.png is not None:
        output_set.add(
            arguments.png,
            lambda path: write_png(path, frame, arguments.exposure),
        )
    if arguments.report is not None:
        report = _composite_report(frame, loaded, timings, output_path, arguments.png)
        output_set.add(arguments.report, lambda path, data=report: _write_json_target(path, data))
    output_set.commit()
    if not arguments.quiet:
        print(
            "composited %dx%d layers=%d hash=%s total=%.1fms"
            % (
                frame.shape[1],
                frame.shape[0],
                len(loaded.layers),
                pixel_result_hash(frame),
                timings["total"],
            )
        )
    return 0


def _command_generate(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if arguments.width <= 0 or arguments.height <= 0:
        parser.error("width and height must be positive")
    scene = build_indoor_scene(arguments.width, arguments.height)
    payload = _json_bytes(scene)
    output_set = AtomicOutputSet()
    output_set.add(
        arguments.output,
        lambda path: _write_binary_target(path, payload),
    )
    output_set.commit()
    if not arguments.quiet:
        print(arguments.output)
    return 0


def _write_binary_target(path: str, payload: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soft-render",
        description="Deterministic CPU-only layered soft rasterizer.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render", help="render a scene to a linear HDR PFM")
    render.add_argument("scene", nargs="?", help="scene JSON path")
    render.add_argument("--scene", dest="scene_option", help="scene JSON path")
    render.add_argument("output", nargs="?", help="output PFM path")
    render.add_argument("-o", "--output", dest="output_option", help="output PFM path")
    render.add_argument("--block-size", type=int, default=4, help="screen tile size")
    render.add_argument(
        "--no-tiling",
        action="store_true",
        help="render the supersampled frame as one block",
    )
    render.add_argument(
        "--preset", choices=("low", "medium", "high"), default="medium"
    )
    render.add_argument("--png", help="tone-mapped PNG preview path")
    render.add_argument("--layers-dir", help="write one PFM and depth PFM per material")
    render.add_argument(
        "--material", type=int, default=None, metavar="ID", help="render only one material"
    )
    render.add_argument("--report", help="write a JSON timing and hash report")
    render.add_argument("--exposure", type=float, default=1.0)
    render.add_argument("--quiet", action="store_true")

    composite = subparsers.add_parser(
        "composite", help="merge material layers by depth into one HDR PFM"
    )
    composite.add_argument("output", nargs="?", help="output PFM path")
    composite.add_argument("-o", "--output", dest="output_option", help="output PFM path")
    composite.add_argument("--layers-dir", required=True)
    composite.add_argument("--png", help="tone-mapped PNG preview path")
    composite.add_argument("--report", help="write a JSON timing and hash report")
    composite.add_argument("--exposure", type=float, default=1.0)
    composite.add_argument("--quiet", action="store_true")

    generate = subparsers.add_parser("generate", help="write the deterministic indoor sample")
    generate.add_argument("output", help="destination scene JSON")
    generate.add_argument("--width", type=int, default=320)
    generate.add_argument("--height", type=int, default=240)
    generate.add_argument("--quiet", action="store_true")
    return parser


def _normalized_argv(argv: Sequence[str]) -> List[str]:
    values = list(argv)
    if not values:
        return values
    if values[0] in ("render", "composite", "generate", "-h", "--help", "--version"):
        return values
    return ["render"] + values


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments_list = sys.argv[1:] if argv is None else list(argv)
    parser = _build_parser()
    arguments = parser.parse_args(_normalized_argv(arguments_list))
    try:
        if arguments.command == "render":
            return _command_render(arguments, parser)
        if arguments.command == "composite":
            return _command_composite(arguments, parser)
        if arguments.command == "generate":
            return _command_generate(arguments, parser)
        parser.error("no command selected")
    except RenderError as error:
        print("error: %s" % error, file=sys.stderr)
        return 2 if isinstance(error, UsageError) else 1
    except (OSError, ValueError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 1
    except Exception as error:
        print(
            "error: internal failure (%s): %s" % (type(error).__name__, error),
            file=sys.stderr,
        )
        return 1
    return 0
