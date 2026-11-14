from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from .audio import (
    analyze_scene_audio,
    build_object_bus,
)
from .constants import MEMORY_LIMIT_MB
from .container import (
    ContainerReader,
    build_container_comments,
    write_container,
)
from .errors import PIPELINE_UNEXPECTED, PipelineError
from .metrics import PeakMemoryTracker
from .renderer import render_container
from .scene import load_scene
from .timeline import block_count


def _add_common_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stats", metavar="PATH", help="write runtime JSON statistics here")
    parser.add_argument("--chunk-samples", type=int, default=1_048_576)


def _resolve_pair(
    parser: argparse.ArgumentParser,
    first: str | None,
    second: str | None,
    first_option: str | None,
    second_option: str | None,
    first_name: str,
    second_name: str,
) -> tuple[str, str]:
    first_value = first_option or first
    second_value = second_option or second
    if not first_value or not second_value:
        parser.error(f"{first_name} and {second_name} are required")
    return first_value, second_value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="immersive-audio",
        description="Deterministic object-based panoramic audio pack/render pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack = subparsers.add_parser("pack", help="pack a scene and clips into a container")
    pack.add_argument("scene", nargs="?", help="scene JSON path")
    pack.add_argument("output", nargs="?", help="container output path")
    pack.add_argument("--scene", dest="scene_option", help="scene JSON path")
    pack.add_argument("--output", dest="output_option", help="container output path")
    _add_common_output(pack)

    render = subparsers.add_parser("render", help="render a container to 8-channel WAV")
    render.add_argument("input", nargs="?", help="container input path")
    render.add_argument("output", nargs="?", help="WAV output path")
    render.add_argument("--input", dest="input_option", help="container input path")
    render.add_argument("--output", dest="output_option", help="WAV output path")
    render.add_argument(
        "--min-distance",
        type=float,
        default=0.1,
        help="minimum distance in meters used by attenuation",
    )
    render.add_argument(
        "--no-verify-checksum",
        action="store_true",
        help="skip the container SHA-256 payload verification",
    )
    _add_common_output(render)
    return parser


def _emit_stats(stats: dict[str, object], stats_path: str | None) -> None:
    encoded = json.dumps(stats, ensure_ascii=False, sort_keys=True, indent=2)
    if stats_path:
        path = Path(stats_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def _pack(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, object]:
    scene_path, output_path = _resolve_pair(
        parser,
        args.scene,
        args.output,
        args.scene_option,
        args.output_option,
        "scene",
        "output",
    )
    if args.chunk_samples <= 0:
        parser.error("--chunk-samples must be positive")
    tracker = PeakMemoryTracker()
    stages: dict[str, float] = {}
    started = time.perf_counter()

    stage_start = time.perf_counter()
    scene = load_scene(scene_path)
    stages["load_scene"] = time.perf_counter() - stage_start
    tracker.sample()

    stage_start = time.perf_counter()
    analysis = analyze_scene_audio(scene)
    stages["analyze_audio"] = time.perf_counter() - stage_start
    tracker.sample()

    stage_start = time.perf_counter()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.buses-", dir=output.parent
    ) as temporary:
        directory = Path(temporary)
        buses = []
        for obj in scene.objects:
            bus_path = directory / f"object-{obj.index:02d}.f32"
            buses.append(
                build_object_bus(
                    scene,
                    obj,
                    analysis,
                    bus_path,
                    chunk_samples=args.chunk_samples,
                )
            )
            tracker.sample()
        stages["build_object_buses"] = time.perf_counter() - stage_start
        comments = build_container_comments(scene, analysis)
        stage_start = time.perf_counter()
        write_stats = write_container(
            output_path,
            scene,
            analysis,
            tuple(buses),
            comments,
            chunk_samples=args.chunk_samples,
            progress=tracker.sample,
        )
        stages["write_container"] = time.perf_counter() - stage_start
    stages["write_metadata"] = write_stats.metadata_seconds
    stages["write_comments"] = write_stats.comment_seconds
    stages["write_audio"] = write_stats.audio_seconds
    stages["total"] = time.perf_counter() - started
    tracker.sample()

    clip_count = sum(len(obj.clips) for obj in scene.objects)
    return {
        "command": "pack",
        "counts": {
            "clips": clip_count,
            "channels": scene.object_count,
            "frame_length": scene.frame_length,
            "metadata_frames": block_count(analysis.total_frames, scene.frame_length),
            "objects": scene.object_count,
            "sample_rate": scene.sample_rate,
            "samples": analysis.total_frames,
        },
        "memory_limit_mb": MEMORY_LIMIT_MB,
        "object_bus_ranges": [
            {
                "index": bus.object_index,
                "object_id": bus.object_id,
                "valid_end_frame": bus.valid_end_frame,
                "valid_start_frame": bus.valid_start_frame,
            }
            for bus in buses
        ],
        "output_bytes": output.stat().st_size,
        "output_path": str(output),
        "payload_sha256": write_stats.sha256,
        "peak_memory_mb": round(tracker.peak_mb, 3),
        "stages_seconds": {key: round(value, 6) for key, value in stages.items()},
        "within_memory_limit": tracker.peak_mb <= MEMORY_LIMIT_MB,
    }


def _render(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, object]:
    input_path, output_path = _resolve_pair(
        parser,
        args.input,
        args.output,
        args.input_option,
        args.output_option,
        "input",
        "output",
    )
    if args.chunk_samples <= 0:
        parser.error("--chunk-samples must be positive")
    tracker = PeakMemoryTracker()
    stages: dict[str, float] = {}
    started = time.perf_counter()

    stage_start = time.perf_counter()
    reader = ContainerReader(input_path)
    stages["open_and_validate_container"] = time.perf_counter() - stage_start
    tracker.sample()

    render_stats = render_container(
        reader,
        output_path,
        verify_checksum=not args.no_verify_checksum,
        minimum_distance=args.min_distance,
        progress=tracker.sample,
    )
    stages["verify_checksum"] = render_stats.checksum_seconds
    stages["render_audio"] = render_stats.render_seconds
    stages["write_wav"] = render_stats.write_seconds
    stages["total"] = time.perf_counter() - started
    tracker.sample()

    output = Path(output_path)
    return {
        "command": "render",
        "counts": {
            "container_channels": reader.header.channel_count,
            "frame_length": reader.header.frame_length,
            "metadata_frames": reader.header.frame_count,
            "objects": reader.header.object_count,
            "output_channels": 8,
            "sample_rate": reader.header.sample_rate,
            "samples": reader.header.sample_count,
        },
        "memory_limit_mb": MEMORY_LIMIT_MB,
        "output_bytes": output.stat().st_size,
        "output_path": str(output),
        "peak_memory_mb": round(tracker.peak_mb, 3),
        "stages_seconds": {key: round(value, 6) for key, value in stages.items()},
        "within_memory_limit": tracker.peak_mb <= MEMORY_LIMIT_MB,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "pack":
            stats = _pack(args, parser)
        elif args.command == "render":
            stats = _render(args, parser)
        else:
            parser.error("unknown command")
            return 2
        _emit_stats(stats, args.stats)
        return 0
    except PipelineError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_status
    except Exception as exc:
        print(f"{PIPELINE_UNEXPECTED}: unexpected pipeline failure ({type(exc).__name__})", file=sys.stderr)
        return 1
