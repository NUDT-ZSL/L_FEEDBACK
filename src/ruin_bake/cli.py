"""Command-line interface for full, incremental, and summary operations."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .bake import BakeOutput, BakeParams, bake_full, bake_incremental
from .canonical import canonical_json
from .config import ConfigError, RunConfig, load_run_config, read_json_object
from .layout import LayoutParams, ProbeLayout, build_probe_layout
from .scene import (
    Scene,
    SceneGeneratorConfig,
    apply_scene_patch,
    generate_scene,
)
from .snapshot import (
    FORMAT_VERSION,
    Snapshot,
    SnapshotError,
    load_snapshot,
    write_snapshot_atomic,
)


def _add_bake_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="JSON run configuration")
    parser.add_argument("--patch", type=Path, help="JSON scene patch (overrides config patch)")
    parser.add_argument("--seed", type=int, help="procedural scene seed")
    parser.add_argument("--bounds-min", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--bounds-max", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--extra-boxes", type=int, help="number of generated non-floor boxes")
    parser.add_argument("--point-light-count", type=int, help="number of point lights (0..4)")
    parser.add_argument("--emissive-box-count", type=int, help="number of emissive non-floor boxes")
    parser.add_argument("--spacing", type=float, help="probe grid spacing")
    parser.add_argument("--cull-distance", type=float, help="surface exclusion distance")
    parser.add_argument("--samples", type=int, help="hemisphere samples per probe")
    parser.add_argument("--ray-epsilon", "--integration-tolerance", dest="ray_epsilon", type=float, help="ray/self-intersection and integration tolerance")
    parser.add_argument("--workers", type=int, help="worker processes; 0 selects up to four")
    parser.add_argument("--chunk-size", type=int, help="probe count per submitted work chunk")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ruin-bake",
        description="Deterministic offline one-bounce irradiance probe baker",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    full = subparsers.add_parser("full", aliases=["bake-full"], help="bake all probes")
    _add_bake_options(full)
    full.add_argument("--out", type=Path, required=True, help="output snapshot path")

    incremental = subparsers.add_parser(
        "incremental",
        aliases=["bake-incremental"],
        help="bake only probes affected by scene changes",
    )
    _add_bake_options(incremental)
    incremental.add_argument("--base", type=Path, required=True, help="previous complete snapshot")
    incremental.add_argument("--out", type=Path, required=True, help="output snapshot path")

    summary = subparsers.add_parser("summary", aliases=["inspect"], help="inspect a snapshot")
    summary.add_argument("snapshot_pos", nargs="?", type=Path, help="snapshot path")
    summary.add_argument("--snapshot", type=Path, help="snapshot path")
    return parser


def _resolve_generator(args: argparse.Namespace, base_scene: Scene | None) -> SceneGeneratorConfig:
    if args.config is None and base_scene is not None:
        try:
            generator = SceneGeneratorConfig.from_dict(base_scene.generation)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"base snapshot generator config is invalid: {exc}") from exc
    else:
        generator = load_run_config(args.config).generator

    overrides: dict[str, Any] = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.bounds_min is not None:
        overrides["bounds_min"] = tuple(args.bounds_min)
    if args.bounds_max is not None:
        overrides["bounds_max"] = tuple(args.bounds_max)
    if args.extra_boxes is not None:
        overrides["extra_boxes"] = args.extra_boxes
    if args.point_light_count is not None:
        overrides["point_light_count"] = args.point_light_count
    if args.emissive_box_count is not None:
        overrides["emissive_box_count"] = args.emissive_box_count
    if overrides:
        try:
            generator = replace(generator, **overrides)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid scene generator override: {exc}") from exc
    return generator


def _resolve_layout(args: argparse.Namespace, base: Snapshot | None) -> LayoutParams:
    if args.config is None and base is not None:
        layout = base.layout_params
    else:
        layout = load_run_config(args.config).layout
    overrides: dict[str, Any] = {}
    if args.spacing is not None:
        overrides["spacing"] = args.spacing
    if args.cull_distance is not None:
        overrides["cull_distance"] = args.cull_distance
    if overrides:
        try:
            layout = replace(layout, **overrides)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid layout override: {exc}") from exc
    return layout


def _resolve_bake(args: argparse.Namespace, base: Snapshot | None) -> BakeParams:
    if args.config is None and base is not None:
        evaluation = base.evaluation
        try:
            bake = BakeParams(
                samples=int(evaluation["samples"]),
                ray_epsilon=float(evaluation["ray_epsilon"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"base snapshot evaluation metadata is invalid: {exc}") from exc
    else:
        bake = load_run_config(args.config).bake
    overrides: dict[str, Any] = {}
    if args.samples is not None:
        overrides["samples"] = args.samples
    if args.ray_epsilon is not None:
        overrides["ray_epsilon"] = args.ray_epsilon
    if args.workers is not None:
        overrides["workers"] = max(1, min(4, os.cpu_count() or 1)) if args.workers == 0 else args.workers
    if args.chunk_size is not None:
        overrides["chunk_size"] = args.chunk_size
    if overrides:
        try:
            bake = replace(bake, **overrides)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid bake override: {exc}") from exc
    return bake


def _resolve_patch(args: argparse.Namespace) -> dict[str, Any]:
    if args.patch is not None:
        return read_json_object(args.patch, "scene patch")
    return dict(load_run_config(args.config).patch)


def _resolve_scene(
    args: argparse.Namespace,
    base: Snapshot | None,
) -> tuple[Scene, LayoutParams, BakeParams]:
    patch = _resolve_patch(args)
    generator_overridden = any(
        value is not None
        for value in (
            args.seed,
            args.bounds_min,
            args.bounds_max,
            args.extra_boxes,
            args.point_light_count,
            args.emissive_box_count,
        )
    )
    if args.config is None and base is not None and not generator_overridden:
        # Without a fresh generator config, incremental patches are applied to
        # the complete scene recovered from the validated base snapshot.  This
        # preserves earlier patches rather than silently regenerating the
        # pristine procedural scene.
        scene = base.scene
    else:
        generator = _resolve_generator(args, None if base is None else base.scene)
        scene = generate_scene(generator)
    scene = apply_scene_patch(scene, patch)
    return scene, _resolve_layout(args, base), _resolve_bake(args, base)


def _next_revision_for_output(path: Path) -> int:
    try:
        return load_snapshot(path).revision + 1
    except SnapshotError:
        return 1


def _bake_report(
    *,
    command: str,
    output: BakeOutput,
    snapshot: Snapshot,
    total_elapsed: float,
    parent: Snapshot | None,
) -> dict[str, Any]:
    return {
        "ok": True,
        "command": command,
        "probe_count": len(output.layout.probes),
        "probe_total": len(output.layout.probes),
        "generated_probe_count": output.layout.generated_count,
        "culled_probe_count": output.layout.culled_count,
        "culled_count": output.layout.culled_count,
        "affected_probe_count": output.affected_probe_count,
        "affected_count": output.affected_probe_count,
        "reused_probe_count": output.reused_probe_count,
        "new_probe_count": output.new_probe_count,
        "reuse_allowed": output.reuse_allowed,
        "changed_geometry_count": len(output.changed_geometry_ids),
        "changed_light_count": len(output.changed_light_ids),
        "samples": output.evaluation["samples"],
        "ray_epsilon": output.evaluation["ray_epsilon"],
        "integration_tolerance": output.evaluation["integration_tolerance"],
        "bounce_depth": output.evaluation["bounce_depth"],
        "integrator": output.evaluation["integrator"],
        "sampling": output.evaluation["sampling"],
        "workers_used": output.workers_used,
        "chunk_count": output.chunk_count,
        "bake_elapsed_seconds": output.elapsed_seconds,
        "elapsed_seconds": total_elapsed,
        "snapshot_version": FORMAT_VERSION,
        "snapshot_format_version": FORMAT_VERSION,
        "snapshot_revision": snapshot.revision,
        "snapshot_path": str(snapshot.path),
        "snapshot_checksum": snapshot.checksum,
        "scene_hash": output.scene.scene_hash,
        "layout_hash": output.layout.layout_hash,
        "layout_params": output.layout.params.to_dict(),
        "parent_snapshot_revision": None if parent is None else parent.revision,
    }


def _parent_metadata(snapshot: Snapshot | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "snapshot_revision": snapshot.revision,
        "scene_hash": snapshot.scene_hash,
        "snapshot_checksum": snapshot.checksum,
    }


def _run_full(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.perf_counter()
    scene, layout_params, bake_params = _resolve_scene(args, None)
    layout = build_probe_layout(scene, layout_params)
    output = bake_full(scene, layout, bake_params)
    revision = _next_revision_for_output(args.out)
    snapshot = write_snapshot_atomic(
        args.out,
        scene=scene,
        layout=layout,
        irradiances=output.irradiances,
        evaluation=output.evaluation,
        revision=revision,
    )
    report = _bake_report(
        command="full",
        output=output,
        snapshot=snapshot,
        total_elapsed=time.perf_counter() - started,
        parent=None,
    )
    return report, 0


def _run_incremental(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.perf_counter()
    base = load_snapshot(args.base)
    scene, layout_params, bake_params = _resolve_scene(args, base)
    layout = build_probe_layout(scene, layout_params)
    output = bake_incremental(scene, layout, bake_params, base)
    revision = base.revision + 1
    snapshot = write_snapshot_atomic(
        args.out,
        scene=scene,
        layout=layout,
        irradiances=output.irradiances,
        evaluation=output.evaluation,
        revision=revision,
        parent=_parent_metadata(base),
    )
    report = _bake_report(
        command="incremental",
        output=output,
        snapshot=snapshot,
        total_elapsed=time.perf_counter() - started,
        parent=base,
    )
    return report, 0


def _run_summary(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.snapshot is not None and args.snapshot_pos is not None:
        raise ValueError("provide the snapshot path either positionally or with --snapshot, not both")
    snapshot_path = args.snapshot if args.snapshot is not None else args.snapshot_pos
    if snapshot_path is None:
        raise ValueError("summary requires a snapshot path")
    started = time.perf_counter()
    snapshot = load_snapshot(snapshot_path)
    report = snapshot.summary()
    report["affected_probe_count"] = 0
    report["affected_count"] = 0
    report["probe_total"] = report["probe_count"]
    report["culled_count"] = report["culled_probe_count"]
    report["snapshot_format_version"] = report["snapshot_version"]
    report["elapsed_seconds"] = time.perf_counter() - started
    return report, 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = str(args.command)
    if command in {"bake-full"}:
        command = "full"
    elif command in {"bake-incremental"}:
        command = "incremental"
    elif command == "inspect":
        command = "summary"
    try:
        if command == "full":
            report, exit_code = _run_full(args)
        elif command == "incremental":
            report, exit_code = _run_incremental(args)
        elif command == "summary":
            report, exit_code = _run_summary(args)
        else:
            raise ValueError(f"unknown command {command!r}")
    except (ConfigError, SnapshotError, ValueError, TypeError, OSError, RuntimeError) as exc:
        print(
            canonical_json(
                {
                    "ok": False,
                    "command": command,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            ),
            file=sys.stderr,
        )
        return 1

    print(canonical_json(report))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

