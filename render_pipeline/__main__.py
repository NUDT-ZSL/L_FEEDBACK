"""Command-line entry point for the offline tiled deferred renderer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .errors import PipelineError, SnapshotCorruptError
from .pipeline import PipelineOptions, RenderPipeline


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m render_pipeline",
        description="Render a scene with a tiled deferred pipeline and persistent tile snapshots.",
    )
    parser.add_argument("--scene", required=True, help="scene JSON path")
    parser.add_argument("--output", required=True, help="output binary PPM path")
    parser.add_argument("--report", help="diagnostic report path (default: <output stem>.report.json)")
    parser.add_argument("--snapshot-dir", help="snapshot directory (default: <output stem>.snapshots)")
    parser.add_argument("--no-snapshot", action="store_true", help="disable snapshot writes and reads")
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=True,
        help="reuse compatible snapshots (the default)",
    )
    parser.add_argument("--no-incremental", action="store_true", help="ignore existing snapshots")
    parser.add_argument("--quality", choices=("high", "fast"), help="shadow quality preset")
    parser.add_argument("--tile-width", type=_positive_int)
    parser.add_argument("--tile-height", type=_positive_int)
    parser.add_argument("--workers", type=_positive_int, default=4)
    parser.add_argument("--pcf-kernel", type=_positive_int)
    parser.add_argument(
        "--shadow-resolution",
        "--shadow-map-resolution",
        dest="shadow_resolution",
        type=_positive_int,
    )
    parser.add_argument("--directional-resolution", type=_positive_int)
    parser.add_argument("--point-resolution", type=_positive_int)
    parser.add_argument("--memory-limit-bytes", type=_positive_int)
    parser.add_argument("--memory-limit-mb", type=_positive_int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.with_name(output_path.stem + ".report.json")
    snapshot_dir = None
    if not args.no_snapshot:
        snapshot_dir = (
            Path(args.snapshot_dir)
            if args.snapshot_dir
            else output_path.with_name(output_path.stem + ".snapshots")
        )
    options = PipelineOptions(
        quality=args.quality,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        workers=args.workers,
        snapshot_dir=snapshot_dir,
        incremental=not args.no_incremental,
        report_path=report_path,
        memory_limit_bytes=args.memory_limit_bytes,
        memory_limit_mb=args.memory_limit_mb,
        pcf_kernel=args.pcf_kernel,
        shadow_resolution=args.shadow_resolution,
        directional_resolution=args.directional_resolution,
        point_resolution=args.point_resolution,
    )
    try:
        pipeline = RenderPipeline(args.scene, options)
        result = pipeline.render_to(output_path, report_path)
    except PipelineError as error:
        print(json.dumps({"status": "error", **error.as_dict()}, sort_keys=True), file=sys.stderr)
        return 2
    summary = {
        "status": result.report["status"],
        "output": str(output_path),
        "report": str(report_path),
        "snapshot_dir": None if snapshot_dir is None else str(snapshot_dir),
        "output_sha256": result.report["output_sha256"],
        "tiles": result.report["summary"],
        "business_codes": result.report["business_codes"],
        "failed_tiles": [
            {"id": tile["id"], "code": tile.get("error", {}).get("code")}
            for tile in result.report["failed_tiles"]
        ],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
