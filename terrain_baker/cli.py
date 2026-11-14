"""Command line entry point.

Returns integer exit codes compatible with Windows process semantics:
0 success, 2 bad arguments, 3 baking failure, 4 I/O failure, 6 when the
software render backend is unavailable (core artifacts still produced).
"""

from __future__ import annotations

import argparse
import json
import sys

from .baker import DEFAULT_MEMORY_LIMIT, DEFAULT_TILE, bake
from .errors import BakerError, EXIT_PREVIEW_UNAVAILABLE
from .heightfield import MAX_RESOLUTION, MIN_RESOLUTION, HeightfieldSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="terrain-baker",
        description="Bake a deterministic heightfield into normal map and LOD meshes.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--resolution", type=int, required=True,
        help=f"odd grid resolution within [{MIN_RESOLUTION}, {MAX_RESOLUTION}]",
    )
    parser.add_argument("--height-scale", type=float, required=True)
    parser.add_argument("--world-size", type=float, default=1000.0)
    parser.add_argument("--octaves", type=int, default=5)
    parser.add_argument("--frequency", type=float, default=2.0)
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[0, 1, 2],
        help="LOD levels, e.g. 0 1 2 (full, half, quarter)",
    )
    parser.add_argument("--output", required=True, help="target output directory")
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE)
    parser.add_argument(
        "--memory-limit-mb", type=int,
        default=DEFAULT_MEMORY_LIMIT // (1024 * 1024),
    )
    parser.add_argument("--preview-width", type=int, default=512)
    parser.add_argument("--preview-height", type=int, default=512)
    parser.add_argument("--consistency-samples", type=int, default=2048)
    parser.add_argument(
        "--skip-preview", action="store_true",
        help="skip offscreen rendering entirely",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    spec = HeightfieldSpec(
        seed=args.seed,
        resolution=args.resolution,
        height_scale=args.height_scale,
        world_size=args.world_size,
        octaves=args.octaves,
        frequency=args.frequency,
    )
    try:
        result = bake(
            spec,
            output_dir=args.output,
            levels=args.levels,
            tile=args.tile,
            memory_limit_bytes=args.memory_limit_mb * 1024 * 1024,
            preview_width=args.preview_width,
            preview_height=args.preview_height,
            consistency_samples=args.consistency_samples,
            skip_preview=args.skip_preview,
        )
    except BakerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code

    manifest = result["manifest"]
    preview = manifest["preview"]
    print(json.dumps(
        {
            "output": args.output,
            "verification": result["verification"],
            "preview": {
                "rendered": preview.get("rendered", False),
                "available": preview.get("available", False),
            },
        },
        indent=2,
    ))
    if preview.get("rendered") is False and preview.get("skipped") is not True:
        print(
            "warning: offscreen preview unavailable; core artifacts complete",
            file=sys.stderr,
        )
        return EXIT_PREVIEW_UNAVAILABLE
    return 0


if __name__ == "__main__":
    sys.exit(main())
