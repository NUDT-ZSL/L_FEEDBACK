#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from softdeferred.errors import RendererError, SceneConfigError
from softdeferred.renderer import render_scene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only deferred renderer with cascaded, cube, and spot shadow maps."
        )
    )
    parser.add_argument("--scene", required=True, help="scene JSON file")
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--width", type=int, help="override output width")
    parser.add_argument("--height", type=int, help="override output height")
    parser.add_argument(
        "--shadow-resolution", type=int, help="override shadow-map resolution"
    )
    parser.add_argument(
        "--depth-tolerance",
        type=float,
        help="override shadow comparison tolerance",
    )
    parser.add_argument("--depth-bias", type=float, help="override depth bias")
    parser.add_argument(
        "--slope-scale", type=float, help="override slope-scaled depth bias"
    )
    parser.add_argument(
        "--cascade-count", type=int, help="override directional cascade count (>= 2)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing output directory",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    overrides = {
        "width": args.width,
        "height": args.height,
        "shadow_resolution": args.shadow_resolution,
        "depth_tolerance": args.depth_tolerance,
        "depth_bias": args.depth_bias,
        "slope_scale": args.slope_scale,
        "cascade_count": args.cascade_count,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    try:
        report = render_scene(
            args.scene,
            args.output,
            overrides=overrides,
            overwrite=args.overwrite,
        )
    except SceneConfigError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except RendererError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print("error: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print("error: unexpected %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1

    print(
        "rendered %dx%d, %d covered pixels, %d shadow maps -> %s"
        % (
            report["resolution"]["width"],
            report["resolution"]["height"],
            report["gbuffer"]["covered_pixels"],
            report["shadow_maps"]["map_count"],
            args.output,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
