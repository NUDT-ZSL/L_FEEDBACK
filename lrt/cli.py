"""Command-line entry point for the offline layered CPU renderer."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional, Sequence

from .errors import ERROR_OUTPUT, SceneError
from .materials import calibration_report
from .renderer import Renderer
from .scene import DEFAULT_HEIGHT, DEFAULT_SEED, DEFAULT_WIDTH, create_default_scene, parse_scene


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lrt",
        description="Deterministic CPU ray tracer with layered coatings and soft area lights.",
    )
    parser.add_argument("--scene", help="input JSON scene; omit to generate the seeded default scene")
    parser.add_argument("--output", default="render.png", help="8-bit RGB PNG output path")
    parser.add_argument("--stats", default="render.json", help="JSON statistics output path")
    parser.add_argument("--seed", type=int, help="override/generated scene seed")
    parser.add_argument("--width", type=int, help="override image width")
    parser.add_argument("--height", type=int, help="override image height")
    parser.add_argument("--spp", type=int, help="paths per subpixel")
    parser.add_argument("--aa", type=int, choices=(1, 2, 4), help="subpixel samples per axis")
    parser.add_argument("--shadow-samples", type=int, help="area-light samples per light and hit")
    parser.add_argument("--max-depth", type=int, help="maximum surface depth")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="include the fixed-angle layered reflectance calibration report",
    )
    return parser


def _load_description(path: Optional[str], seed: int, width: int, height: int) -> Dict[str, Any]:
    if path is None:
        return create_default_scene(seed=seed, width=width, height=height)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise SceneError(2, "cannot read scene %r: %s" % (path, exc)) from exc
    except json.JSONDecodeError as exc:
        raise SceneError(
            2,
            "invalid JSON in %r at line %d column %d: %s"
            % (path, exc.lineno, exc.colno, exc.msg),
        ) from exc
    if not isinstance(data, dict):
        raise SceneError(3, "scene root must be a JSON object")
    return data


def _apply_overrides(
    description: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if args.seed is not None:
        description["seed"] = args.seed
    if args.width is not None:
        description["width"] = args.width
    if args.height is not None:
        description["height"] = args.height
    overrides = {
        "samples_per_pixel": args.spp,
        "aa_samples": args.aa,
        "shadow_samples": args.shadow_samples,
        "max_depth": args.max_depth,
    }
    if any(value is not None for value in overrides.values()):
        render = description.setdefault("render", {})
        if not isinstance(render, dict):
            raise SceneError(8, "render must be an object")
        for key, value in overrides.items():
            if value is not None:
                render[key] = value
    return description


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    seed = DEFAULT_SEED if args.seed is None else args.seed
    width = DEFAULT_WIDTH if args.width is None else args.width
    height = DEFAULT_HEIGHT if args.height is None else args.height
    try:
        description = _load_description(args.scene, seed, width, height)
        description = _apply_overrides(description, args)
        scene = parse_scene(description)
        result = Renderer(scene).render()
        if args.calibrate:
            report = calibration_report(scene.materials.values(), scene.camera.position)
            result.stats["calibration"] = report
            for warning in report["warnings"]:
                print("WARNING: %s" % warning, file=sys.stderr)
            if not report["all_in_range"]:
                print(
                    "WARNING: calibration found layered reflectance outside [0,1]; "
                    "values were not silently clamped",
                    file=sys.stderr,
                )
        for warning in result.stats.get("energy_warnings", []):
            print("WARNING: %s" % warning, file=sys.stderr)
        try:
            result.save_png(args.output)
            result.save_stats(args.stats)
        except OSError as exc:
            raise SceneError(ERROR_OUTPUT, "cannot write output: %s" % exc) from exc
        return 0
    except SceneError as exc:
        print("ERROR %d: %s" % (exc.code, exc.message), file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())

