"""Command line entry point.

Usage:
    python -m terrain_baker SCENE.json --output DIR [--tile-size N]

Exit codes:
    0  success
    1  unexpected internal error (E_INTERNAL)
    2  E_SCENE_MISSING      3  E_SCENE_INVALID
    4  E_UNKNOWN_MATERIAL   5  E_RESOLUTION_LIMIT
    6  E_TARGET_EXISTS      7  E_PATH_UNSAFE
    8  E_CACHE_INVALID
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__, errors
from .baker import run_bake
from .scene import load_scene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="terrain_baker",
        description="Offline deterministic terrain map baker "
                    "(diffuse / normal / material weights + manifest).")
    parser.add_argument("scene", help="path to the JSON scene description")
    parser.add_argument("--output", "-o", required=True,
                        help="output directory (created if missing)")
    parser.add_argument("--tile-size", type=int, default=None,
                        help="override bake tile size (chunked baking)")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scene = load_scene(args.scene)
        if args.tile_size is not None:
            if args.tile_size < 16:
                raise errors.scene_invalid(
                    "--tile-size must be an integer >= 16")
            from dataclasses import replace
            scene = replace(scene, tile_size=args.tile_size)
        manifest = run_bake(scene, args.output)
    except errors.BakeError as exc:
        print(json.dumps(exc.to_json_dict(), ensure_ascii=False),
              file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # noqa: BLE001 - structured fallback
        payload = {"error": "E_INTERNAL", "message": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
        return 1
    summary = {
        "status": "ok",
        "params_hash": manifest["params_hash"],
        "outputs": {k: v["sha256"] for k, v in manifest["outputs"].items()},
        "normal_correction_factor": manifest["normal_correction_factor"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
