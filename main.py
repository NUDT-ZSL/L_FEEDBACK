"""CLI entry point for the track-gate software rendering pipeline.

Usage (Windows):  python main.py run --seed 7 --segments 6 --out artifacts\\run
                  python main.py edges

Exit codes: 0 success, 2 deterministic business error, 1 internal error.
"""

from __future__ import annotations

import argparse
import json
import sys

from trackgate import errors
from trackgate.camera import Camera
from trackgate.errors import GateError
from trackgate.pipeline import run_pipeline
from trackgate.scene import generate_scene
from trackgate.tiles import make_tiles


def _print_json(payload) -> None:
    json.dump(payload, sys.stdout, indent=1)
    sys.stdout.write("\n")


def cmd_run(args) -> int:
    summary = run_pipeline(
        seed=args.seed,
        num_segments=args.segments,
        width=args.width,
        height=args.height,
        tile_width=args.tile_width,
        out_dir=args.out,
    )
    _print_json(
        {
            "ok": summary["ok"],
            "out": args.out,
            "frames": len(summary["frames"]),
            "total_diffs": summary["total_diffs"],
            "cache": summary["cache"],
            "scene_version_initial": summary["scene_version_initial"],
        }
    )
    return errors.EXIT_OK if summary["ok"] else errors.EXIT_BUSINESS


def cmd_edges(args) -> int:
    """Exercise the deterministic edge-case / error-code paths."""
    results = []

    # 1. Empty scene -> E_EMPTY_SCENE
    try:
        generate_scene(args.seed, 0)
        results.append({"case": "empty_scene", "ok": False, "error": None})
    except GateError as exc:
        results.append({"case": "empty_scene", "ok": True, "error": exc.payload()["error"]})

    # 2. Degenerate fov (tiny and huge) -> E_DEGENERATE_FOV
    for vfov in (0.0001, 179.9):
        try:
            Camera(position=(0, 0, 10), yaw_deg=0, pitch_deg=0, vfov_deg=vfov, aspect=16 / 9)
            results.append({"case": f"degenerate_fov_{vfov}", "ok": False, "error": None})
        except GateError as exc:
            results.append(
                {"case": f"degenerate_fov_{vfov}", "ok": True, "error": exc.payload()["error"]}
            )

    # 3. Tile width that does not divide image width -> defined result
    tiles = make_tiles(200, 64)
    results.append(
        {
            "case": "tile_width_not_dividing",
            "ok": tiles == [(0, 64), (64, 128), (128, 192), (192, 200)],
            "tiles": tiles,
        }
    )

    # 4. Invalid tile width -> E_BAD_TILE_WIDTH
    try:
        make_tiles(200, 0)
        results.append({"case": "bad_tile_width", "ok": False, "error": None})
    except GateError as exc:
        results.append({"case": "bad_tile_width", "ok": True, "error": exc.payload()["error"]})

    # 5. Camera inside a gate -> defined business result (renders, reports ids)
    scene = generate_scene(args.seed, 3)
    from trackgate.pipeline import camera_inside_segments

    g = scene.segments[0]
    cam = Camera.look_at(
        (g.translation[0], 1.5, g.translation[2]),
        (g.translation[0] + 5.0, 2.0, g.translation[2]),
        60.0,
        16 / 9,
    )
    inside = camera_inside_segments(scene, cam)
    results.append(
        {"case": "camera_inside_gate", "ok": inside == [0], "inside_segments": inside}
    )

    ok = all(r["ok"] for r in results)
    _print_json({"ok": ok, "results": results})
    return errors.EXIT_OK if ok else errors.EXIT_BUSINESS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="trackgate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the deterministic render sequence")
    p_run.add_argument("--seed", type=int, default=7)
    p_run.add_argument("--segments", type=int, default=6)
    p_run.add_argument("--width", type=int, default=320)
    p_run.add_argument("--height", type=int, default=180)
    p_run.add_argument("--tile-width", type=int, default=64)
    p_run.add_argument("--out", default="artifacts/run")
    p_run.set_defaults(func=cmd_run)

    p_edges = sub.add_parser("edges", help="exercise edge cases and error codes")
    p_edges.add_argument("--seed", type=int, default=7)
    p_edges.set_defaults(func=cmd_edges)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except GateError as exc:
        _print_json(exc.payload())
        return exc.exit_code
    except Exception as exc:  # unexpected internal failure
        _print_json({"ok": False, "error": {"code": "E_INTERNAL", "message": repr(exc)}})
        return errors.EXIT_INTERNAL


if __name__ == "__main__":
    sys.exit(main())
