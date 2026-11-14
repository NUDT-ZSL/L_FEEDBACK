"""Command line interface: render / resume / make-example."""

import argparse
import json
import os
import sys

from . import EXIT_OK
from .engine import Engine
from .errors import PipelineError
from .scene import load_scene
from .snapshot import (
    load_snapshot,
    save_snapshot,
    verify_scene_match,
)


def _cmd_render(args):
    scene = load_scene(args.scene)
    if args.snapshot_interval is not None:
        scene["config"]["snapshot_interval"] = args.snapshot_interval
    engine = Engine(scene, args.out, raw_mode="wb")
    try:
        engine.initialize()
        hook = lambda eng: save_snapshot(eng, args.scene)
        completed = engine.run(max_frames=args.max_frames, snapshot_hook=hook)
        if completed:
            wav_path, log_path = engine.finalize()
            return {"status": "ok", "wav": wav_path, "events": log_path,
                    "frames": engine.frames_completed}
        return {"status": "partial", "frames": engine.frames_completed,
                "total_frames": engine.total_frames,
                "snapshot": os.path.join(args.out, "snapshot.json")}
    finally:
        engine.close()


def _cmd_resume(args):
    state = load_snapshot(args.snapshot)
    scene_path = args.scene or state["scene_path"]
    verify_scene_match(state, scene_path)
    scene = load_scene(scene_path)
    out_dir = args.out or state["out_dir"]
    if os.path.abspath(out_dir) != os.path.abspath(state["out_dir"]):
        raise PipelineError(
            "E_SNAPSHOT_MISMATCH",
            "resume output directory must match the snapshot out_dir "
            f"({state['out_dir']}), got {os.path.abspath(out_dir)}",
        )
    engine = Engine(scene, out_dir, raw_mode="ab")
    try:
        engine.restore_materials()
        engine.load_snapshot_state(state)
        hook = lambda eng: save_snapshot(eng, scene_path)
        completed = engine.run(max_frames=args.max_frames, snapshot_hook=hook)
        if completed:
            wav_path, log_path = engine.finalize()
            return {"status": "ok", "wav": wav_path, "events": log_path,
                    "frames": engine.frames_completed,
                    "resumed_from": state["frames_completed"]}
        return {"status": "partial", "frames": engine.frames_completed,
                "total_frames": engine.total_frames,
                "resumed_from": state["frames_completed"]}
    finally:
        engine.close()


def _cmd_make_example(args):
    from .example import build_example_scene

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "example_scene.json")
    scene = build_example_scene()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(scene, fh, indent=2)
    return {"status": "ok", "scene": path}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="spatmix",
        description="Deterministic offline spatial-audio mixing pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_render = sub.add_parser("render", help="render a scene to a multichannel WAV")
    p_render.add_argument("scene", help="path to the scene JSON file")
    p_render.add_argument("--out", required=True, help="output directory")
    p_render.add_argument("--snapshot-interval", type=int, default=None,
                          help="frames between snapshots (overrides scene config)")
    p_render.add_argument("--max-frames", type=int, default=None,
                          help="stop after this many frames (partial render)")
    p_render.set_defaults(func=_cmd_render)

    p_resume = sub.add_parser("resume", help="resume rendering from a snapshot")
    p_resume.add_argument("--snapshot", required=True, help="snapshot JSON file")
    p_resume.add_argument("--scene", default=None,
                          help="scene path override (hash must match snapshot)")
    p_resume.add_argument("--out", default=None,
                          help="output directory (must match snapshot)")
    p_resume.add_argument("--max-frames", type=int, default=None)
    p_resume.set_defaults(func=_cmd_resume)

    p_ex = sub.add_parser("make-example", help="write an example scene JSON")
    p_ex.add_argument("--out", required=True, help="output directory")
    p_ex.set_defaults(func=_cmd_make_example)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except PipelineError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False))
        return exc.exit_code
    print(json.dumps(result, ensure_ascii=False))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
