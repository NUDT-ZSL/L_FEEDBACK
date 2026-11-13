"""Command line entry: gen / run / verify."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

import numpy as np

from . import errors, scene_gen, snapshot
from .config import load_config
from .renderer import Pipeline

_INTERRUPT = {"received": None}


def _install_signal_handlers():
    def handler(signum, _frame):
        _INTERRUPT["received"] = signum

    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), handler)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_ppm(path, frame_rgba):
    height, width = frame_rgba.shape[0], frame_rgba.shape[1]
    rgb = frame_rgba[:, :, :3].astype(np.uint8)
    with open(path, "wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(rgb.tobytes())


def _save_frame(out_dir, step, frame_rgba):
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    np.save(os.path.join(frames_dir, f"frame_{step:04d}.npy"), frame_rgba)
    _write_ppm(os.path.join(frames_dir, f"frame_{step:04d}.ppm"), frame_rgba)


def cmd_gen(args):
    if args.anomaly:
        scene = scene_gen.generate_anomaly_scene(args.seed, frames=args.frames)
    else:
        scene = scene_gen.generate_scene(args.seed, frames=args.frames)
    scene_gen.write_scene(args.out, scene)
    print(f"wrote scene config to {args.out}")
    return 0


def _run_pipeline(config, log, out_dir, incremental, resume, delay, recovery_events):
    pipeline = Pipeline(config, incremental=incremental)
    snap_dir = os.path.join(out_dir, "snapshots")
    start_frame = 0
    if resume:
        loaded = snapshot.load_latest_snapshot(snap_dir, log)
        if loaded is not None:
            frame, acc = loaded
            pipeline.load_accumulator(acc, frame)
            start_frame = frame + 1
            recovery_events.append(
                {"event": "recover", "source": "snapshot", "frame": frame}
            )
        else:
            recovery_events.append(
                {
                    "event": "recover",
                    "source": "config",
                    "frame": 0,
                    "reason": "no valid snapshot; recomputing from frame 0",
                }
            )
    frames_written = []
    interrupted = False
    for step in range(start_frame, config.frames):
        if _INTERRUPT["received"] is not None:
            signum = _INTERRUPT["received"]
            name = signal.Signals(signum).name
            last = step - 1
            if last >= 0 and pipeline.acc is not None:
                snapshot.save_snapshot(snap_dir, last, pipeline.acc)
            recovery_events.append(
                {"event": "interrupted", "signal": name, "frame": last}
            )
            interrupted = True
            break
        frame_rgba = pipeline.render_frame(step)
        _save_frame(out_dir, step, frame_rgba)
        frames_written.append(step)
        if step % config.snapshot_every == 0:
            snapshot.save_snapshot(snap_dir, step, pipeline.acc)
        if delay > 0.0:
            time.sleep(delay)
    if not interrupted and pipeline.acc is not None:
        snapshot.save_snapshot(snap_dir, config.frames - 1, pipeline.acc)
    report = {
        "mode": "incremental" if incremental else "full",
        "resumed_from": start_frame,
        "frames_written": frames_written,
        "total_shaded_fragments": sum(r.shaded for r in pipeline.records),
        "per_frame": [
            {
                "frame": r.frame,
                "shaded": r.shaded,
                "dirty_rect": list(r.dirty_rect) if r.dirty_rect else None,
                "changed_crystals": r.changed_crystals,
                "draw_sequence": r.draw_sequence,
            }
            for r in pipeline.records
        ],
        "errors": log.errors,
    }
    return report, interrupted


def cmd_run(args):
    log = errors.ErrorLog()
    config = load_config(args.config, log)
    os.makedirs(args.out, exist_ok=True)
    _install_signal_handlers()
    recovery_events = []
    report, interrupted = _run_pipeline(
        config,
        log,
        args.out,
        incremental=(args.mode == "incremental"),
        resume=args.resume,
        delay=args.delay,
        recovery_events=recovery_events,
    )
    _write_json(os.path.join(args.out, "render_report.json"), report)
    if recovery_events:
        _write_json(
            os.path.join(args.out, "recovery_report.json"),
            {"events": recovery_events},
        )
    return 3 if interrupted else 0


def cmd_verify(args):
    log = errors.ErrorLog()
    config = load_config(args.config, log)
    os.makedirs(args.out, exist_ok=True)
    full = Pipeline(config, incremental=False)
    incr = Pipeline(config, incremental=True)
    per_frame = []
    mismatches = 0
    for step in range(config.frames):
        full_frame = full.render_frame(step)
        incr_frame = incr.render_frame(step)
        equal = bool(np.array_equal(full_frame, incr_frame))
        if not equal:
            mismatches += 1
            diff = int(np.abs(full_frame.astype(int) - incr_frame.astype(int)).max())
        else:
            diff = 0
        per_frame.append(
            {
                "frame": step,
                "identical": equal,
                "max_channel_diff": diff,
                "full_shaded": full.records[-1].shaded,
                "incr_shaded": incr.records[-1].shaded,
                "dirty_rect": list(incr.records[-1].dirty_rect)
                if incr.records[-1].dirty_rect
                else None,
                "changed_crystals": incr.records[-1].changed_crystals,
            }
        )
    full_total = sum(r.shaded for r in full.records)
    incr_total = sum(r.shaded for r in incr.records)
    single_change_frames = [p for p in per_frame if len(p["changed_crystals"]) == 1]
    single_change_ok = all(
        p["incr_shaded"] < p["full_shaded"] for p in single_change_frames
    )
    report = {
        "frames": config.frames,
        "all_frames_identical": mismatches == 0,
        "mismatch_count": mismatches,
        "full_total_shaded": full_total,
        "incr_total_shaded": incr_total,
        "single_change_frames": len(single_change_frames),
        "single_change_incr_always_cheaper": single_change_ok,
        "per_frame": per_frame,
        "errors": log.errors,
    }
    _write_json(os.path.join(args.out, "verify_report.json"), report)
    print(
        f"verify: {config.frames} frames, mismatches={mismatches}, "
        f"full_shaded={full_total}, incr_shaded={incr_total}"
    )
    return 0 if mismatches == 0 else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="crystal_renderer")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("gen", help="generate a deterministic scene config")
    gen.add_argument("--out", required=True)
    gen.add_argument("--seed", type=int, default=7)
    gen.add_argument("--frames", type=int, default=40)
    gen.add_argument("--anomaly", action="store_true")
    gen.set_defaults(func=cmd_gen)
    run = sub.add_parser("run", help="render a scene config")
    run.add_argument("--config", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--mode", choices=["full", "incremental"], default="incremental")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--delay", type=float, default=0.0)
    run.set_defaults(func=cmd_run)
    verify = sub.add_parser("verify", help="compare full vs incremental rendering")
    verify.add_argument("--config", required=True)
    verify.add_argument("--out", required=True)
    verify.set_defaults(func=cmd_verify)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
