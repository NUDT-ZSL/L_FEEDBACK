"""End-to-end pipeline: deterministic change sequence, incremental render,
full-recompute consistency verification, and snapshot artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import List

import numpy as np

from .arch import segment_aabb
from .camera import Camera
from .incremental import IncrementalRenderer, load_cache
from .raster import render
from .scene import Scene, generate_scene
from .tiles import make_tiles, recount_from_ids


def build_camera_sequence(scene: Scene, width: int, height: int) -> List[Camera]:
    """Deterministic camera change sequence derived from the scene layout."""
    aspect = width / height
    xs = [s.translation[0] for s in scene.segments]
    x_mid = 0.5 * (min(xs) + max(xs))
    gate0 = scene.segments[0]
    gx, _, gz = gate0.translation
    cams = [
        # 0: establish shot, looking down the row of gates
        Camera.look_at((x_mid - 6.0, 3.0, 16.0), (x_mid, 2.0, 0.0), 55.0, aspect),
        # 1: dolly sideways + closer (camera move only)
        Camera.look_at((x_mid - 14.0, 2.2, 10.0), (x_mid + 4.0, 2.0, 0.0), 55.0, aspect),
        # 2: camera placed inside the first gate frame (defined result)
        Camera.look_at((gx, 1.6, gz), (gx + 8.0, 2.2, gz), 60.0, aspect),
        # 3: pulled back with a wider fov
        Camera.look_at((x_mid, 6.0, 26.0), (x_mid, 1.5, 0.0), 80.0, aspect),
    ]
    return cams


def build_step_sequence(scene: Scene, width: int, height: int):
    """Return [(scene, camera, note), ...] mixing camera and scene changes."""
    cams = build_camera_sequence(scene, width, height)
    seg1 = scene.segments[min(1, len(scene.segments) - 1)]
    mutated_stripe = replace(
        seg1.stripe, count=seg1.stripe.count + 3, duty=round(seg1.stripe.duty * 0.8, 6)
    )
    scene_b = scene.mutate_segment(seg1.index, stripe=mutated_stripe)
    shared_mat = scene.materials[0]
    scene_c = scene.mutate_material(
        shared_mat.material_id,
        (255 - shared_mat.base_color[0],) + tuple(shared_mat.base_color[1:]),
    )
    return [
        (scene, cams[0], "initial frame"),
        (scene, cams[0], "no change: full reuse"),
        (scene, cams[1], "camera move only"),
        (scene_b, cams[1], "segment stripe mutation (scene version bump)"),
        (scene_b, cams[2], "camera inside first gate"),
        (scene_c, cams[3], "shared material recolor + wide fov pullback"),
    ]


def camera_inside_segments(scene: Scene, camera: Camera) -> List[int]:
    """Deterministic business result: ids of gates whose AABB contains the camera."""
    pos = np.asarray(camera.position, dtype=np.float64)
    inside = []
    for seg in scene.segments:
        mins, maxs = segment_aabb(seg)
        if np.all(pos >= mins - 1e-9) and np.all(pos <= maxs + 1e-9):
            inside.append(seg.index)
    return inside


def full_recompute(scene: Scene, camera: Camera, width: int, height: int, tiles):
    """From-scratch full-frame render plus manifest (ground truth)."""
    res = render(scene, camera, width, height)
    manifest = recount_from_ids(res.ids, tiles)
    return res, manifest


def verify_frame(frame, scene, camera, width, height, tiles) -> List[dict]:
    """Byte-compare an incremental frame against a full recompute.

    Returns a structured diff report (empty list = identical)."""
    ref, ref_manifest = full_recompute(scene, camera, width, height, tiles)
    diffs = []
    for i, (x0, x1) in enumerate(tiles):
        color_eq = np.array_equal(frame.color[:, x0:x1], ref.color[:, x0:x1])
        depth_eq = np.array_equal(
            frame.depth[:, x0:x1].view(np.int32),
            ref.depth[:, x0:x1].view(np.int32),
        )
        ids_eq = np.array_equal(frame.ids[:, x0:x1], ref.ids[:, x0:x1])
        manifest_eq = frame.manifest[i]["segments"] == ref_manifest[i]["segments"]
        if not (color_eq and depth_eq and ids_eq and manifest_eq):
            exp = {int(k): v for k, v in frame.manifest[i]["segments"].items()}
            act = {int(k): v for k, v in ref_manifest[i]["segments"].items()}
            all_ids = sorted(set(exp) | set(act))
            diffs.append(
                {
                    "tile": i,
                    "x0": x0,
                    "x1": x1,
                    "color_equal": bool(color_eq),
                    "depth_equal": bool(depth_eq),
                    "ids_equal": bool(ids_eq),
                    "segments": [
                        {
                            "segment": sid,
                            "incremental_pixels": exp.get(sid, 0),
                            "full_recompute_pixels": act.get(sid, 0),
                            "delta": exp.get(sid, 0) - act.get(sid, 0),
                        }
                        for sid in all_ids
                        if exp.get(sid, 0) != act.get(sid, 0)
                    ],
                }
            )
    return diffs


def _write_ppm(path: str, color: np.ndarray) -> None:
    h, w = color.shape[0], color.shape[1]
    with open(path, "wb") as fh:
        fh.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        fh.write(color[:, :, 0:3].tobytes())


def run_pipeline(
    seed: int,
    num_segments: int,
    width: int,
    height: int,
    tile_width: int,
    out_dir: str,
    save_snapshots: bool = True,
) -> dict:
    """Run the deterministic demo sequence. Raises GateError on business
    errors (empty scene, degenerate fov, bad tile width, ...)."""
    scene = generate_scene(seed, num_segments)
    tiles = make_tiles(width, tile_width)
    steps = build_step_sequence(scene, width, height)
    renderer = IncrementalRenderer(width, height, tile_width)

    os.makedirs(out_dir, exist_ok=True)
    frames_summary = []
    all_diffs = []

    for step_idx, (sc, cam, note) in enumerate(steps):
        frame = renderer.render_frame(sc, cam)
        inside = camera_inside_segments(sc, cam)
        diffs = verify_frame(frame, sc, cam, width, height, tiles)
        for d in diffs:
            d["step"] = step_idx
        all_diffs.extend(diffs)
        frames_summary.append(
            {
                "step": step_idx,
                "note": note,
                "generation": frame.generation,
                "scene_version": frame.scene_version,
                "camera_version": frame.camera_version,
                "dirty_tiles": frame.dirty_tiles,
                "reused_tiles": frame.reused_tiles,
                "camera_inside_segments": inside,
                "manifest_buffer_diffs": frame.manifest_diffs,
                "incremental_vs_full_diffs": diffs,
            }
        )
        if save_snapshots:
            np.save(os.path.join(out_dir, f"frame_{step_idx:03d}_color.npy"), frame.color)
            np.save(os.path.join(out_dir, f"frame_{step_idx:03d}_depth.npy"), frame.depth)
            _write_ppm(os.path.join(out_dir, f"frame_{step_idx:03d}.ppm"), frame.color)
            with open(
                os.path.join(out_dir, f"frame_{step_idx:03d}_manifest.json"),
                "w",
                encoding="utf-8",
            ) as fh:
                json.dump(frame.manifest, fh, indent=1)

    cache_dir = os.path.join(out_dir, "cache")
    cache_path = renderer.save_cache(cache_dir)
    meta = renderer.cache_meta()
    cache_hit = load_cache(cache_dir, meta) is not None
    stale_meta = dict(meta, generation=meta["generation"] + 1)
    stale_rejected = load_cache(cache_dir, stale_meta) is None

    summary = {
        "ok": not all_diffs
        and all(not f["manifest_buffer_diffs"] for f in frames_summary),
        "seed": seed,
        "num_segments": num_segments,
        "image": {"width": width, "height": height, "tile_width": tile_width},
        "tiles": [{"tile": i, "x0": a, "x1": b} for i, (a, b) in enumerate(tiles)],
        "scene_version_initial": scene.version,
        "frames": frames_summary,
        "cache": {
            "path": cache_path,
            "self_hit": cache_hit,
            "stale_generation_rejected": stale_rejected,
        },
        "total_diffs": len(all_diffs),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    return summary
