"""Frame batch scheduling: bake the full timeline to disk."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .animator import SkeletalAnimator
from .clip import OrthoFrustum, clip_frame
from .errors import ErrorLog, E_FRAME_FULLY_CULLED
from .normals import recompute_normals
from .raster import (
    BACKGROUND,
    VIEWPORT_WIDTH,
    build_overview,
    rasterize,
    viewport_height,
)
from .skinning import skin_frame


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write via a same-directory temp file, then atomically rename."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def atomic_write_array(path: Path, array: np.ndarray) -> None:
    """Dump an array as little-endian float32 bytes, atomically."""
    atomic_write_bytes(path, np.ascontiguousarray(array, dtype="<f4").tobytes())


def atomic_write_png(path: Path, image: Image.Image) -> None:
    tmp = path.with_name(path.name + ".tmp")
    image.save(tmp, format="PNG")
    os.replace(tmp, path)


def bake(output_dir: str, frames: int, seed: int, fps: float) -> int:
    """Run the full bake. Returns the process exit code (always 0 here;
    business errors are logged, never fatal)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    animator = SkeletalAnimator(seed=seed, fps=fps)
    frustum = OrthoFrustum()
    height = viewport_height(frustum)
    error_log = ErrorLog()
    stats_entries: list[dict] = []
    frame_images: list[Image.Image] = []
    blank = Image.new("RGB", (VIEWPORT_WIDTH, height), BACKGROUND)

    for frame in range(frames):
        world = animator.frame_world_matrices(frame, error_log)
        skinned = None
        if world is not None:
            skinned = skin_frame(
                animator.vertices,
                animator.weights,
                animator.skin_matrices(world),
                frame,
                error_log,
            )
        if skinned is None:
            # Frame skipped due to E_NON_FINITE_MATRIX /
            # E_NON_INVERTIBLE_MATRIX; keep the overview grid aligned.
            frame_images.append(blank.copy())
            stats_entries.append({"frame": frame, "skipped": True})
            continue

        positions, _skinned_normals = skinned
        normals, face_normals = recompute_normals(
            positions, animator.triangles, frame, error_log
        )
        ndc = frustum.to_ndc(positions)
        clip_pos, clip_nrm, face_idx, stats = clip_frame(
            ndc, normals, animator.triangles
        )
        if stats.triangles_out == 0:
            error_log.report(
                frame,
                E_FRAME_FULLY_CULLED,
                "all triangles clipped away; empty buffers written",
            )

        stem = f"frame_{frame:04d}"
        atomic_write_array(out / f"{stem}.vert.bin", clip_pos)
        atomic_write_array(out / f"{stem}.norm.bin", clip_nrm)
        image = rasterize(clip_pos, face_normals, face_idx,
                          VIEWPORT_WIDTH, height)
        atomic_write_png(out / f"{stem}.png", image)
        frame_images.append(image)
        stats_entries.append({"frame": frame, "skipped": False,
                              **stats.as_dict()})

    atomic_write_png(out / "overview.png", build_overview(frame_images))
    atomic_write_bytes(out / "errors.log", error_log.to_jsonl())
    stats_jsonl = "".join(
        json.dumps(e, sort_keys=True) + "\n" for e in stats_entries
    )
    atomic_write_bytes(out / "stats.jsonl", stats_jsonl.encode("utf-8"))
    return 0
