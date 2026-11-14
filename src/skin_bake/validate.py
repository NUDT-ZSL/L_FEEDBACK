"""Deterministic asset validation and per-frame software preview."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .errors import schema
from .runtime import load_baked_asset
from .software import render_software_frame, write_png


def run_check(
    scene_path: str | Path,
    bake_dir: str | Path,
    *,
    frame: int,
    output: Path | None,
) -> int:
    asset = load_baked_asset(scene_path, bake_dir)
    if frame < 0 or frame >= asset.frame_count:
        raise schema(f"frame {frame} is outside [0, {asset.frame_count - 1}]")
    render = render_software_frame(asset, frame)
    if output is None:
        output = Path(bake_dir) / f"preview_frame_{frame:04d}.png"
    write_png(output, render.image)

    decoded = asset.frame_displacements(asset.frame_time(frame))
    if not np.all(np.isfinite(decoded)):
        raise schema("decoded texture contains non-finite displacement values")
    summary = {
        "status": "ok",
        "backend": "software-reference",
        "frame": frame,
        "frameCount": asset.frame_count,
        "frameTime": render.frame_time,
        "sampleIndex": render.sample_index,
        "preview": str(output),
        "coveredPixels": render.covered_pixels,
        "decodedDisplacementAbsMax": float(np.max(np.abs(decoded))),
        "fallbackVertices": list(asset.bind_basis.fallback_vertices),
        "atlasSize": [asset.texture.width, asset.texture.height],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0

