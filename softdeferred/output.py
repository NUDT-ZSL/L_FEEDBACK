from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .errors import RendererError
from .gbuffer import GBuffer
from .shadows import ShadowCollection


def _save_depth_png(path: Path, depth: np.ndarray, coverage: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RendererError("Pillow is required for PNG output: %s" % exc)
    clipped = np.clip(np.asarray(depth, dtype=np.float64), 0.0, 1.0)
    values = np.sqrt(clipped)
    image = (values * 255.0).round().astype(np.uint8)
    Image.fromarray(image, mode="L").save(path)


def _save_shadow_png(path: Path, depth: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RendererError("Pillow is required for PNG output: %s" % exc)
    clipped = np.clip(np.asarray(depth, dtype=np.float64), 0.0, 1.0)
    values = np.sqrt(clipped)
    image = (values * 255.0).round().astype(np.uint8)
    Image.fromarray(image, mode="L").save(path)


def _write_into_staging(
    staging: Path,
    gbuffer: GBuffer,
    hdr: np.ndarray,
    depth: np.ndarray,
    shadows: ShadowCollection,
    report: Dict[str, Any],
) -> None:
    stage_start = time.perf_counter_ns()
    gbuffer.save(staging / "gbuffer")
    np.save(staging / "hdr_color.npy", hdr, allow_pickle=False)
    _save_depth_png(staging / "depth.png", depth, gbuffer.coverage)
    shadow_dir = staging / "shadow_maps"
    shadow_dir.mkdir(parents=True, exist_ok=True)
    for shadow_map in shadows.maps:
        _save_shadow_png(shadow_dir / shadow_map.filename, shadow_map.depth)
    stage_end = time.perf_counter_ns()
    report.setdefault("timings_ms", {})["output_write"] = round(
        (stage_end - stage_start) / 1.0e6, 6
    )
    with (staging / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_outputs(
    output_directory: Path,
    gbuffer: GBuffer,
    hdr: np.ndarray,
    depth: np.ndarray,
    shadows: ShadowCollection,
    report: Dict[str, Any],
    overwrite: bool,
) -> None:
    target = output_directory.expanduser().resolve()
    if target.exists() and not overwrite:
        raise RendererError(
            "output directory already exists (use --overwrite to replace it): %s" % target
        )
    parent = target.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RendererError("cannot create output parent directory %s: %s" % (parent, exc))

    staging = parent / (".%s.tmp-%s" % (target.name, uuid.uuid4().hex))
    backup = parent / (".%s.backup-%s" % (target.name, uuid.uuid4().hex))
    try:
        staging.mkdir(parents=False, exist_ok=False)
        _write_into_staging(staging, gbuffer, hdr, depth, shadows, report)
        if target.exists():
            if not overwrite:
                raise RendererError(
                    "output directory already exists: %s" % target
                )
            os.replace(str(target), str(backup))
            try:
                os.replace(str(staging), str(target))
            except OSError:
                if backup.exists():
                    os.replace(str(backup), str(target))
                raise
            shutil.rmtree(str(backup), ignore_errors=True)
        else:
            os.replace(str(staging), str(target))
    except RendererError:
        shutil.rmtree(str(staging), ignore_errors=True)
        if backup.exists() and not target.exists():
            try:
                os.replace(str(backup), str(target))
            except OSError:
                pass
        raise
    except Exception as exc:
        shutil.rmtree(str(staging), ignore_errors=True)
        if backup.exists() and not target.exists():
            try:
                os.replace(str(backup), str(target))
            except OSError:
                pass
        if isinstance(exc, (OSError, ValueError, TypeError)):
            raise RendererError("cannot write output directory %s: %s" % (target, exc))
        raise
