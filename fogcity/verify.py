"""Read-only verification of previously rendered output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import (
    EXIT_VERIFY_MISMATCH,
    EXIT_VERIFY_OUTPUT_MISSING,
    VERIFY_MISMATCH,
    VERIFY_OUTPUT_MISSING,
    BusinessError,
)
from .output import frame_json_name, frame_png_name
from .renderer import FrameArtifact
from .reporting import make_manifest
from .runner import iter_frames
from .scene import Scene
from .util import canonical_json_bytes, strip_volatile


def expected_output_paths(output_dir: str, frame_count: int) -> list[Path]:
    directory = Path(output_dir)
    paths = [directory / "stats.json"]
    for frame_index in range(frame_count):
        paths.append(directory / frame_png_name(frame_index))
        paths.append(directory / frame_json_name(frame_index))
    return paths


def _raise_missing(paths: list[Path], output_dir: str) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise BusinessError(
            VERIFY_OUTPUT_MISSING,
            "verification requires an existing complete output set",
            EXIT_VERIFY_OUTPUT_MISSING,
            {"output_dir": output_dir, "missing": missing[:16], "missing_count": len(missing)},
        )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BusinessError(
            VERIFY_MISMATCH,
            f"could not read comparable JSON output {path}: {exc}",
            EXIT_VERIFY_MISMATCH,
            {"path": str(path)},
        ) from exc


def _mismatch(frame_index: int | None, path: Path, message: str) -> BusinessError:
    context: dict[str, Any] = {"path": str(path), "reason": message}
    if frame_index is not None:
        context["frame_index"] = frame_index
    return BusinessError(
        VERIFY_MISMATCH,
        "rendered output does not match the deterministic recomputation",
        EXIT_VERIFY_MISMATCH,
        context,
    )


def verify_existing_output(
    scene: Scene,
    output_dir: str,
    *,
    backend: str,
    parallel_frames: int,
) -> dict[str, Any]:
    directory = Path(output_dir)
    if not directory.is_dir():
        raise BusinessError(
            VERIFY_OUTPUT_MISSING,
            "verification output directory does not exist",
            EXIT_VERIFY_OUTPUT_MISSING,
            {"output_dir": output_dir},
        )
    paths = expected_output_paths(output_dir, len(scene.frame_times))
    _raise_missing(paths, output_dir)

    frame_stats: list[dict[str, Any]] = [{} for _ in scene.frame_times]
    gl_info: dict[str, str] | None = None
    iterator = iter_frames(scene, backend=backend, parallel_frames=parallel_frames)
    try:
        for artifact in iterator:
            gl_info = artifact.gl_info
            frame_stats[artifact.frame_index] = artifact.frame_stats
            png_path = directory / frame_png_name(artifact.frame_index)
            json_path = directory / frame_json_name(artifact.frame_index)
            try:
                existing_png = png_path.read_bytes()
            except OSError as exc:
                raise _mismatch(
                    artifact.frame_index,
                    png_path,
                    f"could not read PNG: {exc}",
                ) from exc
            if existing_png != artifact.png_bytes:
                raise _mismatch(
                    artifact.frame_index,
                    png_path,
                    "PNG bytes differ from deterministic recomputation",
                )
            existing_stats = _read_json(json_path)
            if canonical_json_bytes(strip_volatile(existing_stats)) != canonical_json_bytes(
                strip_volatile(artifact.frame_stats)
            ):
                raise _mismatch(
                    artifact.frame_index,
                    json_path,
                    "per-frame statistics differ from recomputation",
                )
    finally:
        iterator.close()

    if gl_info is None:
        raise _mismatch(None, directory, "no frame was rendered for verification")
    recomputed_manifest = make_manifest(scene, frame_stats, gl_info)
    manifest_path = directory / "stats.json"
    existing_manifest = _read_json(manifest_path)
    if canonical_json_bytes(strip_volatile(existing_manifest)) != canonical_json_bytes(
        strip_volatile(recomputed_manifest)
    ):
        raise _mismatch(None, manifest_path, "aggregate statistics differ from recomputation")

    return {
        "output_dir": str(directory.resolve()),
        "frames_verified": len(scene.frame_times),
        "files_verified": len(paths),
    }

