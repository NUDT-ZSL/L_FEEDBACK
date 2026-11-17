from __future__ import annotations

import glob
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .errors import SceneError
from .jsonutil import atomic_write_bytes, dumps_fixed, sha256_file
from .png import encode_png_rgb8
from .render import render_frame
from .scene import Scene


@dataclass(frozen=True)
class ExportResult:
    rendered_frames: tuple[int, ...]
    skipped_frames: tuple[int, ...]
    output_dir: Path
    scene_hash: str


def _frame_names(scene: Scene, frame: int) -> tuple[str, str]:
    try:
        png_name = scene.frame_pattern.format(frame=frame)
        stats_name = scene.stats_pattern.format(frame=frame)
    except (KeyError, IndexError, ValueError) as exc:
        raise SceneError(
            "E_SCHEMA", "output", f"cannot format frame file name: {exc}"
        ) from exc
    if (
        png_name == stats_name
        or png_name == scene.manifest_name
        or stats_name == scene.manifest_name
    ):
        raise SceneError(
            "E_SCHEMA", "output", "frame, stats and manifest file names must differ"
        )
    return png_name, stats_name


def _validate_frame_names(scene: Scene) -> None:
    seen: set[str] = {scene.manifest_name}
    for frame in scene.frames:
        for name in _frame_names(scene, frame):
            if name in seen:
                raise SceneError(
                    "E_SCHEMA",
                    "output",
                    f"generated output name {name!r} is not unique",
                )
            seen.add(name)


def _safe_child(output_dir: Path, name: str) -> Path:
    if "/" in name or "\\" in name or name in {"", ".", ".."}:
        raise SceneError("E_SCHEMA", "output", "generated file names must be basenames")
    path = output_dir / name
    try:
        path.resolve().relative_to(output_dir.resolve())
    except ValueError as exc:
        raise SceneError("E_OUTPUT", str(path), "generated file escapes output directory") from exc
    return path


def _pattern_glob(pattern: str) -> str:
    result = pattern
    while "{" in result and "}" in result:
        start = result.index("{")
        end = result.index("}", start)
        result = result[:start] + "*" + result[end + 1 :]
    return result


def _clear_generated(scene: Scene, output_dir: Path, manifest: dict[str, Any] | None) -> None:
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SceneError(
                "E_OUTPUT", str(output_dir), f"cannot create output directory: {exc}"
            ) from exc
        return
    if not output_dir.is_dir():
        raise SceneError("E_OUTPUT", str(output_dir), "output path is not a directory")
    names: set[str] = {scene.manifest_name}
    if manifest is not None:
        files = manifest.get("files")
        if isinstance(files, list):
            names.update(name for name in files if isinstance(name, str))
        frames = manifest.get("frames")
        if isinstance(frames, list):
            for entry in frames:
                if isinstance(entry, dict):
                    for key in ("png", "stats"):
                        if isinstance(entry.get(key), str):
                            names.add(entry[key])
    for pattern in (scene.frame_pattern, scene.stats_pattern):
        names.update(
            Path(match).name
            for match in glob.glob(str(output_dir / _pattern_glob(pattern)))
        )
    for name in sorted(names):
        if name.startswith("..") or "/" in name or "\\" in name:
            continue
        path = output_dir / name
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise SceneError("E_OUTPUT", str(path), f"cannot remove stale output: {exc}") from exc


def _remove_unexpected_generated(
    scene: Scene,
    output_dir: Path,
    manifest: dict[str, Any] | None,
) -> None:
    expected = {scene.manifest_name}
    for frame in scene.frames:
        expected.update(_frame_names(scene, frame))
    candidates: set[str] = set()
    for pattern in (scene.frame_pattern, scene.stats_pattern):
        candidates.update(
            Path(match).name
            for match in glob.glob(str(output_dir / _pattern_glob(pattern)))
        )
    if manifest is not None and isinstance(manifest.get("files"), list):
        candidates.update(
            Path(name).name
            for name in manifest["files"]
            if isinstance(name, str)
        )
    for name in sorted(candidates - expected):
        if name in {"", ".", ".."} or "/" in name or "\\" in name:
            continue
        path = output_dir / name
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise SceneError(
                "E_OUTPUT", str(path), f"cannot remove stale output: {exc}"
            ) from exc


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _hash_matches(path: Path, expected: str) -> bool:
    if not path.is_file():
        return False
    try:
        return sha256_file(path) == expected
    except OSError:
        return False


def _compatible_prefix(
    scene: Scene,
    output_dir: Path,
    manifest: dict[str, Any] | None,
) -> tuple[list[int], dict[int, dict[str, Any]]] | None:
    if manifest is None:
        return None
    if manifest.get("format_version") != 1:
        return None
    if manifest.get("renderer_version") != __version__:
        return None
    if manifest.get("structure_hash") != scene.structure_hash:
        return None
    if (
        manifest.get("scene_id") != scene.scene_id
        or manifest.get("width") != scene.width
        or manifest.get("height") != scene.height
        or manifest.get("shadow_resolution") != scene.shadow.resolution
    ):
        return None
    frames_value = manifest.get("frames")
    if not isinstance(frames_value, list) or not frames_value:
        return None
    entries: dict[int, dict[str, Any]] = {}
    for value in frames_value:
        if not isinstance(value, dict):
            return None
        frame = value.get("frame")
        if isinstance(frame, bool) or not isinstance(frame, int):
            return None
        if frame in entries:
            return None
        if value.get("frame_hash") != scene.frame_hashes.get(frame):
            return None
        png_name = value.get("png")
        stats_name = value.get("stats")
        png_hash = value.get("png_sha256")
        stats_hash = value.get("stats_sha256")
        if not all(
            isinstance(item, str)
            for item in (png_name, stats_name, png_hash, stats_hash)
        ):
            return None
        expected_png, expected_stats = _frame_names(scene, frame)
        if png_name != expected_png or stats_name != expected_stats:
            return None
        if not _hash_matches(_safe_child(output_dir, png_name), png_hash):
            return None
        if not _hash_matches(_safe_child(output_dir, stats_name), stats_hash):
            return None
        entries[frame] = value
    ordered_frames = sorted(entries)
    if manifest.get("frame_numbers") != ordered_frames:
        return None
    expected_files: list[str] = []
    for frame in ordered_frames:
        value = entries[frame]
        expected_files.extend([value["png"], value["stats"]])
    expected_files.append(scene.manifest_name)
    if manifest.get("files") != expected_files:
        return None
    max_exported = max(entries)
    expected_prefix = {frame for frame in scene.frames if frame <= max_exported}
    if expected_prefix != set(entries):
        return None
    return ordered_frames, entries


def _write_frame(scene: Scene, frame: int, output_dir: Path) -> dict[str, Any]:
    png_name, stats_name = _frame_names(scene, frame)
    png_path = _safe_child(output_dir, png_name)
    stats_path = _safe_child(output_dir, stats_name)
    try:
        result = render_frame(scene, frame)
        png_data = encode_png_rgb8(result.image)
        stats_data = (dumps_fixed(result.stats) + "\n").encode("utf-8")
        atomic_write_bytes(png_path, png_data)
        atomic_write_bytes(stats_path, stats_data)
    except SceneError:
        raise
    except OSError as exc:
        raise SceneError(
            "E_OUTPUT",
            f"frame[{frame}]",
            f"cannot write frame output: {exc}",
        ) from exc
    except Exception as exc:
        raise SceneError(
            "E_RENDER",
            f"frame[{frame}]",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    return {
        "frame": frame,
        "frame_hash": scene.frame_hashes[frame],
        "png": png_name,
        "stats": stats_name,
        "png_sha256": hashlib.sha256(png_data).hexdigest(),
        "stats_sha256": hashlib.sha256(stats_data).hexdigest(),
    }


def _manifest(
    scene: Scene,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    files: list[str] = []
    for entry in entries:
        files.extend([entry["png"], entry["stats"]])
    files.append(scene.manifest_name)
    return {
        "format_version": 1,
        "renderer_version": __version__,
        "scene_id": scene.scene_id,
        "scene_hash": scene.scene_hash,
        "structure_hash": scene.structure_hash,
        "width": scene.width,
        "height": scene.height,
        "shadow_resolution": scene.shadow.resolution,
        "frame_numbers": [int(entry["frame"]) for entry in entries],
        "files": files,
        "frames": entries,
    }


def export_scene(
    scene: Scene,
    output_dir: Path,
    *,
    incremental: bool,
    threads: int,
) -> ExportResult:
    try:
        output_dir = output_dir.resolve()
    except OSError as exc:
        raise SceneError("E_OUTPUT", str(output_dir), f"invalid output path: {exc}") from exc
    _validate_frame_names(scene)
    if output_dir.exists() and not output_dir.is_dir():
        raise SceneError("E_OUTPUT", str(output_dir), "output path is not a directory")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SceneError(
            "E_OUTPUT", str(output_dir), f"cannot create output directory: {exc}"
        ) from exc
    manifest_path = _safe_child(output_dir, scene.manifest_name)
    previous_manifest = _read_manifest(manifest_path) if incremental else None
    compatible = (
        _compatible_prefix(scene, output_dir, previous_manifest)
        if incremental
        else None
    )
    if compatible is not None:
        exported_frames, entries_by_frame = compatible
        _remove_unexpected_generated(scene, output_dir, previous_manifest)
        max_exported = max(exported_frames)
        frames_to_render = tuple(
            frame for frame in scene.frames if frame > max_exported
        )
        skipped_frames = tuple(
            frame for frame in scene.frames if frame <= max_exported
        )
    else:
        _clear_generated(scene, output_dir, previous_manifest)
        frames_to_render = scene.frames
        skipped_frames = ()
        entries_by_frame = {}

    rendered_entries: dict[int, dict[str, Any]] = {}
    if frames_to_render:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {
                executor.submit(_write_frame, scene, frame, output_dir): frame
                for frame in frames_to_render
            }
            errors: list[SceneError] = []
            for future in as_completed(futures):
                try:
                    entry = future.result()
                    rendered_entries[int(entry["frame"])] = entry
                except SceneError as exc:
                    errors.append(exc)
                except Exception as exc:
                    errors.append(
                        SceneError(
                            "E_RENDER",
                            f"frame[{futures[future]}]",
                            f"{type(exc).__name__}: {exc}",
                        )
                    )
        if errors:
            raise errors[0]
    entries_by_frame.update(rendered_entries)
    entries = [
        entries_by_frame[frame]
        for frame in scene.frames
        if frame in entries_by_frame
    ]
    manifest = _manifest(scene, entries)
    try:
        atomic_write_bytes(
            manifest_path,
            (dumps_fixed(manifest) + "\n").encode("utf-8"),
        )
    except OSError as exc:
        raise SceneError(
            "E_OUTPUT", str(manifest_path), f"cannot write manifest: {exc}"
        ) from exc
    return ExportResult(
        rendered_frames=tuple(sorted(rendered_entries)),
        skipped_frames=tuple(sorted(skipped_frames)),
        output_dir=output_dir,
        scene_hash=scene.scene_hash,
    )
