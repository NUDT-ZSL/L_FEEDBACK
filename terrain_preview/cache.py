"""Manifest handling, content-addressed tile cache, and atomic JSON helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .errors import cache_error


MANIFEST_FORMAT = 2


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise cache_error(f"cannot write JSON file {destination}: {exc}") from None


def load_manifest(cache_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = cache_dir / "manifest.json"
    if not path.is_file():
        return None, None
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"ignoring unreadable cache manifest: {exc}"
    if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
        return None, "ignoring incompatible cache manifest"
    return manifest, None


def save_manifest(cache_dir: Path, manifest: dict[str, Any]) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise cache_error(f"cannot create cache directory {cache_dir}: {exc}") from None
    write_json_atomic(cache_dir / "manifest.json", manifest)


def _safe_cache_path(cache_dir: Path, relative: str) -> Path:
    path = (cache_dir / relative).resolve()
    root = cache_dir.resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise cache_error(f"cache manifest contains an unsafe path: {relative}") from None
    return path


def screen_tile_shape(
    row: int, col: int, width: int, height: int, tile_size: int
) -> tuple[int, int]:
    x0 = col * tile_size
    y0 = row * tile_size
    return (
        max(0, min(tile_size, height - y0)),
        max(0, min(tile_size, width - x0)),
    )


def load_cached_tile(
    cache_dir: Path,
    entry: dict[str, Any],
    expected_shape: tuple[int, int],
    *,
    include_geometry: bool = False,
) -> dict[str, np.ndarray] | None:
    relative = entry.get("file")
    if not isinstance(relative, str):
        return None
    path = _safe_cache_path(cache_dir, relative)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            rgb = np.asarray(archive["rgb"])
            normal = np.asarray(archive["normal"])
            shadow = np.asarray(archive["shadow"])
            covered = np.asarray(archive["covered"]).astype(bool)
            if include_geometry:
                positions = np.asarray(archive["positions"])
                normals_float = np.asarray(archive["normals"])
    except (OSError, KeyError, ValueError):
        return None
    tile_height, tile_width = expected_shape
    if (
        rgb.shape != (tile_height, tile_width, 3)
        or normal.shape != (tile_height, tile_width, 3)
        or shadow.shape != (tile_height, tile_width)
        or covered.shape != (tile_height, tile_width)
        or rgb.dtype != np.uint8
        or normal.dtype != np.uint8
        or shadow.dtype != np.uint8
    ):
        return None
    result = {
        "rgb": np.ascontiguousarray(rgb),
        "normal": np.ascontiguousarray(normal),
        "shadow": np.ascontiguousarray(shadow),
        "covered": np.ascontiguousarray(covered),
    }
    if include_geometry:
        if (
            positions.shape != (tile_height, tile_width, 3)
            or normals_float.shape != (tile_height, tile_width, 3)
            or positions.dtype != np.float32
            or normals_float.dtype != np.float32
        ):
            return None
        result["positions"] = np.ascontiguousarray(positions)
        result["normals"] = np.ascontiguousarray(normals_float)
    return result


def save_cached_tile(
    cache_dir: Path,
    rgb: np.ndarray,
    normal: np.ndarray,
    shadow: np.ndarray,
    covered: np.ndarray,
    positions: np.ndarray,
    normals: np.ndarray,
) -> str:
    tiles_dir = cache_dir / "tiles"
    try:
        tiles_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise cache_error(f"cannot create tile cache directory {tiles_dir}: {exc}") from None
    temporary = tiles_dir / f".staging-{os.getpid()}-{id(rgb)}.npz"
    try:
        np.savez(
            temporary,
            rgb=np.ascontiguousarray(rgb),
            normal=np.ascontiguousarray(normal),
            shadow=np.ascontiguousarray(shadow),
            covered=np.ascontiguousarray(covered.astype(np.uint8)),
            positions=np.ascontiguousarray(positions, dtype=np.float32),
            normals=np.ascontiguousarray(normals, dtype=np.float32),
        )
        payload = temporary.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        destination = tiles_dir / f"{digest}.npz"
        if destination.exists():
            temporary.unlink(missing_ok=True)
        else:
            os.replace(temporary, destination)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise cache_error(f"cannot write tile cache in {tiles_dir}: {exc}") from None
    return f"tiles/{destination.name}"


def load_shadow_map(
    cache_dir: Path, manifest: dict[str, Any], resolution: int
) -> np.ndarray | None:
    relative = manifest.get("shadow_file")
    if not isinstance(relative, str):
        return None
    path = _safe_cache_path(cache_dir, relative)
    if not path.is_file():
        return None
    try:
        depth = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if depth.shape != (resolution, resolution) or depth.dtype != np.float32:
        return None
    return np.ascontiguousarray(depth)


def save_shadow_map(cache_dir: Path, depth_map: np.ndarray, shadow_key: str) -> str:
    directory = cache_dir / "shadows"
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise cache_error(f"cannot create shadow cache directory {directory}: {exc}") from None
    file_name = f"shadow-{shadow_key[:20]}.npy"
    destination = directory / file_name
    temporary = directory / f".staging-{os.getpid()}.npy"
    try:
        np.save(temporary, np.ascontiguousarray(depth_map, dtype=np.float32))
        os.replace(temporary, destination)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise cache_error(f"cannot write shadow map cache {destination}: {exc}") from None
    return f"shadows/{file_name}"


def cleanup_cache(cache_dir: Path, manifest: dict[str, Any]) -> None:
    referenced_tiles: set[Path] = set()
    for entry in manifest.get("screen_tiles", {}).values():
        if isinstance(entry, dict) and isinstance(entry.get("file"), str):
            referenced_tiles.add(_safe_cache_path(cache_dir, entry["file"]))
    referenced_shadows: set[Path] = set()
    shadow_file = manifest.get("shadow_file")
    if isinstance(shadow_file, str):
        referenced_shadows.add(_safe_cache_path(cache_dir, shadow_file))

    tiles_dir = cache_dir / "tiles"
    if tiles_dir.is_dir():
        for path in tiles_dir.glob("*.npz"):
            if path.resolve() not in referenced_tiles:
                try:
                    path.unlink()
                except OSError:
                    pass
    shadows_dir = cache_dir / "shadows"
    if shadows_dir.is_dir():
        for path in shadows_dir.glob("*.npy"):
            if path.resolve() not in referenced_shadows:
                try:
                    path.unlink()
                except OSError:
                    pass

