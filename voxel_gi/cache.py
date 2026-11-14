"""Versioned binary cache with atomic replacement and strict validation."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from .baker import BakeResult
from .errors import BusinessError, ErrorCode
from .scene import (
    ProbeLayout,
    Scene,
    canonical_json_bytes,
    compute_geometry_hash,
    compute_layout_hash,
    compute_scene_hash,
)

MAGIC = b"VXGI9SH1"
FORMAT_VERSION = 1
ALGORITHM_VERSION = "1.0.0"
_HEADER = struct.Struct("<8sIII")
_MAX_METADATA_BYTES = 16 * 1024 * 1024
_RECORD_DTYPE = np.dtype(
    [("coefficients", "<f4", (9, 3)), ("residual", "<f4")],
    align=False,
)


@dataclass(frozen=True)
class CacheData:
    metadata: Dict[str, Any]
    coefficients: np.ndarray
    residuals: np.ndarray
    payload_sha256: str
    file_sha256: str
    format_version: int
    file_size: int

    @property
    def probe_count(self) -> int:
        return int(self.coefficients.shape[0])

    @property
    def mean_residual(self) -> float:
        return float(np.mean(self.residuals, dtype=np.float64))

    @property
    def max_residual(self) -> float:
        return float(np.max(self.residuals))


@dataclass(frozen=True)
class WriteResult:
    file_sha256: str
    payload_sha256: str
    file_size: int
    metadata: Dict[str, Any]


def _cache_error(code: str, message: str, path: Path, **details: Any) -> BusinessError:
    merged = {"path": str(path)}
    merged.update(details)
    return BusinessError(code, message, merged)


def _payload_from_result(result: BakeResult) -> bytes:
    count = result.probe_count
    records = np.empty(count, dtype=_RECORD_DTYPE)
    records["coefficients"] = np.asarray(result.coefficients, dtype="<f4")
    records["residual"] = np.asarray(result.residuals, dtype="<f4")
    return records.tobytes(order="C")


def build_metadata(
    scene: Scene,
    layout: ProbeLayout,
    result: BakeResult,
    payload_sha256: str,
    payload_size: int,
) -> Dict[str, Any]:
    positions_sha256 = hashlib.sha256(
        np.asarray(layout.positions, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "bake": {
            "ambient_radiance": [
                float(value) for value in scene.bake.ambient_radiance
            ],
            "ray_count": int(scene.bake.ray_count),
            "seed": int(scene.bake.seed),
            "voxel_padding": int(scene.bake.padding_voxels),
            "voxel_resolution": int(scene.bake.voxel_resolution),
        },
        "energy": {
            "albedo_upper_bound": float(scene.albedo_upper_bound),
            "bound": float(result.energy_bound),
            "clamped_probes": int(result.clamped_probes),
            "max_channel_irradiance": float(result.max_channel_irradiance),
            "max_probe_irradiance": float(result.max_probe_irradiance),
            "measure": "sum_of_rgb_channels",
            "total_light_power": float(scene.total_light_power),
        },
        "format": "voxel-gi-sh9",
        "format_version": FORMAT_VERSION,
        "geometry_hash": result.geometry_hash,
        "layout": {
            "bounds_max": [float(value) for value in layout.bounds_max],
            "bounds_min": [float(value) for value in layout.bounds_min],
            "counts": [int(value) for value in layout.counts],
            "hash": result.layout_hash,
            "positions_sha256": positions_sha256,
            "spacing": float(layout.spacing),
            "up": [float(value) for value in layout.up],
        },
        "lighting_hash": result.lighting_hash,
        "payload": {
            "channels": 3,
            "coefficient_order": "l0,l1m-1,l1m0,l1m1,l2m-2,l2m-1,l2m0,l2m1,l2m2",
            "coefficients": 9,
            "dtype": "<f4",
            "probe_count": int(result.probe_count),
            "record_bytes": int(_RECORD_DTYPE.itemsize),
            "residual_kind": "relative_rms",
            "sha256": payload_sha256,
            "size_bytes": int(payload_size),
        },
        "precision": "float32",
        "residual_summary": {
            "max": float(result.max_residual),
            "mean": float(result.mean_residual),
        },
        "scene_hash": result.scene_hash,
        "scene_id": scene.scene_id,
        "voxelization": result.grid_info,
    }


def _atomic_write(path: Path, data: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        try:
            os.chmod(destination, 0o644)
        except OSError:
            pass
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def write_cache(
    path: Path,
    scene: Scene,
    layout: ProbeLayout,
    result: BakeResult,
) -> WriteResult:
    payload = _payload_from_result(result)
    payload_sha = hashlib.sha256(payload).hexdigest()
    metadata = build_metadata(scene, layout, result, payload_sha, len(payload))
    metadata_bytes = canonical_json_bytes(metadata)
    if len(metadata_bytes) > _MAX_METADATA_BYTES:
        raise BusinessError(
            ErrorCode.INTERNAL_ERROR,
            "Cache metadata exceeds the size limit",
            {"metadata_bytes": len(metadata_bytes)},
        )
    prefix = _HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        len(metadata_bytes),
        len(payload),
    )
    file_bytes = prefix + metadata_bytes + payload
    file_sha = hashlib.sha256(file_bytes).hexdigest()
    try:
        _atomic_write(Path(path), file_bytes)
    except OSError as exc:
        raise _cache_error(
            ErrorCode.IO_ERROR,
            "Unable to write cache atomically",
            Path(path),
            reason=str(exc),
        ) from exc
    return WriteResult(
        file_sha256=file_sha,
        payload_sha256=payload_sha,
        file_size=len(file_bytes),
        metadata=metadata,
    )


def _parse_header_and_metadata(
    raw: bytes,
    path: Path,
) -> Tuple[Dict[str, Any], int, int, int]:
    if len(raw) < _HEADER.size:
        raise _cache_error(
            ErrorCode.CACHE_CORRUPT,
            "Cache is shorter than its fixed header",
            path,
            size=len(raw),
        )
    magic, version, metadata_length, payload_length = _HEADER.unpack_from(raw, 0)
    if magic != MAGIC:
        raise _cache_error(
            ErrorCode.CACHE_BAD_MAGIC,
            "Cache magic does not match VXGI9SH1",
            path,
            actual=magic.decode("ascii", errors="replace"),
        )
    if version != FORMAT_VERSION:
        raise _cache_error(
            ErrorCode.CACHE_VERSION_UNSUPPORTED,
            "Cache format version is not supported",
            path,
            expected=FORMAT_VERSION,
            actual=int(version),
        )
    if metadata_length <= 0 or metadata_length > _MAX_METADATA_BYTES:
        raise _cache_error(
            ErrorCode.CACHE_CORRUPT,
            "Cache metadata length is invalid",
            path,
            metadata_length=int(metadata_length),
        )
    total_length = _HEADER.size + int(metadata_length) + int(payload_length)
    if total_length != len(raw):
        raise _cache_error(
            ErrorCode.CACHE_CORRUPT,
            "Cache file length does not match declared lengths",
            path,
            expected=total_length,
            actual=len(raw),
        )
    metadata_start = _HEADER.size
    metadata_end = metadata_start + int(metadata_length)
    try:
        metadata = json.loads(raw[metadata_start:metadata_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _cache_error(
            ErrorCode.CACHE_CORRUPT,
            "Cache metadata is not valid canonical UTF-8 JSON",
            path,
            reason=str(exc),
        ) from exc
    if not isinstance(metadata, dict):
        raise _cache_error(
            ErrorCode.CACHE_CORRUPT,
            "Cache metadata root must be an object",
            path,
        )
    if int(metadata.get("format_version", -1)) != FORMAT_VERSION:
        raise _cache_error(
            ErrorCode.CACHE_VERSION_UNSUPPORTED,
            "Metadata format_version does not match the header",
            path,
            expected=FORMAT_VERSION,
            actual=metadata.get("format_version"),
        )
    return metadata, int(metadata_length), int(payload_length), metadata_end


def read_cache(path: Path) -> CacheData:
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        if isinstance(exc, FileNotFoundError):
            raise _cache_error(
                ErrorCode.CACHE_NOT_FOUND,
                "Cache file does not exist",
                source,
                reason=str(exc),
            ) from exc
        raise _cache_error(
            ErrorCode.IO_ERROR,
            "Unable to read cache",
            source,
            reason=str(exc),
        ) from exc
    metadata, _, payload_length, payload_start = _parse_header_and_metadata(raw, source)
    payload = raw[payload_start:]
    payload_info = metadata.get("payload")
    if not isinstance(payload_info, dict):
        raise _cache_error(
            ErrorCode.CACHE_CORRUPT,
            "Cache metadata has no payload descriptor",
            source,
        )
    try:
        probe_count = int(metadata.get("payload", {}).get("probe_count", -1))
        record_bytes = int(metadata.get("payload", {}).get("record_bytes", -1))
        declared_size = int(payload_info.get("size_bytes", -1))
    except (TypeError, ValueError) as exc:
        raise _cache_error(
            ErrorCode.CACHE_CORRUPT,
            "Cache payload descriptor contains invalid integers",
            source,
        ) from exc
    try:
        coefficient_count = int(payload_info.get("coefficients", -1))
        channel_count = int(payload_info.get("channels", -1))
    except (TypeError, ValueError) as exc:
        raise _cache_error(
            ErrorCode.CACHE_CORRUPT,
            "Cache SH payload descriptor has invalid counts",
            source,
        ) from exc
    if (
        coefficient_count != 9
        or channel_count != 3
        or payload_info.get("dtype") not in ("<f4", "float32")
    ):
        raise _cache_error(
            ErrorCode.CACHE_CORRUPT,
            "Cache SH payload descriptor is incompatible",
            source,
            coefficients=payload_info.get("coefficients"),
            channels=payload_info.get("channels"),
            dtype=payload_info.get("dtype"),
        )
    if probe_count <= 0 or record_bytes != _RECORD_DTYPE.itemsize:
        raise _cache_error(
            ErrorCode.CACHE_CORRUPT,
            "Cache probe count or record size is invalid",
            source,
            probe_count=probe_count,
            record_bytes=record_bytes,
            expected_record_bytes=_RECORD_DTYPE.itemsize,
        )
    expected_size = probe_count * _RECORD_DTYPE.itemsize
    if (
        expected_size != payload_length
        or declared_size != expected_size
        or len(payload) != expected_size
    ):
        raise _cache_error(
            ErrorCode.CACHE_CORRUPT,
            "Cache payload length does not match probe count",
            source,
            expected=expected_size,
            declared=declared_size,
            actual=len(payload),
        )
    payload_sha = hashlib.sha256(payload).hexdigest()
    declared_sha = payload_info.get("sha256")
    if not isinstance(declared_sha, str) or payload_sha != declared_sha:
        raise _cache_error(
            ErrorCode.CACHE_CHECKSUM_MISMATCH,
            "Cache payload SHA-256 checksum mismatch",
            source,
            expected=declared_sha,
            actual=payload_sha,
        )
    records = np.frombuffer(payload, dtype=_RECORD_DTYPE, count=probe_count).copy()
    coefficients = np.ascontiguousarray(records["coefficients"], dtype=np.float32)
    residuals = np.ascontiguousarray(records["residual"], dtype=np.float32)
    if not np.all(np.isfinite(coefficients)):
        raise _cache_error(
            ErrorCode.CACHE_CORRUPT,
            "Cache contains non-finite SH coefficients",
            source,
        )
    if not np.all(np.isfinite(residuals)) or np.any(residuals < 0.0):
        raise _cache_error(
            ErrorCode.CACHE_CORRUPT,
            "Cache contains invalid residuals",
            source,
        )
    return CacheData(
        metadata=metadata,
        coefficients=coefficients,
        residuals=residuals,
        payload_sha256=payload_sha,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        format_version=FORMAT_VERSION,
        file_size=len(raw),
    )


def validate_cache(scene: Scene, layout: ProbeLayout, cache: CacheData) -> None:
    if cache.metadata.get("format") != "voxel-gi-sh9":
        raise BusinessError(
            ErrorCode.CACHE_VERSION_UNSUPPORTED,
            "Cache payload format identifier is not supported",
            {"actual": cache.metadata.get("format")},
        )
    if cache.metadata.get("algorithm_version") != ALGORITHM_VERSION:
        raise BusinessError(
            ErrorCode.CACHE_VERSION_UNSUPPORTED,
            "Cache was produced by a different bake algorithm version",
            {
                "expected": ALGORITHM_VERSION,
                "actual": cache.metadata.get("algorithm_version"),
            },
        )
    geometry_hash = compute_geometry_hash(scene)
    layout_hash = compute_layout_hash(layout)
    metadata_geometry = cache.metadata.get("geometry_hash")
    if metadata_geometry != geometry_hash:
        raise BusinessError(
            ErrorCode.CACHE_GEOMETRY_MISMATCH,
            "Cache geometry hash does not match the current scene",
            {
                "cache_geometry_hash": metadata_geometry,
                "scene_geometry_hash": geometry_hash,
            },
        )
    scene_hash = compute_scene_hash(
        scene,
        layout,
        geometry_hash=geometry_hash,
        layout_hash=layout_hash,
    )
    if cache.metadata.get("scene_hash") != scene_hash:
        raise BusinessError(
            ErrorCode.CACHE_SCENE_MISMATCH,
            "Cache scene hash does not match lighting or bake parameters",
            {
                "cache_scene_hash": cache.metadata.get("scene_hash"),
                "scene_hash": scene_hash,
            },
        )
    metadata_layout = cache.metadata.get("layout")
    if not isinstance(metadata_layout, dict):
        raise BusinessError(
            ErrorCode.CACHE_CORRUPT,
            "Cache metadata has no valid layout descriptor",
        )
    if metadata_layout.get("hash") != layout_hash:
        raise BusinessError(
            ErrorCode.CACHE_LAYOUT_MISMATCH,
            "Cache probe layout hash does not match the current layout",
            {
                "cache_layout_hash": metadata_layout.get("hash"),
                "layout_hash": layout_hash,
            },
        )
    payload_metadata = cache.metadata.get("payload")
    if not isinstance(payload_metadata, dict):
        raise BusinessError(
            ErrorCode.CACHE_CORRUPT,
            "Cache metadata has no valid payload descriptor",
        )
    try:
        payload_layout_count = int(payload_metadata.get("probe_count", -1))
    except (TypeError, ValueError) as exc:
        raise BusinessError(
            ErrorCode.CACHE_CORRUPT,
            "Cache payload probe count is not an integer",
        ) from exc
    if (
        payload_layout_count != layout.probe_count
        or cache.probe_count != layout.probe_count
    ):
        raise BusinessError(
            ErrorCode.CACHE_LAYOUT_MISMATCH,
            "Cache probe count does not match the current layout",
            {
                "cache_probe_count": cache.probe_count,
                "layout_probe_count": layout.probe_count,
                "metadata_probe_count": payload_layout_count,
            },
        )


def peek_cache(path: Path) -> Dict[str, Any]:
    """Read only the header and metadata, used to classify stale caches."""

    source = Path(path)
    try:
        raw = source.read_bytes()
    except FileNotFoundError as exc:
        raise _cache_error(
            ErrorCode.CACHE_NOT_FOUND,
            "Cache file does not exist",
            source,
            reason=str(exc),
        ) from exc
    except OSError as exc:
        raise _cache_error(
            ErrorCode.IO_ERROR,
            "Unable to read cache metadata",
            source,
            reason=str(exc),
        ) from exc
    metadata, _, payload_length, _ = _parse_header_and_metadata(raw, source)
    if _HEADER.size + len(canonical_json_bytes(metadata)) + payload_length != len(raw):
        # A non-canonical but parseable cache is still structurally acceptable;
        # full validation is performed by read_cache when needed.
        pass
    return metadata
