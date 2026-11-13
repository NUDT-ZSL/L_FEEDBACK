"""Versioned, checksummed, atomically replaced probe snapshots."""

from __future__ import annotations

import json
import hashlib
import math
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .canonical import canonical_bytes, canonical_json, sha256_bytes
from .layout import LayoutParams, ProbeLayout
from .math3d import Vec3
from .scene import Scene

MAGIC = b"RUINBK1\0"
FORMAT_VERSION = 1
PREFIX = struct.Struct("<8sIQQ")
RECORD = struct.Struct("<II3d3d3d")
RECORD_STRIDE = RECORD.size
FOOTER_SIZE = 32
SNAPSHOT_FORMAT = "ruin-bake-snapshot"
_TEST_PAUSE_DONE = False


class SnapshotError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ProbeRecord:
    probe_id: int
    position: Vec3
    normal: Vec3
    irradiance: Vec3

    def key(self) -> str:
        return struct.pack(
            "<6d",
            self.position[0],
            self.position[1],
            self.position[2],
            self.normal[0],
            self.normal[1],
            self.normal[2],
        ).hex()


@dataclass(frozen=True, slots=True)
class Snapshot:
    path: Path
    header: dict[str, Any]
    scene: Scene
    probes: tuple[ProbeRecord, ...]
    checksum: str
    file_size: int

    @property
    def revision(self) -> int:
        return int(self.header["revision"])

    @property
    def scene_hash(self) -> str:
        return str(self.header["scene_hash"])

    @property
    def layout_hash(self) -> str:
        return str(self.header["layout_hash"])

    @property
    def probe_count(self) -> int:
        return len(self.probes)

    @property
    def culled_probe_count(self) -> int:
        return int(self.header["counts"]["culled_probe_count"])

    @property
    def generated_probe_count(self) -> int:
        return int(self.header["counts"]["generated_probe_count"])

    @property
    def layout_params(self) -> LayoutParams:
        return LayoutParams.from_dict(self.header["layout_params"])

    @property
    def evaluation(self) -> dict[str, Any]:
        return dict(self.header["evaluation"])

    def probes_by_key(self) -> dict[str, ProbeRecord]:
        return {record.key(): record for record in self.probes}

    def summary(self) -> dict[str, Any]:
        scene = self.scene
        return {
            "ok": True,
            "command": "summary",
            "snapshot_path": str(self.path),
            "snapshot_version": FORMAT_VERSION,
            "snapshot_revision": self.revision,
            "snapshot_checksum": self.checksum,
            "snapshot_bytes": self.file_size,
            "scene_hash": self.scene_hash,
            "layout_hash": self.layout_hash,
            "probe_count": self.probe_count,
            "generated_probe_count": self.generated_probe_count,
            "culled_probe_count": self.culled_probe_count,
            "layout_params": self.header["layout_params"],
            "light_params": {
                "directional": scene.directional.to_dict(),
                "point_lights": [light.to_dict() for light in scene.point_lights],
            },
            "evaluation": self.header["evaluation"],
            "probe_directory": self.header["probe_directory"],
        }


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SnapshotError(f"duplicate JSON key in snapshot header: {key}")
        result[key] = value
    return result


def _finite_vec3(value: Any, label: str) -> Vec3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise SnapshotError(f"{label} must contain three numbers")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise SnapshotError(f"{label} contains a non-finite value")
    return result


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write while creating snapshot")
        view = view[written:]


def _test_pause_after_temp_write(bytes_written: int) -> None:
    """Test-only hook; it never runs unless explicitly enabled by the harness."""

    global _TEST_PAUSE_DONE
    threshold_text = os.environ.get("RUIN_BAKE_TEST_HOLD_AFTER_TEMP_BYTES")
    if threshold_text is None or _TEST_PAUSE_DONE:
        return
    try:
        threshold = int(threshold_text)
        hold_ms = int(os.environ.get("RUIN_BAKE_TEST_HOLD_MS", "15000"))
    except ValueError as exc:
        raise SnapshotError("invalid snapshot crash-test environment variable") from exc
    if threshold >= 0 and bytes_written >= threshold:
        _TEST_PAUSE_DONE = True
        time.sleep(max(0, hold_ms) / 1000.0)


def _build_header(
    *,
    scene: Scene,
    layout: ProbeLayout,
    evaluation: dict[str, Any],
    revision: int,
    payload: bytes,
    parent: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "format": SNAPSHOT_FORMAT,
        "snapshot_version": FORMAT_VERSION,
        "revision": revision,
        "scene_hash": scene.scene_hash,
        "layout_hash": layout.layout_hash,
        "layout_params": layout.params.to_dict(),
        "evaluation": dict(evaluation),
        "counts": {
            "probe_count": len(layout.probes),
            "generated_probe_count": layout.generated_count,
            "culled_probe_count": layout.culled_count,
        },
        "scene": scene.to_dict(),
        "components": {
            "geometry_hashes": scene.geometry_hashes(),
            "light_hashes": scene.light_hashes(),
        },
        "probe_directory": {
            "records_offset": 0,
            "record_count": len(layout.probes),
            "record_stride": RECORD_STRIDE,
            "encoding": "<II3d3d3d",
        },
        "payload": {
            "length": len(payload),
            "sha256": sha256_bytes(payload),
        },
        "parent": parent,
    }


def _encode_payload(layout: ProbeLayout, irradiances: Sequence[Vec3]) -> bytes:
    if len(irradiances) != len(layout.probes):
        raise SnapshotError("irradiance count does not match probe layout")
    chunks: list[bytes] = []
    for probe, irradiance in zip(layout.probes, irradiances, strict=True):
        values = (*probe.position, *probe.normal, *irradiance)
        if not all(math.isfinite(value) for value in values):
            raise SnapshotError(f"probe {probe.probe_id} contains non-finite data")
        chunks.append(RECORD.pack(probe.probe_id, 0, *values))
    return b"".join(chunks)


def build_snapshot_bytes(
    *,
    scene: Scene,
    layout: ProbeLayout,
    irradiances: Sequence[Vec3],
    evaluation: dict[str, Any],
    revision: int,
    parent: dict[str, Any] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    payload = _encode_payload(layout, irradiances)
    header = _build_header(
        scene=scene,
        layout=layout,
        evaluation=evaluation,
        revision=revision,
        payload=payload,
        parent=parent,
    )
    header_bytes = canonical_bytes(header)
    prefix = PREFIX.pack(MAGIC, FORMAT_VERSION, len(header_bytes), len(payload))
    body = prefix + header_bytes + payload
    digest = hashlib.sha256(body).digest()
    return body + digest, header


def write_snapshot_atomic(
    path: str | os.PathLike[str],
    *,
    scene: Scene,
    layout: ProbeLayout,
    irradiances: Sequence[Vec3],
    evaluation: dict[str, Any],
    revision: int,
    parent: dict[str, Any] | None = None,
) -> Snapshot:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    body, _header = build_snapshot_bytes(
        scene=scene,
        layout=layout,
        irradiances=irradiances,
        evaluation=evaluation,
        revision=revision,
        parent=parent,
    )

    temp_path = destination.with_name(destination.name + ".tmp")
    try:
        temp_path.unlink()
    except FileNotFoundError:
        pass

    fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        try:
            write_chunk_size = int(os.environ.get("RUIN_BAKE_TEST_WRITE_CHUNK_BYTES", str(64 * 1024)))
        except ValueError as exc:
            raise SnapshotError("invalid snapshot crash-test write chunk size") from exc
        if write_chunk_size <= 0:
            raise SnapshotError("snapshot crash-test write chunk size must be positive")
        total = 0
        for start in range(0, len(body), write_chunk_size):
            chunk = body[start : start + write_chunk_size]
            _write_all(fd, chunk)
            total += len(chunk)
            _test_pause_after_temp_write(total)
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        raise
    else:
        os.close(fd)

    # The destination name changes atomically.  A SIGKILL before this point
    # leaves the previous complete snapshot at `destination` untouched.
    os.replace(temp_path, destination)
    try:
        directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        directory_fd = None
    if directory_fd is not None:
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
        finally:
            os.close(directory_fd)
    return load_snapshot(destination)


def load_snapshot(path: str | os.PathLike[str]) -> Snapshot:
    snapshot_path = Path(path)
    try:
        file_size = snapshot_path.stat().st_size
        data = snapshot_path.read_bytes()
    except OSError as exc:
        raise SnapshotError(f"cannot read snapshot {snapshot_path}: {exc}") from exc

    minimum = PREFIX.size + FOOTER_SIZE
    if file_size < minimum:
        raise SnapshotError("snapshot is truncated before its prefix/footer")
    if len(data) != file_size:
        raise SnapshotError("snapshot changed size while being read")
    try:
        magic, file_version, header_len, payload_len = PREFIX.unpack_from(data, 0)
    except struct.error as exc:
        raise SnapshotError("snapshot prefix is invalid") from exc
    if magic != MAGIC:
        raise SnapshotError("snapshot magic does not match")
    if file_version != FORMAT_VERSION:
        raise SnapshotError(f"unsupported snapshot format version: {file_version}")

    expected_size = PREFIX.size + header_len + payload_len + FOOTER_SIZE
    if expected_size != file_size:
        raise SnapshotError("snapshot length fields do not match file size")

    header_start = PREFIX.size
    header_end = header_start + header_len
    payload_end = header_end + payload_len
    header_bytes = data[header_start:header_end]
    payload = data[header_end:payload_end]
    file_digest = data[payload_end:]

    body = data[:payload_end]
    calculated_digest = sha256_bytes(body)
    checksum = file_digest.hex()
    if calculated_digest != checksum:
        raise SnapshotError("snapshot footer checksum mismatch")

    try:
        decoded = header_bytes.decode("utf-8")
        header = json.loads(decoded, object_pairs_hook=_no_duplicate_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"snapshot header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise SnapshotError("snapshot header must be an object")
    if header.get("format") != SNAPSHOT_FORMAT:
        raise SnapshotError("snapshot header format marker is invalid")
    if header.get("snapshot_version") != FORMAT_VERSION:
        raise SnapshotError("snapshot header version does not match prefix")

    payload_info = header.get("payload")
    if not isinstance(payload_info, dict) or payload_info.get("sha256") != sha256_bytes(payload):
        raise SnapshotError("snapshot payload checksum mismatch")
    if payload_info.get("length") != payload_len:
        raise SnapshotError("snapshot payload length does not match header")

    scene_value = header.get("scene")
    if not isinstance(scene_value, dict):
        raise SnapshotError("snapshot does not contain a scene")
    try:
        scene = Scene.from_dict(scene_value)
    except (ValueError, TypeError) as exc:
        raise SnapshotError(f"snapshot scene is invalid: {exc}") from exc
    if scene.scene_hash != header.get("scene_hash"):
        raise SnapshotError("scene content hash does not match snapshot header")

    layout_value = header.get("layout_params")
    try:
        layout_params = LayoutParams.from_dict(layout_value)
    except (ValueError, TypeError) as exc:
        raise SnapshotError(f"snapshot layout parameters are invalid: {exc}") from exc
    if layout_params.layout_hash(scene.bounds_min, scene.bounds_max) != header.get("layout_hash"):
        raise SnapshotError("layout parameter hash does not match snapshot header")

    counts = header.get("counts")
    directory = header.get("probe_directory")
    if not isinstance(counts, dict) or not isinstance(directory, dict):
        raise SnapshotError("snapshot counts or probe directory is missing")
    probe_count = counts.get("probe_count")
    if isinstance(probe_count, bool) or not isinstance(probe_count, int) or probe_count < 0:
        raise SnapshotError("snapshot probe_count is invalid")
    generated = counts.get("generated_probe_count")
    culled = counts.get("culled_probe_count")
    if not isinstance(generated, int) or not isinstance(culled, int):
        raise SnapshotError("snapshot probe counts are invalid")
    if generated < probe_count or culled < 0 or generated != probe_count + culled:
        raise SnapshotError("snapshot probe counts are internally inconsistent")

    try:
        records_offset = int(directory["records_offset"])
        record_count = int(directory["record_count"])
        record_stride = int(directory["record_stride"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotError("snapshot probe directory fields are invalid") from exc
    if directory.get("encoding") != "<II3d3d3d":
        raise SnapshotError("snapshot probe record encoding is unsupported")
    if record_stride != RECORD_STRIDE:
        raise SnapshotError("snapshot probe record stride is invalid")
    if record_count != probe_count:
        raise SnapshotError("snapshot probe directory count is inconsistent")
    if records_offset != 0 or records_offset + record_count * record_stride != payload_len:
        raise SnapshotError("snapshot probe directory does not exactly cover its payload")

    records: list[ProbeRecord] = []
    for index in range(record_count):
        offset = records_offset + index * record_stride
        try:
            probe_id, flags, *values = RECORD.unpack_from(payload, offset)
        except struct.error as exc:
            raise SnapshotError("snapshot probe payload is truncated") from exc
        if flags != 0:
            raise SnapshotError("snapshot probe record flags are unsupported")
        if probe_id != index:
            raise SnapshotError("snapshot probe ids are not contiguous in evaluation order")
        position = tuple(values[0:3])
        normal = tuple(values[3:6])
        irradiance = tuple(values[6:9])
        if not all(math.isfinite(value) for value in values):
            raise SnapshotError(f"snapshot probe {probe_id} contains non-finite data")
        records.append(ProbeRecord(probe_id, position, normal, irradiance))

    components = header.get("components")
    if not isinstance(components, dict):
        raise SnapshotError("snapshot component hashes are missing")
    if components.get("geometry_hashes") != scene.geometry_hashes():
        raise SnapshotError("snapshot geometry component hashes do not match scene")
    if components.get("light_hashes") != scene.light_hashes():
        raise SnapshotError("snapshot light component hashes do not match scene")
    if not isinstance(header.get("evaluation"), dict):
        raise SnapshotError("snapshot evaluation metadata is missing")
    if not isinstance(header.get("revision"), int) or header["revision"] < 1:
        raise SnapshotError("snapshot revision is invalid")

    return Snapshot(
        path=snapshot_path,
        header=header,
        scene=scene,
        probes=tuple(records),
        checksum=checksum,
        file_size=file_size,
    )


def snapshot_header_json(snapshot: Snapshot) -> str:
    return canonical_json(snapshot.header)

