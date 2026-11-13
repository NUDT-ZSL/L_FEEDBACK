"""Versioned, checksummed tile snapshot format."""

from __future__ import annotations

from array import array
import base64
import binascii
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from .errors import E_SNAPSHOT_CORRUPT, E_SNAPSHOT_IO, PipelineError, SnapshotCorruptError
from .gbuffer import GBufferTile
from .scene import Scene, TileSpec, canonical_json, digest
from .shadows import DirectionalShadowMap, PointShadowFace, PointShadowMap, ShadowMaps


SNAPSHOT_FORMAT = "render_pipeline.snapshot"
SNAPSHOT_VERSION = 1


def _encode_float64(values: tuple[float, ...] | list[float]) -> dict[str, Any]:
    packed = array("d", values)
    if sys.byteorder != "little":
        packed.byteswap()
    return {
        "dtype": "float64-le",
        "count": len(values),
        "data": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def _decode_float64(raw: Any, label: str) -> tuple[float, ...]:
    if not isinstance(raw, dict) or raw.get("dtype") != "float64-le":
        raise SnapshotCorruptError(f"{label} is not float64-le encoded")
    count = raw.get("count")
    data = raw.get("data")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0 or not isinstance(data, str):
        raise SnapshotCorruptError(f"{label} has invalid count or data")
    try:
        packed = base64.b64decode(data.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError, binascii.Error) as error:
        raise SnapshotCorruptError(f"{label} base64 data is invalid") from error
    if len(packed) != count * 8:
        raise SnapshotCorruptError(f"{label} byte count does not match")
    values = array("d")
    values.frombytes(packed)
    if sys.byteorder != "little":
        values.byteswap()
    return tuple(float(value) for value in values)


def _encode_int32(values: tuple[int, ...] | list[int]) -> dict[str, Any]:
    packed = array("i", values)
    if packed.itemsize != 4:
        raise RuntimeError("platform int array is not 32-bit")
    if sys.byteorder != "little":
        packed.byteswap()
    return {
        "dtype": "int32-le",
        "count": len(values),
        "data": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def _decode_int32(raw: Any, label: str) -> tuple[int, ...]:
    if not isinstance(raw, dict) or raw.get("dtype") != "int32-le":
        raise SnapshotCorruptError(f"{label} is not int32-le encoded")
    count = raw.get("count")
    data = raw.get("data")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0 or not isinstance(data, str):
        raise SnapshotCorruptError(f"{label} has invalid count or data")
    try:
        packed = base64.b64decode(data.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError, binascii.Error) as error:
        raise SnapshotCorruptError(f"{label} base64 data is invalid") from error
    if len(packed) != count * 4:
        raise SnapshotCorruptError(f"{label} byte count does not match")
    values = array("i")
    values.frombytes(packed)
    if sys.byteorder != "little":
        values.byteswap()
    return tuple(int(value) for value in values)


def shadow_maps_to_payload(shadow_maps: ShadowMaps) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for light_id in sorted(shadow_maps):
        shadow_map = shadow_maps[light_id]
        if isinstance(shadow_map, DirectionalShadowMap):
            payload[light_id] = {
                "kind": "directional",
                "projection": "orthographic",
                "resolution": shadow_map.resolution,
                "eye": list(shadow_map.eye),
                "right": list(shadow_map.right),
                "up": list(shadow_map.up),
                "forward": list(shadow_map.forward),
                "min_x": shadow_map.min_x,
                "max_x": shadow_map.max_x,
                "min_y": shadow_map.min_y,
                "max_y": shadow_map.max_y,
                "min_z": shadow_map.min_z,
                "max_z": shadow_map.max_z,
                "depths": _encode_float64(shadow_map.depths),
            }
        else:
            payload[light_id] = {
                "kind": "point",
                "projection": "cube",
                "position": list(shadow_map.position),
                "far": shadow_map.far,
                "resolution": shadow_map.resolution,
                "faces": [
                    {
                        "forward": list(face.forward),
                        "right": list(face.right),
                        "up": list(face.up),
                        "depths": _encode_float64(face.depths),
                    }
                    for face in shadow_map.faces
                ],
            }
    return payload


def _require_vector3(raw: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise SnapshotCorruptError(f"{label} must have three components")
    result = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise SnapshotCorruptError(f"{label} contains a non-number")
        converted = float(item)
        if not math.isfinite(converted):
            raise SnapshotCorruptError(f"{label} contains a non-finite number")
        result.append(converted)
    return (result[0], result[1], result[2])


def _require_float(raw: Any, label: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise SnapshotCorruptError(f"{label} must be a number")
    converted = float(raw)
    if not math.isfinite(converted):
        raise SnapshotCorruptError(f"{label} must be finite")
    return converted


def shadow_maps_from_payload(payload: Any) -> ShadowMaps:
    if not isinstance(payload, dict):
        raise SnapshotCorruptError("shadows must be an object")
    shadow_maps: ShadowMaps = {}
    for light_id in sorted(payload):
        raw_map = payload[light_id]
        if not isinstance(raw_map, dict):
            raise SnapshotCorruptError(f"shadow {light_id!r} must be an object")
        kind = raw_map.get("kind")
        resolution = raw_map.get("resolution")
        if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution <= 0:
            raise SnapshotCorruptError(f"shadow {light_id!r} resolution is invalid")
        if kind == "directional":
            depths = _decode_float64(raw_map.get("depths"), f"shadow {light_id!r} depths")
            if len(depths) != resolution * resolution:
                raise SnapshotCorruptError(f"shadow {light_id!r} directional map has wrong size")
            if any(not math.isfinite(depth) for depth in depths):
                raise SnapshotCorruptError(f"shadow {light_id!r} contains a non-finite depth")
            shadow_maps[light_id] = DirectionalShadowMap(
                light_id=light_id,
                resolution=resolution,
                eye=_require_vector3(raw_map.get("eye"), "shadow eye"),
                right=_require_vector3(raw_map.get("right"), "shadow right"),
                up=_require_vector3(raw_map.get("up"), "shadow up"),
                forward=_require_vector3(raw_map.get("forward"), "shadow forward"),
                min_x=_require_float(raw_map.get("min_x"), "shadow min_x"),
                max_x=_require_float(raw_map.get("max_x"), "shadow max_x"),
                min_y=_require_float(raw_map.get("min_y"), "shadow min_y"),
                max_y=_require_float(raw_map.get("max_y"), "shadow max_y"),
                min_z=_require_float(raw_map.get("min_z"), "shadow min_z"),
                max_z=_require_float(raw_map.get("max_z"), "shadow max_z"),
                depths=depths,
            )
        elif kind == "point":
            raw_faces = raw_map.get("faces")
            if not isinstance(raw_faces, list) or len(raw_faces) != 6:
                raise SnapshotCorruptError(f"shadow {light_id!r} must contain six cube faces")
            faces = []
            for raw_face in raw_faces:
                if not isinstance(raw_face, dict):
                    raise SnapshotCorruptError(f"shadow {light_id!r} face must be an object")
                depths = _decode_float64(raw_face.get("depths"), f"shadow {light_id!r} face depths")
                if len(depths) != resolution * resolution:
                    raise SnapshotCorruptError(f"shadow {light_id!r} cube face has wrong size")
                if any(not math.isfinite(depth) for depth in depths):
                    raise SnapshotCorruptError(f"shadow {light_id!r} contains a non-finite depth")
                faces.append(
                    PointShadowFace(
                        forward=_require_vector3(raw_face.get("forward"), "face forward"),
                        right=_require_vector3(raw_face.get("right"), "face right"),
                        up=_require_vector3(raw_face.get("up"), "face up"),
                        depths=depths,
                    )
                )
            shadow_maps[light_id] = PointShadowMap(
                light_id=light_id,
                position=_require_vector3(raw_map.get("position"), "point shadow position"),
                far=_require_float(raw_map.get("far"), "point shadow far"),
                resolution=resolution,
                faces=tuple(faces),
            )
        else:
            raise SnapshotCorruptError(f"shadow {light_id!r} has unsupported kind {kind!r}")
    return shadow_maps


@dataclass(frozen=True)
class Snapshot:
    version: int
    tile: TileSpec
    scene: dict[str, Any]
    render: dict[str, Any]
    gbuffer: GBufferTile
    shadow_maps: ShadowMaps
    lights: dict[str, Any]
    dependencies: dict[str, Any]
    checksum: str
    path: str


def _gbuffer_payload(gbuffer: GBufferTile) -> dict[str, Any]:
    return {
        "x": gbuffer.x,
        "y": gbuffer.y,
        "width": gbuffer.width,
        "height": gbuffer.height,
        "albedo": _encode_float64(gbuffer.albedo),
        "normal": _encode_float64(gbuffer.normal),
        "depth": _encode_float64(gbuffer.depth),
        "material_ids": _encode_int32(gbuffer.material_ids),
        "primitive_ids": _encode_int32(gbuffer.primitive_ids),
    }


def _gbuffer_from_payload(raw: Any) -> GBufferTile:
    if not isinstance(raw, dict):
        raise SnapshotCorruptError("gbuffer must be an object")
    values = {}
    for key in ("x", "y", "width", "height"):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise SnapshotCorruptError(f"gbuffer {key} must be an integer")
        values[key] = value
    width = values["width"]
    height = values["height"]
    if width <= 0 or height <= 0:
        raise SnapshotCorruptError("gbuffer dimensions must be positive")
    pixel_count = width * height
    albedo = _decode_float64(raw.get("albedo"), "gbuffer albedo")
    normal = _decode_float64(raw.get("normal"), "gbuffer normal")
    depth = _decode_float64(raw.get("depth"), "gbuffer depth")
    material_ids = _decode_int32(raw.get("material_ids"), "gbuffer material_ids")
    primitive_ids = _decode_int32(raw.get("primitive_ids"), "gbuffer primitive_ids")
    if len(albedo) != pixel_count * 3 or len(normal) != pixel_count * 3:
        raise SnapshotCorruptError("gbuffer color channels have wrong size")
    if len(depth) != pixel_count or len(material_ids) != pixel_count or len(primitive_ids) != pixel_count:
        raise SnapshotCorruptError("gbuffer scalar channels have wrong size")
    if any(not math.isfinite(value) for value in albedo) or any(
        not math.isfinite(value) for value in normal
    ):
        raise SnapshotCorruptError("gbuffer contains a non-finite material or normal value")
    if any(math.isnan(value) or value < 0.0 for value in depth):
        raise SnapshotCorruptError("gbuffer contains an invalid depth")
    if any(material_id < -1 for material_id in material_ids):
        raise SnapshotCorruptError("gbuffer contains an invalid material id")
    return GBufferTile(
        x=values["x"],
        y=values["y"],
        width=width,
        height=height,
        albedo=list(albedo),
        normal=list(normal),
        depth=list(depth),
        material_ids=list(material_ids),
        primitive_ids=list(primitive_ids),
    )


def build_snapshot_payload(
    scene: Scene,
    tile: TileSpec,
    gbuffer: GBufferTile,
    shadow_maps: ShadowMaps,
    material_dependencies: tuple[str, ...],
    light_dependencies: tuple[str, ...],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": SNAPSHOT_FORMAT,
        "version": SNAPSHOT_VERSION,
        "tile": tile.as_dict(),
        "tile_id": tile.tile_id,
        "scene_summary": scene.summary(),
        "scene": scene.snapshot_state(),
        "render": {
            "quality": scene.shadow.quality,
            "shadow": scene.shadow.as_dict(),
            "effective_lights": {
                light.light_id: {
                    "resolution": scene.shadow.effective_resolution(light),
                    "pcf_kernel": scene.shadow.effective_pcf_kernel(light),
                }
                for light in scene.lights
            },
        },
        "gbuffer": _gbuffer_payload(gbuffer),
        "shadows": shadow_maps_to_payload(shadow_maps),
        "lights": {light.light_id: light.state() for light in scene.lights},
        "dependencies": {
            "materials": sorted(material_dependencies),
            "lights": sorted(light_dependencies),
            "has_coverage": gbuffer.has_coverage(),
        },
    }
    return payload


def snapshot_filename(tile: TileSpec) -> str:
    return f"{tile.tile_id}.snapshot.json"


def snapshot_path(snapshot_dir: str | Path, tile: TileSpec) -> Path:
    return Path(snapshot_dir) / snapshot_filename(tile)


def write_snapshot(path: str | Path, payload: dict[str, Any]) -> str:
    destination = Path(path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise PipelineError(
            E_SNAPSHOT_IO,
            f"cannot create snapshot directory {destination.parent}: {error}",
            "snapshot",
        ) from error
    payload_without_checksum = {key: value for key, value in payload.items() if key != "checksum"}
    checksum = "sha256:" + digest(payload_without_checksum)
    complete_payload = dict(payload_without_checksum)
    complete_payload["checksum"] = checksum
    encoded = canonical_json(complete_payload) + "\n"
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=destination.name + ".",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    except OSError as error:
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
        raise PipelineError(E_SNAPSHOT_IO, f"cannot write snapshot {destination}: {error}", "snapshot") from error
    return checksum


def load_snapshot(path: str | Path) -> Snapshot:
    source = Path(path)
    try:
        encoded = source.read_text(encoding="utf-8")
    except OSError as error:
        raise PipelineError(E_SNAPSHOT_IO, f"cannot read snapshot {source}: {error}", "snapshot") from error
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise SnapshotCorruptError(f"snapshot JSON is invalid: {error}") from error
    if not isinstance(payload, dict):
        raise SnapshotCorruptError("snapshot root must be an object")
    if payload.get("format") != SNAPSHOT_FORMAT:
        raise SnapshotCorruptError("snapshot format marker does not match")
    snapshot_version = payload.get("version")
    if isinstance(snapshot_version, bool) or snapshot_version != SNAPSHOT_VERSION:
        raise SnapshotCorruptError(
            f"snapshot version {snapshot_version!r} is not supported (expected {SNAPSHOT_VERSION})"
        )
    expected_checksum = payload.get("checksum")
    if not isinstance(expected_checksum, str):
        raise SnapshotCorruptError("snapshot checksum is missing")
    payload_without_checksum = {key: value for key, value in payload.items() if key != "checksum"}
    try:
        actual_checksum = "sha256:" + digest(payload_without_checksum)
    except (TypeError, ValueError, OverflowError) as error:
        raise SnapshotCorruptError("snapshot content cannot be canonicalized") from error
    if actual_checksum != expected_checksum:
        raise SnapshotCorruptError("snapshot content checksum does not match")
    raw_tile = payload.get("tile")
    if not isinstance(raw_tile, dict):
        raise SnapshotCorruptError("snapshot tile is missing")
    for field in ("index", "x", "y", "width", "height"):
        if isinstance(raw_tile.get(field), bool) or not isinstance(raw_tile.get(field), int):
            raise SnapshotCorruptError(f"snapshot tile {field} must be an integer")
    try:
        tile = TileSpec(
            index=raw_tile["index"],
            x=raw_tile["x"],
            y=raw_tile["y"],
            width=raw_tile["width"],
            height=raw_tile["height"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SnapshotCorruptError("snapshot tile metadata is invalid") from error
    if tile.index < 0 or tile.x < 0 or tile.y < 0 or tile.width <= 0 or tile.height <= 0:
        raise SnapshotCorruptError("snapshot tile metadata has invalid values")
    if payload.get("tile_id") != tile.tile_id:
        raise SnapshotCorruptError("snapshot tile id does not match tile metadata")
    gbuffer = _gbuffer_from_payload(payload.get("gbuffer"))
    if gbuffer.x != tile.x or gbuffer.y != tile.y or gbuffer.width != tile.width or gbuffer.height != tile.height:
        raise SnapshotCorruptError("snapshot tile and G-buffer dimensions differ")
    scene_payload = payload.get("scene")
    render_payload = payload.get("render")
    dependencies = payload.get("dependencies")
    lights = payload.get("lights")
    if not isinstance(scene_payload, dict) or not isinstance(render_payload, dict):
        raise SnapshotCorruptError("snapshot scene or render metadata is missing")
    if payload.get("scene_summary") != scene_payload.get("summary"):
        raise SnapshotCorruptError("snapshot scene summary entries do not match")
    if not isinstance(dependencies, dict) or not isinstance(lights, dict):
        raise SnapshotCorruptError("snapshot dependencies or lights are missing")
    shadow_maps = shadow_maps_from_payload(payload.get("shadows"))
    return Snapshot(
        version=SNAPSHOT_VERSION,
        tile=tile,
        scene=scene_payload,
        render=render_payload,
        gbuffer=gbuffer,
        shadow_maps=shadow_maps,
        lights=lights,
        dependencies=dependencies,
        checksum=expected_checksum,
        path=str(source),
    )


class SnapshotStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def path_for(self, tile: TileSpec) -> Path:
        return snapshot_path(self.directory, tile)

    def save(
        self,
        scene: Scene,
        tile: TileSpec,
        gbuffer: GBufferTile,
        shadow_maps: ShadowMaps,
        material_dependencies: tuple[str, ...],
        light_dependencies: tuple[str, ...],
    ) -> str:
        payload = build_snapshot_payload(
            scene, tile, gbuffer, shadow_maps, material_dependencies, light_dependencies
        )
        return write_snapshot(self.path_for(tile), payload)

    def load(self, tile: TileSpec) -> Snapshot:
        return load_snapshot(self.path_for(tile))
