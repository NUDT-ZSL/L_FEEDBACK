"""Scene loading and strict validation."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import (
    EXIT_KEYFRAME_TIME_DUPLICATE,
    EXIT_KEYFRAME_TIME_ORDER,
    EXIT_SCENE_FIELD,
    EXIT_SCENE_IO,
    EXIT_SCENE_JSON,
    EXIT_SCENE_NOT_FOUND,
    KEYFRAME_TIME_DUPLICATE,
    KEYFRAME_TIME_ORDER,
    SCENE_FIELD,
    SCENE_IO,
    SCENE_JSON,
    SCENE_NOT_FOUND,
    BusinessError,
    scene_field_error,
)
from .util import sha256_bytes


Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


@dataclass(frozen=True)
class CameraKeyframe:
    time: float
    position: Vec3
    orientation: Quat
    fov_y_deg: float
    near: float
    far: float


@dataclass(frozen=True)
class DirectionalLight:
    direction: Vec3
    color: Vec3
    intensity: float


@dataclass(frozen=True)
class Fog:
    color: Vec3
    density: float
    extinction: float
    scattering: float
    anisotropy: float
    ambient: float
    steps: int


@dataclass(frozen=True)
class ShadowSettings:
    resolution: int
    bias: float


@dataclass
class Building:
    id: int
    color: Vec3
    vertices: np.ndarray
    faces: tuple[tuple[int, ...], ...]
    centroid: Vec3


@dataclass
class GlassPanel:
    id: int
    color: Vec3
    transmittance: float
    vertices: np.ndarray
    normal: Vec3
    center: Vec3


@dataclass
class Scene:
    source_path: str
    source_sha256: str
    name: str
    width: int
    height: int
    fps: float
    start_time: float
    end_time: float
    keyframes: tuple[CameraKeyframe, ...]
    light: DirectionalLight
    fog: Fog
    shadow: ShadowSettings
    buildings: tuple[Building, ...]
    glass_panels: tuple[GlassPanel, ...]
    frame_times: tuple[float, ...]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(value: Any, path: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if not _is_number(value):
        raise scene_field_error(path, "expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise scene_field_error(path, "expected a finite number")
    if minimum is not None and result < minimum:
        raise scene_field_error(path, f"must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise scene_field_error(path, f"must be <= {maximum}")
    return result


def _integer(value: Any, path: str, *, minimum: int = 1, maximum: int = 2**31 - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise scene_field_error(path, "expected an integer")
    if value < minimum or value > maximum:
        raise scene_field_error(path, f"must be in [{minimum}, {maximum}]")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise scene_field_error(path, "expected a non-empty string")
    return value


def _vector(value: Any, path: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise scene_field_error(path, f"expected an array of {length} numbers")
    return tuple(_number(item, f"{path}[{index}]") for index, item in enumerate(value))


def _color(value: Any, path: str) -> Vec3:
    return tuple(
        _number(component, path, minimum=0.0, maximum=1.0)
        for component in _vector(value, path, 3)
    )  # type: ignore[return-value]


def _normalize(values: tuple[float, ...], path: str) -> tuple[float, ...]:
    length = math.sqrt(sum(value * value for value in values))
    if length <= 1e-12:
        raise scene_field_error(path, "must not be the zero vector")
    return tuple(value / length for value in values)


def _face_normal(points: np.ndarray) -> np.ndarray:
    """Newell normal for a planar polygon."""

    normal = np.zeros(3, dtype=np.float64)
    for index, point in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        normal[0] += (point[1] - nxt[1]) * (point[2] + nxt[2])
        normal[1] += (point[2] - nxt[2]) * (point[0] + nxt[0])
        normal[2] += (point[0] - nxt[0]) * (point[1] + nxt[1])
    length = float(np.linalg.norm(normal))
    if length <= 1e-12:
        raise ValueError("degenerate polygon")
    return normal / length


def _validate_convex_polyhedron(
    vertices: np.ndarray,
    faces: tuple[tuple[int, ...], ...],
    path: str,
) -> Vec3:
    if len(vertices) < 4:
        raise scene_field_error(path, "convex polyhedron needs at least 4 vertices")
    if len(faces) < 4:
        raise scene_field_error(path, "convex polyhedron needs at least 4 faces")

    centroid = vertices.mean(axis=0)
    tolerance = 1e-6
    edge_uses: dict[tuple[int, int], int] = {}

    for face_index, face in enumerate(faces):
        face_path = f"{path}.faces[{face_index}]"
        if len(face) < 3:
            raise scene_field_error(face_path, "face needs at least 3 indices")
        if len(set(face)) != len(face):
            raise scene_field_error(face_path, "face contains a repeated vertex index")
        points = vertices[np.asarray(face, dtype=np.int64)]
        try:
            normal = _face_normal(points)
        except ValueError as exc:
            raise scene_field_error(face_path, str(exc)) from exc

        max_plane_distance = float(np.max(np.abs((points - points[0]) @ normal)))
        if max_plane_distance > 2e-5:
            raise scene_field_error(face_path, "face is not planar")

        sample_negative = False
        sample_positive = False
        for vertex_index, point in enumerate(vertices):
            if vertex_index in face:
                continue
            signed = float((point - points[0]) @ normal)
            if signed < -tolerance:
                sample_negative = True
            elif signed > tolerance:
                sample_positive = True
            if sample_negative and sample_positive:
                raise scene_field_error(
                    face_path,
                    "polyhedron is not convex (face plane splits other vertices)",
                )

        for index, vertex_index in enumerate(face):
            nxt = face[(index + 1) % len(face)]
            edge = (min(vertex_index, nxt), max(vertex_index, nxt))
            edge_uses[edge] = edge_uses.get(edge, 0) + 1

    if any(count != 2 for count in edge_uses.values()):
        raise scene_field_error(path, "polyhedron is not a closed two-manifold")

    return tuple(float(value) for value in centroid)  # type: ignore[return-value]


def _validate_planar_panel(vertices: np.ndarray, path: str) -> tuple[Vec3, Vec3]:
    if len(vertices) < 3:
        raise scene_field_error(path, "glass panel needs at least 3 vertices")
    try:
        normal = _face_normal(vertices)
    except ValueError as exc:
        raise scene_field_error(path, str(exc)) from exc
    max_plane_distance = float(np.max(np.abs((vertices - vertices[0]) @ normal)))
    if max_plane_distance > 2e-5:
        raise scene_field_error(path, "glass panel is not planar")

    center = vertices.mean(axis=0)
    # A fan triangulation is valid only for a convex planar polygon.  Project
    # to the dominant 2D plane and require all edge turns to have one sign.
    axis = int(np.argmax(np.abs(normal)))
    keep = [index for index in range(3) if index != axis]
    projected = vertices[:, keep]
    signs: list[int] = []
    for index in range(len(projected)):
        a = projected[(index + 1) % len(projected)] - projected[index]
        b = projected[(index + 2) % len(projected)] - projected[(index + 1) % len(projected)]
        cross = float(a[0] * b[1] - a[1] * b[0])
        if abs(cross) > 1e-9:
            signs.append(1 if cross > 0.0 else -1)
    if not signs or min(signs) != max(signs):
        raise scene_field_error(path, "glass panel polygon must be strictly convex")
    return (
        tuple(float(value) for value in normal),  # type: ignore[return-value]
        tuple(float(value) for value in center),  # type: ignore[return-value]
    )


def _load_buildings(raw: Any) -> tuple[Building, ...]:
    if not isinstance(raw, list) or not raw:
        raise scene_field_error("buildings", "expected a non-empty array")
    seen_ids: set[int] = set()
    result: list[Building] = []
    for index, item in enumerate(raw):
        path = f"buildings[{index}]"
        if not isinstance(item, dict):
            raise scene_field_error(path, "expected an object")
        try:
            building_id = _integer(item["id"], f"{path}.id", maximum=2**24 - 1)
            color = _color(item["color"], f"{path}.color")
            raw_vertices = item["vertices"]
            raw_faces = item["faces"]
        except KeyError as exc:
            raise scene_field_error(path, f"missing field {exc.args[0]!r}") from exc
        if building_id in seen_ids:
            raise scene_field_error(f"{path}.id", "building IDs must be unique")
        seen_ids.add(building_id)

        if not isinstance(raw_vertices, list) or not raw_vertices:
            raise scene_field_error(f"{path}.vertices", "expected a non-empty array")
        vertices = np.asarray(
            [
                _vector(vertex, f"{path}.vertices[{vertex_index}]", 3)
                for vertex_index, vertex in enumerate(raw_vertices)
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(vertices)):
            raise scene_field_error(f"{path}.vertices", "vertices must be finite")

        if not isinstance(raw_faces, list) or not raw_faces:
            raise scene_field_error(f"{path}.faces", "expected a non-empty array")
        faces: list[tuple[int, ...]] = []
        for face_index, face in enumerate(raw_faces):
            face_path = f"{path}.faces[{face_index}]"
            if not isinstance(face, list) or len(face) < 3:
                raise scene_field_error(face_path, "expected an index array of length >= 3")
            indices = tuple(
                _integer(vertex_index, f"{face_path}[{index}]", minimum=0)
                for index, vertex_index in enumerate(face)
            )
            if any(vertex_index >= len(vertices) for vertex_index in indices):
                raise scene_field_error(face_path, "vertex index is out of range")
            faces.append(indices)
        centroid = _validate_convex_polyhedron(vertices, tuple(faces), path)
        result.append(
            Building(
                id=building_id,
                color=color,
                vertices=vertices,
                faces=tuple(faces),
                centroid=centroid,
            )
        )
    return tuple(result)


def _load_glass(raw: Any) -> tuple[GlassPanel, ...]:
    if not isinstance(raw, list) or not raw:
        raise scene_field_error("glass_panels", "expected a non-empty array")
    seen_ids: set[int] = set()
    result: list[GlassPanel] = []
    for index, item in enumerate(raw):
        path = f"glass_panels[{index}]"
        if not isinstance(item, dict):
            raise scene_field_error(path, "expected an object")
        try:
            panel_id = _integer(item["id"], f"{path}.id")
            color = _color(item["color"], f"{path}.color")
            transmittance = _number(
                item["transmittance"], f"{path}.transmittance", minimum=0.0, maximum=1.0
            )
            raw_vertices = item["vertices"]
        except KeyError as exc:
            raise scene_field_error(path, f"missing field {exc.args[0]!r}") from exc
        if panel_id in seen_ids:
            raise scene_field_error(f"{path}.id", "glass panel IDs must be unique")
        seen_ids.add(panel_id)

        if not isinstance(raw_vertices, list) or len(raw_vertices) < 3:
            raise scene_field_error(f"{path}.vertices", "expected an array with >= 3 vertices")
        vertices = np.asarray(
            [
                _vector(vertex, f"{path}.vertices[{vertex_index}]", 3)
                for vertex_index, vertex in enumerate(raw_vertices)
            ],
            dtype=np.float64,
        )
        normal, center = _validate_planar_panel(vertices, path)
        result.append(
            GlassPanel(
                id=panel_id,
                color=color,
                transmittance=transmittance,
                vertices=vertices,
                normal=normal,
                center=center,
            )
        )
    return tuple(result)


def _load_keyframes(raw: Any) -> tuple[CameraKeyframe, ...]:
    if not isinstance(raw, list) or not raw:
        raise scene_field_error("camera_keyframes", "expected a non-empty array")
    keyframes: list[CameraKeyframe] = []
    for index, item in enumerate(raw):
        path = f"camera_keyframes[{index}]"
        if not isinstance(item, dict):
            raise scene_field_error(path, "expected an object")
        try:
            time = _number(item["time"], f"{path}.time")
            position = _vector(item["position"], f"{path}.position", 3)
            orientation = _vector(item["orientation"], f"{path}.orientation", 4)
            fov_y_deg = _number(
                item["fov_y_deg"], f"{path}.fov_y_deg", minimum=1.0, maximum=179.0
            )
            near = _number(item["near"], f"{path}.near", minimum=1e-4)
            far = _number(item["far"], f"{path}.far", minimum=near + 1e-4)
        except KeyError as exc:
            raise scene_field_error(path, f"missing field {exc.args[0]!r}") from exc
        orientation = _normalize(orientation, f"{path}.orientation")
        keyframes.append(
            CameraKeyframe(
                time=time,
                position=position,
                orientation=orientation,  # type: ignore[arg-type]
                fov_y_deg=fov_y_deg,
                near=near,
                far=far,
            )
        )

    for index in range(1, len(keyframes)):
        previous = keyframes[index - 1].time
        current = keyframes[index].time
        if current == previous:
            raise BusinessError(
                KEYFRAME_TIME_DUPLICATE,
                "camera keyframe timestamps must be unique and strictly increasing",
                EXIT_KEYFRAME_TIME_DUPLICATE,
                {
                    "field": f"camera_keyframes[{index}].time",
                    "previous_time": previous,
                    "current_time": current,
                },
            )
        if current < previous:
            raise BusinessError(
                KEYFRAME_TIME_ORDER,
                "camera keyframe timestamps must be strictly increasing",
                EXIT_KEYFRAME_TIME_ORDER,
                {
                    "field": f"camera_keyframes[{index}].time",
                    "previous_time": previous,
                    "current_time": current,
                },
            )
    return tuple(keyframes)


def _frame_times(keyframes: tuple[CameraKeyframe, ...], fps: float, start: float, end: float) -> tuple[float, ...]:
    if start < keyframes[0].time or end > keyframes[-1].time:
        raise scene_field_error(
            "render.start_time/end_time",
            "playback interval must lie inside the keyframe time range",
        )
    if start > end:
        raise scene_field_error("render.start_time", "start_time must be <= end_time")
    span = end - start
    count = int(math.floor(span * fps + 1e-9)) + 1
    if count > 10000:
        raise scene_field_error("render", "frame count exceeds the 10000-frame safety limit")
    times = tuple(start + index / fps for index in range(count))
    if times[-1] > end + 1e-7:
        raise scene_field_error("render.end_time", "frame time exceeds end_time")
    return times


def validate_scene_object(document: Any, source_path: str, source_sha256: str) -> Scene:
    if not isinstance(document, dict):
        raise scene_field_error("$", "scene root must be an object")

    if _integer(document.get("schema_version"), "schema_version") != 1:
        raise scene_field_error("schema_version", "only schema_version 1 is supported")
    name = _string(document.get("name"), "name")

    render = document.get("render")
    if not isinstance(render, dict):
        raise scene_field_error("render", "expected an object")
    width = _integer(render.get("width"), "render.width", maximum=4096)
    height = _integer(render.get("height"), "render.height", maximum=4096)
    fps = _number(render.get("fps"), "render.fps", minimum=1e-3, maximum=240.0)
    start_time = _number(render.get("start_time"), "render.start_time")
    end_time = _number(render.get("end_time"), "render.end_time")

    keyframes = _load_keyframes(document.get("camera_keyframes"))
    frame_times = _frame_times(keyframes, fps, start_time, end_time)

    light = document.get("directional_light")
    if not isinstance(light, dict):
        raise scene_field_error("directional_light", "expected an object")
    light_direction = _normalize(
        _vector(light.get("direction"), "directional_light.direction", 3),
        "directional_light.direction",
    )
    light_color = _color(light.get("color"), "directional_light.color")
    light_intensity = _number(
        light.get("intensity"), "directional_light.intensity", minimum=0.0
    )

    fog_raw = document.get("fog")
    if not isinstance(fog_raw, dict):
        raise scene_field_error("fog", "expected an object")
    fog = Fog(
        color=_color(fog_raw.get("color"), "fog.color"),
        density=_number(fog_raw.get("density"), "fog.density", minimum=0.0),
        extinction=_number(fog_raw.get("extinction"), "fog.extinction", minimum=0.0),
        scattering=_number(
            fog_raw.get("scattering"), "fog.scattering", minimum=0.0, maximum=1.0
        ),
        anisotropy=_number(
            fog_raw.get("anisotropy"), "fog.anisotropy", minimum=-0.99, maximum=0.99
        ),
        ambient=_number(fog_raw.get("ambient"), "fog.ambient", minimum=0.0),
        steps=_integer(fog_raw.get("steps"), "fog.steps", minimum=4, maximum=256),
    )

    shadow_raw = document.get("shadow")
    if not isinstance(shadow_raw, dict):
        raise scene_field_error("shadow", "expected an object")
    shadow = ShadowSettings(
        resolution=_integer(
            shadow_raw.get("resolution"), "shadow.resolution", minimum=64, maximum=4096
        ),
        bias=_number(shadow_raw.get("bias"), "shadow.bias", minimum=0.0),
    )

    buildings = _load_buildings(document.get("buildings"))
    panels = _load_glass(document.get("glass_panels"))
    return Scene(
        source_path=source_path,
        source_sha256=source_sha256,
        name=name,
        width=width,
        height=height,
        fps=fps,
        start_time=start_time,
        end_time=end_time,
        keyframes=keyframes,
        light=DirectionalLight(
            direction=light_direction,  # type: ignore[arg-type]
            color=light_color,
            intensity=light_intensity,
        ),
        fog=fog,
        shadow=shadow,
        buildings=buildings,
        glass_panels=panels,
        frame_times=frame_times,
    )


def load_scene(path: str) -> Scene:
    if not os.path.isfile(path):
        raise BusinessError(
            SCENE_NOT_FOUND,
            "scene JSON file does not exist",
            EXIT_SCENE_NOT_FOUND,
            {"path": path},
        )
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise BusinessError(
            SCENE_IO,
            f"could not read scene JSON: {exc}",
            EXIT_SCENE_IO,
            {"path": path},
        ) from exc

    try:
        document = json.loads(raw.decode("utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise BusinessError(
            SCENE_JSON,
            "scene JSON is not valid UTF-8",
            EXIT_SCENE_JSON,
            {"path": path},
        ) from exc
    except json.JSONDecodeError as exc:
        raise BusinessError(
            SCENE_JSON,
            f"scene JSON syntax error: {exc.msg}",
            EXIT_SCENE_JSON,
            {"path": path, "line": exc.lineno, "column": exc.colno},
        ) from exc

    return validate_scene_object(document, path, sha256_bytes(raw))


def scene_public_metadata(scene: Scene) -> dict[str, Any]:
    return {
        "name": scene.name,
        "sha256": scene.source_sha256,
        "resolution": [scene.width, scene.height],
        "fps": scene.fps,
        "start_time": scene.start_time,
        "end_time": scene.end_time,
        "frame_count": len(scene.frame_times),
        "keyframe_count": len(scene.keyframes),
        "building_count": len(scene.buildings),
        "glass_panel_count": len(scene.glass_panels),
    }


__all__ = [
    "Building",
    "BusinessError",
    "CameraKeyframe",
    "DirectionalLight",
    "Fog",
    "GlassPanel",
    "Scene",
    "ShadowSettings",
    "load_scene",
    "scene_public_metadata",
    "validate_scene_object",
    "SCENE_FIELD",
    "EXIT_SCENE_FIELD",
]

