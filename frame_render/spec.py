from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .constants import (
    RENDERER_VERSION,
    SAMPLER_VERSION,
    SCHEMA_VERSION,
)
from .jsonio import JsonFormatError, canonical_dumps, load_json, sha256_text, sha256_value


class SpecError(ValueError):
    pass


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError(f"{label} must be an object")
    return value


def _check_keys(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise SpecError(f"{label} has unknown key(s): {', '.join(unknown)}")


def _integer(value: Any, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise SpecError(f"{label} must be >= {minimum}")
    return value


def _number(value: Any, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise SpecError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise SpecError(f"{label} must be >= {minimum}")
    return result


def _vector3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise SpecError(f"{label} must be an array of three numbers")
    return tuple(_number(component, f"{label}[{index}]") for index, component in enumerate(value))  # type: ignore[return-value]


def _color3(value: Any, label: str) -> tuple[float, float, float]:
    result = _vector3(value, label)
    for index, component in enumerate(result):
        if component < 0.0 or component > 1.0:
            raise SpecError(f"{label}[{index}] must be in [0, 1]")
    return result


@dataclass(frozen=True)
class Sphere:
    sphere_id: str
    center: tuple[float, float, float]
    radius: float
    albedo: tuple[float, float, float]
    specular: float
    shininess: float

    def canonical(self) -> dict[str, Any]:
        return {
            "id": self.sphere_id,
            "center": list(self.center),
            "radius": self.radius,
            "albedo": list(self.albedo),
            "specular": self.specular,
            "shininess": self.shininess,
        }


@dataclass(frozen=True)
class Light:
    position: tuple[float, float, float]
    color: tuple[float, float, float]
    intensity: float

    def canonical(self) -> dict[str, Any]:
        return {
            "position": list(self.position),
            "color": list(self.color),
            "intensity": self.intensity,
        }


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fov_y_deg: float
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]

    def canonical(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fov_y_deg": self.fov_y_deg,
            "position": list(self.position),
            "target": list(self.target),
            "up": list(self.up),
        }


@dataclass(frozen=True)
class FrameState:
    frame: int
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    seed: int

    def canonical(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "position": list(self.position),
            "target": list(self.target),
            "up": list(self.up),
            "seed": self.seed,
        }


@dataclass(frozen=True)
class SamplingConfig:
    samples_per_axis: int
    jitter: bool
    seed_algorithm: str

    def canonical(self) -> dict[str, Any]:
        return {
            "samples_per_axis": self.samples_per_axis,
            "jitter": self.jitter,
            "seed_algorithm": self.seed_algorithm,
        }


@dataclass(frozen=True)
class Scene:
    spheres: tuple[Sphere, ...]
    light: Light
    camera: Camera
    ambient: tuple[float, float, float]
    background: tuple[float, float, float]

    def canonical(self) -> dict[str, Any]:
        return {
            "spheres": [sphere.canonical() for sphere in self.spheres],
            "light": self.light.canonical(),
            "camera": self.camera.canonical(),
            "ambient": list(self.ambient),
            "background": list(self.background),
        }


@dataclass(frozen=True)
class ChunkSpec:
    chunk_id: str
    frame_start: int
    frame_end: int
    frames: tuple[FrameState, ...]
    geometry_digest: str
    sampling_digest: str
    fingerprint: str


@dataclass(frozen=True)
class RenderSpec:
    source_path: Path
    scene: Scene
    sampling: SamplingConfig
    frames: tuple[FrameState, ...]
    frames_by_number: dict[int, FrameState]
    geometry_digest: str
    sampling_digest: str
    frame_start: int
    frame_end: int

    @property
    def width(self) -> int:
        return self.scene.camera.width

    @property
    def height(self) -> int:
        return self.scene.camera.height

    def chunks(self, chunk_size: int) -> list[ChunkSpec]:
        if chunk_size <= 0:
            raise SpecError("chunk_size must be positive")
        result: list[ChunkSpec] = []
        cursor = 0
        while cursor < len(self.frames):
            selected = self.frames[cursor : cursor + chunk_size]
            result.append(self._make_chunk(selected))
            cursor += chunk_size
        return result

    def chunk_for_range(self, frame_start: int, frame_end: int) -> ChunkSpec:
        selected = tuple(
            frame
            for frame in self.frames
            if frame_start <= frame.frame <= frame_end
        )
        if not selected or selected[0].frame != frame_start or selected[-1].frame != frame_end:
            raise SpecError(
                f"frame range {frame_start}..{frame_end} is not represented in input"
            )
        if any(b.frame != a.frame + 1 for a, b in zip(selected, selected[1:])):
            raise SpecError(f"frame range {frame_start}..{frame_end} is not contiguous")
        return self._make_chunk(selected)

    def _make_chunk(self, selected: Iterable[FrameState]) -> ChunkSpec:
        frames = tuple(selected)
        frame_start = frames[0].frame
        frame_end = frames[-1].frame
        frame_payload = [frame.canonical() for frame in frames]
        fingerprint_payload = {
            "schema_version": SCHEMA_VERSION,
            "renderer_version": RENDERER_VERSION,
            "sampler_version": SAMPLER_VERSION,
            "geometry_digest": self.geometry_digest,
            "sampling_digest": self.sampling_digest,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frames": frame_payload,
        }
        return ChunkSpec(
            chunk_id=make_chunk_id(frame_start, frame_end),
            frame_start=frame_start,
            frame_end=frame_end,
            frames=frames,
            geometry_digest=self.geometry_digest,
            sampling_digest=self.sampling_digest,
            fingerprint=sha256_text(canonical_dumps(fingerprint_payload)),
        )


def make_chunk_id(frame_start: int, frame_end: int) -> str:
    return f"chunk_{frame_start:06d}_{frame_end:06d}"


def load_spec(path: str | Path) -> RenderSpec:
    source_path = Path(path).resolve()
    try:
        document = load_json(source_path)
    except JsonFormatError as exc:
        raise SpecError(str(exc)) from exc
    root = _mapping(document, "document")
    _check_keys(
        root,
        {"schema_version", "scene", "animation", "sampling", "metadata"},
        "document",
    )
    version = _integer(root.get("schema_version"), "schema_version")
    if version != SCHEMA_VERSION:
        raise SpecError(
            f"unsupported schema_version {version}; expected {SCHEMA_VERSION}"
        )

    scene_data = _mapping(root.get("scene"), "scene")
    _check_keys(
        scene_data,
        {"spheres", "light", "camera", "ambient", "background"},
        "scene",
    )
    spheres_raw = scene_data.get("spheres")
    if not isinstance(spheres_raw, list) or not spheres_raw:
        raise SpecError("scene.spheres must be a non-empty array")
    spheres: list[Sphere] = []
    seen_ids: set[str] = set()
    for index, raw_sphere in enumerate(spheres_raw):
        item = _mapping(raw_sphere, f"scene.spheres[{index}]")
        _check_keys(
            item,
            {"id", "center", "radius", "albedo", "color", "specular", "shininess"},
            f"scene.spheres[{index}]",
        )
        sphere_id = item.get("id", f"sphere-{index}")
        if not isinstance(sphere_id, str) or not sphere_id:
            raise SpecError(f"scene.spheres[{index}].id must be a non-empty string")
        if sphere_id in seen_ids:
            raise SpecError(f"duplicate sphere id: {sphere_id!r}")
        seen_ids.add(sphere_id)
        if ("albedo" in item) == ("color" in item):
            raise SpecError(
                f"scene.spheres[{index}] must contain exactly one of albedo or color"
            )
        albedo = _color3(
            item.get("albedo", item.get("color")),
            f"scene.spheres[{index}].albedo",
        )
        sphere = Sphere(
            sphere_id=sphere_id,
            center=_vector3(item.get("center"), f"scene.spheres[{index}].center"),
            radius=_number(
                item.get("radius"), f"scene.spheres[{index}].radius", minimum=0.0
            ),
            albedo=albedo,
            specular=_number(
                item.get("specular", 0.25),
                f"scene.spheres[{index}].specular",
                minimum=0.0,
            ),
            shininess=_number(
                item.get("shininess", 32.0),
                f"scene.spheres[{index}].shininess",
                minimum=1.0,
            ),
        )
        if sphere.radius <= 0.0:
            raise SpecError(f"scene.spheres[{index}].radius must be > 0")
        spheres.append(sphere)

    light_data = _mapping(scene_data.get("light"), "scene.light")
    _check_keys(light_data, {"position", "color", "intensity"}, "scene.light")
    light = Light(
        position=_vector3(light_data.get("position"), "scene.light.position"),
        color=_color3(light_data.get("color", [1.0, 1.0, 1.0]), "scene.light.color"),
        intensity=_number(
            light_data.get("intensity", 1.0), "scene.light.intensity", minimum=0.0
        ),
    )

    camera_data = _mapping(scene_data.get("camera"), "scene.camera")
    _check_keys(
        camera_data,
        {"width", "height", "fov_y_deg", "position", "target", "up"},
        "scene.camera",
    )
    width = _integer(camera_data.get("width"), "scene.camera.width", minimum=1)
    height = _integer(camera_data.get("height"), "scene.camera.height", minimum=1)
    fov = _number(
        camera_data.get("fov_y_deg"), "scene.camera.fov_y_deg", minimum=0.0
    )
    if not 0.0 < fov < 180.0:
        raise SpecError("scene.camera.fov_y_deg must be in (0, 180)")
    camera = Camera(
        width=width,
        height=height,
        fov_y_deg=fov,
        position=_vector3(camera_data.get("position"), "scene.camera.position"),
        target=_vector3(camera_data.get("target"), "scene.camera.target"),
        up=_vector3(camera_data.get("up", [0.0, 1.0, 0.0]), "scene.camera.up"),
    )
    scene = Scene(
        spheres=tuple(spheres),
        light=light,
        camera=camera,
        ambient=_color3(
            scene_data.get("ambient", [0.04, 0.04, 0.04]), "scene.ambient"
        ),
        background=_color3(
            scene_data.get("background", [0.0, 0.0, 0.0]), "scene.background"
        ),
    )

    animation = _mapping(root.get("animation"), "animation")
    _check_keys(
        animation, {"frame_start", "frame_end", "frames"}, "animation"
    )
    frame_start = _integer(animation.get("frame_start"), "animation.frame_start", 0)
    frame_end = _integer(animation.get("frame_end"), "animation.frame_end", 0)
    if frame_end < frame_start:
        raise SpecError("animation.frame_end must be >= animation.frame_start")
    raw_frames = animation.get("frames")
    if not isinstance(raw_frames, list):
        raise SpecError("animation.frames must be an array")
    expected_count = frame_end - frame_start + 1
    if len(raw_frames) != expected_count:
        raise SpecError(
            f"animation.frames has {len(raw_frames)} entries; expected {expected_count}"
        )

    frames: list[FrameState] = []
    seen_frames: set[int] = set()
    for index, raw_frame in enumerate(raw_frames):
        item = _mapping(raw_frame, f"animation.frames[{index}]")
        _check_keys(
            item, {"frame", "position", "target", "up", "seed"}, f"animation.frames[{index}]"
        )
        frame_number = _integer(
            item.get("frame"), f"animation.frames[{index}].frame", 0
        )
        expected_number = frame_start + index
        if frame_number != expected_number:
            raise SpecError(
                f"animation.frames[{index}].frame is {frame_number}; expected {expected_number}"
            )
        if frame_number in seen_frames:
            raise SpecError(f"duplicate frame number: {frame_number}")
        seen_frames.add(frame_number)
        seed = _integer(
            item.get("seed"), f"animation.frames[{index}].seed", 0
        )
        if seed > 0xFFFFFFFFFFFFFFFF:
            raise SpecError(f"animation.frames[{index}].seed must fit in uint64")
        frames.append(
            FrameState(
                frame=frame_number,
                position=_vector3(
                    item.get("position"), f"animation.frames[{index}].position"
                ),
                target=_vector3(
                    item.get("target"), f"animation.frames[{index}].target"
                ),
                up=_vector3(
                    item.get("up", camera.up), f"animation.frames[{index}].up"
                ),
                seed=seed,
            )
        )

    sampling_data = _mapping(root.get("sampling"), "sampling")
    _check_keys(
        sampling_data,
        {"samples_per_axis", "jitter", "seed_algorithm"},
        "sampling",
    )
    samples_per_axis = _integer(
        sampling_data.get("samples_per_axis"),
        "sampling.samples_per_axis",
        minimum=1,
    )
    jitter = sampling_data.get("jitter", True)
    if not isinstance(jitter, bool):
        raise SpecError("sampling.jitter must be a boolean")
    seed_algorithm = sampling_data.get("seed_algorithm", SAMPLER_VERSION)
    if seed_algorithm != SAMPLER_VERSION:
        raise SpecError(
            f"unsupported sampling.seed_algorithm {seed_algorithm!r}; "
            f"expected {SAMPLER_VERSION!r}"
        )
    sampling = SamplingConfig(
        samples_per_axis=samples_per_axis,
        jitter=jitter,
        seed_algorithm=seed_algorithm,
    )

    geometry_digest = sha256_value(scene.canonical())
    sampling_digest = sha256_value(sampling.canonical())
    return RenderSpec(
        source_path=source_path,
        scene=scene,
        sampling=sampling,
        frames=tuple(frames),
        frames_by_number={frame.frame: frame for frame in frames},
        geometry_digest=geometry_digest,
        sampling_digest=sampling_digest,
        frame_start=frame_start,
        frame_end=frame_end,
    )

