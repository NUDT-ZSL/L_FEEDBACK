"""Load and validate the deterministic JSON scene configuration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from . import errors


@dataclass
class CrystalSpec:
    crystal_id: str
    layer: int
    vertices: list[tuple[float, float]]
    ior: float
    color: tuple[float, float, float, float]
    translate: tuple[float, float]
    rotate: float
    scale: float
    depth_gradient: tuple[float, float]
    refraction_strength: float
    visible: bool
    keyframes: list[dict]


@dataclass
class SceneConfig:
    width: int
    height: int
    frames: int
    seed: int
    camera: dict
    env: dict
    snapshot_every: int
    crystals: list[CrystalSpec] = field(default_factory=list)


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _polygon_area(vertices: list[tuple[float, float]]) -> float:
    area = 0.0
    count = len(vertices)
    for i in range(count):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % count]
        area += x0 * y1 - x1 * y0
    return 0.5 * area


def _validate_keyframes(raw, frames: int, crystal_id: str, log: errors.ErrorLog) -> list[dict]:
    valid = []
    for kf in raw:
        step = kf.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0 or step >= frames:
            log.emit(errors.E_KEYFRAME_RANGE, crystal_id, f"step={step!r} frames={frames}")
            continue
        action = kf.get("set")
        if not isinstance(action, dict) or not action:
            log.emit(errors.E_KEYFRAME_RANGE, crystal_id, "keyframe requires a non-empty 'set' object")
            continue
        valid.append({"step": step, "set": dict(action)})
    valid.sort(key=lambda k: k["step"])
    return valid


def _parse_crystal(raw: dict, frames: int, log: errors.ErrorLog) -> CrystalSpec | None:
    crystal_id = str(raw.get("id", "<unnamed>"))

    raw_vertices = raw.get("vertices")
    vertices: list[tuple[float, float]] = []
    if isinstance(raw_vertices, list):
        for pt in raw_vertices:
            if (
                isinstance(pt, list)
                and len(pt) == 2
                and _finite(pt[0])
                and _finite(pt[1])
            ):
                vertices.append((float(pt[0]), float(pt[1])))
    if len(vertices) < 3:
        log.emit(errors.E_DEGENERATE_POLYGON, crystal_id, f"vertices={len(vertices)}")
        return None
    if abs(_polygon_area(vertices)) < 1e-9:
        log.emit(errors.E_ZERO_AREA, crystal_id)
        return None

    ior = raw.get("ior")
    if not _finite(ior) or not (1.0 < float(ior) <= 3.0):
        log.emit(errors.E_INVALID_IOR, crystal_id, f"ior={ior!r}")
        return None

    color = raw.get("color")
    if (
        not isinstance(color, list)
        or len(color) != 4
        or not all(_finite(c) and 0.0 <= float(c) <= 1.0 for c in color)
    ):
        log.emit(errors.E_INVALID_COLOR, crystal_id, f"color={color!r}")
        return None

    transform = raw.get("transform", {})
    translate = transform.get("translate", [0.0, 0.0])
    rotate = transform.get("rotate", 0.0)
    scale = transform.get("scale", 1.0)
    if (
        not isinstance(translate, list)
        or len(translate) != 2
        or not all(_finite(v) for v in translate)
        or not _finite(rotate)
        or not _finite(scale)
        or float(scale) == 0.0
    ):
        log.emit(errors.E_INVALID_TRANSFORM, crystal_id, f"transform={transform!r}")
        return None

    gradient = raw.get("depth_gradient", [0.0, 0.0])
    if (
        not isinstance(gradient, list)
        or len(gradient) != 2
        or not all(_finite(v) for v in gradient)
    ):
        log.emit(errors.E_INVALID_TRANSFORM, crystal_id, f"depth_gradient={gradient!r}")
        return None

    strength = raw.get("refraction_strength", 8.0)
    if not _finite(strength) or float(strength) < 0.0:
        log.emit(errors.E_INVALID_TRANSFORM, crystal_id, f"refraction_strength={strength!r}")
        return None

    keyframes = _validate_keyframes(raw.get("keyframes", []), frames, crystal_id, log)

    return CrystalSpec(
        crystal_id=crystal_id,
        layer=int(raw.get("layer", 0)),
        vertices=vertices,
        ior=float(ior),
        color=tuple(float(c) for c in color),
        translate=(float(translate[0]), float(translate[1])),
        rotate=float(rotate),
        scale=float(scale),
        depth_gradient=(float(gradient[0]), float(gradient[1])),
        refraction_strength=float(strength),
        visible=bool(raw.get("visible", True)),
        keyframes=keyframes,
    )


def load_config(path: str, log: errors.ErrorLog) -> SceneConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    try:
        width = int(raw["width"])
        height = int(raw["height"])
        frames = int(raw["frames"])
        seed = int(raw.get("seed", 0))
        if width <= 0 or height <= 0 or frames <= 0:
            raise ValueError("width/height/frames must be positive")
    except (KeyError, TypeError, ValueError) as exc:
        log.emit(errors.E_INVALID_CONFIG, None, str(exc))
        raise SystemExit(2)

    camera = raw.get("camera", {})
    env = raw.get("env", {})
    snapshot_every = int(raw.get("snapshot_every", 10))

    config = SceneConfig(
        width=width,
        height=height,
        frames=frames,
        seed=seed,
        camera={
            "scale": float(camera.get("scale", 1.0)),
            "rotate": float(camera.get("rotate", 0.0)),
            "translate": [float(v) for v in camera.get("translate", [width / 2.0, height / 2.0])],
        },
        env={
            "color_top": [float(v) for v in env.get("color_top", [0.02, 0.03, 0.08])],
            "color_bottom": [float(v) for v in env.get("color_bottom", [0.10, 0.05, 0.16])],
            "stripe_color": [float(v) for v in env.get("stripe_color", [0.16, 0.22, 0.30])],
            "stripe_frequency": float(env.get("stripe_frequency", 0.06)),
            "stripe_amplitude": float(env.get("stripe_amplitude", 0.5)),
        },
        snapshot_every=max(1, snapshot_every),
    )

    for raw_crystal in raw.get("crystals", []):
        spec = _parse_crystal(raw_crystal, frames, log)
        if spec is not None:
            config.crystals.append(spec)

    return config
