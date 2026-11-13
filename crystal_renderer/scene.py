"""Deterministic scene stepping: crystal state and screen-space geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import CrystalSpec, SceneConfig


@dataclass
class CrystalState:
    crystal_id: str
    layer: int
    ior: float
    color: tuple[float, float, float, float]
    translate: tuple[float, float]
    rotate: float
    scale: float
    visible: bool
    depth_gradient: tuple[float, float]
    refraction_strength: float
    screen_vertices: np.ndarray
    refraction_radius: float

    def state_signature(self):
        return (
            self.visible,
            self.translate,
            self.rotate,
            self.scale,
            self.color,
            self.ior,
        )


def _apply_keyframes(spec: CrystalSpec, step: int):
    rotate = spec.rotate
    translate = spec.translate
    scale = spec.scale
    color = spec.color
    visible = spec.visible
    for kf in spec.keyframes:
        if kf["step"] > step:
            break
        action = kf["set"]
        if "rotate" in action:
            rotate = float(action["rotate"])
        if "rotate_by" in action:
            rotate += float(action["rotate_by"])
        if "translate" in action:
            translate = (float(action["translate"][0]), float(action["translate"][1]))
        if "scale" in action:
            scale = float(action["scale"])
        if "color" in action:
            color = tuple(float(c) for c in action["color"])
        if "visible" in action:
            visible = bool(action["visible"])
    return rotate, translate, scale, color, visible


def state_at(spec: CrystalSpec, camera: dict, step: int) -> CrystalState:
    rotate, translate, scale, color, visible = _apply_keyframes(spec, step)

    cos_r = math.cos(rotate)
    sin_r = math.sin(rotate)
    local = np.asarray(spec.vertices, dtype=np.float64)
    rotated = np.empty_like(local)
    rotated[:, 0] = (local[:, 0] * cos_r - local[:, 1] * sin_r) * scale + translate[0]
    rotated[:, 1] = (local[:, 0] * sin_r + local[:, 1] * cos_r) * scale + translate[1]

    cam_scale = float(camera["scale"])
    cam_rot = float(camera["rotate"])
    cam_tx, cam_ty = float(camera["translate"][0]), float(camera["translate"][1])
    cos_c = math.cos(cam_rot)
    sin_c = math.sin(cam_rot)
    screen = np.empty_like(rotated)
    screen[:, 0] = (rotated[:, 0] * cos_c - rotated[:, 1] * sin_c) * cam_scale + cam_tx
    screen[:, 1] = (rotated[:, 0] * sin_c + rotated[:, 1] * cos_c) * cam_scale + cam_ty

    return CrystalState(
        crystal_id=spec.crystal_id,
        layer=spec.layer,
        ior=spec.ior,
        color=color,
        translate=translate,
        rotate=rotate,
        scale=scale,
        visible=visible,
        depth_gradient=spec.depth_gradient,
        refraction_strength=spec.refraction_strength,
        screen_vertices=screen,
        refraction_radius=spec.refraction_strength * (spec.ior - 1.0),
    )


def scene_states(config: SceneConfig, step: int) -> list[CrystalState]:
    return [state_at(spec, config.camera, step) for spec in config.crystals]


def dilated_aabb(state: CrystalState, width: int, height: int) -> tuple[int, int, int, int]:
    pad = int(math.ceil(state.refraction_radius)) + 1
    xs = state.screen_vertices[:, 0]
    ys = state.screen_vertices[:, 1]
    x0 = max(0, int(math.floor(float(xs.min()))) - pad)
    y0 = max(0, int(math.floor(float(ys.min()))) - pad)
    x1 = min(width, int(math.ceil(float(xs.max()))) + pad + 1)
    y1 = min(height, int(math.ceil(float(ys.max()))) + pad + 1)
    return (x0, y0, x1, y1)


def rects_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and a[1] < b[3] and b[0] < a[2] and b[1] < a[3]


def rect_union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def env_background(config: SceneConfig) -> np.ndarray:
    height, width = config.height, config.width
    env = config.env
    top = np.asarray(env["color_top"], dtype=np.float64)
    bottom = np.asarray(env["color_bottom"], dtype=np.float64)
    stripe = np.asarray(env["stripe_color"], dtype=np.float64)
    freq = float(env["stripe_frequency"])
    amp = float(env["stripe_amplitude"])

    ys = np.arange(height, dtype=np.float64)[:, None]
    xs = np.arange(width, dtype=np.float64)[None, :]
    t = ys / max(1, height - 1)
    gradient = top[None, None, :] * (1.0 - t[:, :, None]) + bottom[None, None, :] * t[:, :, None]
    wave = 0.5 + 0.5 * np.sin(xs * freq * 2.0 * math.pi + ys * freq * math.pi)
    background = gradient * (1.0 - amp * wave[:, :, None] * 0.5) + stripe[None, None, :] * (
        amp * wave[:, :, None] * 0.5
    )
    alpha = np.ones((height, width, 1), dtype=np.float64)
    return np.concatenate([background, alpha], axis=2)
