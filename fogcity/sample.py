"""Deterministic, texture-free sample city generator."""

from __future__ import annotations

import math
from typing import Any

import glm


def _box(
    building_id: int,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    color: list[float],
) -> dict[str, Any]:
    cx, cy, cz = center
    hx, hy, hz = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    vertices = [
        [cx - hx, cy - hy, cz - hz],
        [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy - hy, cz + hz],
        [cx - hx, cy - hy, cz + hz],
        [cx - hx, cy + hy, cz - hz],
        [cx + hx, cy + hy, cz - hz],
        [cx + hx, cy + hy, cz + hz],
        [cx - hx, cy + hy, cz + hz],
    ]
    faces = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    return {"id": building_id, "color": color, "vertices": vertices, "faces": faces}


def _prism(
    building_id: int,
    center: tuple[float, float, float],
    radius: float,
    height: float,
    sides: int,
    rotation_radians: float,
    color: list[float],
) -> dict[str, Any]:
    cx, cy, cz = center
    bottom_y = cy
    top_y = cy + height
    vertices: list[list[float]] = []
    for level_y in (bottom_y, top_y):
        for index in range(sides):
            angle = rotation_radians + 2.0 * math.pi * index / sides
            vertices.append(
                [
                    cx + radius * math.cos(angle),
                    level_y,
                    cz + radius * math.sin(angle),
                ]
            )
    faces: list[list[int]] = []
    faces.append(list(reversed(range(sides))))
    faces.append(list(range(sides, 2 * sides)))
    for index in range(sides):
        nxt = (index + 1) % sides
        faces.append([index, nxt, sides + nxt, sides + index])
    return {"id": building_id, "color": color, "vertices": vertices, "faces": faces}


def _glass_panel(
    panel_id: int,
    center: tuple[float, float, float],
    width: float,
    height: float,
    yaw_radians: float,
    tilt_radians: float,
    color: list[float],
    transmittance: float,
    reverse: bool = False,
) -> dict[str, Any]:
    cx, cy, cz = center
    right = glm.normalize(glm.vec3(math.cos(yaw_radians), 0.0, math.sin(yaw_radians)))
    up = glm.normalize(
        glm.vec3(
            0.0,
            math.cos(tilt_radians),
            -math.sin(tilt_radians),
        )
    )
    half_w = width * 0.5
    half_h = height * 0.5
    corners = [
        glm.vec3(cx, cy, cz) - right * half_w - up * half_h,
        glm.vec3(cx, cy, cz) + right * half_w - up * half_h,
        glm.vec3(cx, cy, cz) + right * half_w + up * half_h,
        glm.vec3(cx, cy, cz) - right * half_w + up * half_h,
    ]
    if reverse:
        corners = list(reversed(corners))
    return {
        "id": panel_id,
        "color": color,
        "transmittance": transmittance,
        "vertices": [[float(p.x), float(p.y), float(p.z)] for p in corners],
    }


def _camera_quaternion(
    position: tuple[float, float, float],
    target: tuple[float, float, float],
) -> list[float]:
    forward = glm.normalize(glm.vec3(target) - glm.vec3(position))
    quaternion = glm.normalize(glm.quatLookAt(forward, glm.vec3(0.0, 1.0, 0.0)))
    return [
        float(quaternion.x),
        float(quaternion.y),
        float(quaternion.z),
        float(quaternion.w),
    ]


def build_sample_document(
    *,
    width: int = 480,
    height: int = 270,
    fps: float = 1.0,
    start_time: float = 0.0,
    end_time: float = 6.0,
    fog_steps: int = 32,
    shadow_resolution: int = 768,
) -> dict[str, Any]:
    """Return the complete deterministic sample scene document."""

    buildings = [
        _box(90, (0.0, -0.6, 0.0), (90.0, 1.2, 90.0), [0.26, 0.28, 0.26]),
        _box(1, (-6.0, 4.0, -2.0), (4.0, 8.0, 4.0), [0.34, 0.39, 0.43]),
        _prism(2, (2.0, 0.0, -8.0), 3.3, 9.0, 6, 0.22, [0.40, 0.36, 0.31]),
        _box(3, (9.0, 5.5, -3.0), (3.5, 11.0, 5.0), [0.29, 0.36, 0.40]),
        _box(4, (-11.0, 3.0, 5.0), (5.0, 6.0, 3.5), [0.37, 0.31, 0.27]),
        _prism(5, (0.0, 0.0, 3.8), 2.6, 10.0, 5, 0.0, [0.33, 0.38, 0.35]),
        _box(6, (13.5, 3.0, 7.0), (4.0, 6.0, 4.0), [0.30, 0.33, 0.42]),
        _box(7, (-15.0, 4.5, -8.0), (5.0, 9.0, 5.0), [0.24, 0.30, 0.37]),
    ]

    panels = [
        _glass_panel(
            101,
            (-3.1, 2.15, 11.0),
            5.4,
            2.8,
            math.radians(-12.0),
            math.radians(4.0),
            [0.23, 0.62, 0.68],
            0.64,
        ),
        _glass_panel(
            102,
            (2.2, 2.35, 9.2),
            4.6,
            2.9,
            math.radians(28.0),
            math.radians(-3.0),
            [0.74, 0.58, 0.28],
            0.70,
        ),
        _glass_panel(
            103,
            (6.7, 1.85, 10.2),
            3.2,
            2.2,
            math.radians(68.0),
            math.radians(2.0),
            [0.31, 0.48, 0.78],
            0.57,
        ),
        _glass_panel(
            104,
            (0.3, 2.55, 6.4),
            5.8,
            3.4,
            math.radians(6.0),
            math.radians(-2.0),
            [0.68, 0.73, 0.72],
            0.76,
        ),
        # Winding is intentionally opposite: this panel's normal points away
        # from the sample camera path, so it must never be blended.
        _glass_panel(
            105,
            (-7.5, 2.2, 7.5),
            3.0,
            2.0,
            math.radians(0.0),
            math.radians(0.0),
            [0.50, 0.82, 0.48],
            0.50,
            reverse=True,
        ),
    ]

    keyframe_specs = [
        ((7.0, 2.25, 18.0), (0.0, 2.8, -2.0), 50.0, 0.1, 90.0),
        ((1.5, 2.85, 14.0), (-1.0, 3.2, -1.5), 54.0, 0.1, 90.0),
        ((-4.5, 3.20, 9.0), (1.5, 2.4, -2.5), 58.0, 0.1, 90.0),
        ((10.0, 3.80, 11.0), (2.0, 3.1, -4.0), 52.0, 0.1, 90.0),
    ]
    keyframes = []
    for index, (position, target, fov, near, far) in enumerate(keyframe_specs):
        keyframes.append(
            {
                "time": float(index * 2),
                "position": list(position),
                "orientation": _camera_quaternion(position, target),
                "fov_y_deg": fov,
                "near": near,
                "far": far,
            }
        )

    return {
        "schema_version": 1,
        "name": "deterministic_fog_city_sample",
        "render": {
            "width": width,
            "height": height,
            "fps": fps,
            "start_time": start_time,
            "end_time": end_time,
        },
        "camera_keyframes": keyframes,
        "directional_light": {
            "direction": [-0.42, -0.62, 0.66],
            "color": [1.0, 0.82, 0.58],
            "intensity": 4.6,
        },
        "fog": {
            "color": [0.52, 0.65, 0.74],
            "density": 0.018,
            "extinction": 1.25,
            "scattering": 0.95,
            "anisotropy": 0.62,
            "ambient": 0.50,
            "steps": fog_steps,
        },
        "shadow": {"resolution": shadow_resolution, "bias": 0.0015},
        "buildings": buildings,
        "glass_panels": panels,
    }

