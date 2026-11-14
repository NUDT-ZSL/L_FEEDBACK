"""Camera-projected previews and deterministic bilinear sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .color import display_transform
from .errors import MaterialError
from .pngio import write_png
from .scene import Scene
from .util import ensure_output_name


def camera_uv_grid(
    scene: Scene, resolution: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width, height = resolution
    camera = scene.camera
    position = np.asarray(camera["position"], dtype=np.float64)
    target = np.asarray(camera["target"], dtype=np.float64)
    up = np.asarray(camera["up"], dtype=np.float64)
    forward = target - position
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    right = np.cross(forward, up)
    if float(np.linalg.norm(right)) < 1e-8:
        raise MaterialError("E004", "camera.up must not be parallel to the view direction")
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    y, x = np.mgrid[0:height, 0:width]
    aspect = width / height
    ndc_x = ((x + 0.5) / width * 2.0 - 1.0) * aspect
    ndc_y = 1.0 - (y + 0.5) / height * 2.0
    if camera["projection"] == "orthographic":
        scale = float(camera["ortho_scale"]) / 2.0
        origin = (
            position[None, None, :]
            + right[None, None, :] * (ndc_x[:, :, None] * scale)
            + true_up[None, None, :] * (ndc_y[:, :, None] * scale)
        )
        ray = np.broadcast_to(forward, origin.shape)
        ray_valid = np.ones((height, width), dtype=bool)
    else:
        focal = 1.0 / np.tan(np.deg2rad(float(camera["fov_y_degrees"]) / 2.0))
        direction = (
            forward[None, None, :] * focal
            + right[None, None, :] * ndc_x[:, :, None]
            + true_up[None, None, :] * ndc_y[:, :, None]
        )
        ray = direction / np.maximum(
            np.linalg.norm(direction, axis=2, keepdims=True), 1e-12
        )
        origin = np.broadcast_to(position, ray.shape)
        ray_valid = np.ones((height, width), dtype=bool)

    plane_point = np.asarray(scene.surface["uv_origin"], dtype=np.float64)
    plane_normal = np.asarray(
        scene.surface["geometry_normal"], dtype=np.float64
    )
    denominator = np.sum(ray * plane_normal[None, None, :], axis=2)
    ray_valid &= np.abs(denominator) > 1e-9
    distance = np.sum(
        (plane_point[None, None, :] - origin) * plane_normal[None, None, :],
        axis=2,
    )
    t = np.divide(
        distance,
        denominator,
        out=np.full_like(distance, np.nan),
        where=ray_valid,
    )
    ray_valid &= np.isfinite(t) & (t >= 0.0)
    hit = origin + ray * t[:, :, None]
    offset = hit - plane_point[None, None, :]
    axis_u = np.asarray(scene.surface["uv_axis_u"], dtype=np.float64)
    axis_v = np.asarray(scene.surface["uv_axis_v"], dtype=np.float64)
    size_u, size_v = (float(v) for v in scene.surface["material_size"])
    u = 0.5 + np.sum(offset * axis_u[None, None, :], axis=2) / size_u
    v = 0.5 + np.sum(offset * axis_v[None, None, :], axis=2) / size_v
    ray_valid &= (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    return u.astype(np.float32), v.astype(np.float32), ray_valid


def sample_bilinear(
    texture: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    fill: np.ndarray,
) -> np.ndarray:
    height, width = texture.shape[:2]
    channels = 1 if texture.ndim == 2 else texture.shape[2]
    fill_array = np.asarray(fill, dtype=np.float32).reshape(channels)
    x = u * np.float32(width - 1)
    y = v * np.float32(height - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)
    if texture.ndim == 2:
        p00 = texture[y0, x0][:, :, None]
        p10 = texture[y0, x1][:, :, None]
        p01 = texture[y1, x0][:, :, None]
        p11 = texture[y1, x1][:, :, None]
    else:
        p00 = texture[y0, x0]
        p10 = texture[y0, x1]
        p01 = texture[y1, x0]
        p11 = texture[y1, x1]
    wx = wx[:, :, None]
    wy = wy[:, :, None]
    result = (
        (1.0 - wx) * (1.0 - wy) * p00
        + wx * (1.0 - wy) * p10
        + (1.0 - wx) * wy * p01
        + wx * wy * p11
    ).astype(np.float32)
    if np.any(~valid):
        result[~valid] = fill_array
    return result


def _output_name(scene: Scene, kind: str) -> str:
    name = scene.output["filename_pattern"].replace("{kind}", kind)
    return ensure_output_name(name)


def render_previews(
    scene: Scene,
    out_dir: Path,
    *,
    baked_linear: np.ndarray,
    object_normals: np.ndarray,
    single_light_linear: np.ndarray,
    bit_depth: int,
) -> list[dict[str, Any]]:
    resolution = (
        int(scene.output["preview_resolution"][0]),
        int(scene.output["preview_resolution"][1]),
    )
    u, v, valid = camera_uv_grid(scene, resolution)
    background = np.asarray(scene.output["background"], dtype=np.float32)
    exposure = float(scene.baking["exposure"])

    main_linear = sample_bilinear(baked_linear, u, v, valid, background)
    main_display = display_transform(main_linear, exposure)
    normal_sampled = sample_bilinear(
        object_normals, u, v, valid, np.array([0.0, 0.0, 1.0], np.float32)
    )
    normal_display = np.clip(0.5 + 0.5 * normal_sampled, 0.0, 1.0)
    normal_display[~valid] = np.array([0.045, 0.050, 0.060], np.float32)
    single_linear = sample_bilinear(
        single_light_linear, u, v, valid, background
    )
    single_display = display_transform(single_linear, exposure)

    outputs: list[dict[str, Any]] = []
    for kind, image in (
        ("preview_main", main_display),
        ("preview_normal", normal_display),
        ("preview_single_light", single_display),
    ):
        name = _output_name(scene, kind)
        digest = write_png(out_dir / name, image, bit_depth)
        outputs.append(
            {
                "name": name,
                "kind": kind,
                "role": "camera_preview",
                "width": resolution[0],
                "height": resolution[1],
                "bit_depth": bit_depth,
                "sha256": digest,
            }
        )
    return outputs
