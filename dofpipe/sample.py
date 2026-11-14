from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - edge0) / max(edge1 - edge0, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _ellipse(
    x: np.ndarray, y: np.ndarray, cx: float, cy: float, rx: float, ry: float
) -> np.ndarray:
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


def _rounded_box(
    x: np.ndarray,
    y: np.ndarray,
    left: float,
    right: float,
    top: float,
    bottom: float,
    radius: float,
) -> np.ndarray:
    clamped_x = np.clip(x, left + radius, right - radius)
    clamped_y = np.clip(y, top + radius, bottom - radius)
    inside_core = (x >= left) & (x <= right) & (y >= top) & (y <= bottom)
    near_corner = (x - clamped_x) ** 2 + (y - clamped_y) ** 2 <= radius**2
    return inside_core & near_corner


def _background_texture(
    width: int, height: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    xn = (x + 0.5) / width
    yn = (y + 0.5) / height
    v = 0.50 + 0.16 * np.sin(18.0 * xn + 3.0 * yn) + 0.08 * np.cos(31.0 * yn)
    u = 0.45 + 0.18 * np.cos(15.0 * xn - 8.0 * yn) + 0.06 * np.sin(27.0 * xn)
    checker = ((np.floor(xn * 12.0) + np.floor(yn * 8.0)) % 2.0) * 0.08

    coarse = rng.normal(0.0, 1.0, size=(height // 9 + 2, width // 9 + 2))
    coarse = cv2.resize(
        coarse.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    ).astype(np.float64)
    coarse = np.clip(coarse * 0.035, -0.09, 0.09)

    blue = np.clip(0.18 + 0.35 * v + checker + coarse, 0.0, 1.0)
    green = np.clip(0.20 + 0.30 * u + 0.5 * checker + coarse, 0.0, 1.0)
    red = np.clip(0.26 + 0.18 * v + 0.35 * checker + 0.5 * coarse, 0.0, 1.0)
    color = np.dstack((blue, green, red))

    depth = 0.88 + 0.025 * np.sin(7.0 * xn + 2.0 * yn)
    return color, depth


def _compose_frame(
    frame_index: int,
    frame_count: int,
    width: int,
    height: int,
    background_color: np.ndarray,
    background_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    xn = (x + 0.5) / width
    yn = (y + 0.5) / height
    phase = 0.0 if frame_count <= 1 else frame_index / (frame_count - 1)

    color = background_color.copy()
    depth = background_depth.copy()

    mid_cx = 0.68 - 0.055 * np.sin(2.0 * np.pi * phase)
    mid_cy = 0.54
    mid_mask = _ellipse(xn, yn, mid_cx, mid_cy, 0.19, 0.255)
    mid_edge = _ellipse(xn, yn, mid_cx, mid_cy, 0.205, 0.275) & ~mid_mask
    mid_u = np.clip((xn - (mid_cx - 0.19)) / 0.38, 0.0, 1.0)
    mid_color = np.dstack(
        (
            0.62 - 0.25 * mid_u,
            0.55 + 0.18 * np.sin(4.0 * np.pi * mid_u),
            0.28 + 0.20 * (1.0 - mid_u),
        )
    )
    color[mid_mask] = mid_color[mid_mask]
    color[mid_edge] = np.array([0.90, 0.82, 0.48])
    depth[mid_mask | mid_edge] = 0.52

    center_x = 0.22 + 0.56 * phase
    half_width = 0.075
    top, bottom = 0.20, 0.84
    radius = 0.075
    foreground = _rounded_box(
        xn,
        yn,
        center_x - half_width,
        center_x + half_width,
        top,
        bottom,
        radius,
    )
    head = _ellipse(xn, yn, center_x, 0.225, 0.072, 0.105)
    foreground |= head
    rim = (
        _rounded_box(
            xn,
            yn,
            center_x - half_width - 0.018,
            center_x + half_width + 0.018,
            top - 0.012,
            bottom + 0.012,
            radius + 0.015,
        )
        | _ellipse(xn, yn, center_x, 0.225, 0.088, 0.122)
    ) & ~foreground

    stripe = foreground & (
        np.abs((xn - center_x) * 15.0 - (yn - 0.5) * 7.0) < 1.4
    )
    fg_u = np.clip((xn - (center_x - half_width)) / (2.0 * half_width), 0.0, 1.0)
    foreground_color = np.dstack(
        (
            0.18 + 0.18 * fg_u,
            0.28 + 0.28 * fg_u,
            0.92 - 0.16 * fg_u,
        )
    )
    color[foreground] = foreground_color[foreground]
    color[stripe] = np.array([0.92, 0.88, 0.70])
    color[rim] = np.array([0.86, 0.48, 0.18])
    depth[foreground] = 0.13 + 0.006 * np.sin(5.0 * np.pi * yn[foreground])
    depth[rim] = 0.13

    small_orbs = (
        _ellipse(xn, yn, 0.12 + 0.02 * phase, 0.32, 0.018, 0.032)
        | _ellipse(xn, yn, 0.87, 0.24 + 0.03 * phase, 0.022, 0.028)
    )
    color[small_orbs] = np.array([0.92, 0.72, 0.22])
    depth[small_orbs] = 0.65

    return (
        np.clip(np.rint(color * 255.0), 0, 255).astype(np.uint8),
        np.ascontiguousarray(depth.astype(np.float32)),
    )


def generate_sample(
    output_dir: str | Path,
    *,
    width: int = 240,
    height: int = 135,
    frame_count: int = 8,
    fps: float = 12.0,
    seed: int = 12345,
) -> dict[str, object]:
    if width < 16 or height < 16:
        raise ValueError("sample width and height must be >= 16")
    if frame_count < 1:
        raise ValueError("sample frame_count must be >= 1")
    if fps <= 0:
        raise ValueError("sample fps must be > 0")

    cv2.setNumThreads(1)
    cv2.setRNGSeed(int(seed) & 0x7FFFFFFF)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    background_color, background_depth = _background_texture(width, height, rng)
    for frame_index in range(frame_count):
        color, depth = _compose_frame(
            frame_index,
            frame_count,
            width,
            height,
            background_color,
            background_depth,
        )
        color_path = output_dir / f"color_{frame_index:04d}.png"
        depth_path = output_dir / f"depth_{frame_index:04d}.npy"
        if not cv2.imwrite(
            str(color_path),
            color,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        ):
            raise OSError(f"failed to write sample image {color_path}")
        np.save(depth_path, depth, allow_pickle=False)

    sequence_end_s = (frame_count - 1) / fps
    if frame_count == 1:
        keyframes = [{"time": 0.0, "focus_distance": 3.8}]
    else:
        track_end_s = max(1.0 / fps, sequence_end_s * 0.85)
        keyframes = [
            {"time": 0.0, "focus_distance": 1.8},
            {"time": track_end_s * 0.55, "focus_distance": 3.8},
            {"time": track_end_s, "focus_distance": 9.0},
        ]
    focus_track = {
        "time_unit": "seconds",
        "keyframes": keyframes,
    }
    with (output_dir / "focus_track.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(focus_track, handle, indent=2)
        handle.write("\n")

    config = {
        "frame_rate": float(fps),
        "frame_start": 0,
        "frame_count": int(frame_count),
        "color_pattern": "color_{frame:04d}.png",
        "depth_pattern": "depth_{frame:04d}.npy",
        "focus_track": "focus_track.json",
        "seed": int(seed),
        "lens": {
            "focal_length_mm": 85.0,
            "f_number": 1.2,
            "sensor_width_mm": 36.0,
            "near_plane_m": 0.5,
            "far_plane_m": 20.0,
            "max_blur_radius_px": 10.0,
            "invalid_depth_fallback_m": 20.0,
            "depth_tolerance_m": 0.001,
        },
        "aperture": {
            "blades": 6,
            "rotation_deg": 0.0,
            "supersample": 8,
        },
        "render": {
            "depth_slices": 8,
            "gather_rounds": 2,
            "png_compression": 3,
            "output_pattern": "frame_{frame:04d}.png",
        },
        "validation": {"slice_max_abs_error_8bit": 2.0},
    }
    with (output_dir / "config.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    manifest = {
        "generator": "dofpipe.sample",
        "width": int(width),
        "height": int(height),
        "frame_count": int(frame_count),
        "frame_rate": float(fps),
        "seed": int(seed),
        "color_pattern": "color_{frame:04d}.png",
        "depth_pattern": "depth_{frame:04d}.npy",
        "focus_track": "focus_track.json",
    }
    with (output_dir / "generation_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest
