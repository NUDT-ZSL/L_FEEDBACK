from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import GBuffer, Mesh, build_uv_mesh, rasterize
from .material import MaterialMaps
from .scene import CameraSpec, LightSpec, SceneSpec
from .shading import ShadeResult, shade_gbuffer
from .util import rounded, sha256_file


@dataclass(frozen=True)
class ViewRender:
    camera: CameraSpec
    gbuffer: GBuffer
    shading: ShadeResult


def build_cached_mesh(scene: SceneSpec) -> Mesh:
    return build_uv_mesh(scene.uv_layout.mesh)


def build_gbuffers(scene: SceneSpec, mesh: Mesh) -> dict[str, GBuffer]:
    return {camera.id: rasterize(mesh, camera) for camera in scene.cameras}


def render_views(
    scene: SceneSpec,
    material: MaterialMaps,
    gbuffers: dict[str, GBuffer],
    preset: LightSpec,
) -> dict[str, ViewRender]:
    cameras = {camera.id: camera for camera in scene.cameras}
    views: dict[str, ViewRender] = {}
    for camera_id in sorted(gbuffers):
        gbuffer = gbuffers[camera_id]
        views[camera_id] = ViewRender(
            camera=cameras[camera_id],
            gbuffer=gbuffer,
            shading=shade_gbuffer(gbuffer, material, preset),
        )
    return views


def _nullable(values: np.ndarray, coverage: np.ndarray, digits: int = 6) -> list[Any]:
    if values.ndim != 1:
        raise ValueError("nullable expects a one-dimensional array")
    result: list[Any] = []
    for value, is_covered in zip(values, coverage):
        result.append(rounded(value, digits) if is_covered else None)
    return result


def _nullable_int(values: np.ndarray, coverage: np.ndarray) -> list[Any]:
    return [
        int(value) if is_covered else None
        for value, is_covered in zip(values.reshape(-1), coverage.reshape(-1))
    ]


def _covered_float_column(values: np.ndarray, coverage: np.ndarray, digits: int = 6) -> list[Any]:
    return _nullable(values.reshape(-1), coverage.reshape(-1), digits)


def _key_pixel_record(
    *,
    key_type: str,
    target_uv: tuple[float, float],
    x: int,
    y: int,
    shading: ShadeResult,
    gbuffer: GBuffer,
) -> dict[str, Any]:
    return {
        "type": key_type,
        "target_uv": [round(target_uv[0], 6), round(target_uv[1], 6)],
        "pixel": [x, y],
        "uv": rounded(gbuffer.uv[y, x]),
        "normal": rounded(gbuffer.normal[y, x]),
        "linear_irradiance": rounded(shading.linear_irradiance[y, x]),
        "linear_color_before_tone_map": rounded(shading.linear_color[y, x]),
        "tone_mapped_linear_color": rounded(shading.mapped_linear[y, x]),
        "srgb8": shading.srgb8[y, x].astype(int).tolist(),
    }


def _make_key_records(
    view: ViewRender,
    materials: MaterialMaps,
    landmark_uvs: tuple[tuple[float, float], ...],
) -> list[dict[str, Any]]:
    coverage = view.gbuffer.coverage
    uv_flat = view.gbuffer.uv.reshape(-1, 2)
    records: list[dict[str, Any]] = []
    for target in landmark_uvs:
        distance = np.sum((uv_flat - np.asarray(target)) ** 2.0, axis=1)
        index = int(np.argmin(distance))
        y, x = divmod(index, view.camera.width)
        if not coverage[y, x]:
            continue
        records.append(
            _key_pixel_record(
                key_type="uv_landmark",
                target_uv=target,
                x=x,
                y=y,
                shading=view.shading,
                gbuffer=view.gbuffer,
            )
        )

    boundary_index = metallic_boundary_index(materials)
    if boundary_index is not None:
        by, bx = divmod(boundary_index, materials.width)
        target = ((bx + 0.5) / materials.width, (by + 0.5) / materials.height)
        distance = np.sum((uv_flat - np.asarray(target)) ** 2.0, axis=1)
        index = int(np.argmin(distance))
        y, x = divmod(index, view.camera.width)
        if coverage[y, x]:
            records.append(
                _key_pixel_record(
                    key_type="metallic_boundary_probe",
                    target_uv=target,
                    x=x,
                    y=y,
                    shading=view.shading,
                    gbuffer=view.gbuffer,
                )
            )
    return records


def metallic_boundary_index(materials: MaterialMaps) -> int | None:
    mask = materials.metallic
    crossings: list[tuple[float, int]] = []
    left = mask[:, :-1]
    right = mask[:, 1:]
    crossing_right = ((left - 0.35) * (right - 0.35) <= 0.0) & (np.abs(left - right) > 1e-8)
    for y, x in zip(*np.nonzero(crossing_right)):
        crossings.append((float(abs(right[y, x] - left[y, x])), int(y * materials.width + x)))
    top = mask[:-1, :]
    bottom = mask[1:, :]
    crossing_down = ((top - 0.35) * (bottom - 0.35) <= 0.0) & (np.abs(top - bottom) > 1e-8)
    for y, x in zip(*np.nonzero(crossing_down)):
        crossings.append((float(abs(bottom[y, x] - top[y, x])), int(y * materials.width + x)))
    if not crossings:
        candidates = np.nonzero((mask >= 0.30) & (mask <= 0.40))
        if not len(candidates[0]):
            return None
        return int(candidates[0][0] * materials.width + candidates[1][0])
    crossings.sort(key=lambda pair: (pair[0], pair[1]))
    return crossings[0][1]


def metadata_document(
    scene: SceneSpec,
    preset: LightSpec,
    view: ViewRender,
    materials: MaterialMaps,
    image_relative_path: str,
    image_path: Path,
    consistency_relative_path: str,
    landmark_uvs: tuple[tuple[float, float], ...],
) -> dict[str, Any]:
    """Build a compact columnar trace for every sampled image pixel."""

    height, width = view.gbuffer.coverage.shape
    coverage = view.gbuffer.coverage
    key_records = _make_key_records(view, materials, landmark_uvs)
    return {
        "schema_version": 1,
        "scene_id": scene.scene_id,
        "preset_id": preset.id,
        "camera_id": view.camera.id,
        "image": image_relative_path,
        "image_sha256": sha256_file(image_path),
        "image_size": [width, height],
        "pixel_order": "row-major",
        "pixels": {
            "x": np.tile(np.arange(width, dtype=np.int64), height).tolist(),
            "y": np.repeat(np.arange(height, dtype=np.int64), width).tolist(),
            "coverage": coverage.reshape(-1).astype(int).tolist(),
            "uv_u": _covered_float_column(view.gbuffer.uv[..., 0], coverage),
            "uv_v": _covered_float_column(view.gbuffer.uv[..., 1], coverage),
            "normal_x": _covered_float_column(view.gbuffer.normal[..., 0], coverage),
            "normal_y": _covered_float_column(view.gbuffer.normal[..., 1], coverage),
            "normal_z": _covered_float_column(view.gbuffer.normal[..., 2], coverage),
            "linear_irradiance_r": _covered_float_column(view.shading.linear_irradiance[..., 0], coverage),
            "linear_irradiance_g": _covered_float_column(view.shading.linear_irradiance[..., 1], coverage),
            "linear_irradiance_b": _covered_float_column(view.shading.linear_irradiance[..., 2], coverage),
            "linear_color_before_tone_map_r": _covered_float_column(view.shading.linear_color[..., 0], coverage),
            "linear_color_before_tone_map_g": _covered_float_column(view.shading.linear_color[..., 1], coverage),
            "linear_color_before_tone_map_b": _covered_float_column(view.shading.linear_color[..., 2], coverage),
            "tone_mapped_linear_r": _covered_float_column(view.shading.mapped_linear[..., 0], coverage),
            "tone_mapped_linear_g": _covered_float_column(view.shading.mapped_linear[..., 1], coverage),
            "tone_mapped_linear_b": _covered_float_column(view.shading.mapped_linear[..., 2], coverage),
            "srgb8_r": _nullable_int(view.shading.srgb8[..., 0], coverage),
            "srgb8_g": _nullable_int(view.shading.srgb8[..., 1], coverage),
            "srgb8_b": _nullable_int(view.shading.srgb8[..., 2], coverage),
            "metallic": _covered_float_column(view.shading.metallic, coverage),
            "roughness": _covered_float_column(view.shading.roughness, coverage),
        },
        "key_pixels": key_records,
        "consistency_report": consistency_relative_path,
    }
