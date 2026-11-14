"""Camera-space shadow evaluation: selection, bias, PCF and cascade fading."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .csm import CascadeData
from .errors import ERR_INVALID_DEPTH, PipelineError
from .matrices import normalize
from .rasterizer import validate_depth_map


@dataclass
class ShadowEvaluation:
    factor: np.ndarray  # 0 = fully shadowed, 1 = fully lit
    cascade_index: np.ndarray
    blend: np.ndarray
    world_depth: np.ndarray
    normal_bias_world: np.ndarray
    depth_bias_ndc: np.ndarray
    coverage: np.ndarray  # per-cascade boolean pixel ownership


def select_cascade(world_depth: np.ndarray, cascades: list) -> np.ndarray:
    index = np.full(world_depth.shape, len(cascades) - 1, dtype=np.int32)
    for cascade in cascades:
        unassigned = index == len(cascades) - 1
        index[unassigned & (world_depth <= cascade.far)] = cascade.index
    return index


def _blend_weights(
    world_depth: np.ndarray,
    cascade_index: np.ndarray,
    cascades: list,
    blend_fraction: float,
) -> np.ndarray:
    """0 = own cascade only, 1 = fully faded toward the next cascade."""
    blend = np.zeros(world_depth.shape, dtype=np.float32)
    last = len(cascades) - 1
    for cascade in cascades[:-1]:
        width = max(cascade.far - cascade.near, 1e-9) * blend_fraction
        start = cascade.far - width
        own = cascade_index == cascade.index
        fade = np.clip((world_depth - start) / max(width, 1e-9), 0.0, 1.0)
        blend[own] = fade[own].astype(np.float32)
    blend[cascade_index == last] = 0.0
    return blend


def _sample_shadow_map(
    depth_map: np.ndarray,
    ndc_xy: np.ndarray,
    ref_depth: np.ndarray,
    radius_texels: int,
    valid: np.ndarray,
) -> np.ndarray:
    height, width = depth_map.shape
    uv = (ndc_xy + 1.0) * 0.5
    px = uv[..., 0] * width
    py = (1.0 - uv[..., 1]) * height
    center_x = np.floor(px).astype(np.int64)
    center_y = np.floor(py).astype(np.int64)
    lit_sum = np.zeros(ref_depth.shape, dtype=np.float32)
    weight_sum = np.zeros(ref_depth.shape, dtype=np.float32)
    for dy in range(-radius_texels, radius_texels + 1):
        for dx in range(-radius_texels, radius_texels + 1):
            sx = center_x + dx
            sy = center_y + dy
            in_bounds = (
                (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height) & valid
            )
            sampled = np.ones(ref_depth.shape, dtype=np.float32)
            sampled[in_bounds] = np.where(
                depth_map[sy[in_bounds], sx[in_bounds]]
                >= ref_depth[in_bounds],
                1.0,
                0.0,
            ).astype(np.float32)
            weight = in_bounds.astype(np.float32)
            lit_sum += sampled * weight
            weight_sum += weight
    with np.errstate(invalid="ignore"):
        factor = np.where(
            weight_sum > 0.0, lit_sum / np.maximum(weight_sum, 1.0), 1.0
        )
    return factor.astype(np.float32)


def evaluate_shadows(
    gbuffer: dict,
    camera_view: np.ndarray,
    cascades: list,
    depth_maps: list,
    use_snapped: bool,
    normal_bias_scale: float = 0.02,
    depth_bias: float = 2.0e-4,
    pcf_kernel: int = 3,
    blend_fraction: float = 0.15,
) -> ShadowEvaluation:
    """Evaluate a per-pixel lit factor for every camera pixel that hit geometry."""
    if pcf_kernel < 1 or pcf_kernel % 2 == 0:
        raise PipelineError(ERR_INVALID_DEPTH, "pcf_kernel must be an odd integer >= 1")
    hit = gbuffer["hit"]
    world_pos = gbuffer["world_pos"]
    normals = gbuffer["normal"]

    eye = np.linalg.inv(np.asarray(camera_view, dtype=np.float64))[:3, 3]
    world_depth = np.linalg.norm(world_pos - eye, axis=2).astype(np.float32)
    world_depth[~hit] = 0.0
    cascade_index = select_cascade(world_depth, cascades)
    blend = _blend_weights(
        world_depth, cascade_index, cascades, float(blend_fraction)
    )

    factor = np.ones(world_depth.shape, dtype=np.float32)
    normal_bias_world = np.zeros(world_depth.shape, dtype=np.float32)
    depth_bias_ndc = np.zeros(world_depth.shape, dtype=np.float32)
    coverage = np.zeros((len(cascades),) + world_depth.shape, dtype=bool)
    radius = pcf_kernel // 2

    for cascade in cascades:
        depth_map = np.asarray(depth_maps[cascade.index], dtype=np.float32)
        validate_depth_map(depth_map, np.ones(depth_map.shape, dtype=bool))
        matrix = (
            cascade.snapped_view_proj if use_snapped else cascade.view_proj
        )
        own = hit & (cascade_index == cascade.index)
        coverage[cascade.index] = own
        if not bool(own.any()):
            continue

        unit_normals = np.zeros_like(normals)
        norms = np.linalg.norm(normals, axis=2, keepdims=True)
        safe = norms[..., 0] > 0.0
        unit_normals[safe] = normals[safe] / norms[safe]
        light_dir = np.asarray(_light_direction(cascade), dtype=np.float64)
        facing = np.clip(np.abs(np.sum(unit_normals * light_dir, axis=2)), 0.0, 1.0)
        slope_scale = np.clip(1.0 / np.sqrt(np.maximum(1.0 - facing ** 2, 1e-6)),
                              1.0, 10.0)
        slope_scale = np.where(safe, slope_scale, 1.0)
        n_bias = (
            normal_bias_scale * cascade.texel_world_size
        ) * slope_scale
        offset_pos = world_pos + unit_normals * n_bias[..., None]
        normal_bias_world[own] = n_bias[own].astype(np.float32)

        clip = _to_clip(matrix, offset_pos)
        ndc_xy = clip[..., :2] / clip[..., 3:4]
        ref_depth = (clip[..., 2] / clip[..., 3]).astype(np.float32)
        in_xy = (np.abs(ndc_xy[..., 0]) <= 1.0) & (np.abs(ndc_xy[..., 1]) <= 1.0)
        in_z = (ref_depth >= 0.0) & (ref_depth <= 1.0)
        valid = own & in_xy & in_z
        bias_ndc = (depth_bias * slope_scale).astype(np.float32)
        depth_bias_ndc[own] = bias_ndc[own]
        reference = np.clip(ref_depth - bias_ndc, 0.0, 1.0)

        own_factor = _sample_shadow_map(
            depth_map, ndc_xy, reference, radius, valid
        )
        factor[own] = own_factor[own]

        next_index = cascade.index + 1
        fading = own & (blend > 0.0) & (next_index < len(cascades))
        if bool(fading.any()):
            next_cascade = cascades[next_index]
            next_map = np.asarray(depth_maps[next_index], dtype=np.float32)
            next_matrix = (
                next_cascade.snapped_view_proj
                if use_snapped
                else next_cascade.view_proj
            )
            next_clip = _to_clip(next_matrix, offset_pos)
            next_ndc = next_clip[..., :2] / next_clip[..., 3:4]
            next_ref = (next_clip[..., 2] / next_clip[..., 3]).astype(np.float32)
            next_valid = fading & (
                (np.abs(next_ndc[..., 0]) <= 1.0)
                & (np.abs(next_ndc[..., 1]) <= 1.0)
                & (next_ref >= 0.0)
                & (next_ref <= 1.0)
            )
            next_factor = _sample_shadow_map(
                next_map, next_ndc,
                np.clip(next_ref - bias_ndc, 0.0, 1.0),
                radius, next_valid,
            )
            weight = blend[fading]
            factor[fading] = (
                (1.0 - weight) * factor[fading] + weight * next_factor[fading]
            ).astype(np.float32)

    factor[~hit] = 1.0
    if not np.isfinite(factor).all():
        raise PipelineError(
            ERR_INVALID_DEPTH, "shadow evaluation produced non-finite factors"
        )
    return ShadowEvaluation(
        factor=factor,
        cascade_index=cascade_index,
        blend=blend,
        world_depth=world_depth,
        normal_bias_world=normal_bias_world,
        depth_bias_ndc=depth_bias_ndc,
        coverage=coverage,
    )


def _to_clip(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    shape = points.shape[:-1]
    flat = points.reshape(-1, 3).astype(np.float64)
    homog = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
    clip = homog @ np.asarray(matrix, dtype=np.float64).T
    return clip.reshape(shape + (4,))


def _light_direction(cascade: CascadeData) -> np.ndarray:
    # The light view matrix maps the light travel direction to local -Z.
    rotation = cascade.light_view[:3, :3]
    return normalize(-(rotation.T @ np.array((0.0, 0.0, 1.0))))

