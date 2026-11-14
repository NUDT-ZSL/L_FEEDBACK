"""Cascade frustum splitting, light orthographic matrices and texel snapping."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import ERR_CONFIG, PipelineError
from .matrices import (
    frustum_corners,
    is_finite_matrix,
    look_at,
    multiply,
    orthographic,
    perspective,
)


@dataclass
class CascadeData:
    index: int
    near: float
    far: float
    light_view: np.ndarray
    ortho: np.ndarray
    view_proj: np.ndarray
    snapped_view_proj: np.ndarray
    texel_world_size: float
    frustum_corners_world: np.ndarray
    raw_bounds: tuple = (0.0, 0.0, 0.0, 0.0)
    snapped_bounds: tuple = (0.0, 0.0, 0.0, 0.0)


def cascade_splits(
    near: float,
    far: float,
    cascade_count: int,
    lam: float = 0.5,
) -> np.ndarray:
    """Return ``cascade_count + 1`` split distances using PSSM lambda blending.

    ``lam=0`` is uniform, ``lam=1`` is purely logarithmic.
    """
    if not (0.0 < near < far):
        raise PipelineError(ERR_CONFIG, "camera requires 0 < near < far")
    if cascade_count < 1:
        raise PipelineError(ERR_CONFIG, "cascade_count must be >= 1")
    if not (0.0 <= lam <= 1.0):
        raise PipelineError(ERR_CONFIG, "cascade lambda must be in [0, 1]")
    uniform = np.linspace(near, far, cascade_count + 1, dtype=np.float64)
    log = near * (far / near) ** (
        np.arange(cascade_count + 1, dtype=np.float64) / cascade_count
    )
    splits = lam * log + (1.0 - lam) * uniform
    splits[0] = near
    splits[-1] = far
    return splits


def _sub_frustum_corners(
    full_corners: np.ndarray,
    view: np.ndarray,
    proj: np.ndarray,
    split_near: float,
    split_far: float,
) -> np.ndarray:
    """Intersect the camera frustum with two depth planes by geometric scaling."""
    view_proj = multiply(proj, view)
    inverse = np.linalg.inv(view_proj)
    ndc_xy = np.array(
        [(-1, -1), (1, -1), (1, 1), (-1, 1)], dtype=np.float64
    )
    corners: list = []
    for depth in (split_near, split_far):
        point_eye = np.array((0.0, 0.0, -depth, 1.0), dtype=np.float64)
        point_clip = proj @ point_eye
        ndc_z = point_clip[2] / point_clip[3]
        for x, y in ndc_xy:
            world_h = inverse @ np.array((x, y, ndc_z, 1.0), dtype=np.float64)
            corners.append(world_h[:3] / world_h[3])
    return np.asarray(corners, dtype=np.float64)


def _ortho_bounds(
    corners: np.ndarray,
    light_view: np.ndarray,
    depth_extension_fraction: float,
    scene_bounds: np.ndarray = None,
) -> tuple:
    if scene_bounds is not None:
        view_pts = np.concatenate([corners, scene_bounds], axis=0)
    else:
        view_pts = corners.copy()
    homog = np.concatenate(
        [view_pts, np.ones((view_pts.shape[0], 1), dtype=np.float64)], axis=1
    )
    light_pts = (homog @ light_view.T)[:, :3]
    mins = light_pts.min(axis=0)
    maxs = light_pts.max(axis=0)
    # Force square XY extents so light texels are square and snapping lands
    # on a single shared grid for both axes.
    center_x = 0.5 * (mins[0] + maxs[0])
    center_y = 0.5 * (mins[1] + maxs[1])
    half_span = 0.5 * max(float(maxs[0] - mins[0]), float(maxs[1] - mins[1]))
    mins[0] = center_x - half_span
    maxs[0] = center_x + half_span
    mins[1] = center_y - half_span
    maxs[1] = center_y + half_span
    # Extra Z range keeps the near/far planes from clipping sloped geometry.
    # Scale it with the frustum footprint so depth precision stays high.
    xy_span = max(float(maxs[0] - mins[0]), float(maxs[1] - mins[1]))
    extension = xy_span * depth_extension_fraction
    mins[2] -= extension
    maxs[2] += extension
    return float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]), float(
        mins[2]
    ), float(maxs[2])


def _snap_value(value: float, texel_size: float) -> float:
    return round(value / texel_size) * texel_size


def build_cascades(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    fovy_rad: float,
    aspect: float,
    camera_near: float,
    camera_far: float,
    light_direction: np.ndarray,
    light_up: np.ndarray,
    cascade_count: int,
    shadow_resolution: int,
    lam: float = 0.5,
    depth_extension_fraction: float = 0.25,
    scene_bounds: np.ndarray = None,
) -> list:
    """Build all cascade matrices, including both raw and snapped variants.

    The two variants share one cascade index; snapping only translates the
    orthographic frustum by whole light-texel increments.
    """
    if shadow_resolution < 2:
        raise PipelineError(ERR_CONFIG, "shadow_resolution must be >= 2")
    view = look_at(eye, target, up)
    proj = perspective(fovy_rad, aspect, camera_near, camera_far)
    full_corners = frustum_corners(multiply(proj, view))
    splits = cascade_splits(camera_near, camera_far, cascade_count, lam)

    # Position the light on the far side of the scene along its travel
    # direction so every occluder lies in front of the light (positive
    # orthographic near distance).  The XY bounds remain per-cascade; only the
    # eye placement is shared.
    if scene_bounds is None:
        scene_center = np.zeros(3, dtype=np.float64)
        scene_radius = 100.0
    else:
        scene_center = (scene_bounds[0] + scene_bounds[1]) * 0.5
        scene_radius = float(
            np.linalg.norm(scene_bounds[1] - scene_bounds[0])
        )
    light_eye = scene_center - light_direction * (scene_radius * 2.0 + 10.0)
    light_target = light_eye + light_direction
    light_view = look_at(light_eye, light_target, light_up)

    cascades: list = []
    for index in range(cascade_count):
        split_near = float(splits[index])
        split_far = float(splits[index + 1])
        corners = _sub_frustum_corners(
            full_corners, view, proj, split_near, split_far
        )
        left, right, bottom, top, l_near, l_far = _ortho_bounds(
            corners, light_view, depth_extension_fraction, scene_bounds
        )
        # Orthographic "near" is measured along the camera's -Z axis, so the
        # positive distances are the negated light-view Z bounds.
        near_dist = -l_far
        far_dist = -l_near
        near_dist = max(near_dist, 0.05)
        if not (near_dist < far_dist):
            raise PipelineError(
                ERR_CONFIG,
                f"cascade {index} produced invalid light near/far bounds",
            )
        ortho = orthographic(
            left, right, bottom, top, near_dist, far_dist
        )
        view_proj = multiply(ortho, light_view)

        texel_x = (right - left) / shadow_resolution
        texel_y = (top - bottom) / shadow_resolution
        texel_world_size = max(texel_x, texel_y)
        # Snap the light-view frustum origin to a whole-texel grid so that
        # camera movement does not shift shadow edges by sub-texel amounts.
        snapped_left = _snap_value(left, texel_x)
        snapped_bottom = _snap_value(bottom, texel_y)
        snapped_right = snapped_left + (right - left)
        snapped_top = snapped_bottom + (top - bottom)
        snapped_ortho = orthographic(
            snapped_left,
            snapped_right,
            snapped_bottom,
            snapped_top,
            near_dist,
            far_dist,
        )
        snapped_view_proj = multiply(snapped_ortho, light_view)

        if not (
            is_finite_matrix(view_proj) and is_finite_matrix(snapped_view_proj)
        ):
            raise PipelineError(
                ERR_CONFIG, f"cascade {index} produced a non-finite matrix"
            )

        cascades.append(
            CascadeData(
                index=index,
                near=split_near,
                far=split_far,
                light_view=light_view,
                ortho=ortho,
                view_proj=view_proj,
                snapped_view_proj=snapped_view_proj,
                texel_world_size=float(texel_world_size),
                frustum_corners_world=corners,
                raw_bounds=(left, right, bottom, top),
                snapped_bounds=(
                    snapped_left,
                    snapped_right,
                    snapped_bottom,
                    snapped_top,
                ),
            )
        )
    return cascades


def validate_cascades(cascades: list, shadow_resolution: int) -> None:
    """Fail loudly before any fake metric can be serialized."""
    if len(cascades) == 0:
        raise PipelineError(ERR_CONFIG, "no cascades were produced")
    for cascade in cascades:
        if not (
            is_finite_matrix(cascade.view_proj)
            and is_finite_matrix(cascade.snapped_view_proj)
        ):
            raise PipelineError(
                ERR_CONFIG,
                f"cascade {cascade.index} contains an invalid matrix",
            )
        if not math.isfinite(cascade.texel_world_size) or cascade.texel_world_size <= 0:
            raise PipelineError(
                ERR_CONFIG,
                f"cascade {cascade.index} has an invalid texel size",
            )

