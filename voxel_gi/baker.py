"""Deterministic threaded SH-9 probe baking."""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .errors import BusinessError, ErrorCode
from .sampler import hemisphere_directions
from .scene import (
    Light,
    ProbeLayout,
    Scene,
    build_probe_layout,
    compute_geometry_hash,
    compute_layout_hash,
    compute_lighting_hash,
    compute_scene_hash,
)
from .sh import IRRADIANCE_FROM_C00, basis
from .voxel import RayHit, VoxelGrid


@dataclass(frozen=True)
class BakeResult:
    coefficients: np.ndarray
    residuals: np.ndarray
    probe_irradiance: np.ndarray
    clamped_probes: int
    energy_bound: float
    voxelization_seconds: float
    probe_bake_seconds: float
    thread_count: int
    grid_info: dict
    geometry_hash: str
    lighting_hash: str
    layout_hash: str
    scene_hash: str

    @property
    def probe_count(self) -> int:
        return int(self.coefficients.shape[0])

    @property
    def mean_residual(self) -> float:
        return float(np.mean(self.residuals, dtype=np.float64))

    @property
    def max_residual(self) -> float:
        return float(np.max(self.residuals))

    @property
    def max_probe_irradiance(self) -> float:
        if self.probe_irradiance.size == 0:
            return 0.0
        return float(np.max(np.sum(self.probe_irradiance, axis=1)))

    @property
    def max_channel_irradiance(self) -> float:
        if self.probe_irradiance.size == 0:
            return 0.0
        return float(np.max(self.probe_irradiance))

    @property
    def within_energy_bound(self) -> bool:
        return self.max_probe_irradiance <= self.energy_bound + 1.0e-9


def _reflected_radiance(
    scene: Scene,
    grid: VoxelGrid,
    hit: RayHit,
    shadow_epsilon: float,
) -> np.ndarray:
    normal = hit.normal
    point = hit.point
    shadow_origin = point + normal * shadow_epsilon
    irradiance = np.zeros(3, dtype=np.float64)

    for light in scene.lights:
        if light.kind == "point":
            sample = light.sample_points[0]
            delta = sample - point
            distance_squared = float(np.dot(delta, delta))
            if distance_squared <= 1.0e-20:
                continue
            distance = math.sqrt(distance_squared)
            incoming = delta / distance
            cosine_surface = float(np.dot(normal, incoming))
            if cosine_surface <= 0.0:
                continue
            if grid.is_occluded(shadow_origin, sample):
                continue
            contribution = (
                light.color
                * (light.radiant_power / (4.0 * math.pi * distance_squared))
                * cosine_surface
            )
            irradiance += contribution
            continue

        sample_area = light.area / float(light.sample_count)
        emitted_radiance = light.color * (light.radiant_power / (math.pi * light.area))
        for sample in light.sample_points:
            delta = sample - point
            distance_squared = float(np.dot(delta, delta))
            if distance_squared <= 1.0e-20:
                continue
            distance = math.sqrt(distance_squared)
            incoming = delta / distance
            cosine_surface = float(np.dot(normal, incoming))
            if cosine_surface <= 0.0:
                continue
            cosine_light = float(np.dot(light.normal, -incoming))
            if cosine_light <= 0.0:
                continue
            if grid.is_occluded(shadow_origin, sample):
                continue
            irradiance += (
                emitted_radiance
                * (cosine_surface * cosine_light * sample_area / distance_squared)
            )
    return hit.albedo * irradiance / math.pi


def _bake_probe(
    scene: Scene,
    grid: VoxelGrid,
    layout: ProbeLayout,
    probe_index: int,
) -> Tuple[np.ndarray, float, np.ndarray, int]:
    position = layout.positions[probe_index]
    sample_count = int(scene.bake.ray_count)
    directions = hemisphere_directions(
        sample_count,
        int(scene.bake.seed),
        probe_index,
        layout.up,
    )
    cosines = directions @ layout.up
    radiance = np.empty((sample_count, 3), dtype=np.float64)
    shadow_epsilon = max(1.0e-5, grid.voxel_size * 0.01)

    for index, direction in enumerate(directions):
        hit = grid.trace_primary(position, direction)
        if hit is None:
            radiance[index, :] = scene.bake.ambient_radiance
        else:
            radiance[index, :] = _reflected_radiance(
                scene, grid, hit, shadow_epsilon
            )

    directional_irradiance = radiance * cosines[:, None]
    basis_matrix = basis(directions)
    solid_angle = 2.0 * math.pi / float(sample_count)
    coefficients = (basis_matrix.T @ directional_irradiance) * solid_angle

    fitted = basis_matrix @ coefficients
    error_rms = math.sqrt(float(np.mean((fitted - directional_irradiance) ** 2)))
    signal_rms = math.sqrt(float(np.mean(directional_irradiance ** 2)))
    residual = 0.0 if signal_rms <= 1.0e-15 else error_rms / signal_rms

    irradiance = coefficients[0] * IRRADIANCE_FROM_C00
    clamped = 0
    total_irradiance = float(np.sum(irradiance))
    # Leave a small margin so float32 serialization cannot push the sum a few
    # ULPs above the declared physical bound.
    guarded_bound = max(0.0, float(scene.energy_bound) * (1.0 - 1.0e-6))
    if total_irradiance > guarded_bound:
        scale = guarded_bound / total_irradiance if total_irradiance > 0.0 else 0.0
        coefficients *= scale
        irradiance *= scale
        clamped = 1

    coefficients32 = np.ascontiguousarray(coefficients, dtype=np.float32)
    return coefficients32, float(residual), irradiance, clamped


def bake_scene(
    scene: Scene,
    layout: Optional[ProbeLayout] = None,
    grid: Optional[VoxelGrid] = None,
    threads: Optional[int] = None,
) -> BakeResult:
    total_start = time.perf_counter()
    voxelization_start = time.perf_counter()
    if grid is None:
        grid = VoxelGrid.build(scene)
    voxelization_seconds = time.perf_counter() - voxelization_start
    if layout is None:
        layout = build_probe_layout(scene)
    if layout.probe_count <= 0:
        raise BusinessError(
            ErrorCode.INVALID_PROBE_BOUNDS,
            "Probe layout is empty",
            {"probe_count": layout.probe_count},
        )
    occupied_indices = [
        int(index)
        for index, position in enumerate(layout.positions)
        if grid.point_is_occupied(position)
    ]
    if occupied_indices:
        raise BusinessError(
            ErrorCode.PROBE_INSIDE_OCCLUDER,
            "Probe positions intersect voxelized occluders",
            {"probe_indices": occupied_indices[:32], "count": len(occupied_indices)},
        )

    worker_count = int(scene.bake.threads if threads is None else threads)
    if worker_count < 1 or worker_count > 256:
        raise BusinessError(
            ErrorCode.INVALID_BAKE_CONFIG,
            "Thread count must be in [1, 256]",
            {"threads": worker_count},
        )
    probe_start = time.perf_counter()
    if worker_count == 1:
        probe_results = [
            _bake_probe(scene, grid, layout, index)
            for index in range(layout.probe_count)
        ]
    else:
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="voxel-gi",
        ) as executor:
            probe_results = list(
                executor.map(
                    lambda index: _bake_probe(scene, grid, layout, index),
                    range(layout.probe_count),
                    chunksize=1,
                )
            )
    probe_bake_seconds = time.perf_counter() - probe_start

    coefficients = np.stack([item[0] for item in probe_results], axis=0)
    residuals = np.asarray([item[1] for item in probe_results], dtype=np.float32)
    # Derive report/metadata energy from the exact float32 payload that users
    # load. This keeps the declared energy self-consistent with the cache.
    probe_irradiance = (
        coefficients[:, 0, :].astype(np.float64) * IRRADIANCE_FROM_C00
    )
    clamped_probes = int(sum(item[3] for item in probe_results))
    if not np.all(np.isfinite(coefficients)):
        raise BusinessError(
            ErrorCode.INTERNAL_ERROR,
            "Bake produced non-finite SH coefficients",
        )
    if not np.all(np.isfinite(residuals)):
        raise BusinessError(
            ErrorCode.INTERNAL_ERROR,
            "Bake produced non-finite residuals",
        )

    geometry_hash = compute_geometry_hash(scene)
    lighting_hash = compute_lighting_hash(scene)
    layout_hash = compute_layout_hash(layout)
    scene_hash = compute_scene_hash(
        scene,
        layout,
        geometry_hash=geometry_hash,
        layout_hash=layout_hash,
        lighting_hash=lighting_hash,
    )
    _ = time.perf_counter() - total_start
    return BakeResult(
        coefficients=np.ascontiguousarray(coefficients),
        residuals=residuals,
        probe_irradiance=np.ascontiguousarray(probe_irradiance),
        clamped_probes=clamped_probes,
        energy_bound=float(scene.energy_bound),
        voxelization_seconds=float(voxelization_seconds),
        probe_bake_seconds=float(probe_bake_seconds),
        thread_count=worker_count,
        grid_info=grid.info(),
        geometry_hash=geometry_hash,
        lighting_hash=lighting_hash,
        layout_hash=layout_hash,
        scene_hash=scene_hash,
    )
