"""Per-frame tangent-space projection and numerically stable accumulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import BindBasis
from .scene import Scene
from .skinning import animation_frame_times, interpolate_curve


@dataclass(frozen=True)
class ProjectionResult:
    frame_times: np.ndarray
    curves: np.ndarray
    cycle_endpoint: np.ndarray
    projected_vectors: np.ndarray
    max_relative_error: np.ndarray
    max_relative_error_by_vertex: dict[int, float]
    overall_max_relative_error: float
    max_accumulation_correction: float
    loop_closure_error: float


def sample_input_curves(scene: Scene, times: np.ndarray) -> np.ndarray:
    loop = scene.loop_mode == "loop"
    result = np.empty((len(times), scene.vertex_count, 3), dtype=np.float64)
    for vertex in range(scene.vertex_count):
        curve = scene.curves[vertex]
        for frame, time in enumerate(times):
            result[frame, vertex] = interpolate_curve(
                curve.times, curve.vectors, float(time), scene.duration, loop
            )
    return result


def project_to_bind_basis(
    scene: Scene,
    basis: BindBasis,
    frame_times: np.ndarray,
    input_vectors: np.ndarray,
) -> ProjectionResult:
    t = basis.tangents[None, :, :]
    b = basis.bitangents[None, :, :]
    n = basis.normals[None, :, :]
    object_vectors = (
        input_vectors[:, :, 0:1] * t
        + input_vectors[:, :, 1:2] * b
        + input_vectors[:, :, 2:3] * n
    )
    u = np.sum(object_vectors * t, axis=2)
    v = np.sum(object_vectors * b, axis=2)
    projected = u[:, :, None] * t + v[:, :, None] * b
    residual = object_vectors - projected
    source_norm = np.linalg.norm(object_vectors, axis=2)
    residual_norm = np.linalg.norm(residual, axis=2)
    relative = np.divide(
        residual_norm,
        source_norm,
        out=np.zeros_like(residual_norm),
        where=source_norm > 1e-15,
    )

    raw = np.stack([u, v], axis=2)
    if scene.displacement_mode == "delta":
        curves, endpoint, correction, closure = _accumulate_delta(
            raw, scene.loop_mode == "loop"
        )
    else:
        curves = raw.copy()
        endpoint = curves[0].copy() if scene.loop_mode == "loop" else curves[-1].copy()
        correction = 0.0
        closure = (
            float(np.max(np.linalg.norm(curves[0] - endpoint, axis=1)))
            if scene.loop_mode == "loop"
            else 0.0
        )

    by_vertex = {
        vertex: float(np.max(relative[:, vertex])) for vertex in range(scene.vertex_count)
    }
    return ProjectionResult(
        frame_times=np.asarray(frame_times, dtype=np.float64),
        curves=curves,
        cycle_endpoint=endpoint,
        projected_vectors=np.stack([u, v], axis=2),
        max_relative_error=np.max(relative, axis=0),
        max_relative_error_by_vertex=by_vertex,
        overall_max_relative_error=float(np.max(relative)),
        max_accumulation_correction=correction,
        loop_closure_error=closure,
    )


def _accumulate_delta(
    raw: np.ndarray, loop: bool
) -> tuple[np.ndarray, np.ndarray, float, float]:
    frames, vertices, _ = raw.shape
    adjusted = raw.copy()
    correction = 0.0
    if loop and frames > 0:
        totals = raw.sum(axis=0)
        adjusted -= totals[None, :, :] / float(frames)
        correction = float(np.max(np.linalg.norm(totals / float(frames), axis=1)))

    total = np.zeros((vertices, 2), dtype=np.float64)
    compensation = np.zeros_like(total)
    out = np.empty_like(adjusted)
    for frame in range(frames):
        out[frame] = total
        increment = adjusted[frame] - compensation
        updated = total + increment
        compensation = (updated - total) - increment
        total = updated
    endpoint = np.zeros_like(total) if loop else total
    closure = (
        float(np.max(np.linalg.norm(out[0] - endpoint, axis=1))) if loop else 0.0
    )
    return out, endpoint, correction, closure


def _loop_closure_error(curves: np.ndarray) -> float:
    if curves.shape[0] < 2:
        return 0.0
    return float(np.max(np.linalg.norm(curves[0] - curves[-1], axis=1)))


def frame_count(scene: Scene) -> int:
    return int(len(animation_frame_times(scene)))


def resample_projection(
    scene: Scene,
    projection: ProjectionResult,
    sample_count: int,
) -> np.ndarray:
    """Resample accumulated animation-frame curves onto texture time samples."""
    frame_times = projection.frame_times
    curves = projection.curves
    if scene.loop_mode == "loop":
        query_times = np.arange(sample_count, dtype=np.float64) / (
            sample_count / scene.duration
        )
        end_time = scene.duration
    else:
        if sample_count < 2:
            query_times = np.zeros(1, dtype=np.float64)
        else:
            query_times = np.linspace(0.0, scene.duration, sample_count, dtype=np.float64)
        end_time = scene.duration

    extended_times = np.concatenate([frame_times, np.array([end_time], dtype=np.float64)])
    extended_values = np.concatenate(
        [curves, projection.cycle_endpoint[None, :, :]], axis=0
    )
    out = np.empty((sample_count, scene.vertex_count, 2), dtype=np.float64)
    for vertex in range(scene.vertex_count):
        series = extended_values[:, vertex, :]
        out[:, vertex, 0] = np.interp(query_times, extended_times, series[:, 0])
        out[:, vertex, 1] = np.interp(query_times, extended_times, series[:, 1])
    return out

