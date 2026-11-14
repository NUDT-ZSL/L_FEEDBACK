"""Deterministic camera keyframe interpolation."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any

from .scene import CameraKeyframe, Scene
from .util import quantize, quantize_vec


Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


@dataclass(frozen=True)
class CameraState:
    frame_index: int
    time: float
    from_keyframe_index: int
    to_keyframe_index: int
    alpha: float
    position: Vec3
    orientation: Quat
    forward: Vec3
    right: Vec3
    up: Vec3
    fov_y_deg: float
    near: float
    far: float

    def stats(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "time": quantize(self.time),
            "segment": {
                "from_keyframe_index": self.from_keyframe_index,
                "to_keyframe_index": self.to_keyframe_index,
                "alpha": quantize(self.alpha),
                "from_time": None,
                "to_time": None,
            },
            "interpolation": {
                "position": "linear",
                "orientation": "shortest_path_slerp_normalized",
                "fov": "linear",
                "near_far": "linear",
            },
            "position": quantize_vec(self.position),
            "orientation_xyzw": quantize_vec(self.orientation),
            "forward": quantize_vec(self.forward),
            "right": quantize_vec(self.right),
            "up": quantize_vec(self.up),
            "fov_y_deg": quantize(self.fov_y_deg),
            "near": quantize(self.near),
            "far": quantize(self.far),
        }


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(left * right for left, right in zip(a, b))


def _normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    length = math.sqrt(_dot(values, values))
    if length <= 1e-15:
        raise ValueError("zero-length vector")
    return tuple(value / length for value in values)


def _lerp_tuple(a: tuple[float, ...], b: tuple[float, ...], alpha: float) -> tuple[float, ...]:
    return tuple((1.0 - alpha) * left + alpha * right for left, right in zip(a, b))


def slerp(q0: Quat, q1: Quat, alpha: float) -> Quat:
    """Normalized shortest-path spherical interpolation of xyzw quaternions."""

    dot = _dot(q0, q1)
    if dot < 0.0:
        q1 = tuple(-value for value in q1)
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalize(_lerp_tuple(q0, q1, alpha))  # type: ignore[return-value]
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    if sin_theta <= 1e-12:
        return _normalize(q0)  # type: ignore[return-value]
    left_weight = math.sin((1.0 - alpha) * theta) / sin_theta
    right_weight = math.sin(alpha * theta) / sin_theta
    result = tuple(
        left_weight * left + right_weight * right for left, right in zip(q0, q1)
    )
    return _normalize(result)  # type: ignore[return-value]


def rotate_vector(quaternion: Quat, vector: Vec3) -> Vec3:
    x, y, z, w = quaternion
    # q * v * q^-1, expanded for a unit quaternion.
    tx = 2.0 * (y * vector[2] - z * vector[1])
    ty = 2.0 * (z * vector[0] - x * vector[2])
    tz = 2.0 * (x * vector[1] - y * vector[0])
    return (
        vector[0] + w * tx + (y * tz - z * ty),
        vector[1] + w * ty + (z * tx - x * tz),
        vector[2] + w * tz + (x * ty - y * tx),
    )


def _segment(keyframes: tuple[CameraKeyframe, ...], time: float) -> tuple[int, float]:
    if len(keyframes) == 1:
        return 0, 0.0
    times = [keyframe.time for keyframe in keyframes]
    index = bisect.bisect_right(times, time) - 1
    if index < 0:
        index = 0
    if index >= len(keyframes) - 1:
        index = len(keyframes) - 2
    start = keyframes[index].time
    end = keyframes[index + 1].time
    alpha = 0.0 if end == start else (time - start) / (end - start)
    alpha = min(1.0, max(0.0, alpha))
    return index, alpha


def interpolate_camera(scene: Scene, frame_index: int) -> CameraState:
    time = scene.frame_times[frame_index]
    segment_index, alpha = _segment(scene.keyframes, time)
    from_keyframe = scene.keyframes[segment_index]
    to_keyframe = scene.keyframes[segment_index + 1] if len(scene.keyframes) > 1 else from_keyframe

    position = _lerp_tuple(from_keyframe.position, to_keyframe.position, alpha)
    orientation = slerp(from_keyframe.orientation, to_keyframe.orientation, alpha)
    fov = (1.0 - alpha) * from_keyframe.fov_y_deg + alpha * to_keyframe.fov_y_deg
    near = (1.0 - alpha) * from_keyframe.near + alpha * to_keyframe.near
    far = (1.0 - alpha) * from_keyframe.far + alpha * to_keyframe.far

    forward = rotate_vector(orientation, (0.0, 0.0, -1.0))
    right = rotate_vector(orientation, (1.0, 0.0, 0.0))
    up = rotate_vector(orientation, (0.0, 1.0, 0.0))
    return CameraState(
        frame_index=frame_index,
        time=time,
        from_keyframe_index=segment_index,
        to_keyframe_index=segment_index + 1 if len(scene.keyframes) > 1 else 0,
        alpha=alpha,
        position=position,  # type: ignore[arg-type]
        orientation=orientation,
        forward=forward,
        right=right,
        up=up,
        fov_y_deg=fov,
        near=near,
        far=far,
    )


def camera_stats_with_segment_times(camera: CameraState, scene: Scene) -> dict[str, Any]:
    stats = camera.stats()
    segment = stats["segment"]
    segment["from_time"] = quantize(scene.keyframes[camera.from_keyframe_index].time)
    segment["to_time"] = quantize(scene.keyframes[camera.to_keyframe_index].time)
    return stats

