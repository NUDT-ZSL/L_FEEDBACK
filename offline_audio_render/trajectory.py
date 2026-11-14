"""Linear-interpolated listener trajectory with endpoint clamping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spatial import ListenerPose


@dataclass(frozen=True)
class TrajectoryKey:
    time: float
    position: tuple[float, float, float]
    yaw: float


class ListenerTrajectory:
    def __init__(self, keys: list[TrajectoryKey]) -> None:
        if not keys:
            raise ValueError("trajectory must contain at least one key")
        self._times = np.array([k.time for k in keys], dtype=np.float64)
        self._positions = np.array(
            [k.position for k in keys], dtype=np.float64
        )
        self._yaws = np.array([k.yaw for k in keys], dtype=np.float64)

    def pose_at(self, time_s: float) -> ListenerPose:
        """Interpolate pose at ``time_s``; clamp outside the keyframe range."""
        t = float(time_s)
        if t <= self._times[0]:
            return ListenerPose(self._positions[0].copy(), float(self._yaws[0]))
        if t >= self._times[-1]:
            return ListenerPose(self._positions[-1].copy(), float(self._yaws[-1]))
        index = int(np.searchsorted(self._times, t, side="right") - 1)
        index = min(index, len(self._times) - 2)
        span = self._times[index + 1] - self._times[index]
        fraction = 0.0 if span <= 0.0 else (t - self._times[index]) / span
        position = (
            self._positions[index]
            + (self._positions[index + 1] - self._positions[index]) * fraction
        )
        yaw = self._yaws[index] + (
            self._yaws[index + 1] - self._yaws[index]
        ) * fraction
        return ListenerPose(position, float(yaw))
