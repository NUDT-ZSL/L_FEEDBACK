"""Camera path file loading, validation and deterministic interpolation.

Path files are JSON documents of the form::

    {
      "version": 1,
      "keyframes": [
        {"t": 0.0, "eye": [x, y, z], "target": [x, y, z], "fovy_deg": 55.0}
      ]
    }
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from .errors import ERR_PATH_FORMAT, ERR_PATH_MISSING, ERR_UNKNOWN, PipelineError
from .scene import CameraKeyframe


def load_path(path: str) -> list:
    if not os.path.exists(path):
        raise PipelineError(
            ERR_PATH_MISSING, f"camera path file does not exist: {path}"
        )
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PipelineError(
            ERR_PATH_FORMAT, f"camera path file is not valid JSON: {exc}"
        ) from exc
    except OSError as exc:
        raise PipelineError(
            ERR_PATH_MISSING, f"cannot read camera path file: {exc}"
        ) from exc
    return parse_path(data)


def parse_path(data: object) -> list:
    if not isinstance(data, dict) or "keyframes" not in data:
        raise PipelineError(
            ERR_PATH_FORMAT, "path document must be an object with 'keyframes'"
        )
    raw_frames = data["keyframes"]
    if not isinstance(raw_frames, list) or len(raw_frames) < 2:
        raise PipelineError(
            ERR_PATH_FORMAT, "'keyframes' must be a list with at least 2 entries"
        )
    frames: list = []
    previous_t = None
    for index, item in enumerate(raw_frames):
        if not isinstance(item, dict):
            raise PipelineError(
                ERR_PATH_FORMAT, f"keyframe {index} must be an object"
            )
        for key in ("t", "eye", "target", "fovy_deg"):
            if key not in item:
                raise PipelineError(
                    ERR_PATH_FORMAT,
                    f"keyframe {index} is missing required field '{key}'",
                )
        try:
            frame = CameraKeyframe.from_dict(item)
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineError(
                ERR_PATH_FORMAT, f"keyframe {index} is invalid: {exc}"
            ) from exc
        if previous_t is not None and frame.t <= previous_t:
            raise PipelineError(
                ERR_PATH_FORMAT,
                f"keyframe {index} time {frame.t} must be strictly increasing",
            )
        previous_t = frame.t
        frames.append(frame)
    return frames


def save_path(path: str, frames: list) -> None:
    document = {
        "version": 1,
        "keyframes": [frame.to_dict() for frame in frames],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")


def interpolate_keyframes(frames: list, time_value: float) -> CameraKeyframe:
    """Piecewise linear interpolation; values outside the range clamp."""
    if not math.isfinite(time_value):
        raise PipelineError(ERR_PATH_FORMAT, "frame time must be finite")
    if time_value <= frames[0].t:
        base = frames[0]
        return CameraKeyframe(float(time_value), base.eye.copy(),
                              base.target.copy(), base.fovy_deg)
    if time_value >= frames[-1].t:
        base = frames[-1]
        return CameraKeyframe(float(time_value), base.eye.copy(),
                              base.target.copy(), base.fovy_deg)
    for index in range(len(frames) - 1):
        a = frames[index]
        b = frames[index + 1]
        if a.t <= time_value <= b.t:
            alpha = (time_value - a.t) / (b.t - a.t)
            eye = (1.0 - alpha) * np.asarray(a.eye, dtype=np.float64) + alpha * np.asarray(
                b.eye, dtype=np.float64
            )
            target = (1.0 - alpha) * np.asarray(a.target, dtype=np.float64) + alpha * np.asarray(
                b.target, dtype=np.float64
            )
            fovy = (1.0 - alpha) * a.fovy_deg + alpha * b.fovy_deg
            return CameraKeyframe(float(time_value), eye, target, float(fovy))
    raise PipelineError(ERR_UNKNOWN, "interpolation failed")

