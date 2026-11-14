from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

from .constants import METADATA_FIELDS
from .scene import AudioObject, Keyframe, Scene


def seconds_to_frames(seconds: Decimal | float | int, sample_rate: int) -> int:
    value = seconds if isinstance(seconds, Decimal) else Decimal(str(seconds))
    return int((value * sample_rate).to_integral_value(rounding=ROUND_HALF_UP))


def frame_to_seconds(frame: int, sample_rate: int) -> Decimal:
    return Decimal(frame) / Decimal(sample_rate)


def block_count(total_samples: int, frame_length: int) -> int:
    if total_samples < 0:
        raise ValueError("total_samples must be non-negative")
    return (total_samples + frame_length - 1) // frame_length


def _interpolate(keyframes: tuple[Keyframe, ...], time: Decimal) -> Keyframe:
    if time <= keyframes[0].time:
        return keyframes[0]
    if time >= keyframes[-1].time:
        return keyframes[-1]
    lo = 0
    hi = len(keyframes) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if keyframes[mid].time <= time:
            lo = mid
        else:
            hi = mid
    left = keyframes[lo]
    right = keyframes[hi]
    ratio = float((time - left.time) / (right.time - left.time))
    return Keyframe(
        time=time,
        azimuth=left.azimuth + (right.azimuth - left.azimuth) * ratio,
        elevation=left.elevation + (right.elevation - left.elevation) * ratio,
        distance=left.distance + (right.distance - left.distance) * ratio,
        gain=left.gain + (right.gain - left.gain) * ratio,
        size=left.size + (right.size - left.size) * ratio,
    )


def sample_object(obj: AudioObject, time: Decimal) -> Keyframe:
    return _interpolate(obj.keyframes, time)


def metadata_row(keyframe: Keyframe) -> tuple[float, float, float, float, float]:
    return (
        keyframe.azimuth,
        keyframe.elevation,
        keyframe.distance,
        keyframe.gain,
        keyframe.size,
    )


def iter_metadata_blocks(scene: Scene, total_samples: int):
    blocks = block_count(total_samples, scene.frame_length)
    for block_index in range(blocks):
        time = frame_to_seconds(block_index * scene.frame_length, scene.sample_rate)
        values = np.asarray(
            [metadata_row(sample_object(obj, time)) for obj in scene.objects],
            dtype=np.float32,
        )
        yield block_index, values


def metadata_array(scene: Scene, total_samples: int) -> np.ndarray:
    blocks = block_count(total_samples, scene.frame_length)
    result = np.empty((blocks, scene.object_count, len(METADATA_FIELDS)), dtype=np.float32)
    for block_index, values in iter_metadata_blocks(scene, total_samples):
        result[block_index] = values
    return result


def metadata_block_time(scene: Scene, block_index: int) -> Decimal:
    return frame_to_seconds(block_index * scene.frame_length, scene.sample_rate)


def expected_wav_frames(duration_seconds: float, sample_rate: int) -> int:
    if not math.isfinite(duration_seconds) or duration_seconds < 0:
        raise ValueError("duration_seconds must be finite and non-negative")
    return seconds_to_frames(duration_seconds, sample_rate)
