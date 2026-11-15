from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .schema import DialogueSpec, ProjectSpec, TrackSpec


@dataclass
class SpatialLine:
    stereo: np.ndarray
    frame_table: np.ndarray
    diagnostics: Dict[str, Any]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _frame_position(track: TrackSpec, frame: int, fps: float) -> np.ndarray:
    count = track.frame_count
    if frame < count:
        return track.positions[frame]
    velocity = _frame_velocity(track, count - 1, fps)
    return track.positions[-1] + velocity / fps


def _frame_forward(track: TrackSpec, frame: int) -> np.ndarray:
    return track.forwards[min(frame, track.frame_count - 1)]


def _frame_velocity(track: TrackSpec, frame: int, fps: float) -> np.ndarray:
    count = track.frame_count
    if count <= 1:
        return np.zeros(3, dtype=np.float64)
    if frame <= 0:
        return (track.positions[1] - track.positions[0]) * fps
    if frame >= count - 1:
        return (track.positions[-1] - track.positions[-2]) * fps
    return (track.positions[frame + 1] - track.positions[frame - 1]) * (fps / 2.0)


def _equal_power_pan(pan: float) -> tuple[float, float]:
    angle = (pan + 1.0) * math.pi / 4.0
    return math.cos(angle), math.sin(angle)


def _fade_envelope(sample_count: int, fade_samples: int) -> np.ndarray:
    envelope = np.ones(sample_count, dtype=np.float64)
    fade = min(max(0, fade_samples), sample_count // 2)
    if fade <= 0:
        return envelope
    phase = (np.arange(fade, dtype=np.float64) + 0.5) / float(fade)
    ramp = 0.5 - 0.5 * np.cos(math.pi * phase)
    envelope[:fade] *= ramp
    envelope[-fade:] *= ramp[::-1]
    return envelope


def spatialize_line(
    project: ProjectSpec,
    dialogue: DialogueSpec,
    source_mono: np.ndarray,
) -> SpatialLine:
    timebase = project.timebase
    fps = float(project.fps)
    sample_count = dialogue.end_sample - dialogue.start_sample
    if source_mono.size != sample_count:
        raise ValueError("source sample length does not match the dialogue interval")
    if sample_count <= 0:
        raise ValueError("dialogue interval must contain at least one sample")

    track = project.tracks[dialogue.speaker]
    camera_position = np.asarray(project.camera["position"], dtype=np.float64)
    camera_right = np.asarray(project.camera["right"], dtype=np.float64)
    sound_speed = project.speed_of_sound_mps
    reference_distance = project.distance_reference_m

    boundary_frames = list(range(dialogue.start_frame, dialogue.end_frame + 1))
    knot_samples = np.asarray(
        [
            timebase.frame_to_sample(frame) - dialogue.start_sample
            for frame in boundary_frames
        ],
        dtype=np.float64,
    )
    knot_samples[-1] = float(sample_count)
    left_gain_knots = np.empty(len(boundary_frames), dtype=np.float64)
    right_gain_knots = np.empty(len(boundary_frames), dtype=np.float64)
    doppler_rate_knots = np.empty(len(boundary_frames), dtype=np.float64)
    frame_table = np.empty((len(boundary_frames), 7), dtype=np.float64)

    for index, frame in enumerate(boundary_frames):
        position = _frame_position(track, frame, fps)
        forward = _frame_forward(track, frame)
        relative = position - camera_position
        distance = float(np.linalg.norm(relative))
        if distance <= 1e-12:
            direction_to_camera = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            distance = 1e-12
        else:
            direction_to_camera = -relative / distance
        distance_gain = min(1.0, reference_distance / max(distance, reference_distance))
        directivity = 0.65 + 0.35 * max(
            0.0, float(np.dot(forward, direction_to_camera))
        )
        lateral = float(np.dot(relative, camera_right)) / max(distance, 1e-12)
        pan = _clamp(project.pan_strength * lateral, -1.0, 1.0)
        pan_left, pan_right = _equal_power_pan(pan)
        left_gain = distance_gain * directivity * pan_left
        right_gain = distance_gain * directivity * pan_right

        velocity = _frame_velocity(track, min(frame, track.frame_count - 1), fps)
        radial_velocity = float(np.dot(velocity, direction_to_camera))
        radial_velocity = _clamp(radial_velocity, -0.9 * sound_speed, 0.9 * sound_speed)
        doppler_factor = sound_speed / (sound_speed - radial_velocity)
        doppler_rate = 1.0 / doppler_factor

        left_gain_knots[index] = left_gain
        right_gain_knots[index] = right_gain
        doppler_rate_knots[index] = doppler_rate
        frame_table[index] = (
            float(frame),
            distance,
            distance_gain,
            pan,
            doppler_factor,
            left_gain,
            right_gain,
        )

    sample_positions = np.arange(sample_count, dtype=np.float64)
    left_envelope = np.interp(sample_positions, knot_samples, left_gain_knots)
    right_envelope = np.interp(sample_positions, knot_samples, right_gain_knots)
    doppler_rate = np.interp(sample_positions, knot_samples, doppler_rate_knots)

    trapezoid_steps = (doppler_rate[:-1] + doppler_rate[1:]) * 0.5
    warp_phase = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(trapezoid_steps, dtype=np.float64))
    )
    if sample_count > 1 and float(warp_phase[-1]) > 0.0:
        source_positions = warp_phase / warp_phase[-1] * (sample_count - 1)
    else:
        source_positions = sample_positions
    warped_mono = np.interp(
        source_positions,
        sample_positions,
        np.asarray(source_mono, dtype=np.float64),
    )

    fade_samples = int(round(project.fade_ms * 48.0))
    fade = _fade_envelope(sample_count, fade_samples)
    left = warped_mono * left_envelope * fade
    right = warped_mono * right_envelope * fade
    stereo = np.ascontiguousarray(np.stack((left, right), axis=1), dtype=np.float64)

    active_table = frame_table[:-1]
    diagnostics = {
        "frame_segment_count": dialogue.end_frame - dialogue.start_frame,
        "distance_m_min": float(np.min(active_table[:, 1])),
        "distance_m_max": float(np.max(active_table[:, 1])),
        "distance_gain_min": float(np.min(active_table[:, 2])),
        "distance_gain_max": float(np.max(active_table[:, 2])),
        "pan_min": float(np.min(active_table[:, 3])),
        "pan_max": float(np.max(active_table[:, 3])),
        "doppler_factor_min": float(np.min(active_table[:, 4])),
        "doppler_factor_max": float(np.max(active_table[:, 4])),
        "left_gain_min": float(np.min(active_table[:, 5])),
        "left_gain_max": float(np.max(active_table[:, 5])),
        "right_gain_min": float(np.min(active_table[:, 6])),
        "right_gain_max": float(np.max(active_table[:, 6])),
        "fade_samples": fade_samples,
        "gain_interpolation": "linear_between_frame_knots",
        "doppler_mode": "aligned_frame_integrated_variable_rate_warp",
    }
    return SpatialLine(stereo=stereo, frame_table=frame_table, diagnostics=diagnostics)


def spatialize_all(
    project: ProjectSpec,
    sources: Dict[str, np.ndarray],
) -> List[SpatialLine]:
    lines: List[SpatialLine] = []
    for dialogue in project.dialogues:
        lines.append(spatialize_line(project, dialogue, sources[dialogue.id]))
    return lines

