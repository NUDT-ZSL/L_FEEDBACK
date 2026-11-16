"""Deterministic overlapping-grain synthesis."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .analysis import round_half_up

GRAIN_SECONDS = 0.050
OVERLAP_RATIO = 0.50
CLOUD_TARGET_PEAK = 0.62


@dataclass(frozen=True)
class GrainCloud:
    audio: np.ndarray
    grains: list[dict[str, object]]
    grain_length_samples: int
    nominal_hop_samples: float
    loop_samples: int
    normalization_gain: float


def _read_rate_shifted_grain(
    source: np.ndarray, source_position: int, rate: float, grain_length: int
) -> np.ndarray:
    source_length = source.shape[0]
    source_offsets = (source_position + np.arange(grain_length) * rate) % source_length
    left = np.floor(source_offsets).astype(np.int64)
    right = (left + 1) % source_length
    fraction = (source_offsets - left)[:, None]
    return source[left] * (1.0 - fraction) + source[right] * fraction


def synthesize_grain_cloud(
    source: np.ndarray,
    sample_rate: int,
    beat_samples: int,
    seed: int,
) -> GrainCloud:
    """Build a seamless grain cloud whose loop is exactly four beats."""

    source_values = np.asarray(source, dtype=np.float64)
    if source_values.ndim != 2 or source_values.shape[0] == 0:
        raise ValueError("source must contain at least one audio frame")
    if source_values.shape[1] not in (1, 2):
        raise ValueError("source must be mono or stereo")

    grain_length = max(2, round_half_up(sample_rate * GRAIN_SECONDS))
    nominal_hop = grain_length * (1.0 - OVERLAP_RATIO)
    loop_samples = int(4 * beat_samples)
    if loop_samples < grain_length:
        raise ValueError("four-beat loop is shorter than one grain")

    starts: list[int] = []
    index = 0
    while True:
        start = round_half_up(index * nominal_hop)
        if start >= loop_samples:
            break
        starts.append(start)
        index += 1

    normalized_seed = int(seed) % (2**128)
    rng = np.random.Generator(np.random.PCG64(normalized_seed))
    grain_count = len(starts)
    semitone_choices = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=np.int64)
    maximum_rate = float(2.0 ** (int(semitone_choices.max()) / 12.0))
    source_span_limit = int(math.ceil((grain_length - 1) * maximum_rate)) + 1
    source_start_limit = source_values.shape[0] - source_span_limit + 1
    if source_start_limit > 0:
        source_positions = rng.integers(
            0, source_start_limit, size=grain_count
        )
    else:
        source_positions = rng.integers(
            0, source_values.shape[0], size=grain_count
        )
    semitones = semitone_choices[
        rng.integers(0, semitone_choices.size, size=grain_count)
    ]
    pan_steps = rng.integers(-6, 7, size=grain_count)
    pans = pan_steps.astype(np.float64) / 10.0

    source_channels = source_values.shape[1]
    cloud = np.zeros((loop_samples, 2), dtype=np.float64)
    window = np.sin(
        np.pi * (np.arange(grain_length, dtype=np.float64) + 0.5) / grain_length
    )
    timeline = np.arange(grain_length, dtype=np.int64)
    grain_records: list[dict[str, object]] = []

    for grain_index, start in enumerate(starts):
        semitone = int(semitones[grain_index])
        rate = float(2.0 ** (semitone / 12.0))
        source_position = int(source_positions[grain_index])
        grain = _read_rate_shifted_grain(
            source_values, source_position, rate, grain_length
        )
        grain *= window[:, None]

        pan = float(pans[grain_index])
        pan_angle = (pan + 1.0) * math.pi / 4.0
        left_gain = math.sqrt(2.0) * math.cos(pan_angle)
        right_gain = math.sqrt(2.0) * math.sin(pan_angle)
        if source_channels == 1:
            stereo_grain = np.column_stack(
                (grain[:, 0] * left_gain, grain[:, 0] * right_gain)
            )
        else:
            stereo_grain = np.column_stack(
                (grain[:, 0] * left_gain, grain[:, 1] * right_gain)
            )

        positions = (start + timeline) % loop_samples
        np.add.at(cloud, positions, stereo_grain)
        end_sample = start + grain_length
        wraps_loop = end_sample > loop_samples
        source_span = (grain_length - 1) * rate
        grain_records.append(
            {
                "index": grain_index,
                "start_sample": int(start),
                "end_sample": int(end_sample),
                "duration_samples": int(grain_length),
                "source_position": int(source_position),
                "source_end_position": round(source_position + source_span, 6),
                "playback_rate": round(rate, 9),
                "pitch_semitones": semitone,
                "pan": round(pan, 6),
                "wraps_loop": bool(wraps_loop),
                "wrapped_end_sample": int(end_sample % loop_samples)
                if wraps_loop
                else None,
            }
        )

    cloud_peak = float(np.max(np.abs(cloud))) if cloud.size else 0.0
    normalization_gain = 1.0
    if cloud_peak > CLOUD_TARGET_PEAK:
        normalization_gain = CLOUD_TARGET_PEAK / cloud_peak
        cloud *= normalization_gain

    return GrainCloud(
        audio=cloud,
        grains=grain_records,
        grain_length_samples=grain_length,
        nominal_hop_samples=round(nominal_hop, 6),
        loop_samples=loop_samples,
        normalization_gain=round(float(normalization_gain), 9),
    )

