"""Pure, deterministic DSP helpers: fades, ducking, gain, limiter, mixing.

Everything in this module is a side-effect-free function of its inputs, which
is what makes block-at-a-time rendering byte-identical to a single full render:
every envelope is computed analytically from absolute frame positions and the
normalized plan, never from accumulated render state.
"""

from __future__ import annotations

import math
import struct
from typing import Dict, List, Optional, Tuple

from .audioio import INT16_MAX, WavData
from .model import PlannedSegment, TimelinePlan

LIMITER_CEILING_DBFS = -1.0


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    if linear <= 1e-12:
        return -120.0
    return 20.0 * math.log10(linear)


def ms_to_frames(ms: float, sample_rate: int) -> int:
    return int(round(ms * sample_rate / 1000.0))


def frames_to_ms(frames: int, sample_rate: int) -> float:
    return frames * 1000.0 / sample_rate


def segment_fade_gain(local_frame: int, frame_count: int, fade_in: int, fade_out: int) -> float:
    """Linear 0->1 fade-in and 1->0 fade-out, exact zero at the outer edges."""
    if fade_in > 0 and local_frame < fade_in:
        return local_frame / fade_in
    if fade_out > 0 and local_frame >= frame_count - fade_out:
        return (frame_count - 1 - local_frame) / fade_out
    return 1.0


def duck_envelope(
    master_frame: int,
    segment: PlannedSegment,
    attack_frames: int,
    release_frames: int,
    duck_gain_linear: float,
) -> float:
    """Piecewise-linear ambience ducking gain around the dialogue interval."""
    start = segment.duck_start_frame
    end = segment.duck_end_frame
    if end <= start:
        return 1.0
    if master_frame < start - attack_frames or master_frame >= end + release_frames:
        return 1.0
    if start - attack_frames <= master_frame < start:
        position = master_frame - (start - attack_frames)
        return 1.0 + (duck_gain_linear - 1.0) * position / max(1, attack_frames)
    if start <= master_frame < end:
        return duck_gain_linear
    position = master_frame - end
    return duck_gain_linear + (1.0 - duck_gain_linear) * position / max(1, release_frames)


def measure_clip(frames: List[int]) -> Tuple[float, float]:
    """Return peak/RMS amplitudes normalized to [0, 1] for 16-bit PCM."""
    if not frames:
        return 0.0, 0.0
    peak = max(abs(sample) for sample in frames) / float(INT16_MAX)
    rms = math.sqrt(sum(sample * sample for sample in frames) / len(frames)) / float(INT16_MAX)
    return peak, rms


def compute_segment_gain(
    peak: float,
    rms: float,
    target_peak_dbfs: float,
    loudness_target_dbfs: float,
) -> Tuple[float, float, float]:
    """Loudness + peak-derived gain, conservative (smaller of the two)."""
    measured_peak_db = linear_to_db(peak) if peak > 0 else -120.0
    measured_rms_db = linear_to_db(rms) if rms > 0 else -120.0
    loudness_gain_db = loudness_target_dbfs - measured_rms_db
    peak_gain_db = target_peak_dbfs - measured_peak_db
    applied_db = min(loudness_gain_db, peak_gain_db)
    applied_db = max(-40.0, min(20.0, applied_db))
    return applied_db, loudness_gain_db, peak_gain_db


def smoothed_gain(
    master_frame: int,
    segment: PlannedSegment,
    current_gain: float,
    previous_gain: Optional[float],
    smoothing_frames: int,
) -> float:
    """Linear ramp from the previous segment gain at the segment boundary."""
    if previous_gain is None or smoothing_frames <= 0:
        return current_gain
    local = master_frame - segment.start_frame
    if local >= smoothing_frames:
        return current_gain
    return previous_gain + (current_gain - previous_gain) * local / smoothing_frames


def limit_int16(value: float, ceiling_linear: float) -> int:
    """Round to int16 and hard-limit to the configured ceiling (default -1 dBFS)."""
    ceiling = math.floor(ceiling_linear * INT16_MAX)
    rounded = int(math.floor(abs(value) * INT16_MAX + 0.5))
    if value < 0:
        rounded = -rounded
    if rounded > ceiling:
        return ceiling
    if rounded < -ceiling:
        return -ceiling
    return rounded


def find_segment(plan: TimelinePlan, master_frame: int) -> Optional[PlannedSegment]:
    for segment in plan.segments:
        if segment.start_frame <= master_frame < segment.end_frame:
            return segment
    return None


def render_block(
    plan: TimelinePlan,
    block_index: int,
    sources: Dict[int, Tuple[WavData, WavData]],
    segment_gains: Dict[int, float],
    previous_gains: Dict[int, Optional[float]],
) -> bytes:
    """Render one master block to interleaved 16-bit stereo little-endian PCM."""
    start = block_index * plan.block_frames
    frame_count = min(plan.block_frames, plan.total_frames - start)
    if frame_count <= 0:
        return b""

    attack = max(1, ms_to_frames(plan.duck_attack_ms, plan.sample_rate))
    release = max(1, ms_to_frames(plan.duck_release_ms, plan.sample_rate))
    smoothing = ms_to_frames(plan.gain_smoothing_ms, plan.sample_rate)
    duck_linear = db_to_linear(plan.duck_gain_db)
    ceiling_linear = db_to_linear(plan.limiter_ceiling_dbfs)

    pcm = bytearray(frame_count * 4)
    out_offset = 0
    for master_frame in range(start, start + frame_count):
        segment = find_segment(plan, master_frame)
        if segment is None:
            out_offset += 4  # gap: digital silence
            continue
        local_frame = master_frame - segment.start_frame
        voice_clip, ambience_clip = sources[segment.index]
        voice_sample = voice_clip.frames[local_frame] / float(INT16_MAX)
        ambience_sample = ambience_clip.frames[local_frame] / float(INT16_MAX)

        fade = segment_fade_gain(
            local_frame,
            segment.frame_count,
            segment.fade_in_frames,
            segment.fade_out_frames,
        )
        gain = smoothed_gain(
            master_frame,
            segment,
            segment_gains[segment.index],
            previous_gains[segment.index],
            smoothing,
        )
        duck = duck_envelope(master_frame, segment, attack, release, duck_linear)

        voice_out = voice_sample * fade * gain
        ambience_out = ambience_sample * fade * duck * gain
        mixed = voice_out + ambience_out
        limited = limit_int16(mixed, ceiling_linear)
        struct.pack_into("<hh", pcm, out_offset, limited, limited)
        out_offset += 4
    return bytes(pcm)
