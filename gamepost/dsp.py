from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction

import numpy as np

from .config import ExportConfig
from .timeline import Segment, Timeline


EPSILON = 1.0e-12
_PI = 3.141592653589793238462643383279502884
_TWO_PI = 2.0 * _PI


def db_to_linear(value_db: float) -> float:
    return float(10.0 ** (value_db / 20.0))


def linear_to_db(value: float, floor_db: float = -120.0) -> float:
    if value <= 0.0 or not math.isfinite(value):
        return floor_db
    return max(floor_db, 20.0 * math.log10(value))


def deterministic_sin(values: np.ndarray | float) -> np.ndarray:
    """A fixed polynomial sine path used for interpolation coefficients.

    The input is first reduced to [-pi, pi]. The degree-21 Taylor series is far
    more accurate than 16-bit output needs and avoids platform libm differences.
    """
    x = np.asarray(values, dtype=np.float64)
    reduced = x - _TWO_PI * np.floor((x + _PI) / _TWO_PI)
    x2 = reduced * reduced
    term = reduced.copy()
    result = reduced.copy()
    for order in range(1, 11):
        denominator = float((2 * order) * (2 * order + 1))
        term = -term * x2 / denominator
        result = result + term
    return result


def _sinc(values: np.ndarray) -> np.ndarray:
    result = np.ones_like(values, dtype=np.float64)
    nonzero = np.abs(values) > 1.0e-15
    result[nonzero] = deterministic_sin(_PI * values[nonzero]) / (
        _PI * values[nonzero]
    )
    return result


@lru_cache(maxsize=32)
def _sinc_table(taps: int, phases: int, cutoff_key: int) -> np.ndarray:
    cutoff = cutoff_key / 1_000_000.0
    radius = taps // 2
    offsets = np.arange(-radius + 1, radius + 1, dtype=np.float64)
    phase_values = np.arange(phases, dtype=np.float64) / float(phases)
    distances = offsets[None, :] - phase_values[:, None]
    kernel = cutoff * _sinc(cutoff * distances) * _sinc(distances / radius)
    row_sums = np.zeros(phases, dtype=np.float64)
    for tap in range(taps):
        row_sums += kernel[:, tap]
    kernel /= row_sums[:, None]
    return kernel.astype(np.float64, copy=False)


def _coefficient_table(rate: float, taps: int, phases: int) -> tuple[np.ndarray, int]:
    if rate > 1.0:
        cutoff = 0.94 / rate
    else:
        cutoff = 0.94
    cutoff = max(0.05, min(1.0, cutoff))
    cutoff_key = int(round(cutoff * 1_000_000.0))
    return _sinc_table(taps, phases, cutoff_key), taps // 2


def _resample_event_segment(
    source: np.ndarray,
    segment: Segment,
    taps: int,
    phases: int,
) -> np.ndarray:
    source_frames = segment.source_frames
    output_frames = segment.output_frames
    source_slice = source[segment.source_start : segment.source_end]
    if output_frames == source_frames:
        return source_slice.copy()
    if source_frames <= 0 or output_frames <= 0:
        return np.zeros((output_frames, source.shape[1]), dtype=np.float64)

    table, radius = _coefficient_table(float(segment.effective_rate), taps, phases)
    offsets = np.arange(-radius + 1, radius + 1, dtype=np.int64)
    result = np.empty((output_frames, source.shape[1]), dtype=np.float64)
    block_frames = 65_536
    for block_start in range(0, output_frames, block_frames):
        block_end = min(output_frames, block_start + block_frames)
        local = np.arange(block_start, block_end, dtype=np.int64)
        numerator = local * source_frames
        base = numerator // output_frames
        remainder = numerator - base * output_frames
        phase = (remainder * phases + output_frames // 2) // output_frames
        carry = phase >= phases
        base = base + carry.astype(np.int64)
        phase = np.where(carry, phase - phases, phase)
        indices = base[:, None] + offsets[None, :]
        np.clip(indices, 0, source_frames - 1, out=indices)

        gathered = source_slice[indices, :]
        weights = table[phase, :]
        accumulator = np.zeros(
            (block_end - block_start, source.shape[1]), dtype=np.float64
        )
        # A fixed tap order avoids dependence on reduction tree/vector width.
        for tap in range(taps):
            accumulator += gathered[:, tap, :] * weights[:, tap, None]
        result[block_start:block_end] = accumulator
    return result


def resample_narration(
    source: np.ndarray,
    timeline: Timeline,
    config: ExportConfig,
) -> np.ndarray:
    output = np.zeros(
        (timeline.output_frames, source.shape[1]), dtype=np.float64
    )
    for segment in timeline.segments:
        if segment.kind == "normal":
            target = output[segment.output_start : segment.output_end]
            np.copyto(
                target,
                source[segment.source_start : segment.source_end],
            )
        else:
            output[segment.output_start : segment.output_end] = (
                _resample_event_segment(
                    source,
                    segment,
                    config.sinc_taps,
                    config.sinc_phases,
                )
            )
    return output


@dataclass
class ClipResult:
    pcm: np.ndarray
    stats: dict


def _catmull_rom(
    x0: float,
    x1: float,
    x2: float,
    x3: float,
    t: float,
) -> float:
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2.0 * x1
        + (-x0 + x2) * t
        + (2.0 * x0 - 5.0 * x1 + 4.0 * x2 - x3) * t2
        + (-x0 + 3.0 * x1 - 3.0 * x2 + x3) * t3
    )


def repair_clipping(pcm: np.ndarray, config: ExportConfig) -> ClipResult:
    repaired = pcm.copy()
    frames, channels = repaired.shape
    runs: list[dict] = []
    per_channel = [0 for _ in range(channels)]
    clipped_mask = np.zeros_like(pcm, dtype=bool)
    original_peak = int(np.max(np.abs(pcm.astype(np.int32)))) if pcm.size else 0

    for channel in range(channels):
        values = repaired[:, channel]
        clipped = np.abs(values.astype(np.int32)) >= config.clip_threshold
        clipped_mask[:, channel] = clipped
        if not np.any(clipped):
            continue
        edges = np.diff(np.concatenate(([0], clipped.astype(np.int8), [0])))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        for start, end in zip(starts.tolist(), ends.tolist()):
            length = end - start
            if length < config.clip_min_run:
                continue
            left_index = start - 1
            right_index = end
            x1 = float(values[left_index]) if left_index >= 0 else float(values[start])
            x0 = (
                float(values[left_index - 1])
                if left_index - 1 >= 0
                else x1
            )
            x2 = (
                float(values[right_index])
                if right_index < frames
                else float(values[end - 1])
            )
            x3 = (
                float(values[right_index + 1])
                if right_index + 1 < frames
                else x2
            )
            for index in range(start, end):
                t = (index - start + 1.0) / (length + 1.0)
                value = _catmull_rom(x0, x1, x2, x3, t)
                repaired[index, channel] = int(
                    max(-32768, min(32767, int(round(value))))
                )
            per_channel[channel] += length
            runs.append(
                {
                    "channel": channel,
                    "start_sample": int(start),
                    "end_sample": int(end),
                    "samples": int(length),
                }
            )

    repaired_peak = (
        int(np.max(np.abs(repaired.astype(np.int32)))) if repaired.size else 0
    )
    unchanged = bool(
        np.array_equal(repaired[~clipped_mask], pcm[~clipped_mask])
    )
    stats = {
        "threshold": config.clip_threshold,
        "min_run_samples": config.clip_min_run,
        "runs": runs,
        "run_count": len(runs),
        "samples_repaired": int(sum(per_channel)),
        "samples_repaired_per_channel": per_channel,
        "original_peak_int16": original_peak,
        "repaired_peak_int16": repaired_peak,
        "non_clipped_samples_unchanged": unchanged,
    }
    return ClipResult(pcm=repaired, stats=stats)


def _merge_quiet_runs(
    quiet: np.ndarray,
    starts: np.ndarray,
    window: int,
    merge_gap_frames: int,
) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    indexes = np.flatnonzero(quiet)
    if indexes.size == 0:
        return runs
    run_start = int(indexes[0])
    previous = int(indexes[0])
    for index in indexes[1:]:
        index = int(index)
        gap = int(starts[index]) - (int(starts[previous]) + window)
        if gap > merge_gap_frames:
            runs.append((int(starts[run_start]), int(starts[previous]) + window))
            run_start = index
        previous = index
    runs.append((int(starts[run_start]), int(starts[previous]) + window))
    return runs


def detect_silence(
    audio: np.ndarray,
    sample_rate: int,
    config: ExportConfig,
) -> list[dict]:
    frames = int(audio.shape[0])
    if frames == 0:
        return []
    mono = np.mean(audio, axis=1, dtype=np.float64)
    window = max(1, int(round(config.silence_window_ms * sample_rate / 1000.0)))
    hop = max(1, int(round(config.silence_hop_ms * sample_rate / 1000.0)))
    if frames < window:
        starts = np.array([0], dtype=np.int64)
        window = frames
    else:
        starts = np.arange(0, frames - window + 1, hop, dtype=np.int64)

    squared = mono * mono
    cumulative = np.concatenate(
        (np.array([0.0], dtype=np.float64), np.cumsum(squared, dtype=np.float64))
    )
    sums = cumulative[starts + window] - cumulative[starts]
    rms = np.sqrt(np.maximum(sums / float(window), 0.0))
    dbfs = 20.0 * np.log10(np.maximum(rms, EPSILON))
    quiet = dbfs <= config.silence_threshold_dbfs
    merge_gap = int(round(config.silence_merge_ms * sample_rate / 1000.0))
    raw_runs = _merge_quiet_runs(quiet, starts, window, merge_gap)

    minimum = int(round(config.silence_min_ms * sample_rate / 1000.0))
    fade = int(round(config.silence_fade_ms * sample_rate / 1000.0))
    segments: list[dict] = []
    for start, end in raw_runs:
        start = max(0, min(frames, start))
        end = max(start, min(frames, end))
        if end - start < minimum:
            continue
        segment_fade = min(fade, (end - start) // 2)
        segments.append(
            {
                "start_sample": int(start),
                "end_sample": int(end),
                "samples": int(end - start),
                "gain_linear": 0.0,
                "gain_db": -120.0,
                "fade_samples": int(segment_fade),
            }
        )
    return segments


def apply_silence_gate(audio: np.ndarray, segments: list[dict]) -> np.ndarray:
    frames = int(audio.shape[0])
    gain = np.ones(frames, dtype=np.float64)
    for segment in segments:
        start = int(segment["start_sample"])
        end = int(segment["end_sample"])
        fade = min(int(segment["fade_samples"]), max(0, (end - start) // 2))
        gain[start:end] = 0.0
        if fade > 0:
            positions = np.arange(fade, dtype=np.float64) / float(fade)
            fade_out = 0.5 + 0.5 * deterministic_sin(_PI * (positions + 0.5))
            fade_in = 0.5 - 0.5 * deterministic_sin(_PI * (positions + 0.5))
            gain[start : start + fade] = fade_out
            gain[end - fade : end] = fade_in
    return audio * gain[:, None]


K_PRE_B = np.array(
    [1.53512485958697, -2.69169618940638, 1.19839281085285],
    dtype=np.float64,
)
K_PRE_A = np.array(
    [1.0, -1.69065929318241, 0.73248077421585],
    dtype=np.float64,
)
K_RLB_B = np.array([1.0, -2.0, 1.0], dtype=np.float64)
K_RLB_A = np.array(
    [1.0, -1.99004745483398, 0.99007225036621],
    dtype=np.float64,
)


def _biquad_channel(samples: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    output = np.empty_like(samples, dtype=np.float64)
    x1 = x2 = y1 = y2 = 0.0
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    a1, a2 = float(a[1]), float(a[2])
    for index, sample in enumerate(samples):
        x0 = float(sample)
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        output[index] = y0
        x2, x1 = x1, x0
        y2, y1 = y1, y0
    return output


def k_weight(audio: np.ndarray) -> np.ndarray:
    weighted = np.empty_like(audio, dtype=np.float64)
    for channel in range(audio.shape[1]):
        first = _biquad_channel(audio[:, channel], K_PRE_B, K_PRE_A)
        weighted[:, channel] = _biquad_channel(first, K_RLB_B, K_RLB_A)
    return weighted


def gated_loudness(audio: np.ndarray, sample_rate: int) -> dict:
    frames = int(audio.shape[0])
    if frames == 0:
        return {
            "integrated_lufs": None,
            "block_count": 0,
            "gated_block_count": 0,
            "method": "ITU-R BS.1770-4 K-weighting, 400 ms blocks, 75% overlap",
        }
    weighted = k_weight(audio)
    block = max(1, int(round(0.4 * sample_rate)))
    hop = max(1, int(round(0.1 * sample_rate)))
    if frames < block:
        starts = np.array([0], dtype=np.int64)
        block = frames
    else:
        starts = np.arange(0, frames - block + 1, hop, dtype=np.int64)
    energies = np.empty(starts.size, dtype=np.float64)
    for index, start in enumerate(starts):
        segment = weighted[start : start + block]
        energies[index] = float(np.mean(np.sum(segment * segment, axis=1)))
    with np.errstate(divide="ignore", invalid="ignore"):
        loudness = -0.691 + 10.0 * np.log10(np.maximum(energies, EPSILON))
    absolute = loudness >= -70.0
    if np.any(absolute):
        absolute_loudness = -0.691 + 10.0 * math.log10(
            float(np.mean(energies[absolute]))
        )
        relative_threshold = absolute_loudness - 10.0
        gated = absolute & (loudness >= relative_threshold)
        integrated = -0.691 + 10.0 * math.log10(
            float(np.mean(energies[gated]))
        )
    else:
        absolute_loudness = None
        relative_threshold = None
        gated = np.zeros_like(absolute)
        integrated = None
    return {
        "integrated_lufs": None if integrated is None else round(float(integrated), 9),
        "absolute_gated_lufs": (
            None
            if absolute_loudness is None
            else round(float(absolute_loudness), 9)
        ),
        "relative_gate_lufs": (
            None
            if relative_threshold is None
            else round(float(relative_threshold), 9)
        ),
        "block_count": int(starts.size),
        "gated_block_count": int(np.count_nonzero(gated)),
        "method": "ITU-R BS.1770-4 K-weighting, 400 ms blocks, 75% overlap",
    }


def loudness_gain(
    measured_lufs: float | None,
    target_lufs: float,
    config: ExportConfig,
) -> dict:
    if measured_lufs is None:
        return {
            "input_lufs": None,
            "target_lufs": target_lufs,
            "gain_db": 0.0,
            "gain_linear": 1.0,
            "limited": False,
        }
    raw_gain = target_lufs - measured_lufs
    gain = max(-config.max_cut_db, min(config.max_boost_db, raw_gain))
    return {
        "input_lufs": round(float(measured_lufs), 9),
        "target_lufs": target_lufs,
        "requested_gain_db": round(float(raw_gain), 9),
        "gain_db": round(float(gain), 9),
        "gain_linear": float(db_to_linear(gain)),
        "limited": bool(abs(gain - raw_gain) > 1.0e-12),
    }


def fit_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    if audio.shape[1] == channels:
        return audio
    if audio.shape[1] == 1 and channels == 2:
        return np.repeat(audio, 2, axis=1)
    if audio.shape[1] == 2 and channels == 1:
        return np.mean(audio, axis=1, keepdims=True, dtype=np.float64)
    raise ValueError(f"cannot map {audio.shape[1]} channels to {channels}")


def fit_frames(audio: np.ndarray, frames: int, channels: int) -> np.ndarray:
    mapped = fit_channels(audio, channels)
    if mapped.shape[0] == frames:
        return mapped
    if mapped.shape[0] > frames:
        return mapped[:frames]
    result = np.zeros((frames, channels), dtype=np.float64)
    result[: mapped.shape[0]] = mapped
    return result


def mix_tracks(
    game: np.ndarray,
    narration: np.ndarray,
    game_gain_db: float,
    mic_gain_db: float,
    peak_limit_dbfs: float,
) -> tuple[np.ndarray, dict]:
    channels = max(game.shape[1], narration.shape[1])
    frames = int(narration.shape[0])
    game_fit = fit_frames(game, frames, channels)
    narration_fit = fit_frames(narration, frames, channels)
    game_gain = db_to_linear(game_gain_db)
    mic_gain = db_to_linear(mic_gain_db)
    raw = game_fit * game_gain + narration_fit * mic_gain
    raw_peak = float(np.max(np.abs(raw))) if raw.size else 0.0
    limit = db_to_linear(peak_limit_dbfs)
    peak_scale = 1.0 if raw_peak <= limit or raw_peak == 0.0 else limit / raw_peak
    protected = raw * peak_scale
    protected_peak = float(np.max(np.abs(protected))) if protected.size else 0.0
    stats = {
        "channels": channels,
        "game_gain_db": round(float(game_gain_db), 9),
        "mic_gain_db": round(float(mic_gain_db), 9),
        "raw_peak_linear": raw_peak,
        "raw_peak_dbfs": (
            None if raw_peak == 0.0 else round(linear_to_db(raw_peak), 9)
        ),
        "peak_limit_dbfs": peak_limit_dbfs,
        "peak_limit_linear": limit,
        "peak_protection_applied": bool(peak_scale < 1.0),
        "peak_protection_scale": float(peak_scale),
        "peak_protection_db": (
            0.0 if peak_scale == 0.0 else round(linear_to_db(peak_scale), 9)
        ),
        "protected_peak_linear": protected_peak,
        "protected_peak_dbfs": (
            None if protected_peak == 0.0 else round(linear_to_db(protected_peak), 9)
        ),
    }
    return protected, stats


def source_position(
    segment: Segment,
    local_output_index: int,
) -> tuple[int, Fraction]:
    source_frames = segment.source_frames
    output_frames = segment.output_frames
    position = Fraction(local_output_index * source_frames, output_frames)
    index = position.numerator // position.denominator
    remainder = position - index
    return segment.source_start + index, remainder
