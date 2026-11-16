"""Sample-accurate timeline construction and click-safe mixing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import soundfile as sf

from .config import CUT_TRANSITION_MS, DUCK_TRANSITION_MS
from .errors import ProcessingError, SceneValidationError
from .scene import DialogueIntervalSpec, SceneSpec


@dataclass(frozen=True)
class DuckIntervalResult:
    index: int
    start_sample: int
    end_sample: int
    attack_samples: int
    release_samples: int
    duck_gain: float
    gain_at_start: float
    gain_at_end: float
    max_adjacent_step: float


@dataclass(frozen=True)
class CutCheck:
    index: int
    cut_time_seconds: float
    cut_sample: int
    fade_samples: int
    window_start_sample: int
    window_end_sample: int
    pre_samples: int
    post_samples: int
    equal_power_max_error: float
    left_boundary_value: float
    left_boundary_expected: float
    right_boundary_value: float
    right_boundary_expected: float
    boundary_continuity_error: float
    max_abs_step_before: float
    max_abs_step_after: float
    peak_in_window: float
    clipped_samples: int
    passed: bool


@dataclass(frozen=True)
class MixResult:
    output_path: Path
    sample_rate: int
    total_samples: int
    peak_abs: float
    clipped_samples: int
    duck_intervals: Tuple[DuckIntervalResult, ...]
    cut_checks: Tuple[CutCheck, ...]


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    """Convert seconds to an output sample index using deterministic rounding."""

    if seconds < 0.0 or not math.isfinite(seconds) or sample_rate <= 0:
        raise ProcessingError("invalid time conversion request")
    return int(math.floor(seconds * sample_rate + 0.5))


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    source_frames = audio.shape[0]
    target_frames = max(1, int(math.floor(source_frames * target_rate / source_rate + 0.5)))
    positions = np.arange(target_frames, dtype=np.float64) * source_rate / target_rate
    left = np.floor(positions).astype(np.int64)
    np.clip(left, 0, source_frames - 1, out=left)
    right = np.minimum(left + 1, source_frames - 1)
    fraction = positions - np.floor(positions)
    output = (
        audio[left] * (1.0 - fraction)[:, None]
        + audio[right] * fraction[:, None]
    )
    return output


def read_audio(path: Path, target_rate: int) -> np.ndarray:
    """Read a WAV as float64 stereo at the target sample rate."""

    try:
        audio, source_rate = sf.read(
            str(path), dtype="float64", always_2d=True, fill_value=0.0
        )
    except Exception as exc:
        raise SceneValidationError(f"cannot read audio file {path}: {exc}") from exc
    if audio.shape[0] == 0:
        raise SceneValidationError(f"audio file is empty: {path}")
    if audio.shape[1] not in (1, 2):
        raise SceneValidationError(
            f"only mono or stereo audio is supported: {path} "
            f"({audio.shape[1]} channels)"
        )
    if not np.all(np.isfinite(audio)):
        raise SceneValidationError(f"audio contains non-finite samples: {path}")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return _resample_linear(np.ascontiguousarray(audio), int(source_rate), target_rate)


def _equal_power_gain(start_gain: float, end_gain: float, position: np.ndarray) -> np.ndarray:
    """Interpolate two amplitudes on a constant-power gain curve."""

    p = np.clip(np.asarray(position, dtype=np.float64), 0.0, 1.0)
    weight_start = np.cos(0.5 * np.pi * p)
    weight_end = np.sin(0.5 * np.pi * p)
    return np.sqrt(
        weight_start * weight_start * start_gain * start_gain
        + weight_end * weight_end * end_gain * end_gain
    )


def _interval_envelope(
    total_samples: int,
    interval: DialogueIntervalSpec,
    sample_rate: int,
    transition_samples: int,
) -> Tuple[np.ndarray, int, int]:
    start = seconds_to_samples(interval.start, sample_rate)
    end = seconds_to_samples(interval.end, sample_rate)
    start = min(max(start, 0), total_samples)
    end = min(max(end, 0), total_samples)
    envelope = np.ones(total_samples, dtype=np.float64)
    if end <= start:
        return envelope, 0, 0

    length = end - start
    if length >= 2 * transition_samples:
        attack_length = transition_samples
        release_length = transition_samples
    else:
        attack_length = max(1, length // 2)
        release_length = max(1, length - attack_length)
        if attack_length + release_length > length:
            release_length = max(0, length - attack_length)

    duck_gain = float(interval.duck_gain)
    envelope[start:end] = duck_gain

    attack_indices = np.arange(start, min(start + attack_length, end), dtype=np.int64)
    if attack_indices.size:
        denominator = max(attack_indices.size - 1, 1)
        position = (attack_indices - start) / denominator
        envelope[attack_indices] = _equal_power_gain(1.0, duck_gain, position)

    release_start = max(start, end - release_length)
    release_indices = np.arange(release_start, end, dtype=np.int64)
    if release_indices.size:
        denominator = max(release_indices.size - 1, 1)
        position = (release_indices - release_start) / denominator
        envelope[release_indices] = _equal_power_gain(duck_gain, 1.0, position)

    return envelope, attack_length, release_length


def build_duck_envelope(
    intervals: Sequence[DialogueIntervalSpec],
    total_samples: int,
    sample_rate: int,
    transition_ms: float = DUCK_TRANSITION_MS,
) -> Tuple[np.ndarray, Tuple[DuckIntervalResult, ...]]:
    """Build a continuous ducking envelope shared by ambience and music."""

    transition_samples = max(1, seconds_to_samples(transition_ms / 1000.0, sample_rate))
    combined = np.ones(total_samples, dtype=np.float64)
    reports: List[DuckIntervalResult] = []
    for index, interval in enumerate(intervals):
        envelope, attack_samples, release_samples = _interval_envelope(
            total_samples, interval, sample_rate, transition_samples
        )
        combined = np.minimum(combined, envelope)
        start = min(seconds_to_samples(interval.start, sample_rate), total_samples)
        end = min(seconds_to_samples(interval.end, sample_rate), total_samples)
        start_value = float(combined[start]) if start < total_samples else 1.0
        end_value = float(combined[end]) if end < total_samples else 1.0
        if end > start:
            step = float(np.max(np.abs(np.diff(combined[start:end])))) if end - start > 1 else 0.0
        else:
            step = 0.0
        reports.append(
            DuckIntervalResult(
                index=index,
                start_sample=start,
                end_sample=end,
                attack_samples=attack_samples,
                release_samples=release_samples,
                duck_gain=float(interval.duck_gain),
                gain_at_start=start_value,
                gain_at_end=end_value,
                max_adjacent_step=step,
            )
        )
    return combined, tuple(reports)


def _load_asset(cache: Dict[Path, np.ndarray], path: Path, sample_rate: int) -> np.ndarray:
    if path not in cache:
        cache[path] = read_audio(path, sample_rate)
    return cache[path]


def _build_ambience(
    scene: SceneSpec, sample_rate: int, total_samples: int, cache: Dict[Path, np.ndarray]
) -> np.ndarray:
    asset = _load_asset(cache, scene.ambience.file, sample_rate)
    start = seconds_to_samples(scene.ambience.loop_start, sample_rate)
    end = seconds_to_samples(scene.ambience.loop_end, sample_rate)
    start = min(max(start, 0), asset.shape[0])
    end = min(max(end, 0), asset.shape[0])
    if end <= start:
        raise SceneValidationError(
            "scene.ambience: loop region is empty after sample-rate conversion"
        )
    region = asset[start:end]
    indices = np.arange(total_samples, dtype=np.int64) % region.shape[0]
    return region[indices] * scene.ambience.gain


def _build_music(
    scene: SceneSpec, sample_rate: int, total_samples: int, cache: Dict[Path, np.ndarray]
) -> np.ndarray:
    music = np.zeros((total_samples, 2), dtype=np.float64)
    for layer in scene.music_layers:
        asset = _load_asset(cache, layer.file, sample_rate)
        start = seconds_to_samples(layer.start, sample_rate)
        if start >= total_samples:
            continue
        available = total_samples - start
        looped_length = min(available, asset.shape[0] * layer.loops)
        if looped_length <= 0:
            continue
        repeats = int(math.ceil(looped_length / asset.shape[0]))
        placed = np.tile(asset, (repeats, 1))[:looped_length]
        music[start : start + looped_length] += placed * layer.gain
    return music


def _build_effects(
    scene: SceneSpec, sample_rate: int, total_samples: int, cache: Dict[Path, np.ndarray]
) -> np.ndarray:
    effects = np.zeros((total_samples, 2), dtype=np.float64)
    for event in scene.events:
        asset = _load_asset(cache, event.file, sample_rate)
        start = seconds_to_samples(event.time, sample_rate)
        if start >= total_samples:
            continue
        available = min(asset.shape[0], total_samples - start)
        if available > 0:
            effects[start : start + available] += asset[:available] * event.gain
    return effects


def _stretch_halves(
    before: np.ndarray, after: np.ndarray, window_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    before_positions = np.linspace(0.0, before.shape[0] - 1, window_length)
    after_positions = np.linspace(0.0, after.shape[0] - 1, window_length)
    before_index = np.arange(before.shape[0], dtype=np.float64)
    after_index = np.arange(after.shape[0], dtype=np.float64)
    before_stretched = np.column_stack(
        [
            np.interp(before_positions, before_index, before[:, channel])
            for channel in range(before.shape[1])
        ]
    )
    after_stretched = np.column_stack(
        [
            np.interp(after_positions, after_index, after[:, channel])
            for channel in range(after.shape[1])
        ]
    )
    return before_stretched, after_stretched


def _apply_cut_crossfades(
    mix: np.ndarray,
    cuts: Sequence[float],
    sample_rate: int,
    transition_ms: float,
) -> Tuple[np.ndarray, Tuple[CutCheck, ...]]:
    fade_samples = max(2, seconds_to_samples(transition_ms / 1000.0, sample_rate))
    if fade_samples % 2:
        fade_samples += 1
    half = fade_samples // 2
    output = mix.copy()
    checks: List[CutCheck] = []
    previous_end = -1

    for index, cut_time in enumerate(cuts):
        center = seconds_to_samples(cut_time, sample_rate)
        start = center - half
        end = center + half
        if start < 0 or end > output.shape[0]:
            raise SceneValidationError(
                f"scene.camera_cuts[{index}] is too close to a scene boundary "
                f"for a centered {transition_ms:g} ms crossfade"
            )
        if start < previous_end:
            raise SceneValidationError(
                f"scene.camera_cuts[{index}] overlaps the previous crossfade window"
            )

        before = output[start:center].copy()
        after = output[center:end].copy()
        before_stretched, after_stretched = _stretch_halves(before, after, fade_samples)
        position = np.linspace(0.0, 1.0, fade_samples)
        weight_before = np.cos(0.5 * np.pi * position)
        weight_after = np.sin(0.5 * np.pi * position)
        equal_power_error = float(
            np.max(np.abs(weight_before**2 + weight_after**2 - 1.0))
        )
        left_expected = float(output[start, 0])
        right_expected = float(output[end - 1, 0])

        output[start:end] = (
            before_stretched * weight_before[:, None]
            + after_stretched * weight_after[:, None]
        )

        left_value = float(output[start, 0])
        right_value = float(output[end - 1, 0])
        continuity_error = max(
            abs(left_value - left_expected), abs(right_value - right_expected)
        )
        original_window = mix[start:end]
        repaired_window = output[start:end]
        original_step = (
            float(np.max(np.abs(np.diff(original_window, axis=0))))
            if original_window.shape[0] > 1
            else 0.0
        )
        repaired_step = (
            float(np.max(np.abs(np.diff(repaired_window, axis=0))))
            if repaired_window.shape[0] > 1
            else 0.0
        )
        peak = float(np.max(np.abs(repaired_window)))
        clipped = int(np.count_nonzero(np.abs(repaired_window) > 1.0))
        passed = (
            math.isfinite(peak)
            and continuity_error <= 1e-12
            and equal_power_error <= 1e-12
            and clipped == 0
        )
        checks.append(
            CutCheck(
                index=index,
                cut_time_seconds=float(cut_time),
                cut_sample=center,
                fade_samples=fade_samples,
                window_start_sample=start,
                window_end_sample=end,
                pre_samples=half,
                post_samples=half,
                equal_power_max_error=equal_power_error,
                left_boundary_value=left_value,
                left_boundary_expected=left_expected,
                right_boundary_value=right_value,
                right_boundary_expected=right_expected,
                boundary_continuity_error=continuity_error,
                max_abs_step_before=original_step,
                max_abs_step_after=repaired_step,
                peak_in_window=peak,
                clipped_samples=clipped,
                passed=passed,
            )
        )
        previous_end = end
    return output, tuple(checks)


def write_pcm16(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    try:
        sf.write(
            str(path),
            clipped,
            sample_rate,
            format="WAV",
            subtype="PCM_16",
        )
    except Exception as exc:
        raise ProcessingError(f"cannot write output WAV {path}: {exc}") from exc


def mix_scene(scene: SceneSpec, output_path: Path, sample_rate: int) -> MixResult:
    """Build and write one sample-aligned stereo scene mix."""

    total_samples = seconds_to_samples(scene.duration, sample_rate)
    if total_samples <= 0:
        raise SceneValidationError("scene.duration is shorter than one output sample")
    cut_samples = [
        seconds_to_samples(cut_time, sample_rate) for cut_time in scene.camera_cuts
    ]
    half_cut = max(1, seconds_to_samples(CUT_TRANSITION_MS / 2000.0, sample_rate))
    for index, center in enumerate(cut_samples):
        if center - half_cut < 0 or center + half_cut > total_samples:
            raise SceneValidationError(
                f"scene.camera_cuts[{index}] is too close to a scene boundary "
                f"for a centered {CUT_TRANSITION_MS:g} ms crossfade"
            )
    for index in range(1, len(cut_samples)):
        if cut_samples[index] - cut_samples[index - 1] < 2 * half_cut:
            raise SceneValidationError(
                f"scene.camera_cuts[{index}] overlaps the previous crossfade window"
            )

    cache: Dict[Path, np.ndarray] = {}
    ambience = _build_ambience(scene, sample_rate, total_samples, cache)
    music = _build_music(scene, sample_rate, total_samples, cache)
    duck_envelope, duck_reports = build_duck_envelope(
        scene.dialogue_intervals, total_samples, sample_rate
    )
    timed_bed = (ambience + music) * duck_envelope[:, None]
    effects = _build_effects(scene, sample_rate, total_samples, cache)
    mix = timed_bed + effects
    mix, cut_checks = _apply_cut_crossfades(
        mix, scene.camera_cuts, sample_rate, CUT_TRANSITION_MS
    )

    if not np.all(np.isfinite(mix)):
        raise ProcessingError("mix contains non-finite samples")
    peak_abs = float(np.max(np.abs(mix))) if mix.size else 0.0
    clipped_samples = int(np.count_nonzero(np.abs(mix) > 1.0))
    write_pcm16(output_path, mix, sample_rate)
    return MixResult(
        output_path=output_path,
        sample_rate=sample_rate,
        total_samples=total_samples,
        peak_abs=peak_abs,
        clipped_samples=clipped_samples,
        duck_intervals=duck_reports,
        cut_checks=cut_checks,
    )
