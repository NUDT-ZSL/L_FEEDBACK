"""Render scheduled placements into a 48kHz mono PCM16 WAV track."""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf

from .config import Config
from .normalize import NormalizationResult, normalize_track
from .scheduler import LanguageSchedule


def total_frames(schedule: LanguageSchedule) -> int:
    """Track length based on the last retained clip, rounded to a whole ms."""
    if not schedule.placements:
        return 0
    last_end = max(placement.end for placement in schedule.placements)
    return int(np.ceil(last_end / 48.0)) * 48


def _equal_power_gains(fade: int) -> tuple[np.ndarray, np.ndarray]:
    # gain_out^2 + gain_in^2 == 1 preserves total energy across the fade.
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    gain_in = np.sin(ramp * (np.pi / 2.0)).astype(np.float32)
    gain_out = np.cos(ramp * (np.pi / 2.0)).astype(np.float32)
    return gain_out, gain_in


def render_raw_track(
    schedule: LanguageSchedule,
    clips: dict[str, LoadedClipLike],
    config: Config,
) -> np.ndarray:
    """Place clips on an integer-sample timeline with silence gaps.

    Two clips whose scheduled ranges touch (no silence gap) are joined with a
    deterministic equal-power crossfade bounded by both clip lengths and the
    configured fade duration.  Clips separated by silence are simply copied,
    leaving the gap as zeros.
    """
    frames = total_frames(schedule)
    track = np.zeros(frames, dtype=np.float32)
    placements = schedule.placements
    if not placements:
        return track

    configured_fade = config.crossfade_frames

    # First pass: copy each clip at its scheduled offset.  Scheduling ensures
    # clips never fully overlap; only the bounded crossfade regions may touch.
    for placement in placements:
        samples = clips[placement.event_id].samples
        track[placement.start : placement.end] = samples[: placement.frames]

    # Second pass: replace every touching boundary with an equal-power blend.
    for index in range(1, len(placements)):
        previous = placements[index - 1]
        current = placements[index]
        if previous.end != current.start or configured_fade == 0:
            continue
        fade = min(
            configured_fade,
            previous.frames // 2,
            current.frames // 2,
        )
        if fade == 0:
            continue
        gain_out, gain_in = _equal_power_gains(fade)
        prev_samples = clips[previous.event_id].samples
        curr_samples = clips[current.event_id].samples
        overlap_start = current.end - fade
        track[overlap_start : current.end] = (
            prev_samples[previous.frames - fade : previous.frames] * gain_out
            + curr_samples[current.frames - fade : current.frames] * gain_in
        )

    return track


# Structural typing keeps the renderer decoupled from the assets module.
try:
    from .assets import LoadedClip as LoadedClipLike  # noqa: F401
except Exception:  # pragma: no cover - import is always valid in practice
    LoadedClipLike = object  # type: ignore[assignment,misc]


def build_language_track(
    schedule: LanguageSchedule,
    clips_by_event: dict[str, LoadedClipLike],
    config: Config,
) -> tuple[np.ndarray, NormalizationResult]:
    """Render then normalize a language track."""
    raw = render_raw_track(schedule, clips_by_event, config)
    result = normalize_track(raw, config)
    return result.samples, result


def write_pcm16_wav(path: str, samples: np.ndarray, config: Config) -> dict:
    """Write 48kHz/16bit mono WAV and verify header/frame consistency."""
    if samples.ndim != 1:
        raise ValueError("only mono tracks are supported")
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = np.round(clipped * 32767.0).astype("<i2")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, pcm16, config.sample_rate, subtype="PCM_16", format="WAV")

    info = sf.info(path)
    if info.samplerate != config.sample_rate:
        raise ValueError(f"sample rate mismatch: {info.samplerate}")
    if info.channels != 1:
        raise ValueError(f"channel count mismatch: {info.channels}")
    if info.subtype != "PCM_16":
        raise ValueError(f"subtype mismatch: {info.subtype}")
    if info.frames != pcm16.shape[0]:
        raise ValueError(
            f"frame count mismatch: header={info.frames} data={pcm16.shape[0]}"
        )

    return {
        "path": path,
        "frames": int(info.frames),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "subtype": info.subtype,
        "duration_ms": round(info.frames / config.ms_frame, 3),
    }
