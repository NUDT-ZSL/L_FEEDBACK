from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from . import SCHEMA_VERSION
from .dsp import (
    DEFAULT_TARGET_LUFS,
    SAMPLE_RATE,
    integrated_loudness,
)
from .io_utils import write_json

DEFAULT_DURATION_SECONDS = 10.0
LEAD_SECONDS = 0.80
CONTENT_SECONDS = 8.0
SCENE_END_SECONDS = (0.0, 2.2, 4.6, 6.8, 8.0)
SCENE_RELATIVE_GAIN = (0.12, 1.0, 0.38, 0.82)
SYNC_POSITIONS_SECONDS = (0.18, 1.05, 2.53, 3.42, 4.75, 5.61, 6.93, 7.58)


def _pinkish_noise(rng: np.random.Generator, shape: Tuple[int, int]) -> np.ndarray:
    # A fixed mild low-frequency tilt gives camera ambience without relying
    # on any external material.
    white = rng.standard_normal(shape)
    frequencies = np.fft.rfftfreq(shape[0], d=1.0 / SAMPLE_RATE)
    shape_filter = 1.0 / np.sqrt(np.maximum(frequencies, 18.0))
    shaped = np.empty_like(white)
    for channel in range(shape[1]):
        shaped[:, channel] = np.fft.irfft(
            np.fft.rfft(white[:, channel]) * shape_filter, n=shape[0]
        )
    shaped -= np.mean(shaped, axis=0, keepdims=True)
    rms = np.sqrt(np.mean(shaped * shaped, axis=0, keepdims=True))
    return shaped / np.maximum(rms, 1e-12)


def _hann(length: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(
        2.0 * np.pi * np.arange(length, dtype=np.float64) / max(1, length - 1)
    )


def _add_sync_chirps(
    source: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    output = source.copy()
    sample_count = output.shape[0]
    chirp_length = int(0.018 * SAMPLE_RATE)
    envelope = _hann(chirp_length)
    for position_seconds in SYNC_POSITIONS_SECONDS:
        start = int(round(position_seconds * SAMPLE_RATE))
        if start + chirp_length > sample_count:
            continue
        t = np.arange(chirp_length, dtype=np.float64) / SAMPLE_RATE
        start_frequency = 1_300.0 + 170.0 * rng.random()
        end_frequency = 7_200.0 + 300.0 * rng.random()
        phase = 2.0 * np.pi * (
            start_frequency * t
            + 0.5 * (end_frequency - start_frequency) * t * t / (chirp_length / SAMPLE_RATE)
        )
        chirp = 0.30 * envelope * np.sin(phase)
        output[start : start + chirp_length, :] += chirp[:, None]
    return output


def _make_common_program(seed: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x43414D]))
    length = int(round(CONTENT_SECONDS * SAMPLE_RATE))
    source = _pinkish_noise(rng, (length, 2))
    source *= 0.10

    # Four scene textures, including a deliberately quiet scene so the
    # downstream ±12 dB clamp is exercised by generated material.
    scene_edges = [int(round(v * SAMPLE_RATE)) for v in SCENE_END_SECONDS]
    t = np.arange(length, dtype=np.float64) / SAMPLE_RATE
    tonal = (
        0.022 * np.sin(2.0 * np.pi * 92.0 * t)
        + 0.014 * np.sin(2.0 * np.pi * 184.0 * t + 0.35)
        + 0.010 * np.sin(2.0 * np.pi * 523.25 * t + 0.8)
    )
    for index, gain in enumerate(SCENE_RELATIVE_GAIN):
        start = scene_edges[index]
        end = scene_edges[index + 1]
        local_t = t[start:end] - t[start]
        source[start:end, :] *= gain
        local_tonal = (
            tonal[start:end]
            * (0.65 + 0.35 * np.sin(2.0 * np.pi * (0.17 + index * 0.03) * local_t) ** 2)
        )
        source[start:end, 0] += local_tonal
        source[start:end, 1] += 0.92 * local_tonal

    source = _add_sync_chirps(source, rng)
    rms = float(np.sqrt(np.mean(source * source)))
    if rms > 0.0:
        source *= 0.11 / rms
    return source


def _shift_content(
    content: np.ndarray, start_sample: int, output_length: int
) -> np.ndarray:
    output = np.zeros((output_length, content.shape[1]), dtype=np.float64)
    if start_sample < 0:
        content = content[-start_sample:, :]
        start_sample = 0
    count = min(content.shape[0], output_length - start_sample)
    if count > 0:
        output[start_sample : start_sample + count, :] = content[:count, :]
    return output


def _unique_marker(
    length: int, track_index: int, rng: np.random.Generator
) -> np.ndarray:
    marker = np.zeros((length, 2), dtype=np.float64)
    frequency = 280.0 + 137.0 * track_index
    duration = int(0.130 * SAMPLE_RATE)
    envelope = _hann(duration)
    t = np.arange(duration, dtype=np.float64) / SAMPLE_RATE
    tone = 0.026 * envelope * np.sin(2.0 * np.pi * frequency * t + 0.2 * track_index)
    for scene_start in SCENE_END_SECONDS[:-1]:
        start = int(round((scene_start + 0.31 + 0.013 * rng.random()) * SAMPLE_RATE))
        count = min(duration, length - start)
        if count > 0:
            marker[start : start + count, 0] += tone[:count]
            marker[start : start + count, 1] += 0.84 * tone[:count]
    return marker


def _target_lufs_plan(
    seed: int,
    channels: int,
    silent_index: Optional[int],
    clipped_index: Optional[int],
) -> Dict[int, Optional[float]]:
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 0x4C554653])
    )
    candidates = (-30.0, -27.0, -24.0, -21.0, -18.0, -15.0, -12.0)
    order = list(candidates)
    rng.shuffle(order)
    targets: Dict[int, Optional[float]] = {}
    audible_index = 0
    for track_number in range(1, channels + 1):
        if silent_index is not None and track_number == silent_index:
            targets[track_number] = None
        elif clipped_index is not None and track_number == clipped_index:
            targets[track_number] = -6.0
        else:
            targets[track_number] = float(order[audible_index % len(order)])
            audible_index += 1
    return targets


def _scale_without_unintended_clipping(
    audio: np.ndarray, target_lufs: float
) -> Tuple[np.ndarray, Optional[float]]:
    current = integrated_loudness(audio)
    if current is None:
        return audio, None
    gain_db = target_lufs - current
    scaled = audio * (10.0 ** (gain_db / 20.0))
    peak = float(np.max(np.abs(scaled))) if scaled.size else 0.0
    if peak > 0.995:
        # Keep non-designated tracks clean. The measured value in the timeline
        # is the actual result after this safety adjustment.
        scaled *= 0.980 / peak
    measured = integrated_loudness(scaled)
    return scaled, measured


def _scale_clipped(audio: np.ndarray, target_lufs: float) -> Tuple[np.ndarray, Optional[float]]:
    current = integrated_loudness(audio)
    if current is None:
        return audio, None
    gain_db = target_lufs - current
    scaled = audio * (10.0 ** (gain_db / 20.0))
    peak = float(np.max(np.abs(scaled))) if scaled.size else 0.0
    if peak < 1.08:
        scaled *= 1.08 / max(peak, 1e-12)
    clipped = np.clip(scaled, -1.0, 1.0)
    measured = integrated_loudness(clipped)
    # Hard clipping changes the integrated loudness. Iteratively adjust the
    # already-clipped waveform while preserving saturated samples so a -6 LUFS
    # target remains inside the requested -30..-6 LUFS range.
    for _ in range(12):
        if measured is None or abs(measured - target_lufs) <= 0.02:
            break
        correction = 10.0 ** ((target_lufs - measured) / 20.0)
        clipped = np.clip(clipped * correction, -1.0, 1.0)
        measured = integrated_loudness(clipped)
    return clipped, measured


def _add_transient(
    audio: np.ndarray, start_sample: int
) -> np.ndarray:
    """Add four near-full-scale slate samples without creating a clip flag."""
    output = audio.copy()
    pattern = np.array([0.97, -0.97, 0.84, -0.84], dtype=np.float64)
    start = int(start_sample)
    count = min(pattern.size, output.shape[0] - start)
    if count > 0:
        output[start : start + count, :] += pattern[:count, None]
        np.clip(output, -1.0, 1.0, out=output)
    return output


def generate_group(
    seed: int,
    output_dir: Path,
    channels: int = 4,
    silent_track: Optional[int] = None,
    clipped_track: Optional[int] = None,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
) -> Dict[str, object]:
    """Generate one deterministic group of WAV files and its timeline JSON."""
    if channels < 1:
        raise ValueError("channels must be at least 1")
    if duration_seconds < 2.0:
        raise ValueError("duration_seconds must be at least 2.0")
    if silent_track is not None and not 1 <= silent_track <= channels:
        raise ValueError("silent_track must be a 1-based valid track number")
    if clipped_track is not None and not 1 <= clipped_track <= channels:
        raise ValueError("clipped_track must be a 1-based valid track number")
    if silent_track is not None and silent_track == clipped_track:
        raise ValueError("silent and clipped tracks must be different")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("cam_*.wav"):
        stale.unlink()
    timeline_path = output / "timeline.json"
    if timeline_path.exists():
        timeline_path.unlink()

    master_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x53454544]))
    total_samples = int(round(duration_seconds * SAMPLE_RATE))
    common = _make_common_program(int(seed))
    common_length = common.shape[0]
    lead_samples = int(round(LEAD_SECONDS * SAMPLE_RATE))

    if common_length + lead_samples > total_samples:
        raise ValueError("duration_seconds is too short for the synth program")

    tracks: List[Dict[str, object]] = []
    file_names = [f"cam_{index + 1:02d}.wav" for index in range(channels)]
    reference_index = 0
    if silent_track is not None:
        for index in range(channels):
            if index + 1 != silent_track:
                reference_index = index
                break
    offsets: List[int] = [
        int(master_rng.integers(-int(0.300 * SAMPLE_RATE), int(0.300 * SAMPLE_RATE) + 1))
        for _ in range(channels)
    ]
    offsets[reference_index] = 0
    if silent_track is not None:
        offsets[silent_track - 1] = 0
    target_plan = _target_lufs_plan(
        int(seed), channels, silent_track, clipped_track
    )

    for index in range(channels):
        track_number = index + 1
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), 0x54524143, track_number])
        )
        target = target_plan[track_number]
        if silent_track is not None and track_number == silent_track:
            audio = np.zeros((total_samples, 2), dtype=np.float64)
            measured = None
        else:
            unique = _unique_marker(common_length, index, rng)
            noise = _pinkish_noise(rng, (common_length, 2)) * 0.0018
            content = common + unique + noise
            delayed = _shift_content(content, lead_samples + offsets[index], total_samples)
            if clipped_track is not None and track_number == clipped_track:
                audio, measured = _scale_clipped(delayed, float(target))
            else:
                audio, measured = _scale_without_unintended_clipping(
                    delayed, float(target)
                )
            transient_start = (
                lead_samples + offsets[index] + int(0.45 * SAMPLE_RATE)
            )
            audio = _add_transient(audio, transient_start)
        if not np.all(np.isfinite(audio)):
            raise RuntimeError("internal synth produced non-finite samples")

        relative_offset = offsets[index]
        track_record: Dict[str, object] = {
            "id": f"cam_{track_number:02d}",
            "file": file_names[index],
            "track_index": track_number,
            "offset_samples": int(relative_offset),
            "source_delay_samples": int(offsets[index]),
            "offset_seconds": round(relative_offset / SAMPLE_RATE, 6),
            "target_lufs": target,
            "measured_lufs": None if measured is None else round(float(measured), 4),
            "silent": bool(silent_track is not None and track_number == silent_track),
            "clipped": bool(clipped_track is not None and track_number == clipped_track),
            "marker_hz": round(280.0 + 137.0 * index, 3),
        }
        tracks.append(track_record)
        sf.write(
            str(output / file_names[index]),
            audio,
            SAMPLE_RATE,
            subtype="PCM_16",
            format="WAV",
        )

    reference_start = lead_samples
    scene_records: List[Dict[str, object]] = []
    for index in range(len(SCENE_END_SECONDS) - 1):
        relative_start = int(round(SCENE_END_SECONDS[index] * SAMPLE_RATE))
        relative_end = int(round(SCENE_END_SECONDS[index + 1] * SAMPLE_RATE))
        start = reference_start + relative_start
        end = reference_start + relative_end
        if end > total_samples:
            raise ValueError("duration_seconds is too short for all scenes")
        scene_records.append(
            {
                "id": f"scene_{index + 1:02d}",
                "start_sample": start,
                "end_sample": end,
                "start_seconds": round(start / SAMPLE_RATE, 6),
                "end_seconds": round(end / SAMPLE_RATE, 6),
                "target_lufs": DEFAULT_TARGET_LUFS,
            }
        )

    timeline: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "sample_rate": SAMPLE_RATE,
        "channels": int(channels),
        "duration_samples": total_samples,
        "reference_track": file_names[reference_index],
        "target_lufs": DEFAULT_TARGET_LUFS,
        "fade_ms": 50,
        "gain_limit_db": 12.0,
        "tracks": tracks,
        "scenes": scene_records,
        "synthesis": {
            "deterministic": True,
            "content_seconds": CONTENT_SECONDS,
            "lead_seconds": LEAD_SECONDS,
            "scene_relative_gain": list(SCENE_RELATIVE_GAIN),
            "offset_convention": "positive means track content is delayed versus reference",
        },
    }
    write_json(timeline_path, timeline)
    return timeline
