"""Deterministic per-entry audio decoding, stretching, and mixing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import EntryError, ExternalToolError
from .fftools import FFmpegTools, atomic_ffmpeg_output
from .model import AudioConfig, DialogueEntry
from .timecode import format_timecode, ms_to_samples


@dataclass
class EntryAudio:
    entry: DialogueEntry
    samples: np.ndarray
    source_probe: Any
    source_span_ms: int
    source_offset_ms: int
    target_duration_ms: int
    stretch_rate: float
    atempo: float
    primary_operation: str
    operations: list[str]
    residual_crop_ms: float = 0.0
    residual_pad_ms: float = 0.0
    clip_gain_db: float = 0.0
    input_peak: float = 0.0
    output_peak: float = 0.0
    error_codes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def operation_label(self) -> str:
        return {
            "stretch": "伸缩",
            "crop": "裁剪",
            "pad_silence": "补静音",
            "skip": "跳过",
            "none": "无",
        }.get(self.primary_operation, self.primary_operation)


def _pan_filter(source_channels: int, target_channels: int) -> str:
    if target_channels == source_channels:
        return (
            "pan=mono|c0=c0"
            if target_channels == 1
            else "pan=stereo|c0=c0|c1=c1"
        )
    if target_channels == 1:
        return "pan=mono|c0=0.5*c0+0.5*c1"
    return "pan=stereo|c0=0.70710678*c0|c1=0.70710678*c0"


def _decode_window(
    tools: FFmpegTools,
    entry: DialogueEntry,
    audio: AudioConfig,
    probe: Any,
    source_span_ms: int,
) -> np.ndarray:
    start_seconds = entry.source_offset_ms / 1_000.0
    duration_seconds = source_span_ms / 1_000.0
    pan = _pan_filter(probe.channels, audio.channels)
    filter_spec = (
        f"atrim=start={start_seconds:.6f}:duration={duration_seconds:.6f},"
        "asetpts=PTS-STARTPTS,"
        f"aresample={audio.sample_rate}:resampler=soxr,{pan}"
    )
    raw = tools.pipe_ffmpeg(
        [
            "-v",
            "error",
            "-i",
            str(entry.source_path),
            "-af",
            filter_spec,
            "-ar",
            str(audio.sample_rate),
            "-ac",
            str(audio.channels),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ]
    )
    if len(raw) % (4 * audio.channels) != 0:
        raise EntryError("E_DECODE", f"decoded audio payload has an invalid length")
    samples = np.frombuffer(raw, dtype="<f4").reshape(-1, audio.channels).copy()
    if samples.size == 0:
        raise EntryError("E_DECODE", "decoded source audio window is empty")
    if not np.all(np.isfinite(samples)):
        raise EntryError("E_DECODE", "decoded source audio contains non-finite samples")
    return samples


def _stretch_samples(
    tools: FFmpegTools, samples: np.ndarray, audio: AudioConfig, atempo: float
) -> np.ndarray:
    raw = np.asarray(samples, dtype="<f4").tobytes(order="C")
    output = tools.pipe_ffmpeg(
        [
            "-v",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(audio.sample_rate),
            "-ac",
            str(audio.channels),
            "-i",
            "pipe:0",
            "-af",
            f"atempo={atempo:.10f}",
            "-ar",
            str(audio.sample_rate),
            "-ac",
            str(audio.channels),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ],
        input_data=raw,
    )
    if len(output) % (4 * audio.channels) != 0:
        raise EntryError("E_PROCESS", "stretched audio payload has an invalid length")
    result = np.frombuffer(output, dtype="<f4").reshape(-1, audio.channels).copy()
    if result.size == 0:
        raise EntryError("E_PROCESS", "time-stretch produced empty audio")
    return result


def _fit_length(
    samples: np.ndarray, target_samples: int
) -> tuple[np.ndarray, int, int]:
    current = samples.shape[0]
    if current > target_samples:
        trim = current - target_samples
        return samples[:target_samples].copy(), trim, 0
    if current < target_samples:
        pad = target_samples - current
        padding = np.zeros((pad, samples.shape[1]), dtype=np.float32)
        return np.concatenate([samples, padding], axis=0), 0, pad
    return samples, 0, 0


def prepare_entry_audio(
    tools: FFmpegTools,
    entry: DialogueEntry,
    audio: AudioConfig,
    source_probe: Any,
) -> EntryAudio:
    """Decode one entry and fit it to its target subtitle duration."""

    if entry.target_duration_ms is None:
        raise EntryError("E_TIMECODE", "entry has no resolvable target duration")
    target_duration_ms = entry.target_duration_ms
    file_duration_ms = int(round(source_probe.duration_seconds * 1_000.0))
    if entry.source_end_ms is not None:
        source_span_ms = entry.source_end_ms - entry.source_start_ms
    else:
        source_span_ms = file_duration_ms - entry.source_offset_ms
    if source_span_ms <= 0:
        raise EntryError("E_RANGE", "source window has zero or negative length")
    if entry.source_offset_ms + source_span_ms > file_duration_ms:
        raise EntryError(
            "E_RANGE",
            f"source file window {format_timecode(entry.source_offset_ms)} + "
            f"{format_timecode(source_span_ms)} exceeds file duration "
            f"{format_timecode(file_duration_ms)}",
        )

    stretch_rate = target_duration_ms / float(source_span_ms)
    if stretch_rate < audio.stretch_min or stretch_rate > audio.stretch_max:
        raise EntryError(
            "E_RANGE",
            f"required duration ratio {stretch_rate:.6f} is outside "
            f"[{audio.stretch_min}, {audio.stretch_max}]",
        )

    samples = _decode_window(tools, entry, audio, source_probe, source_span_ms)
    input_peak = float(np.max(np.abs(samples)))
    duration_delta_ms = target_duration_ms - source_span_ms
    operations = ["resample", "channel_map"]
    primary_operation = "none"
    atempo = 1.0
    residual_trim_samples = 0
    residual_pad_samples = 0

    if abs(duration_delta_ms) > audio.min_stretch_delta_ms:
        atempo = source_span_ms / float(target_duration_ms)
        samples = _stretch_samples(tools, samples, audio, atempo)
        primary_operation = "stretch"
        operations.append("stretch")
    elif duration_delta_ms > 1:
        primary_operation = "pad_silence"
        operations.append("pad_silence")
    elif duration_delta_ms < -1:
        primary_operation = "crop"
        operations.append("crop")

    target_samples = ms_to_samples(target_duration_ms, audio.sample_rate)
    samples, residual_trim_samples, residual_pad_samples = _fit_length(
        samples, target_samples
    )
    if residual_trim_samples:
        operations.append("crop")
    if residual_pad_samples:
        operations.append("pad_silence")

    output_peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    clip_gain = 1.0
    error_codes: list[str] = []
    if output_peak > audio.max_sample_peak:
        clip_gain = audio.max_sample_peak / output_peak
        samples *= clip_gain
        output_peak *= clip_gain
        error_codes.append("E_CLIP")

    operations.append("mix")
    return EntryAudio(
        entry=entry,
        samples=samples.astype(np.float32, copy=False),
        source_probe=source_probe,
        source_span_ms=source_span_ms,
        source_offset_ms=entry.source_offset_ms,
        target_duration_ms=target_duration_ms,
        stretch_rate=stretch_rate,
        atempo=atempo,
        primary_operation=primary_operation,
        operations=_deduplicate(operations),
        residual_crop_ms=residual_trim_samples * 1_000.0 / audio.sample_rate,
        residual_pad_ms=residual_pad_samples * 1_000.0 / audio.sample_rate,
        clip_gain_db=20.0 * math.log10(clip_gain) if clip_gain != 1.0 else 0.0,
        input_peak=input_peak,
        output_peak=output_peak,
        error_codes=error_codes,
    )


def _deduplicate(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def write_fragment_wav(
    tools: FFmpegTools,
    path: Path,
    samples: np.ndarray,
    audio: AudioConfig,
    *,
    raw_temp: Path,
) -> None:
    """Write an atomically replaced 48 kHz/24-bit WAV fragment."""

    import os

    samples = np.ascontiguousarray(samples, dtype="<f4")
    raw = samples.tobytes(order="C")
    temp_target = raw_temp
    temp_target.parent.mkdir(parents=True, exist_ok=True)
    with temp_target.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        atomic_ffmpeg_output(
            tools,
            path,
            [
                "-f",
                "f32le",
                "-ar",
                str(audio.sample_rate),
                "-ac",
                str(audio.channels),
                "-i",
                str(temp_target),
                "-c:a",
                "pcm_s24le",
            ],
        )
    finally:
        try:
            temp_target.unlink(missing_ok=True)
        except OSError:
            pass


def read_fragment(
    tools: FFmpegTools,
    path: Path,
    audio: AudioConfig,
) -> np.ndarray:
    raw = tools.pipe_ffmpeg(
        [
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ar",
            str(audio.sample_rate),
            "-ac",
            str(audio.channels),
            "pipe:1",
        ]
    )
    if len(raw) % (4 * audio.channels) != 0:
        raise EntryError("E_STATE", f"fragment {path} has an invalid audio length")
    return np.frombuffer(raw, dtype="<f4").reshape(-1, audio.channels).copy()


@dataclass
class MixResult:
    samples: np.ndarray
    total_samples: int
    pre_trim_peak: float
    mix_gain_db: float
    clipping_detected: bool
    clipped_entry_ordinals: list[int]


def mix_entries(
    prepared: list[EntryAudio],
    *,
    total_duration_ms: int,
    audio: AudioConfig,
) -> MixResult:
    total_samples = ms_to_samples(total_duration_ms, audio.sample_rate)
    output = np.zeros((total_samples, audio.channels), dtype=np.float64)
    intervals: list[tuple[int, int, int]] = []
    for item in prepared:
        start = ms_to_samples(item.entry.target_start_ms, audio.sample_rate)
        end = start + item.samples.shape[0]
        if end > total_samples:
            overflow = end - total_samples
            if overflow > 2:
                raise EntryError(
                    "E_RANGE",
                    f"entry {item.entry.ordinal} extends beyond total timeline duration",
                )
            item.samples = item.samples[: item.samples.shape[0] - overflow]
            end = total_samples
        output[start:end] += item.samples.astype(np.float64) * item.entry.weight
        intervals.append((item.entry.ordinal, start, end))

    pre_trim_peak = float(np.max(np.abs(output))) if output.size else 0.0
    mix_gain = 1.0
    clipping_detected = pre_trim_peak > audio.max_sample_peak
    clipped_ordinals: list[int] = []
    if clipping_detected:
        mix_gain = audio.max_sample_peak / pre_trim_peak
        output *= mix_gain
        # Report entries that contributed to any pre-trim over-full-scale sample.
        pre_scale = output / mix_gain
        over = np.any(np.abs(pre_scale) > audio.max_sample_peak, axis=1)
        for ordinal, start, end in intervals:
            if np.any(over[start:end]):
                clipped_ordinals.append(ordinal)

    return MixResult(
        samples=output.astype(np.float32),
        total_samples=total_samples,
        pre_trim_peak=pre_trim_peak,
        mix_gain_db=20.0 * math.log10(mix_gain) if mix_gain != 1.0 else 0.0,
        clipping_detected=clipping_detected,
        clipped_entry_ordinals=clipped_ordinals,
    )


def write_mix_wav(
    tools: FFmpegTools,
    path: Path,
    mix: MixResult,
    audio: AudioConfig,
) -> None:
    import os

    raw_temp = path.with_name(f".{path.name}.{os.getpid()}.raw")
    try:
        with raw_temp.open("wb") as handle:
            handle.write(np.ascontiguousarray(mix.samples, dtype="<f4").tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        atomic_ffmpeg_output(
            tools,
            path,
            [
                "-f",
                "f32le",
                "-ar",
                str(audio.sample_rate),
                "-ac",
                str(audio.channels),
                "-i",
                str(raw_temp),
                "-c:a",
                "pcm_s24le",
            ],
        )
    finally:
        try:
            raw_temp.unlink(missing_ok=True)
        except OSError:
            pass


def normalize_loudness(
    tools: FFmpegTools,
    source: Path,
    target: Path,
    audio: AudioConfig,
    total_samples: int,
    *,
    max_iterations: int = 3,
) -> dict[str, Any]:
    """Two-pass EBU R128 normalization, with measured correction iterations."""

    before = tools.measure_loudness(source, audio)
    best_path = target
    stages: list[dict[str, Any]] = []
    current_source = source
    margin = 0.20
    tp_target = audio.max_true_peak_dbtp - margin
    last_measurement = before
    for iteration in range(1, max_iterations + 1):
        measured = tools.measure_loudness(current_source, audio)
        filter_spec = (
            f"loudnorm=I={audio.target_lufs}:TP={tp_target}:LRA={audio.loudness_range}:"
            f"measured_I={measured['input_i']:.6f}:"
            f"measured_TP={measured['input_tp']:.6f}:"
            f"measured_LRA={measured['input_lra']:.6f}:"
            f"measured_thresh={measured['input_thresh']:.6f}:"
            f"offset={measured['target_offset']:.6f}:linear=true,"
            f"aresample={audio.sample_rate}:resampler=soxr,"
            f"apad=whole_len={total_samples},"
            f"atrim=end_sample={total_samples}"
        )
        atomic_ffmpeg_output(
            tools,
            best_path,
            [
                "-i",
                str(current_source),
                "-af",
                filter_spec,
                "-ar",
                str(audio.sample_rate),
                "-ac",
                str(audio.channels),
                "-c:a",
                "pcm_s24le",
            ],
        )
        last_measurement = tools.measure_loudness(best_path, audio)
        stages.append(
            {
                "iteration": iteration,
                "source_integrated_lufs": measured["input_i"],
                "source_true_peak_dbtp": measured["input_tp"],
                "output_integrated_lufs": last_measurement["input_i"],
                "output_true_peak_dbtp": last_measurement["input_tp"],
                "target_true_peak_dbtp": tp_target,
                "linear": True,
            }
        )
        lufs_ok = abs(last_measurement["input_i"] - audio.target_lufs) <= 0.35
        tp_ok = last_measurement["input_tp"] <= audio.max_true_peak_dbtp - 0.01
        if lufs_ok and tp_ok:
            break
        current_source = best_path
        # A repeated measurement typically converges the two-pass offset. If
        # true peak remains high, increase the safety margin for the next pass.
        if not tp_ok:
            tp_target -= 0.20
        if abs(last_measurement["input_i"] - audio.target_lufs) > 0.15:
            tp_target = audio.max_true_peak_dbtp - margin

    ebur = tools.measure_ebur128(best_path)
    fp = tools.measure_loudness(best_path, audio)
    ebur_tp = ebur["true_peak_dbtp"]
    final_tp = (
        max(fp["input_tp"], ebur_tp)
        if math.isfinite(ebur_tp)
        else fp["input_tp"]
    )
    ideal_loudness_gain = audio.target_lufs - before["input_i"]
    peak_limited_gain = final_tp - before["input_tp"]
    extra_peak_trim = min(0.0, peak_limited_gain - ideal_loudness_gain)
    return {
        "method": "FFmpeg loudnorm (EBU R128), iterative two-pass",
        "target_lufs": audio.target_lufs,
        "max_true_peak_dbtp": audio.max_true_peak_dbtp,
        "before": {
            "integrated_lufs": before["input_i"],
            "true_peak_dbtp": before["input_tp"],
            "loudness_range_lu": before["input_lra"],
        },
        "after": {
            "integrated_lufs": fp["input_i"],
            "true_peak_dbtp": fp["input_tp"],
            "loudness_range_lu": fp["input_lra"],
            "ebur128_integrated_lufs": ebur["integrated_lufs"],
            "ebur128_true_peak_dbtp": ebur["true_peak_dbtp"],
        },
        "raw_source_gain_db": audio.target_lufs - before["input_i"],
        "actual_gain_db": fp["input_i"] - before["input_i"],
        "true_peak_correction_db": extra_peak_trim,
        "true_peak_correction": {
            "required": before["input_tp"] > audio.max_true_peak_dbtp,
            "before_dbtp": before["input_tp"],
            "after_dbtp": final_tp,
            "ideal_loudness_gain_db": ideal_loudness_gain,
            "peak_limited_gain_db": peak_limited_gain,
            "additional_trim_db": extra_peak_trim,
            "method": "FFmpeg loudnorm true-peak target with measured correction",
        },
        "true_peak_exceeded_before": before["input_tp"] > audio.max_true_peak_dbtp,
        "stages": stages,
        "within_tolerance": (
            abs(last_measurement["input_i"] - audio.target_lufs) <= 0.5
            and final_tp <= audio.max_true_peak_dbtp + 1e-6
        ),
    }

