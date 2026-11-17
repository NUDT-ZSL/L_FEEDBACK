"""WAV loading, streaming mixing, clipping statistics, and atomic output."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import soundfile as sf

from .config import MixerSettings, TrackSettings
from .dsp import (
    align_loop_indices,
    fade_envelope,
    loop_period_ratio,
    milliseconds_to_frames,
    resample_segment,
    resampled_loop_frames,
    round_ratio,
    scheduled_loop_count,
)
from .errors import E_INVALID_INPUT, E_IO, E_MIX, E_RESAMPLE, MixerError

_BLOCK_FRAMES = 65536
_SOURCE_RATE_MIN = 8000
_SOURCE_RATE_MAX = 192000


@dataclass(frozen=True)
class SourceInfo:
    path: str
    sample_rate: int
    channels: int
    frames: int


@dataclass
class PreparedTrack:
    settings: TrackSettings
    source: SourceInfo
    loop_start_sample: int
    loop_end_sample: int
    loop_source_frames: int
    loop_output_frames: int
    period_numerator: int
    period_denominator: int
    offset_frames: int
    fade_in_frames: int
    fade_out_frames: int
    loop_audio: np.ndarray
    loop_count: int = 0

    @property
    def loop_duration_seconds(self) -> float:
        return self.loop_source_frames / float(self.source.sample_rate)

    @property
    def period_frames(self) -> float:
        return self.period_numerator / float(self.period_denominator)


def _read_source_info(path: str) -> SourceInfo:
    if not os.path.exists(path):
        raise MixerError(E_IO, f"input file does not exist: {path}")
    if not os.path.isfile(path):
        raise MixerError(E_IO, f"input path is not a regular file: {path}")
    try:
        info = sf.info(path)
    except FileNotFoundError as exc:
        raise MixerError(E_IO, f"cannot open input file: {exc}") from exc
    except PermissionError as exc:
        raise MixerError(E_IO, f"cannot read input file: {exc}") from exc
    except OSError as exc:
        raise MixerError(E_IO, f"cannot open input file: {exc}") from exc
    except (RuntimeError, ValueError) as exc:
        raise MixerError(E_INVALID_INPUT, f"invalid or unsupported WAV file: {exc}") from exc

    file_format = str(info.format).upper()
    if not (file_format == "WAV" or file_format == "WAVEX"):
        raise MixerError(E_INVALID_INPUT, f"input format must be WAV, got {info.format}")
    if info.samplerate < _SOURCE_RATE_MIN or info.samplerate > _SOURCE_RATE_MAX:
        raise MixerError(
            E_INVALID_INPUT,
            f"source sample rate must be between {_SOURCE_RATE_MIN} and "
            f"{_SOURCE_RATE_MAX} Hz, got {info.samplerate}",
        )
    if info.channels not in (1, 2):
        raise MixerError(
            E_INVALID_INPUT,
            f"source channel count must be 1 or 2, got {info.channels}",
        )
    if info.frames < 1:
        raise MixerError(E_INVALID_INPUT, "source WAV contains no audio frames")
    return SourceInfo(
        path=path,
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        frames=int(info.frames),
    )


def _read_loop_segment(source: SourceInfo, start: int, end: int) -> np.ndarray:
    frame_count = end - start
    try:
        with sf.SoundFile(source.path, mode="r") as handle:
            handle.seek(start)
            segment = handle.read(
                frames=frame_count,
                dtype="float64",
                always_2d=True,
            )
    except FileNotFoundError as exc:
        raise MixerError(E_IO, f"input file disappeared while reading: {exc}") from exc
    except PermissionError as exc:
        raise MixerError(E_IO, f"cannot read input file: {exc}") from exc
    except OSError as exc:
        raise MixerError(E_IO, f"cannot read input file: {exc}") from exc
    except (RuntimeError, ValueError) as exc:
        raise MixerError(E_INVALID_INPUT, f"cannot decode WAV samples: {exc}") from exc
    if segment.shape[0] != frame_count:
        raise MixerError(
            E_IO,
            f"short read from input WAV: expected {frame_count} frames, got {segment.shape[0]}",
        )
    if segment.shape[1] != source.channels:
        raise MixerError(E_IO, "input WAV channel count changed while reading")
    if not np.all(np.isfinite(segment)):
        raise MixerError(E_RESAMPLE, "source samples contain NaN or infinity")
    return np.ascontiguousarray(segment, dtype=np.float64)


def _prepare_track(
    settings: TrackSettings,
    source: SourceInfo,
    target_rate: int,
    output_channels: int,
) -> PreparedTrack:
    loop_start, loop_end = align_loop_indices(
        settings.loop_start_seconds,
        settings.loop_end_seconds,
        source.sample_rate,
        source.frames,
    )
    loop_source_frames = loop_end - loop_start
    if loop_source_frames < 2:
        raise MixerError(E_INVALID_INPUT, "aligned loop must contain at least two source samples")

    loop_output_frames = resampled_loop_frames(
        loop_source_frames, source.sample_rate, target_rate
    )
    if loop_output_frames < 2:
        raise MixerError(E_RESAMPLE, "aligned loop is shorter than two output samples")

    segment = _read_loop_segment(source, loop_start, loop_end)
    loop_audio = resample_segment(
        segment,
        source.sample_rate,
        target_rate,
        loop_output_frames,
    )

    if source.channels == 1 and output_channels == 2:
        loop_audio = np.concatenate((loop_audio, loop_audio), axis=1)
    elif source.channels != loop_audio.shape[1]:
        raise MixerError(E_MIX, "internal channel-layout mismatch")

    fade_in_frames = milliseconds_to_frames(settings.fade_in_ms, target_rate)
    fade_out_frames = milliseconds_to_frames(settings.fade_out_ms, target_rate)
    envelope = fade_envelope(
        loop_output_frames,
        fade_in_frames,
        fade_out_frames,
        settings.fade_curve,
    )
    loop_audio *= envelope[:, np.newaxis]
    if settings.volume != 1.0:
        loop_audio *= float(settings.volume)
    loop_audio = np.ascontiguousarray(loop_audio, dtype=np.float64)

    numerator, denominator = loop_period_ratio(
        loop_source_frames,
        source.sample_rate,
        target_rate,
    )
    offset_frames = int(math.floor(settings.offset_seconds * target_rate + 0.5))

    return PreparedTrack(
        settings=settings,
        source=source,
        loop_start_sample=loop_start,
        loop_end_sample=loop_end,
        loop_source_frames=loop_source_frames,
        loop_output_frames=loop_output_frames,
        period_numerator=numerator,
        period_denominator=denominator,
        offset_frames=offset_frames,
        fade_in_frames=fade_in_frames,
        fade_out_frames=fade_out_frames,
        loop_audio=loop_audio,
    )


def _mix_track_block(
    track: PreparedTrack,
    mixed: np.ndarray,
    block_start: int,
    block_end: int,
    total_frames: int,
) -> None:
    offset = track.offset_frames
    if offset >= block_end or offset >= total_frames:
        return

    relative_block_start = block_start - offset
    relative_block_end = min(block_end, total_frames) - offset
    if relative_block_end <= 0:
        return

    loop_frames = track.loop_output_frames
    numerator = track.period_numerator
    denominator = track.period_denominator

    # Candidate indices are estimated in integer arithmetic and padded by two
    # on either side; exact intersection checks below reject padding.
    lower = relative_block_start - loop_frames
    if lower <= 0:
        first_k = 0
    else:
        first_k = max(0, (lower * denominator) // numerator - 2)
    last_k = max(0, (relative_block_end * denominator + numerator - 1) // numerator + 2)

    for k in range(first_k, last_k + 1):
        start_relative = round_ratio(k * numerator, denominator)
        start_abs = offset + start_relative
        if start_abs >= total_frames or start_abs >= block_end:
            break

        next_relative = round_ratio((k + 1) * numerator, denominator)
        span = min(
            loop_frames,
            next_relative - start_relative,
            total_frames - start_abs,
        )
        if span <= 0 or start_abs + span <= block_start:
            continue

        destination_start = max(block_start, start_abs)
        destination_end = min(block_end, start_abs + span)
        source_start = destination_start - start_abs
        source_end = source_start + (destination_end - destination_start)

        mixed[destination_start - block_start : destination_end - block_start] += (
            track.loop_audio[source_start:source_end]
        )


def _create_temp_output(output_path: str) -> str:
    parent = os.path.dirname(output_path) or os.curdir
    if not os.path.isdir(parent):
        raise MixerError(E_IO, f"output directory does not exist: {parent}")
    base = os.path.basename(output_path) or "output.wav"
    try:
        descriptor, temp_path = tempfile.mkstemp(
            prefix=f".{base}.",
            suffix=".tmp",
            dir=parent,
        )
        os.close(descriptor)
    except OSError as exc:
        raise MixerError(E_IO, f"output directory is not writable: {exc}") from exc
    return temp_path


def _track_summary(track: PreparedTrack) -> Dict[str, object]:
    return {
        "path": track.source.path,
        "source_sample_rate": track.source.sample_rate,
        "source_channels": track.source.channels,
        "loop_start_sample": track.loop_start_sample,
        "loop_end_sample": track.loop_end_sample,
        "loop_source_frames": track.loop_source_frames,
        "loop_output_frames": track.loop_output_frames,
        "loop_duration_seconds": round(track.loop_duration_seconds, 9),
        "period_frames": round(track.period_frames, 9),
        "offset_frames": track.offset_frames,
        "volume": float(track.settings.volume),
        "fade_in_frames": track.fade_in_frames,
        "fade_out_frames": track.fade_out_frames,
        "fade_curve": track.settings.fade_curve,
        "loop_count": track.loop_count,
    }


def run_mix(
    settings: MixerSettings,
    target_rate: int,
    output_path: str,
) -> Dict[str, object]:
    """Execute one mix and return the deterministic success summary."""

    if isinstance(target_rate, bool) or not isinstance(target_rate, int):
        raise MixerError(E_INVALID_INPUT, "target sample rate must be an integer")
    if target_rate < _SOURCE_RATE_MIN or target_rate > _SOURCE_RATE_MAX:
        raise MixerError(
            E_INVALID_INPUT,
            f"target sample rate must be between {_SOURCE_RATE_MIN} and "
            f"{_SOURCE_RATE_MAX} Hz",
        )
    if not settings.tracks:
        raise MixerError(E_INVALID_INPUT, "at least one track is required")
    if len(settings.tracks) > 8:
        raise MixerError(E_INVALID_INPUT, "at most 8 tracks are supported")

    output_path = os.path.abspath(output_path)
    sources = [_read_source_info(track.path) for track in settings.tracks]
    output_channels = max(source.channels for source in sources)

    # Resolve source-aligned loop durations first.  This allows a missing
    # duration to mean "play each track once, including its start offset".
    max_end_seconds = 0.0
    for track_settings, source in zip(settings.tracks, sources):
        loop_start, loop_end = align_loop_indices(
            track_settings.loop_start_seconds,
            track_settings.loop_end_seconds,
            source.sample_rate,
            source.frames,
        )
        loop_source_frames = loop_end - loop_start
        loop_output_frames = resampled_loop_frames(
            loop_source_frames, source.sample_rate, target_rate
        )
        track_end_seconds = track_settings.offset_seconds + (
            loop_output_frames / float(target_rate)
        )
        max_end_seconds = max(max_end_seconds, track_end_seconds)

    duration = settings.duration_seconds
    if duration is None:
        if max_end_seconds <= 0.0:
            raise MixerError(E_INVALID_INPUT, "could not infer a positive playback duration")
        duration = max_end_seconds
    total_frames = int(math.floor(duration * target_rate + 0.5))
    if total_frames < 1:
        raise MixerError(E_INVALID_INPUT, "playback duration is shorter than one output sample")

    temp_path = _create_temp_output(output_path)
    clip_count = 0
    clipped_frames = 0
    peak_before_limit = 0.0
    try:
        prepared: List[PreparedTrack] = []
        for track_settings, source in zip(settings.tracks, sources):
            prepared.append(
                _prepare_track(track_settings, source, target_rate, output_channels)
            )
        for track in prepared:
            track.loop_count = scheduled_loop_count(
                total_frames,
                track.offset_frames,
                track.period_numerator,
                track.period_denominator,
            )

        try:
            output_handle = sf.SoundFile(
                temp_path,
                mode="w",
                samplerate=target_rate,
                channels=output_channels,
                format="WAV",
                subtype="PCM_24",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise MixerError(E_IO, f"cannot create output WAV: {exc}") from exc

        with output_handle:
            for block_start in range(0, total_frames, _BLOCK_FRAMES):
                block_end = min(total_frames, block_start + _BLOCK_FRAMES)
                mixed = np.zeros(
                    (block_end - block_start, output_channels),
                    dtype=np.float64,
                )
                for track in prepared:
                    _mix_track_block(track, mixed, block_start, block_end, total_frames)
                if not np.all(np.isfinite(mixed)):
                    raise MixerError(E_MIX, "mix produced NaN or infinity")

                absolute = np.abs(mixed)
                block_peak = float(np.max(absolute)) if absolute.size else 0.0
                peak_before_limit = max(peak_before_limit, block_peak)
                over = absolute > 1.0
                clip_count += int(np.count_nonzero(over))
                clipped_frames += int(np.count_nonzero(np.any(over, axis=1)))
                np.clip(mixed, -1.0, 1.0, out=mixed)
                try:
                    output_handle.write(mixed)
                except (OSError, RuntimeError, ValueError) as exc:
                    raise MixerError(E_IO, f"cannot write output WAV: {exc}") from exc

        try:
            os.replace(temp_path, output_path)
        except OSError as exc:
            raise MixerError(
                E_IO,
                f"cannot finalize output file: {exc}",
            ) from exc
        temp_path = ""
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            except OSError:
                pass

    primary = prepared[0]
    peak_after_limit = min(1.0, peak_before_limit)
    return {
        "status": "ok",
        "error_code": None,
        "output": output_path,
        "output_sample_rate": target_rate,
        "output_frames": total_frames,
        "output_channels": output_channels,
        "output_duration_seconds": round(total_frames / float(target_rate), 9),
        "clip_count": clip_count,
        "clipped_frames": clipped_frames,
        "loop_count": primary.loop_count,
        "fade_curve": primary.settings.fade_curve,
        "fade_in_ms": float(primary.settings.fade_in_ms),
        "fade_out_ms": float(primary.settings.fade_out_ms),
        "peak_before_limit": round(peak_before_limit, 12),
        "peak_after_limit": round(peak_after_limit, 12),
        "tracks": [_track_summary(track) for track in prepared],
    }
