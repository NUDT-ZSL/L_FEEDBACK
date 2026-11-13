"""Per-file transcode, QC, and concurrent batch orchestration."""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    E_LOUDNESS_FAIL,
    E_MEDIA_CORRUPT,
    E_OUTPUT_WRITE,
    E_SYNC_FAIL,
    E_TIMEOUT,
    E_UNEXPECTED,
    GOP_SIZE,
    MAX_START_DELTA_MS,
    TRANSCODE_TIMEOUT_SECONDS,
    W_AUDIO_DELAY,
    W_FPS_DEFAULT,
    W_FPS_INFERRED,
)
from .loudness import (
    LoudnessMeasurement,
    LoudnessMeasurementError,
    LoudnessMeasurementTimeout,
    loudness_passes,
    measure_loudness,
    normalization_gain_db,
)
from .probe import (
    MediaInfo,
    ProbeError,
    ProbeTimeout,
    keyframe_times,
    parse_float,
    probe_media,
)
from .rates import FrameRateResolution, is_fraction_rate, resolve_frame_rate
from .report import empty_file_record
from .process import run_filtered_command


@dataclass(frozen=True)
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    report: Path
    max_workers: int
    ffmpeg: str
    ffprobe: str
    transcode_timeout: float = TRANSCODE_TIMEOUT_SECONDS
    probe_timeout: float = 15.0


@dataclass
class FileResult:
    record: dict[str, Any]
    error_code: str | None = None


def _warning_sort_key(warning: str) -> tuple[int, str]:
    order = {W_FPS_INFERRED: 0, W_FPS_DEFAULT: 1, W_AUDIO_DELAY: 2}
    return order.get(warning, 99), warning


def _add_warning(record: dict[str, Any], warning: str | None) -> None:
    if warning and warning not in record["warnings"]:
        record["warnings"].append(warning)
        record["warnings"].sort(key=_warning_sort_key)


def _safe_float(value: Any, default: float = 0.0) -> float:
    parsed = parse_float(value)
    return default if parsed is None else parsed


def _profile_level(profile: str | None, level: Any) -> tuple[str | None, str | None]:
    level_number = None
    try:
        if level is not None and str(level).strip():
            level_number = float(str(level)) / 10.0
    except (TypeError, ValueError):
        level_number = None
    if profile and level_number is not None:
        return f"{profile} {level_number:.1f}", f"{level_number:.1f}"
    return None, None


def _source_start_times(info: MediaInfo) -> tuple[float, float]:
    video = info.video or {}
    audio = info.audio or {}
    return (
        _safe_float(video.get("start_time", info.format.get("start_time", 0))),
        _safe_float(audio.get("start_time", info.format.get("start_time", 0))),
    )


def _audio_filter(channels: int, gain_db: float | None) -> str:
    if channels == 2:
        layout_filter = "aformat=channel_layouts=stereo"
    elif channels == 6:
        layout_filter = (
            "pan=5.1|FL=c0|FR=c1|FC=c2|LFE=c3|BL=c4|BR=c5"
        )
    else:
        raise ValueError("unsupported audio channel count")
    gain_text = "0.000" if gain_db is None else f"{gain_db:.6f}"
    return (
        f"{layout_filter},"
        "aresample=48000,"
        "asetpts=PTS-STARTPTS,"
        f"volume={gain_text}dB"
    )


def _temporary_output_path(output_dir: Path, stem: str) -> Path:
    token = uuid.uuid4().hex
    return output_dir / f".{stem}.{token}.partial.mp4"


def _remove(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _build_transcode_command(
    config: PipelineConfig,
    source: Path,
    temporary: Path,
    info: MediaInfo,
    fps: FrameRateResolution,
    channels: int,
    gain_db: float | None,
) -> list[str]:
    audio_bitrate = "384k" if channels == 6 else "192k"
    command = [
        config.ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
    ]
    if info.subtitle_streams:
        command.extend(["-map", "0:s", "-c:s", "mov_text"])
    command.extend(
        [
            "-vf",
            "setpts=PTS-STARTPTS",
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-level:v",
            "4.1",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-g",
            str(GOP_SIZE),
            "-keyint_min",
            str(GOP_SIZE),
            "-sc_threshold",
            "0",
            "-x264-params",
            f"keyint={GOP_SIZE}:min-keyint={GOP_SIZE}:scenecut=0:open-gop=0",
            "-r",
            fps.text,
            "-af",
            _audio_filter(channels, gain_db),
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            "-write_tmcd",
            "0",
            "-map_metadata",
            "-1",
            "-f",
            "mp4",
            str(temporary),
        ]
    )
    return command


def _validate_video_contract(
    source_info: MediaInfo,
    output_info: MediaInfo,
    fps: FrameRateResolution,
) -> bool:
    source_video = source_info.video or {}
    output_video = output_info.video
    if output_video is None:
        return False
    profile = str(output_video.get("profile", ""))
    level = output_video.get("level")
    try:
        level_value = int(str(level)) if level is not None else None
    except (TypeError, ValueError):
        level_value = None
    expected_width = source_video.get("width")
    expected_height = source_video.get("height")
    return (
        output_video.get("codec_name") == "h264"
        and profile == "High"
        and level_value == 41
        and output_video.get("pix_fmt") == "yuv420p"
        and output_video.get("width") == expected_width
        and output_video.get("height") == expected_height
        and is_fraction_rate(
            str(output_video.get("avg_frame_rate", "")), fps.text
        )
    )


def _validate_gop(
    ffprobe: str,
    path: Path,
    fps: FrameRateResolution,
    timeout: float,
) -> bool:
    times = keyframe_times(ffprobe, path, timeout=timeout)
    if not times:
        return False
    times.sort()
    for previous, current in zip(times, times[1:]):
        interval = (current - previous) * float(fps.value)
        if interval > GOP_SIZE + 1.0:
            return False
    return True


def _measurement_fields(
    record: dict[str, Any], measurement: LoudnessMeasurement | None
) -> None:
    if measurement is None:
        return
    record["loudness_lufs"] = measurement.integrated_lufs
    record["loudness_lra"] = measurement.lra
    record["true_peak_dbtp"] = measurement.true_peak_dbtp


def process_file(config: PipelineConfig, source: Path) -> FileResult:
    record = empty_file_record(source.name)
    final_path = config.output_dir / f"{source.stem}.mp4"
    temporary_path: Path | None = None
    _remove(final_path)

    try:
        try:
            source_info = probe_media(
                config.ffprobe, source, timeout=config.probe_timeout
            )
        except ProbeTimeout:
            record["error_code"] = E_TIMEOUT
            return FileResult(record, E_TIMEOUT)
        except ProbeError:
            record["error_code"] = E_MEDIA_CORRUPT
            return FileResult(record, E_MEDIA_CORRUPT)

        if source_info.video is None or source_info.audio is None:
            record["error_code"] = E_MEDIA_CORRUPT
            return FileResult(record, E_MEDIA_CORRUPT)

        source_video = source_info.video
        source_audio = source_info.audio
        fps = resolve_frame_rate(
            source_video, source_info.format.get("duration")
        )
        _add_warning(record, fps.warning)

        width = source_video.get("width")
        height = source_video.get("height")
        try:
            width_int = int(width)
            height_int = int(height)
        except (TypeError, ValueError):
            record["error_code"] = E_MEDIA_CORRUPT
            return FileResult(record, E_MEDIA_CORRUPT)
        if width_int <= 0 or height_int <= 0 or width_int % 2 or height_int % 2:
            record["error_code"] = E_MEDIA_CORRUPT
            return FileResult(record, E_MEDIA_CORRUPT)

        try:
            channels = int(str(source_audio.get("channels", "0")))
        except (TypeError, ValueError):
            channels = 0
        if channels not in (2, 6):
            record["error_code"] = E_MEDIA_CORRUPT
            return FileResult(record, E_MEDIA_CORRUPT)

        video_start, audio_start = _source_start_times(source_info)
        delay = audio_start - video_start
        record["source_video_start_time"] = round(video_start, 6)
        record["source_audio_start_time"] = round(audio_start, 6)
        record["source_audio_delay_sec"] = round(delay, 6)
        if abs(delay) > 1.0:
            _add_warning(record, W_AUDIO_DELAY)

        source_loudness: LoudnessMeasurement | None = None
        gain_db: float | None = None
        try:
            source_loudness = measure_loudness(
                config.ffmpeg, source, timeout=config.transcode_timeout
            )
            gain_db = normalization_gain_db(source_loudness)
        except LoudnessMeasurementTimeout:
            record["error_code"] = E_TIMEOUT
            return FileResult(record, E_TIMEOUT)
        except LoudnessMeasurementError:
            # A measurable source is not required before encoding. The output
            # is still produced and independently rejected by the second pass.
            gain_db = None

        temporary_path = _temporary_output_path(config.output_dir, source.stem)
        command = _build_transcode_command(
            config,
            source,
            temporary_path,
            source_info,
            fps,
            channels,
            gain_db,
        )
        try:
            result = run_filtered_command(
                command,
                timeout=config.transcode_timeout,
            )
        except OSError:
            _remove(temporary_path)
            record["error_code"] = E_MEDIA_CORRUPT
            return FileResult(record, E_MEDIA_CORRUPT)
        if result.timed_out:
            _remove(temporary_path)
            record["error_code"] = E_TIMEOUT
            return FileResult(record, E_TIMEOUT)
        if result.returncode != 0 or not temporary_path.is_file():
            _remove(temporary_path)
            record["error_code"] = E_MEDIA_CORRUPT
            return FileResult(record, E_MEDIA_CORRUPT)

        try:
            os.replace(temporary_path, final_path)
        except OSError:
            _remove(temporary_path)
            record["error_code"] = E_OUTPUT_WRITE
            return FileResult(record, E_OUTPUT_WRITE)
        temporary_path = None
        record["output_path"] = str(final_path)

        try:
            output_info = probe_media(
                config.ffprobe, final_path, timeout=config.probe_timeout
            )
        except (ProbeError, ProbeTimeout):
            _remove(final_path)
            record["output_path"] = None
            record["error_code"] = E_MEDIA_CORRUPT
            return FileResult(record, E_MEDIA_CORRUPT)

        output_video = output_info.video or {}
        output_audio = output_info.audio or {}
        profile_level, level_text = _profile_level(
            output_video.get("profile"), output_video.get("level")
        )
        record.update(
            {
                "video_codec": output_video.get("codec_name"),
                "video_profile": output_video.get("profile"),
                "video_profile_level": profile_level,
                "video_level": level_text,
                "pix_fmt": output_video.get("pix_fmt"),
                "frame_rate": fps.text,
                "width": output_video.get("width"),
                "height": output_video.get("height"),
                "gop_size": GOP_SIZE,
                "audio_codec": output_audio.get("codec_name"),
                "channel_layout": output_audio.get("channel_layout"),
                "audio_channels": output_audio.get("channels"),
            }
        )

        if not _validate_video_contract(source_info, output_info, fps):
            _remove(final_path)
            record["output_path"] = None
            record["error_code"] = E_MEDIA_CORRUPT
            return FileResult(record, E_MEDIA_CORRUPT)

        try:
            gop_ok = _validate_gop(
                config.ffprobe,
                final_path,
                fps,
                config.probe_timeout,
            )
        except (ProbeError, ProbeTimeout):
            gop_ok = False
        if not gop_ok:
            _remove(final_path)
            record["output_path"] = None
            record["error_code"] = E_MEDIA_CORRUPT
            return FileResult(record, E_MEDIA_CORRUPT)

        output_video_start = _safe_float(output_video.get("start_time", 0))
        output_audio_start = _safe_float(output_audio.get("start_time", 0))
        output_delta_ms = abs(output_audio_start - output_video_start) * 1000.0
        record["output_video_start_time"] = round(output_video_start, 6)
        record["output_audio_start_time"] = round(output_audio_start, 6)
        record["output_start_delta_ms"] = round(output_delta_ms, 3)

        sync_ok = output_delta_ms <= MAX_START_DELTA_MS

        output_loudness: LoudnessMeasurement | None = None
        loudness_ok = False
        try:
            output_loudness = measure_loudness(
                config.ffmpeg, final_path, timeout=config.transcode_timeout
            )
            loudness_ok = loudness_passes(output_loudness)
        except LoudnessMeasurementTimeout:
            loudness_ok = False
        except LoudnessMeasurementError:
            loudness_ok = False
        _measurement_fields(record, output_loudness)

        if not sync_ok:
            record["status"] = "failed"
            record["error_code"] = E_SYNC_FAIL
            return FileResult(record, E_SYNC_FAIL)
        if not loudness_ok:
            record["status"] = "failed"
            record["error_code"] = E_LOUDNESS_FAIL
            return FileResult(record, E_LOUDNESS_FAIL)

        record["status"] = "success"
        record["error_code"] = None
        return FileResult(record, None)
    except Exception:
        if temporary_path is not None:
            _remove(temporary_path)
        _remove(final_path)
        record["status"] = "failed"
        record["output_path"] = None
        record["error_code"] = E_UNEXPECTED
        return FileResult(record, E_UNEXPECTED)


def run_batch(config: PipelineConfig) -> list[dict[str, Any]]:
    sources = sorted(
        (
            entry
            for entry in config.input_dir.iterdir()
            if entry.is_file() and entry.suffix.lower() == ".mxf"
        ),
        key=lambda entry: entry.name,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if not sources:
        return []

    worker_count = min(config.max_workers, len(sources))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(process_file, config, source)
            for source in sources
        ]
        results = [future.result().record for future in futures]
    return sorted(results, key=lambda item: str(item["source_file"]))
