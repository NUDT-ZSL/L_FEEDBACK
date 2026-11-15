"""FFmpeg/FFprobe invocation, MP4 packaging, and duration verification."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from .errors import MasterError


def _tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise MasterError(
            "E_FFMPEG_NOT_FOUND",
            "Required media tool is not available on PATH",
            {"tool": name},
        )
    return path


def package_mp4(
    wav_path: str,
    mp4_path: str,
    sample_rate: int,
    expected_duration: float,
    layout_name: str,
    target_lufs: float,
    measured_lufs: float,
    true_peak_dbtp: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    ffmpeg = _tool("ffmpeg")
    output = config["output"]
    codec = output["audio_codec"]
    command: List[str] = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        wav_path,
        "-map",
        "0:a:0",
        "-vn",
        "-map_metadata",
        "-1",
        "-ar",
        str(sample_rate),
        "-fflags",
        "+bitexact",
        "-flags:a",
        "+bitexact",
    ]
    if codec == "aac":
        command.extend(["-c:a", "aac", "-b:a", output["aac_bitrate"]])
    else:
        command.extend(
            [
                "-c:a",
                "pcm_s24le" if output["bits_per_sample"] == 24 else "pcm_s16le",
            ]
        )
    if output.get("write_metadata", True):
        loudness_comment = (
            "channel_layout={};loudness_lufs={:.6f};"
            "loudness_target_lufs={:.6f};true_peak_dbtp={:.6f}"
        ).format(
            layout_name,
            measured_lufs,
            target_lufs,
            true_peak_dbtp,
        )
        command.extend(
            [
                "-metadata",
                "title=Dialogue-priority cinematic master",
                "-metadata",
                "creation_time=1970-01-01T00:00:00Z",
                "-metadata",
                "comment={}".format(loudness_comment),
                "-metadata",
                "description=Dialogue-priority broadcast master",
                "-metadata:s:a:0",
                "channel_layout={}".format(layout_name),
                "-metadata:s:a:0",
                "loudness_lufs={:.6f}".format(measured_lufs),
                "-metadata:s:a:0",
                "loudness_target_lufs={:.6f}".format(target_lufs),
                "-metadata:s:a:0",
                "true_peak_dbtp={:.6f}".format(true_peak_dbtp),
            ]
        )
    command.extend(["-f", "mp4", mp4_path])

    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
        )
    except FileNotFoundError:
        raise MasterError(
            "E_FFMPEG_NOT_FOUND", "FFmpeg executable could not be started"
        )
    except subprocess.TimeoutExpired:
        raise MasterError("E_FFMPEG_FAILED", "FFmpeg timed out")
    if result.returncode != 0:
        raise MasterError(
            "E_FFMPEG_FAILED",
            "FFmpeg returned a non-zero exit status",
            {
                "returncode": result.returncode,
                "stderr": result.stderr[-8000:],
                "command": command,
            },
        )
    if not os.path.isfile(mp4_path) or os.path.getsize(mp4_path) == 0:
        raise MasterError(
            "E_FFMPEG_FAILED", "FFmpeg did not produce an MP4 file"
        )

    probe = probe_media(mp4_path)
    stream = _first_audio_stream(probe)
    measured_duration = _stream_duration(stream, probe)
    stream_duration_delta = measured_duration - expected_duration
    configured_tolerance_ms = output.get("duration_tolerance_ms")
    if configured_tolerance_ms is None:
        if codec == "aac":
            tolerance = 1024.0 / sample_rate + 0.005
        else:
            tolerance = 1.0 / sample_rate + 0.001
    else:
        tolerance = configured_tolerance_ms / 1000.0
    if abs(stream_duration_delta) > tolerance:
        raise MasterError(
            "E_DURATION_MISMATCH",
            "Packaged media duration differs from the WAV",
            {
                "wav_duration_sec": expected_duration,
                "mp4_duration_sec": measured_duration,
                "delta_sec": stream_duration_delta,
                "tolerance_sec": tolerance,
            },
        )

    actual_layout = stream.get("channel_layout")
    stream_tags = {
        str(key).lower(): value
        for key, value in (stream.get("tags") or {}).items()
    }
    format_tags = {
        str(key).lower(): value
        for key, value in (probe.get("format", {}).get("tags") or {}).items()
    }
    tags = dict(format_tags)
    tags.update(stream_tags)
    metadata_verified = {
        "channel_layout": bool(actual_layout),
        "loudness_metadata": False,
    }
    comment = str(tags.get("comment", ""))
    if comment:
        metadata_verified["loudness_metadata"] = all(
            token in comment
            for token in (
                "loudness_lufs=",
                "loudness_target_lufs=",
                "true_peak_dbtp=",
                "channel_layout=",
            )
        )
    metadata_verified["loudness_lufs"] = (
        "loudness_lufs" in tags or "loudness_lufs=" in comment
    )
    metadata_verified["loudness_target_lufs"] = (
        "loudness_target_lufs" in tags or "loudness_target_lufs=" in comment
    )
    metadata_verified["true_peak_dbtp"] = (
        "true_peak_dbtp" in tags or "true_peak_dbtp=" in comment
    )
    if output.get("write_metadata", True) and not all(
        metadata_verified.values()
    ):
        raise MasterError(
            "E_FFMPEG_FAILED",
            "Packaged metadata could not be verified",
            {"metadata_verified": metadata_verified, "tags": tags},
        )
    return {
        "path": mp4_path,
        "container": probe.get("format", {}).get("format_name"),
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate", sample_rate)),
        "channels": int(stream.get("channels", 0)),
        "channel_layout": actual_layout,
        "duration_sec": round(float(measured_duration), 9),
        "wav_duration_sec": round(float(expected_duration), 9),
        "duration_delta_sec": round(float(stream_duration_delta), 9),
        "duration_tolerance_sec": round(float(tolerance), 9),
        "duration_match": True,
        "metadata_tags": tags,
        "stream_tags": stream_tags,
        "format_tags": format_tags,
        "metadata_verified": metadata_verified,
        "size_bytes": os.path.getsize(mp4_path),
    }


def probe_media(path: str) -> Dict[str, Any]:
    ffprobe = _tool("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        (
            "stream=codec_name,codec_type,channels,channel_layout,sample_rate,"
            "duration,duration_ts,time_base:stream_tags:"
            "format=duration,format_name,size:format_tags"
        ),
        "-of",
        "json",
        path,
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        raise MasterError(
            "E_FFMPEG_NOT_FOUND", "FFprobe executable could not be started"
        )
    except subprocess.TimeoutExpired:
        raise MasterError("E_FFPROBE_FAILED", "FFprobe timed out")
    if result.returncode != 0:
        raise MasterError(
            "E_FFPROBE_FAILED",
            "FFprobe returned a non-zero exit status",
            {
                "returncode": result.returncode,
                "stderr": result.stderr[-8000:],
                "command": command,
            },
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MasterError(
            "E_FFPROBE_FAILED",
            "Unable to parse FFprobe JSON output",
            {"error": str(exc)},
        )


def _first_audio_stream(probe: Dict[str, Any]) -> Dict[str, Any]:
    streams = probe.get("streams") or []
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if not audio:
        raise MasterError(
            "E_FFPROBE_FAILED", "Packaged MP4 contains no audio stream"
        )
    return audio[0]


def _stream_duration(
    stream: Dict[str, Any], probe: Dict[str, Any]
) -> float:
    candidates: List[Optional[float]] = []
    for key in ("duration",):
        value = stream.get(key)
        try:
            candidates.append(float(value))
        except (TypeError, ValueError):
            pass
    if stream.get("duration_ts") is not None and stream.get("time_base"):
        try:
            numerator, denominator = str(stream["time_base"]).split("/")
            candidates.append(
                float(stream["duration_ts"]) * float(numerator) / float(denominator)
            )
        except (ValueError, ZeroDivisionError):
            pass
    try:
        candidates.append(float(probe.get("format", {}).get("duration")))
    except (TypeError, ValueError):
        pass
    if not candidates:
        raise MasterError(
            "E_FFPROBE_FAILED", "Unable to determine packaged media duration"
        )
    return min(value for value in candidates if value is not None and value >= 0.0)

