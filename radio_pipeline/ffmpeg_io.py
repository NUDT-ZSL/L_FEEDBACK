from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ProcessingError


METADATA_PREFIX = "RADIO_PIPELINE_METADATA="
OGG_SERIAL_OFFSET = 4242


@dataclass
class EncodeResult:
    wav_path: Path
    ogg_path: Path
    pcm_path: Path
    commands: list[list[str]]
    wav_probe: dict[str, Any]
    ogg_probe: dict[str, Any]


def find_tool(name: str) -> str:
    env_name = f"{name.upper()}_BIN"
    override = os.environ.get(env_name)
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return str(candidate)
        found = shutil.which(override)
        if found:
            return found
        raise ProcessingError(f"{env_name} does not point to an executable: {override}")
    found = shutil.which(name)
    if found:
        return found
    if os.name == "nt":
        suffix = ".exe"
        for directory in os.environ.get("PATH", "").split(os.pathsep):
            candidate = Path(directory) / f"{name}{suffix}"
            if candidate.is_file():
                return str(candidate)
    raise ProcessingError(
        f"{name} was not found. Install FFmpeg 6.0+ and add its bin directory to PATH "
        f"or set {env_name}."
    )


def _run(command: list[str], capture_stdout: bool = False) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=False,
        )
    except OSError as exc:
        raise ProcessingError(f"failed to execute {command[0]}: {exc}") from exc


def _require_success(result: subprocess.CompletedProcess, label: str) -> None:
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise ProcessingError(f"{label} failed with exit code {result.returncode}: {stderr}")


def ffmpeg_version() -> str:
    ffmpeg = find_tool("ffmpeg")
    result = _run([ffmpeg, "-hide_banner", "-version"], capture_stdout=True)
    _require_success(result, "ffmpeg -version")
    first_line = result.stdout.decode("utf-8", errors="replace").splitlines()
    return first_line[0].strip() if first_line else "unknown"


def metadata_comment(metadata: dict[str, Any]) -> str:
    encoded = json.dumps(
        metadata, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return METADATA_PREFIX + encoded


def encode_with_ffmpeg(
    pcm: np.ndarray,
    output_dir: Path,
    metadata: dict[str, Any],
    wav_name: str,
    ogg_name: str,
    opus_bitrate_bps: int,
) -> EncodeResult:
    ffmpeg = find_tool("ffmpeg")
    ffprobe = find_tool("ffprobe")
    output_dir.mkdir(parents=True, exist_ok=True)
    pcm_path = output_dir / "processed.pcm"
    wav_path = output_dir / wav_name
    ogg_path = output_dir / ogg_name
    pcm_path.write_bytes(np.asarray(pcm, dtype="<i2").tobytes(order="C"))
    comment = metadata_comment(metadata)
    common_input = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-threads",
        "1",
        "-f",
        "s16le",
        "-ar",
        "48000",
        "-ac",
        "1",
        "-i",
        str(pcm_path),
        "-map_metadata",
        "-1",
        "-metadata",
        f"comment={comment}",
        "-metadata",
        f"sample_rate={metadata['sample_rate']}",
        "-metadata",
        f"channels={metadata['channels']}",
        "-metadata",
        f"duration_ms={metadata['duration_ms']}",
        "-fflags",
        "+bitexact",
        "-flags",
        "+bitexact",
    ]
    wav_command = common_input + ["-c:a", "pcm_s16le", str(wav_path)]
    ogg_command = common_input + [
        "-c:a",
        "libopus",
        "-b:a",
        str(opus_bitrate_bps),
        "-vbr",
        "off",
        "-compression_level",
        "10",
        "-application",
        "audio",
        "-frame_duration",
        "20",
        "-serial_offset",
        str(OGG_SERIAL_OFFSET),
        str(ogg_path),
    ]
    for command, label in ((wav_command, "WAV encode"), (ogg_command, "Ogg/Opus encode")):
        _require_success(_run(command), label)

    return EncodeResult(
        wav_path=wav_path,
        ogg_path=ogg_path,
        pcm_path=pcm_path,
        commands=[wav_command, ogg_command],
        wav_probe=probe_audio(wav_path, ffprobe),
        ogg_probe=probe_audio(ogg_path, ffprobe),
    )


def probe_audio(path: Path, ffprobe: str | None = None) -> dict[str, Any]:
    tool = ffprobe or find_tool("ffprobe")
    result = _run(
        [
            tool,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_stdout=True,
    )
    _require_success(result, f"ffprobe {path.name}")
    try:
        return json.loads(result.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ProcessingError(f"ffprobe returned invalid JSON for {path}: {exc}") from exc


def embedded_metadata(probe: dict[str, Any]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    format_block = probe.get("format")
    if isinstance(format_block, dict) and isinstance(format_block.get("tags"), dict):
        candidates.append(format_block["tags"])
    streams = probe.get("streams")
    if isinstance(streams, list):
        for stream in streams:
            if isinstance(stream, dict) and isinstance(stream.get("tags"), dict):
                candidates.append(stream["tags"])
    for tags in candidates:
        for key, value in tags.items():
            if (
                str(key).lower() == "comment"
                and isinstance(value, str)
                and value.startswith(METADATA_PREFIX)
            ):
                try:
                    parsed = json.loads(value[len(METADATA_PREFIX) :])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed
    return None


def decode_audio_int16(path: Path) -> np.ndarray:
    ffmpeg = find_tool("ffmpeg")
    result = _run(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "48000",
            "-ac",
            "1",
            "pipe:1",
        ],
        capture_stdout=True,
    )
    _require_success(result, f"decode {path.name}")
    if len(result.stdout) % 2 != 0:
        raise ProcessingError(f"decoded {path.name} has an odd byte count")
    return np.frombuffer(result.stdout, dtype="<i2").copy()


def stream_summary(probe: dict[str, Any]) -> dict[str, Any]:
    streams = probe.get("streams")
    stream = streams[0] if isinstance(streams, list) and streams else {}
    format_block = probe.get("format") if isinstance(probe.get("format"), dict) else {}
    return {
        "codec_name": stream.get("codec_name"),
        "sample_rate": _integer_or_none(stream.get("sample_rate")),
        "channels": _integer_or_none(stream.get("channels")),
        "duration_ts": _integer_or_none(stream.get("duration_ts")),
        "duration_seconds": _float_or_none(stream.get("duration")),
        "initial_padding": _integer_or_none(stream.get("initial_padding")),
        "format_name": format_block.get("format_name"),
        "format_duration_seconds": _float_or_none(format_block.get("duration")),
        "format_bit_rate": _integer_or_none(format_block.get("bit_rate")),
        "size_bytes": _integer_or_none(format_block.get("size")),
    }


def _integer_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
