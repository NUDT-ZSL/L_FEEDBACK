from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .errors import BuilderError, ConfigError


@dataclass(frozen=True)
class AudioProbe:
    sample_rate: int
    channels: int
    start_time: Decimal
    duration: Optional[Decimal]
    codec_name: str


@dataclass(frozen=True)
class EncodedProbe:
    sample_rate: int
    channels: int
    packet_count: int


def find_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise BuilderError(f"required executable not found in PATH: {name}")
    return path


def run_command(
    command: Sequence[str],
    label: str,
    error_type: type[Exception] = BuilderError,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise error_type(f"{label} failed to start: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise error_type(f"{label} failed: {detail[:4000]}")
    return result


def _decimal_value(value: Any, default: str = "0") -> Decimal:
    if value is None or value == "N/A" or value == "":
        value = default
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ConfigError(f"ffprobe returned a non-numeric value: {value!r}") from exc


def _stream_and_format(data: Dict[str, Any], source: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    streams = data.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ConfigError(f"no audio stream found in input: {source}")
    stream = streams[0]
    fmt = data.get("format", {})
    if not isinstance(stream, dict) or not isinstance(fmt, dict):
        raise ConfigError(f"ffprobe returned malformed data for input: {source}")
    return stream, fmt


def probe_audio(ffprobe: str, source: Path) -> AudioProbe:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,start_time,duration:"
        "format=start_time,duration",
        "-of",
        "json",
        str(source),
    ]
    result = run_command(
        command, f"ffprobe input {source}", error_type=ConfigError
    )
    try:
        import json

        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"ffprobe produced invalid JSON for input: {source}") from exc

    stream, fmt = _stream_and_format(data, source)
    try:
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"input has invalid audio parameters: {source}") from exc
    if sample_rate <= 0 or channels <= 0:
        raise ConfigError(f"input has invalid audio parameters: {source}")

    start_value = stream.get("start_time", fmt.get("start_time"))
    duration_value = stream.get("duration", fmt.get("duration"))
    return AudioProbe(
        sample_rate=sample_rate,
        channels=channels,
        start_time=_decimal_value(start_value),
        duration=(
            None
            if duration_value in (None, "N/A", "")
            else _decimal_value(duration_value)
        ),
        codec_name=str(stream.get("codec_name", "unknown")),
    )


def decode_to_pcm(
    ffmpeg: str,
    source: Path,
    destination: Path,
    sample_rate: int,
    channels: int,
) -> None:
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-acodec",
        "pcm_s16le",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(destination),
    ]
    run_command(command, f"decode input {source}", error_type=ConfigError)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise ConfigError(f"decoded PCM is empty for input: {source}")


def encode_aac_segment(
    ffmpeg: str,
    aligned_pcm: Path,
    destination: Path,
    sample_rate: int,
    channels: int,
    bitrate_kbps: int,
    start_seconds: str,
    duration_seconds: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-ss",
        start_seconds,
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-i",
        str(aligned_pcm),
        "-t",
        duration_seconds,
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        "-threads",
        "1",
        "-c:a",
        "aac",
        "-profile:a",
        "aac_low",
        "-b:a",
        f"{bitrate_kbps}k",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-f",
        "adts",
        str(destination),
    ]
    run_command(
        command,
        f"AAC encode {bitrate_kbps}kbps",
        error_type=BuilderError,
    )


def parse_adts_frames(data: bytes) -> List[bytes]:
    frames: List[bytes] = []
    offset = 0
    while offset < len(data):
        if offset + 7 > len(data):
            raise BuilderError(f"truncated ADTS header at byte {offset}")
        if data[offset] != 0xFF or (data[offset + 1] & 0xF6) != 0xF0:
            raise BuilderError(f"invalid ADTS syncword at byte {offset}")
        protection_absent = data[offset + 1] & 0x01
        frame_length = (
            ((data[offset + 3] & 0x03) << 11)
            | (data[offset + 4] << 3)
            | (data[offset + 5] >> 5)
        )
        header_length = 7 if protection_absent else 9
        if frame_length < header_length:
            raise BuilderError(f"invalid ADTS frame length at byte {offset}")
        end = offset + frame_length
        if end > len(data):
            raise BuilderError(f"truncated ADTS frame at byte {offset}")
        frames.append(data[offset:end])
        offset = end
    if not frames:
        raise BuilderError("AAC output contains no ADTS frames")
    return frames


def strip_aac_priming(source: Path, destination: Path) -> int:
    frames = parse_adts_frames(source.read_bytes())
    if len(frames) < 2:
        raise BuilderError(f"AAC output is too short to remove priming: {source}")
    destination.write_bytes(b"".join(frames[1:]))
    return len(frames) - 1


def count_adts_frames(source: Path) -> int:
    return len(parse_adts_frames(source.read_bytes()))


def probe_encoded_segment(ffprobe: str, source: Path) -> EncodedProbe:
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_packets",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,nb_read_packets",
        "-of",
        "json",
        str(source),
    ]
    result = run_command(
        command, f"ffprobe output {source}", error_type=BuilderError
    )
    try:
        import json

        data = json.loads(result.stdout)
        stream = data["streams"][0]
        return EncodedProbe(
            sample_rate=int(stream["sample_rate"]),
            channels=int(stream["channels"]),
            packet_count=int(stream["nb_read_packets"]),
        )
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BuilderError(f"cannot read encoded audio parameters: {source}") from exc

