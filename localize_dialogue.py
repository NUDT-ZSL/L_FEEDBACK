#!/usr/bin/env python3
"""Replace a cutscene dialogue track while preserving and realigning effects."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np


ERROR_FILE_NOT_FOUND = "FILE_NOT_FOUND"
ERROR_UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
ERROR_INVALID_ARGUMENT = "INVALID_ARGUMENT"
ERROR_ENCODING_FAILED = "ENCODING_FAILED"
ERROR_TOOL_NOT_AVAILABLE = "TOOL_NOT_AVAILABLE"
ERROR_PROCESSING_FAILED = "PROCESSING_FAILED"

STATUS_SUCCESS = "success"
STATUS_FAILURE = "failure"

ALIGNMENT_TOLERANCE_MS = 5.0
PCM_CODECS = ("pcm_f32le", "pcm_s24le", "pcm_s16le")


class ToolError(Exception):
    """A user-facing error with a stable business error code."""

    def __init__(self, code: str, message: str, details: Optional[str] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


class ArgumentParsingError(Exception):
    """Raised instead of argparse terminating before a report can be written."""


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ArgumentParsingError(message)

    def print_help(self, file: Any = None) -> None:
        super().print_help(file if file is not None else sys.stderr)


@dataclass(frozen=True)
class AudioStream:
    ordinal: int
    index: int
    codec_name: str
    sample_rate: int
    channels: int
    duration_seconds: Optional[float]
    sample_fmt: Optional[str]
    bits_per_raw_sample: Optional[int]


@dataclass
class CliArguments:
    input_container: Path
    localized_dialogue: Path
    output_container: Path
    report: Path
    sound_offset_ms: float


class Progress:
    """Ten-percent progress messages on stderr."""

    def __init__(self) -> None:
        self._next_percent = 0

    def update(self, percent: int, message: str) -> None:
        if percent < self._next_percent:
            return
        print(f"[progress] {percent:3d}% - {message}", file=sys.stderr, flush=True)
        self._next_percent = min(100, percent + 10)

    def finish(self) -> None:
        if self._next_percent <= 100:
            print("[progress] 100% - complete", file=sys.stderr, flush=True)
            self._next_percent = 110

    def stop(self) -> None:
        self._next_percent = 110


def utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be a finite number")
    return parsed


def build_parser() -> JsonArgumentParser:
    parser = JsonArgumentParser(
        prog="localize_dialogue.py",
        description=(
            "Replace track 0 dialogue with a localized mono file, realign track 1 "
            "sound effects, write a new container and a JSON report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Both flag-based and positional forms are accepted. Offsets are in "
            "milliseconds relative to the localized dialogue start; negative "
            "values start effects earlier and trim their leading samples."
        ),
    )
    parser.add_argument("input_container", nargs="?", help="original container")
    parser.add_argument("localized_dialogue", nargs="?", help="localized mono audio")
    parser.add_argument("output_container", nargs="?", help="output container")
    parser.add_argument("report", nargs="?", help="output JSON report")

    parser.add_argument(
        "-i",
        "--input",
        "--input-container",
        "--original",
        "--original-container",
        dest="input_opt",
        help="original two-track container",
    )
    parser.add_argument(
        "-l",
        "--localized-dialogue",
        "--localized",
        "--dialogue",
        dest="localized_opt",
        help="localized mono dialogue file",
    )
    parser.add_argument(
        "-o",
        "--output",
        "--output-container",
        dest="output_opt",
        help="output container path",
    )
    parser.add_argument(
        "-r",
        "--report",
        "--report-file",
        dest="report_opt",
        help="output JSON report path",
    )
    parser.add_argument(
        "--sound-offset-ms",
        "--sfx-offset-ms",
        "--offset-ms",
        dest="sound_offset_ms",
        type=finite_float,
        default=0.0,
        help="effects start offset in milliseconds; negative starts earlier",
    )
    return parser


def extract_report_argument(argv: Sequence[str]) -> Optional[Path]:
    """Best-effort report path extraction used when argument parsing fails."""

    for pos, arg in enumerate(argv):
        if arg in ("-r", "--report", "--report-file"):
            if pos + 1 < len(argv):
                return Path(argv[pos + 1])
        if arg.startswith("--report="):
            return Path(arg.split("=", 1)[1])
        if arg.startswith("--report-file="):
            return Path(arg.split("=", 1)[1])
        if arg.startswith("-r") and not arg.startswith("--") and len(arg) > 2:
            return Path(arg[2:])

    non_options = [arg for arg in argv if not arg.startswith("-")]
    if len(non_options) >= 4:
        return Path(non_options[3])
    return None


def resolve_cli_arguments(args: argparse.Namespace) -> CliArguments:
    input_value = args.input_opt or args.input_container
    localized_value = args.localized_opt or args.localized_dialogue
    output_value = args.output_opt or args.output_container
    report_value = args.report_opt or args.report

    missing = []
    if not input_value:
        missing.append("input container")
    if not localized_value:
        missing.append("localized dialogue")
    if not output_value:
        missing.append("output container")
    if not report_value:
        missing.append("report path")
    if missing:
        raise ToolError(
            ERROR_INVALID_ARGUMENT,
            "missing required argument(s): " + ", ".join(missing),
        )

    offset = float(args.sound_offset_ms)
    if not math.isfinite(offset):
        raise ToolError(ERROR_INVALID_ARGUMENT, "sound offset must be finite")

    return CliArguments(
        input_container=Path(input_value),
        localized_dialogue=Path(localized_value),
        output_container=Path(output_value),
        report=Path(report_value),
        sound_offset_ms=offset,
    )


def absolute_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def require_tool(name: str) -> str:
    location = shutil.which(name)
    if location is None:
        raise ToolError(
            ERROR_TOOL_NOT_AVAILABLE,
            f"required executable not found in PATH: {name}",
        )
    return location


def run_capture(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        raise ToolError(
            ERROR_TOOL_NOT_AVAILABLE,
            f"could not execute {command[0]}: {exc}",
        ) from exc


def parse_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def probe_media(path: Path) -> dict[str, Any]:
    ffprobe = require_tool("ffprobe")
    result = run_capture(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "ffprobe failed"
        raise ToolError(
            ERROR_UNSUPPORTED_FORMAT,
            f"cannot probe media file: {path}",
            detail,
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ToolError(
            ERROR_UNSUPPORTED_FORMAT,
            f"ffprobe returned invalid JSON for: {path}",
            str(exc),
        ) from exc
    if not isinstance(payload, dict):
        raise ToolError(
            ERROR_UNSUPPORTED_FORMAT,
            f"ffprobe returned an unexpected payload for: {path}",
        )
    return payload


def stream_duration_seconds(
    stream: dict[str, Any], format_info: dict[str, Any]
) -> Optional[float]:
    duration = parse_float(stream.get("duration"))
    if duration is not None and duration >= 0:
        return duration

    duration_ts = parse_int(stream.get("duration_ts"))
    time_base = stream.get("time_base")
    if duration_ts is not None and isinstance(time_base, str) and "/" in time_base:
        numerator, denominator = time_base.split("/", 1)
        try:
            base = float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError):
            base = 0.0
        if base > 0:
            return duration_ts * base

    tag_duration = stream.get("tags", {}).get("DURATION")
    if isinstance(tag_duration, str):
        parts = tag_duration.split(":")
        if len(parts) == 3:
            try:
                hours, minutes, seconds = (float(part) for part in parts)
                return hours * 3600 + minutes * 60 + seconds
            except ValueError:
                pass

    format_duration = parse_float(format_info.get("duration"))
    return format_duration if format_duration is not None else None


def audio_streams(payload: dict[str, Any]) -> list[AudioStream]:
    format_info = payload.get("format", {})
    streams = payload.get("streams", [])
    if not isinstance(streams, list):
        return []

    parsed: list[AudioStream] = []
    audio_ordinal = 0
    for stream in streams:
        if not isinstance(stream, dict) or stream.get("codec_type") != "audio":
            continue
        sample_rate = parse_int(stream.get("sample_rate"))
        channels = parse_int(stream.get("channels"))
        if sample_rate is None or sample_rate <= 0 or channels is None or channels <= 0:
            raise ToolError(
                ERROR_UNSUPPORTED_FORMAT,
                "audio stream is missing sample rate or channel information",
            )
        stream_index = parse_int(stream.get("index"))
        if stream_index is None:
            raise ToolError(
                ERROR_UNSUPPORTED_FORMAT,
                "audio stream is missing its stream index",
            )
        parsed.append(
            AudioStream(
                ordinal=audio_ordinal,
                index=stream_index,
                codec_name=str(stream.get("codec_name") or "unknown"),
                sample_rate=sample_rate,
                channels=channels,
                duration_seconds=stream_duration_seconds(stream, format_info),
                sample_fmt=stream.get("sample_fmt"),
                bits_per_raw_sample=parse_int(stream.get("bits_per_raw_sample")),
            )
        )
        audio_ordinal += 1
    return parsed


def validate_inputs(args: CliArguments) -> tuple[Path, Path, Path, Path]:
    input_path = absolute_path(args.input_container)
    localized_path = absolute_path(args.localized_dialogue)
    output_path = absolute_path(args.output_container)
    report_path = absolute_path(args.report)

    for label, path in (
        ("input container", input_path),
        ("localized dialogue", localized_path),
    ):
        if not path.exists():
            raise ToolError(ERROR_FILE_NOT_FOUND, f"{label} does not exist: {path}")
        if not path.is_file():
            raise ToolError(
                ERROR_INVALID_ARGUMENT,
                f"{label} is not a regular file: {path}",
            )

    for label, path in (
        ("output container", output_path),
        ("report", report_path),
    ):
        if path.exists() and path.is_dir():
            raise ToolError(
                ERROR_INVALID_ARGUMENT,
                f"{label} path is a directory: {path}",
            )

    protected = {
        input_path,
        localized_path,
    }
    if output_path in protected:
        raise ToolError(
            ERROR_INVALID_ARGUMENT,
            "output container must not overwrite an input file",
        )
    if report_path in protected:
        raise ToolError(
            ERROR_INVALID_ARGUMENT,
            "report path must not overwrite an input file",
        )
    if report_path == output_path:
        raise ToolError(
            ERROR_INVALID_ARGUMENT,
            "report path and output container must be different files",
        )

    return input_path, localized_path, output_path, report_path


def decode_audio(
    path: Path,
    audio_ordinal: int,
    sample_rate: int,
    target_channels: int,
    description: str,
) -> np.ndarray:
    ffmpeg = require_tool("ffmpeg")
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        f"0:a:{audio_ordinal}",
        "-vn",
        "-sn",
        "-dn",
        "-ar",
        str(sample_rate),
    ]
    command.extend(["-ac", str(target_channels)])
    command.extend(["-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"])

    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise ToolError(
            ERROR_TOOL_NOT_AVAILABLE,
            f"could not execute ffmpeg while decoding {description}: {exc}",
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ToolError(
            ERROR_UNSUPPORTED_FORMAT,
            f"cannot decode {description}",
            detail or "ffmpeg decode failed",
        )

    raw = np.frombuffer(result.stdout, dtype="<f4")
    if raw.size % target_channels != 0:
        raise ToolError(
            ERROR_UNSUPPORTED_FORMAT,
            f"decoded {description} has an incomplete sample frame",
        )
    decoded = raw.reshape(-1, target_channels).astype(np.float32, copy=False)
    if decoded.size == 0:
        raise ToolError(
            ERROR_UNSUPPORTED_FORMAT,
            f"decoded {description} contains no audio samples",
        )
    return decoded


def build_sound_effect_timeline(
    effect: np.ndarray,
    target_samples: int,
    offset_ms: float,
    sample_rate: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Place effects on a dialogue-anchored timeline.

    Dialogue always starts at sample zero. A positive offset inserts leading
    silence. A negative offset trims the amount of effect audio that would
    otherwise precede the dialogue start. Any effect samples outside the
    dialogue duration are not copied.
    """

    if effect.ndim != 2 or effect.shape[1] != 2:
        raise AssertionError("effect timeline requires a stereo array")

    offset_scale = offset_ms * sample_rate / 1000.0
    if not math.isfinite(offset_scale):
        raise ToolError(
            ERROR_INVALID_ARGUMENT,
            "sound offset is too large for the target sample rate",
        )
    offset_samples = int(round(offset_scale))
    output = np.zeros((target_samples, 2), dtype=np.float32)

    leading_padding_samples = 0
    truncated_leading_samples = 0
    truncated_trailing_samples = 0

    if offset_samples >= 0:
        leading_padding_samples = min(offset_samples, target_samples)
        available_output = max(0, target_samples - leading_padding_samples)
        retained_samples = min(effect.shape[0], available_output)
        if retained_samples:
            output[
                leading_padding_samples : leading_padding_samples + retained_samples
            ] = effect[:retained_samples]
        truncated_trailing_samples = effect.shape[0] - retained_samples
    else:
        truncated_leading_samples = min(-offset_samples, effect.shape[0])
        retained_candidates = effect.shape[0] - truncated_leading_samples
        retained_samples = min(retained_candidates, target_samples)
        if retained_samples:
            output[:retained_samples] = effect[
                truncated_leading_samples : truncated_leading_samples
                + retained_samples
            ]
        truncated_trailing_samples = retained_candidates - retained_samples

    retained_total = effect.shape[0] - (
        truncated_leading_samples + truncated_trailing_samples
    )
    trailing_padding_samples = target_samples - leading_padding_samples - retained_total
    alignment_error_ms = abs(
        (offset_samples / sample_rate * 1000.0) - offset_ms
    )

    details = {
        "offset_samples": offset_samples,
        "leading_padding_samples": leading_padding_samples,
        "trailing_padding_samples": trailing_padding_samples,
        "retained_samples": retained_total,
        "truncated_samples": truncated_leading_samples
        + truncated_trailing_samples,
        "truncated_leading_samples": truncated_leading_samples,
        "truncated_trailing_samples": truncated_trailing_samples,
        "alignment_error_ms": round(alignment_error_ms, 6),
    }
    return output, details


def write_raw_f32(path: Path, audio: np.ndarray) -> None:
    payload = np.ascontiguousarray(audio, dtype="<f4")
    payload.tofile(str(path))


def unique_in_order(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def codec_candidates(original_effect_codec: str) -> list[str]:
    candidates: list[str] = []
    if original_effect_codec.startswith("pcm_") and original_effect_codec not in (
        "pcm_alaw",
        "pcm_mulaw",
    ):
        candidates.append(original_effect_codec)
    candidates.extend(PCM_CODECS)
    return unique_in_order(candidates)


def encode_output(
    dialogue: np.ndarray,
    effect: np.ndarray,
    sample_rate: int,
    output_path: Path,
    original_effect_codec: str,
    temporary_dir: Path,
) -> tuple[Path, str, Optional[str]]:
    ffmpeg = require_tool("ffmpeg")
    dialogue_raw = temporary_dir / "dialogue.f32"
    effect_raw = temporary_dir / "effect.f32"
    write_raw_f32(dialogue_raw, dialogue)
    write_raw_f32(effect_raw, effect)

    suffix = output_path.suffix.lower() or ".mkv"
    encoded_path = temporary_dir / f"encoded{suffix}"
    errors: list[str] = []

    for codec in codec_candidates(original_effect_codec):
        if encoded_path.exists():
            encoded_path.unlink()
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            "2",
            "-channel_layout",
            "stereo",
            "-i",
            str(dialogue_raw),
            "-f",
            "f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            "2",
            "-channel_layout",
            "stereo",
            "-i",
            str(effect_raw),
            "-map",
            "0:a:0",
            "-map",
            "1:a:0",
            "-c:a",
            codec,
            "-metadata:s:a:0",
            "title=Localized Dialogue",
            "-metadata:s:a:1",
            "title=Sound Effects",
            "-map_metadata",
            "-1",
            str(encoded_path),
        ]
        result = run_capture(command)
        if result.returncode == 0 and encoded_path.exists():
            fallback_error = None
            if codec != original_effect_codec:
                fallback_error = (
                    "PCM output selected to preserve decoded samples exactly; "
                    f"source effects codec was {original_effect_codec!r}"
                )
            return encoded_path, codec, fallback_error
        errors.append(
            f"{codec}: "
            + (result.stderr.strip() or "ffmpeg returned a non-zero exit code")
        )

    raise ToolError(
        ERROR_ENCODING_FAILED,
        "failed to encode the output container with all compatible PCM codecs",
        "\n".join(errors[-4:]),
    )


def verify_output(
    path: Path,
    expected_rate: int,
    expected_samples: int,
    expected_dialogue: np.ndarray,
    expected_effect: np.ndarray,
    sample_tolerance: float,
) -> dict[str, Any]:
    payload = probe_media(path)
    streams = audio_streams(payload)
    if len(streams) != 2:
        raise ToolError(
            ERROR_ENCODING_FAILED,
            f"encoded output contains {len(streams)} audio streams instead of 2",
        )
    for ordinal, stream in enumerate(streams):
        if stream.sample_rate != expected_rate:
            raise ToolError(
                ERROR_ENCODING_FAILED,
                f"output audio stream {ordinal} has sample rate "
                f"{stream.sample_rate}, expected {expected_rate}",
            )
        if stream.channels != 2:
            raise ToolError(
                ERROR_ENCODING_FAILED,
                f"output audio stream {ordinal} has {stream.channels} channels, "
                "expected 2",
            )

    dialogue_actual = decode_audio(
        path, 0, expected_rate, 2, "encoded dialogue track"
    )
    effect_actual = decode_audio(
        path, 1, expected_rate, 2, "encoded effects track"
    )
    for label, actual, expected in (
        ("dialogue", dialogue_actual, expected_dialogue),
        ("effects", effect_actual, expected_effect),
    ):
        if actual.shape != expected.shape:
            raise ToolError(
                ERROR_ENCODING_FAILED,
                f"encoded {label} track has shape {actual.shape}, "
                f"expected {expected.shape}",
            )
        max_error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
        if max_error > sample_tolerance:
            raise ToolError(
                ERROR_ENCODING_FAILED,
                f"encoded {label} track differs from the prepared timeline "
                f"by {max_error:.9g}",
            )

    duration_seconds = expected_samples / expected_rate
    return {
        "audio_stream_count": len(streams),
        "dialogue_samples": int(dialogue_actual.shape[0]),
        "effects_samples": int(effect_actual.shape[0]),
        "duration_seconds": round(duration_seconds, 6),
        "sample_comparison_tolerance": sample_tolerance,
        "passed": True,
    }


def sample_tolerance_for_codec(codec: str) -> float:
    if codec == "pcm_f32le":
        return 1e-7
    if codec == "pcm_s24le":
        return 2e-7
    if codec in ("pcm_s16le", "pcm_u8", "pcm_s8"):
        return 1e-4
    return 1e-6


def duration_round(value: Optional[float]) -> Optional[float]:
    return round(value, 6) if value is not None else None


def empty_report(
    args: Optional[CliArguments],
    output_path: Optional[Path],
) -> dict[str, Any]:
    return {
        "status": STATUS_FAILURE,
        "error_code": None,
        "message": "",
        "output_file": str(output_path) if output_path is not None else None,
        "input_container": (
            str(absolute_path(args.input_container)) if args is not None else None
        ),
        "localized_dialogue_file": (
            str(absolute_path(args.localized_dialogue)) if args is not None else None
        ),
        "dialogue_track": None,
        "sound_effect_track": None,
        "ignored_audio_tracks": 0,
        "alignment": {
            "anchor": "localized_dialogue_start",
            "tolerance_ms": ALIGNMENT_TOLERANCE_MS,
            "within_tolerance": None,
        },
        "output_format": None,
        "verification": None,
        "processing_timestamp": utc_timestamp(),
    }


def failure_report(
    args: Optional[CliArguments],
    output_path: Optional[Path],
    error: ToolError,
) -> dict[str, Any]:
    report = empty_report(args, output_path)
    report["status"] = STATUS_FAILURE
    report["error_code"] = error.code
    report["message"] = error.message
    if error.details:
        report["details"] = error.details
    return report


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def print_json(payload: dict[str, Any]) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
    sys.stdout.write("\n")
    sys.stdout.flush()


def process(args: CliArguments, progress: Progress) -> tuple[dict[str, Any], Path]:
    progress.update(0, "validating arguments and input files")
    input_path, localized_path, output_path, _ = validate_inputs(args)
    progress.update(10, "probing input media")

    input_payload = probe_media(input_path)
    localized_payload = probe_media(localized_path)
    input_streams = audio_streams(input_payload)
    localized_streams = audio_streams(localized_payload)

    if len(input_streams) < 2:
        raise ToolError(
            ERROR_UNSUPPORTED_FORMAT,
            "input container must contain at least two audio tracks",
            f"found {len(input_streams)} audio track(s)",
        )
    if len(localized_streams) != 1:
        raise ToolError(
            ERROR_UNSUPPORTED_FORMAT,
            "localized dialogue file must contain exactly one audio track",
            f"found {len(localized_streams)} audio track(s)",
        )

    original_dialogue = input_streams[0]
    original_effect = input_streams[1]
    localized = localized_streams[0]
    ignored_tracks = max(0, len(input_streams) - 2)

    if original_dialogue.channels != 2:
        raise ToolError(
            ERROR_UNSUPPORTED_FORMAT,
            "input dialogue track (track 0) must be stereo",
            f"found {original_dialogue.channels} channel(s)",
        )
    if localized.channels != 1:
        raise ToolError(
            ERROR_UNSUPPORTED_FORMAT,
            "localized dialogue must be mono",
            f"found {localized.channels} channel(s)",
        )
    if localized.sample_rate <= 0 or original_dialogue.sample_rate <= 0:
        raise ToolError(
            ERROR_UNSUPPORTED_FORMAT,
            "dialogue tracks must have a positive sample rate",
        )

    output_rate = original_dialogue.sample_rate
    progress.update(20, "decoding and converting localized dialogue")
    localized_mono = decode_audio(
        localized_path,
        0,
        output_rate,
        1,
        "localized dialogue",
    )[:, 0]
    # Mono-to-stereo expansion is an exact channel copy, not a gain-changing mix.
    localized_stereo = np.repeat(localized_mono[:, np.newaxis], 2, axis=1)
    target_samples = int(localized_stereo.shape[0])
    if target_samples <= 0:
        raise ToolError(
            ERROR_UNSUPPORTED_FORMAT,
            "localized dialogue contains no usable samples",
        )
    progress.update(30, "localized dialogue conversion complete")

    progress.update(40, "decoding and normalizing sound effects")
    effect = decode_audio(
        input_path,
        original_effect.ordinal,
        output_rate,
        2,
        "sound effects track",
    )
    progress.update(50, "sound effects decoded")

    progress.update(60, "aligning effects to the localized dialogue timeline")
    aligned_effect, timeline = build_sound_effect_timeline(
        effect,
        target_samples,
        args.sound_offset_ms,
        output_rate,
    )
    within_tolerance = timeline["alignment_error_ms"] <= ALIGNMENT_TOLERANCE_MS
    if not within_tolerance:
        raise ToolError(
            ERROR_PROCESSING_FAILED,
            "computed alignment error exceeds the 5 ms tolerance",
            f"error={timeline['alignment_error_ms']} ms",
        )

    progress.update(70, "encoding output container")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".localize_dialogue_",
            dir=str(output_path.parent),
        ) as temporary_name:
            temporary_dir = Path(temporary_name)
            encoded_path, actual_codec, fallback_reason = encode_output(
                localized_stereo,
                aligned_effect,
                output_rate,
                output_path,
                original_effect.codec_name,
                temporary_dir,
            )

            progress.update(80, "verifying encoded stream layout and samples")
            verification = verify_output(
                encoded_path,
                output_rate,
                target_samples,
                localized_stereo,
                aligned_effect,
                sample_tolerance_for_codec(actual_codec),
            )

            progress.update(90, "publishing output container")
            os.replace(encoded_path, output_path)
    except ToolError:
        raise
    except (OSError, ValueError) as exc:
        raise ToolError(
            ERROR_ENCODING_FAILED,
            f"failed to prepare or publish output container: {output_path}",
            str(exc),
        ) from exc

    output_payload = probe_media(output_path)
    output_streams = audio_streams(output_payload)
    if len(output_streams) != 2:
        raise ToolError(
            ERROR_ENCODING_FAILED,
            "published output does not contain exactly two audio streams",
        )

    output_format_info = output_payload.get("format", {})
    source_dialogue_duration = original_dialogue.duration_seconds
    localized_duration = localized.duration_seconds
    effect_source_duration = original_effect.duration_seconds
    converted_duration = target_samples / output_rate

    report = empty_report(args, output_path)
    report["status"] = STATUS_SUCCESS
    report["error_code"] = None
    report["message"] = "localized dialogue replacement completed"
    report["dialogue_track"] = {
        "track_index": 0,
        "sample_rate": output_rate,
        "channels": 2,
        "duration_seconds": round(converted_duration, 6),
        "sample_count": target_samples,
        "source_sample_rate": localized.sample_rate,
        "source_channels": localized.channels,
        "source_duration_seconds": duration_round(localized_duration),
        "converted_sample_rate": output_rate,
        "converted_channels": 2,
        "converted_duration_seconds": round(converted_duration, 6),
        "resampled": localized.sample_rate != output_rate,
        "channel_expanded": True,
    }
    report["sound_effect_track"] = {
        "track_index": 1,
        "sample_rate": output_rate,
        "channels": 2,
        "duration_seconds": round(converted_duration, 6),
        "sample_count": target_samples,
        "source_sample_rate": original_effect.sample_rate,
        "source_channels": original_effect.channels,
        "source_duration_seconds": duration_round(effect_source_duration),
        "converted_sample_rate": output_rate,
        "converted_channels": 2,
        "converted_duration_seconds": round(converted_duration, 6),
        "offset_ms": args.sound_offset_ms,
        "resampled": original_effect.sample_rate != output_rate,
        "remixed": original_effect.channels != 2,
        "offset_samples": timeline["offset_samples"],
        "leading_padding_samples": timeline["leading_padding_samples"],
        "trailing_padding_samples": timeline["trailing_padding_samples"],
        "retained_samples": timeline["retained_samples"],
        "truncated_samples": timeline["truncated_samples"],
        "truncated_leading_samples": timeline["truncated_leading_samples"],
        "truncated_trailing_samples": timeline["truncated_trailing_samples"],
        "padded_samples": timeline["leading_padding_samples"]
        + timeline["trailing_padding_samples"],
    }
    report["ignored_audio_tracks"] = ignored_tracks
    report["source_track_info"] = {
        "dialogue_codec": original_dialogue.codec_name,
        "effects_codec": original_effect.codec_name,
        "dialogue_sample_rate": original_dialogue.sample_rate,
        "dialogue_channels": original_dialogue.channels,
        "dialogue_duration_seconds": duration_round(source_dialogue_duration),
        "effects_sample_rate": original_effect.sample_rate,
        "effects_channels": original_effect.channels,
        "effects_duration_seconds": duration_round(effect_source_duration),
        "audio_track_count": len(input_streams),
    }
    report["alignment"] = {
        "anchor": "localized_dialogue_start",
        "requested_offset_ms": args.sound_offset_ms,
        "realized_offset_ms": round(
            timeline["offset_samples"] / output_rate * 1000.0, 6
        ),
        "alignment_error_ms": timeline["alignment_error_ms"],
        "tolerance_ms": ALIGNMENT_TOLERANCE_MS,
        "within_tolerance": True,
    }
    report["output_format"] = {
        "container": output_format_info.get("format_name", "unknown"),
        "audio_codec": output_streams[0].codec_name,
        "effects_audio_codec": output_streams[1].codec_name,
        "pcm": output_streams[0].codec_name.startswith("pcm_"),
        "source_effects_codec": original_effect.codec_name,
        "fallback_reason": fallback_reason,
    }
    report["verification"] = verification
    return report, output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    report_hint = extract_report_argument(arguments)
    progress = Progress()
    parser = build_parser()
    args: Optional[CliArguments] = None

    try:
        namespace = parser.parse_args(arguments)
        args = resolve_cli_arguments(namespace)
        report_hint = absolute_path(args.report)
        report, output_path = process(args, progress)
        write_json_atomic(report_hint, report)
        progress.finish()
        print_json(report)
        return 0
    except ArgumentParsingError as exc:
        error = ToolError(ERROR_INVALID_ARGUMENT, str(exc))
        progress.stop()
        _handle_failure(args, report_hint, error)
        return 2
    except ToolError as exc:
        progress.stop()
        _handle_failure(args, report_hint, exc)
        return 1
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        error = ToolError(
            ERROR_PROCESSING_FAILED,
            "unexpected processing failure",
            f"{type(exc).__name__}: {exc}",
        )
        progress.stop()
        _handle_failure(args, report_hint, error)
        return 1


def _handle_failure(
    args: Optional[CliArguments],
    report_hint: Optional[Path],
    error: ToolError,
) -> None:
    if report_hint is not None:
        report_hint = absolute_path(report_hint)
    output_path = absolute_path(args.output_container) if args is not None else None
    report = failure_report(args, output_path, error)

    if report_hint is not None:
        try:
            write_json_atomic(report_hint, report)
        except OSError as exc:
            print(
                f"ERROR: cannot write report {report_hint}: {exc}",
                file=sys.stderr,
            )
    else:
        print(
            "ERROR: no report path was provided; failure report is stdout only",
            file=sys.stderr,
        )

    print(f"ERROR {error.code}: {error.message}", file=sys.stderr)
    if error.details:
        print(error.details, file=sys.stderr)
    print_json(report)


if __name__ == "__main__":
    raise SystemExit(main())
