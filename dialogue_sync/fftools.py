"""FFmpeg/ffprobe wrappers with explicit version and error handling."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ExternalToolError, StateError


@dataclass(frozen=True)
class AudioProbe:
    codec_name: str
    sample_rate: int
    channels: int
    bit_depth: int
    duration_seconds: float
    sample_fmt: str


class FFmpegTools:
    def __init__(
        self,
        ffmpeg: str | None = None,
        ffprobe: str | None = None,
        *,
        check_version: bool = True,
    ) -> None:
        self.ffmpeg = ffmpeg or os.environ.get("UID110_FFMPEG", "ffmpeg")
        self.ffprobe = ffprobe or os.environ.get("UID110_FFPROBE", "ffprobe")
        self.ffmpeg_path = shutil.which(self.ffmpeg)
        self.ffprobe_path = shutil.which(self.ffprobe)
        if self.ffmpeg_path is None:
            raise ExternalToolError(f"ffmpeg executable not found: {self.ffmpeg}")
        if self.ffprobe_path is None:
            raise ExternalToolError(f"ffprobe executable not found: {self.ffprobe}")
        if check_version:
            self._check_version()

    def _check_version(self) -> None:
        output = self._run([self.ffmpeg_path, "-version"], allow_failure=False)
        match = re.search(r"ffmpeg version ([0-9]+(?:\.[0-9]+)+)", output.stdout)
        if not match:
            raise ExternalToolError("unable to determine FFmpeg version")
        version = tuple(int(part) for part in match.group(1).split("."))
        if version < (5, 0):
            raise ExternalToolError(
                f"FFmpeg 5.0 or newer is required, found {match.group(1)}"
            )

    def _run(
        self,
        args: list[str],
        *,
        input_data: bytes | None = None,
        allow_failure: bool = False,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                args,
                input=input_data,
                text=False,
                capture_output=True,
                check=False,
                timeout=timeout,
                env={**os.environ, "LC_ALL": "C"},
            )
        except OSError as exc:
            raise ExternalToolError(f"cannot execute {args[0]}: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ExternalToolError(f"external command timed out: {args[0]}") from exc
        text_out = result.stdout.decode("utf-8", errors="replace")
        text_err = result.stderr.decode("utf-8", errors="replace")
        completed = subprocess.CompletedProcess(
            args=args, returncode=result.returncode, stdout=text_out, stderr=text_err
        )
        if result.returncode != 0 and not allow_failure:
            tail = text_err.strip().splitlines()[-20:]
            detail = "\n".join(tail) if tail else "(no stderr)"
            raise ExternalToolError(
                f"external command failed with exit {result.returncode}: {args[0]}\n{detail}"
            )
        return completed

    def command(
        self,
        args: list[str],
        *,
        allow_failure: bool = False,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return self._run(
            [self.ffmpeg_path, "-hide_banner", "-nostdin", *args],
            allow_failure=allow_failure,
            timeout=timeout,
        )

    def pipe_ffmpeg(
        self,
        args: list[str],
        *,
        input_data: bytes | None = None,
    ) -> bytes:
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-hide_banner", "-nostdin", *args],
                input=input_data,
                capture_output=True,
                check=False,
                env={**os.environ, "LC_ALL": "C"},
            )
        except OSError as exc:
            raise ExternalToolError(f"cannot execute ffmpeg: {exc}") from exc
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise ExternalToolError(f"ffmpeg pipe command failed:\n{detail}")
        return result.stdout

    def probe_audio(self, path: Path) -> AudioProbe:
        command = [
            self.ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            (
                "stream=codec_name,sample_rate,channels,bits_per_sample,"
                "bits_per_raw_sample,duration,sample_fmt"
            ),
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
        result = self._run(command)
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ExternalToolError(f"invalid ffprobe JSON for {path}") from exc
        streams = payload.get("streams") or []
        if not streams:
            raise ExternalToolError(f"{path}: no audio stream found")
        stream = streams[0]
        sample_rate = int(stream.get("sample_rate") or 0)
        channels = int(stream.get("channels") or 0)
        bit_depth = int(stream.get("bits_per_raw_sample") or 0)
        if not bit_depth:
            bit_depth = int(stream.get("bits_per_sample") or 0)
        duration_text = stream.get("duration") or (payload.get("format") or {}).get(
            "duration"
        )
        try:
            duration = float(duration_text)
        except (TypeError, ValueError) as exc:
            raise ExternalToolError(f"{path}: audio duration is unavailable") from exc
        if duration <= 0:
            raise ExternalToolError(f"{path}: audio duration must be positive")
        return AudioProbe(
            codec_name=str(stream.get("codec_name") or ""),
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
            duration_seconds=duration,
            sample_fmt=str(stream.get("sample_fmt") or ""),
        )

    def probe_json(self, path: Path, extra_args: list[str] | None = None) -> dict[str, Any]:
        command = [
            self.ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration,size,format_name:stream=index,codec_name,codec_type,"
            "sample_rate,channels,bits_per_sample,bits_per_raw_sample:stream_tags=language,title",
            "-show_chapters",
            "-of",
            "json",
            *(extra_args or []),
            str(path),
        ]
        result = self._run(command)
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ExternalToolError(f"invalid ffprobe JSON for {path}") from exc

    def measure_loudness(self, path: Path, audio: Any) -> dict[str, Any]:
        filter_spec = (
            f"loudnorm=I={audio.target_lufs}:TP={audio.max_true_peak_dbtp}:"
            f"LRA={audio.loudness_range}:print_format=json"
        )
        result = self.command(
            ["-i", str(path), "-af", filter_spec, "-f", "null", "-"],
            allow_failure=False,
        )
        payload = _last_json_object(result.stderr)
        if payload is None:
            raise ExternalToolError(f"could not parse loudnorm output for {path}")
        return _normalize_loudness_payload(payload)

    def measure_ebur128(self, path: Path) -> dict[str, float]:
        result = self.command(
            ["-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
            allow_failure=False,
        )
        integrated = _find_last_float(result.stderr, r"I:\s+(-?[\d.]+|-inf)\s+LUFS")
        peak = _find_last_float(result.stderr, r"Peak:\s+(-?[\d.]+|-inf)\s+dBFS")
        lra = _find_last_float(result.stderr, r"LRA:\s+(-?[\d.]+|-inf)\s+LU")
        return {
            "integrated_lufs": integrated,
            "true_peak_dbtp": peak,
            "loudness_range_lu": lra,
        }


def _last_json_object(text: str) -> dict[str, Any] | None:
    starts = [match.start() for match in re.finditer(r"\{", text)]
    for start in reversed(starts):
        depth = 0
        in_string = False
        escaped = False
        end: int | None = None
        for position in range(start, len(text)):
            character = text[position]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    end = position + 1
                    break
        if end is None:
            continue
        fragment = text[start:end]
        try:
            candidate = json.loads(fragment)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    return None


def _to_float(value: Any) -> float:
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"-inf", "inf"}:
            return float(stripped)
    return float(value)


def _normalize_loudness_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    result["input_i"] = _to_float(payload.get("input_i", "-inf"))
    result["input_tp"] = _to_float(payload.get("input_tp", "-inf"))
    result["input_lra"] = _to_float(payload.get("input_lra", "0.0"))
    result["input_thresh"] = _to_float(payload.get("input_thresh", "-inf"))
    result["target_offset"] = _to_float(payload.get("target_offset", "0.0"))
    result["normalization_type"] = payload.get("normalization_type")
    return result


def _find_last_float(text: str, pattern: str) -> float:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if not matches:
        return float("nan")
    value = matches[-1].group(1)
    return float(value)


def atomic_ffmpeg_output(
    tools: FFmpegTools,
    target: Path,
    args: list[str],
    *,
    timeout: float | None = None,
) -> None:
    """Run FFmpeg into a sibling temporary file and atomically replace target."""

    temp = target.with_name(
        f".{target.stem}.{os.getpid()}.partial{target.suffix}"
    )
    try:
        temp.unlink(missing_ok=True)
    except OSError as exc:
        raise StateError(f"cannot remove stale temporary output {temp}: {exc}") from exc
    try:
        tools.command(["-y", *args, str(temp)], timeout=timeout)
        with temp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp, target)
        _fsync_directory(target.parent)
    except Exception:
        try:
            temp.unlink(missing_ok=True)
        finally:
            raise


def atomic_write_bytes(path: Path, data: bytes) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            temp.unlink(missing_ok=True)
        finally:
            raise


def atomic_write_json(path: Path, payload: Any) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    atomic_write_bytes(path, data)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)

