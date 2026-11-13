"""Thin, validated wrappers around ffmpeg and ffprobe."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
from pathlib import Path
from typing import Any, Optional

from . import codes
from .model import JobFailure, TrackInfo


def _tail(value: str, limit: int = 1200) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return "..." + value[-limit:]


def _command_group_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


class Toolchain:
    def __init__(self, ffmpeg: Path, ffprobe: Path, timeout_sec: int = 600) -> None:
        self.ffmpeg = Path(ffmpeg)
        self.ffprobe = Path(ffprobe)
        self.timeout_sec = timeout_sec
        self.versions: dict[str, str] = {}

    @staticmethod
    def _resolve_executable(path: Path, tool_name: str) -> Path:
        candidate = Path(path)
        if candidate.is_dir():
            candidate = candidate / (f"{tool_name}.exe" if os.name == "nt" else tool_name)
        if os.name == "nt" and not candidate.exists() and candidate.suffix == "":
            exe_candidate = candidate.with_suffix(".exe")
            if exe_candidate.exists():
                return exe_candidate
        return candidate

    def validate(self) -> None:
        self.ffmpeg = self._resolve_executable(self.ffmpeg, "ffmpeg")
        self.ffprobe = self._resolve_executable(self.ffprobe, "ffprobe")
        if not self.ffmpeg.is_file():
            raise JobFailure(
                codes.FFMPEG_UNAVAILABLE,
                f"ffmpeg executable does not exist: {self.ffmpeg}",
                fatal=True,
            )
        if not self.ffprobe.is_file():
            raise JobFailure(
                codes.FFPROBE_UNAVAILABLE,
                f"ffprobe executable does not exist: {self.ffprobe}",
                fatal=True,
            )
        for tool, code in ((self.ffmpeg, codes.FFMPEG_UNAVAILABLE), (self.ffprobe, codes.FFPROBE_UNAVAILABLE)):
            try:
                completed = self._run([str(tool), "-version"], timeout=10, check=False)
            except JobFailure as exc:
                raise JobFailure(code, f"{tool.name} is not runnable: {exc.message}", fatal=True) from exc
            if completed.returncode != 0:
                raise JobFailure(
                    code,
                    f"{tool.name} -version exited with {completed.returncode}: {_tail(completed.stderr)}",
                    fatal=True,
                )
            self.versions[tool.name] = (completed.stdout or completed.stderr).splitlines()[0] if (
                completed.stdout or completed.stderr
            ) else tool.name

    def _run(
        self,
        args: list[str],
        *,
        timeout: Optional[int] = None,
        check: bool = True,
        error_code: str = codes.INTERNAL_ERROR,
        error_context: str = "",
    ) -> subprocess.CompletedProcess[str]:
        timeout_value = timeout if timeout is not None else self.timeout_sec
        try:
            process = subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                **_command_group_kwargs(),
            )
        except FileNotFoundError as exc:
            raise JobFailure(
                codes.FFMPEG_UNAVAILABLE if "ffmpeg" in Path(args[0]).name else codes.FFPROBE_UNAVAILABLE,
                f"cannot launch {args[0]}: {exc}",
                fatal=True,
            ) from exc
        except OSError as exc:
            raise JobFailure(
                error_code,
                f"{error_context or 'cannot launch process'}: {exc}",
                details={"command": args[0]},
            ) from exc
        try:
            stdout, stderr = process.communicate(timeout=timeout_value)
        except subprocess.TimeoutExpired as exc:
            if os.name == "nt":
                process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    process.kill()
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            raise JobFailure(
                error_code,
                f"{error_context or 'process'} exceeded timeout of {timeout_value} seconds",
                details={"stderr_tail": _tail(stderr or "")},
            ) from exc
        completed = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
        if check and completed.returncode != 0:
            combined = _tail(f"{completed.stderr}\n{completed.stdout}")
            if "no space left on device" in combined.lower():
                code = codes.TEMP_SPACE_INSUFFICIENT
            else:
                code = error_code
            raise JobFailure(
                code,
                f"{error_context or 'command failed'} (exit {completed.returncode})",
                details={"stderr_tail": combined},
            )
        return completed

    def probe(self, role: str, path: Path) -> TrackInfo:
        if not path.exists():
            raise JobFailure(
                codes.SOURCE_MISSING,
                f"{role} source does not exist: {path}",
                details={"role": role, "path": str(path)},
            )
        if not path.is_file():
            raise JobFailure(
                codes.SOURCE_UNREADABLE,
                f"{role} source is not a regular file: {path}",
                details={"role": role, "path": str(path)},
            )
        if not os.access(path, os.R_OK):
            raise JobFailure(
                codes.SOURCE_UNREADABLE,
                f"{role} source is not readable: {path}",
                details={"role": role, "path": str(path)},
            )
        command = [
            str(self.ffprobe),
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index,codec_name,sample_rate,channels,duration:format=duration",
            "-of",
            "json",
            str(path),
        ]
        completed = self._run(
            command,
            error_code=codes.INPUT_PROBE_FAILED,
            error_context=f"ffprobe failed for {role} source",
        )
        try:
            data = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise JobFailure(
                codes.INPUT_PROBE_FAILED,
                f"ffprobe returned invalid JSON for {role} source: {exc}",
                details={"stderr_tail": _tail(completed.stderr)},
            ) from exc
        streams = data.get("streams")
        if not isinstance(streams, list) or not streams:
            raise JobFailure(
                codes.SOURCE_HAS_NO_AUDIO,
                f"{role} source has no audio stream: {path}",
                details={"role": role, "path": str(path)},
            )
        stream = streams[0]
        try:
            sample_rate = int(stream["sample_rate"])
            channels = int(stream["channels"])
        except (KeyError, TypeError, ValueError) as exc:
            raise JobFailure(
                codes.INPUT_PROBE_FAILED,
                f"ffprobe did not report sample rate/channels for {role} source",
            ) from exc
        if sample_rate not in (44100, 48000) or channels not in (1, 2):
            raise JobFailure(
                codes.UNSUPPORTED_AUDIO_FORMAT,
                f"{role} source uses unsupported format: {sample_rate} Hz, {channels} channel(s)",
                details={
                    "role": role,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "supported_sample_rates": [44100, 48000],
                    "supported_channels": [1, 2],
                },
            )
        duration: Optional[float] = None
        for value in (stream.get("duration"), (data.get("format") or {}).get("duration")):
            try:
                candidate = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(candidate) and candidate > 0.0:
                duration = candidate
                break
        return TrackInfo(
            role=role,
            path=path,
            sample_rate=sample_rate,
            channels=channels,
            duration_sec=duration,
            codec_name=stream.get("codec_name"),
        )

    def decode_to_pcm48_stereo(self, source: Path, destination: Path, role: str) -> None:
        command = [
            str(self.ffmpeg),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            "-map_metadata",
            "-1",
            "-fflags",
            "+bitexact",
            "-flags",
            "+bitexact",
            "-f",
            "wav",
            str(destination),
        ]
        self._run(
            command,
            error_code=codes.AUDIO_DECODE_FAILED,
            error_context=f"failed to decode/normalize {role} track",
        )
        if not destination.is_file() or destination.stat().st_size <= 44:
            raise JobFailure(
                codes.AUDIO_DECODE_FAILED,
                f"decoder produced no audio for {role} track",
            )

    def mix_tracks(
        self,
        dialogue: Path,
        music: Path,
        ambience: Path,
        destination: Path,
        fade_start_sec: float,
        fade_duration_sec: float,
        music_gain_db: float,
        ambience_gain_db: float,
    ) -> None:
        def number(value: float) -> str:
            return format(float(value), ".10g")

        filter_graph = (
            "[0:a]anull[dialogue];"
            f"[1:a]afade=t=out:st={number(fade_start_sec)}:d={number(fade_duration_sec)}:curve=tri,"
            f"volume={number(music_gain_db)}dB[music];"
            f"[2:a]volume={number(ambience_gain_db)}dB[ambience];"
            "[dialogue][music][ambience]amix=inputs=3:duration=longest:dropout_transition=0:normalize=0,"
            "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[mix]"
        )
        command = [
            str(self.ffmpeg),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(dialogue),
            "-i",
            str(music),
            "-i",
            str(ambience),
            "-filter_complex",
            filter_graph,
            "-map",
            "[mix]",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_f32le",
            "-map_metadata",
            "-1",
            "-fflags",
            "+bitexact",
            "-flags",
            "+bitexact",
            "-f",
            "wav",
            str(destination),
        ]
        self._run(
            command,
            error_code=codes.MIX_FAILED,
            error_context="three-track mix failed",
        )
        if not destination.is_file() or destination.stat().st_size <= 44:
            raise JobFailure(codes.MIX_FAILED, "mixer produced no audio")

    def measure_integrated_lufs(self, track: Path) -> float:
        command = [
            str(self.ffmpeg),
            "-nostdin",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "info",
            "-i",
            str(track),
            "-vn",
            "-af",
            "loudnorm=I=-23:TP=-2:LRA=11:print_format=json",
            "-f",
            "null",
            "-",
        ]
        completed = self._run(
            command,
            error_code=codes.LOUDNESS_MEASUREMENT_FAILED,
            error_context="dialogue loudness measurement failed",
        )
        stderr = completed.stderr
        start = stderr.rfind("{")
        end = stderr.rfind("}")
        if start < 0 or end <= start:
            raise JobFailure(
                codes.LOUDNESS_MEASUREMENT_FAILED,
                "loudness analyzer emitted no JSON summary",
                details={"stderr_tail": _tail(stderr)},
            )
        try:
            data = json.loads(stderr[start : end + 1])
            value = float(data["input_i"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise JobFailure(
                codes.LOUDNESS_MEASUREMENT_FAILED,
                "could not parse integrated loudness from analyzer output",
                details={"stderr_tail": _tail(stderr)},
            ) from exc
        if not math.isfinite(value):
            raise JobFailure(
                codes.LOUDNESS_MEASUREMENT_FAILED,
                f"dialogue integrated loudness is not finite ({data.get('input_i')!r})",
            )
        return value

    def apply_gain_to_raw_pcm(self, source: Path, destination: Path, gain_db: float) -> None:
        command = [
            str(self.ffmpeg),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            f"volume={format(float(gain_db), '.10g')}dB",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            "-f",
            "s16le",
            str(destination),
        ]
        self._run(
            command,
            error_code=codes.FINALIZE_FAILED,
            error_context="final gain/16-bit PCM render failed",
        )
        if not destination.is_file() or destination.stat().st_size == 0:
            raise JobFailure(codes.FINALIZE_FAILED, "final renderer produced no PCM data")
