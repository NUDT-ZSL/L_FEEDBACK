from __future__ import annotations

import math
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .errors import ProcessingError


_METADATA_FILTER = (
    "ebur128=peak=true:metadata=1,"
    "astats=metadata=1:reset=1:length=0.1:"
    "measure_perchannel=none:measure_overall=Peak_level,"
    "ametadata=mode=print:file=-"
)
_INTEGRATED_RE = re.compile(
    r"Integrated loudness:.*?\bI:\s*([-+]?(?:\d+(?:\.\d*)?|inf|nan))\s*LUFS",
    re.DOTALL,
)
_LRA_RE = re.compile(
    r"Loudness range:.*?\bLRA:\s*([-+]?(?:\d+(?:\.\d*)?|inf|nan))\s*LU",
    re.DOTALL,
)
_TRUE_PEAK_RE = re.compile(
    r"True peak:.*?\bPeak:\s*([-+]?(?:\d+(?:\.\d*)?|inf|nan))\s*dBFS",
    re.DOTALL,
)
_FRAME_RE = re.compile(
    r"^frame:\d+\s+pts:\S*\s+pts_time:([-+]?(?:\d+(?:\.\d*)?|inf|nan))\s*$"
)


@dataclass
class LoudnessAnalysis:
    integrated_lufs: float
    loudness_range_lu: float
    true_peak_dbtp: float
    short_term_min_lufs: float | None
    short_term_max_lufs: float | None
    short_term_mean_lufs: float | None
    short_term_window_count: int
    true_peak_violations: list[dict]
    duration_seconds: float

    def to_report(self) -> dict:
        return {
            "integrated_lufs": _round(self.integrated_lufs),
            "loudness_range_lu": _round(self.loudness_range_lu),
            "true_peak_dbtp": _round(self.true_peak_dbtp),
            "short_term": {
                "window_length_seconds": 3.0,
                "window_hop_seconds": 0.1,
                "min_lufs": _round_optional(self.short_term_min_lufs),
                "max_lufs": _round_optional(self.short_term_max_lufs),
                "mean_lufs": _round_optional(self.short_term_mean_lufs),
                "window_count": self.short_term_window_count,
            },
            "true_peak_violation_count": len(self.true_peak_violations),
            "true_peak_violations": self.true_peak_violations,
            "duration_seconds": _round(self.duration_seconds),
        }


def _round(value: float, digits: int = 3) -> float:
    if not math.isfinite(value):
        return -120.0 if value < 0 else 120.0
    return round(float(value), digits)


def _round_optional(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else _round(value, digits)


def _parse_db(value: str, *, default: float) -> float:
    try:
        number = float(value)
    except ValueError:
        return default
    if math.isnan(number):
        return default
    if math.isinf(number):
        return -120.0 if number < 0 else 120.0
    return number


def _ffmpeg_binary() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def _run_stream(
    cmd: list[str],
    audio: np.ndarray | None = None,
    *,
    capture_stdout: bool = True,
    chunk_frames: int = 262144,
) -> tuple[bytes, bytes]:
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if audio is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    writer_error: list[BaseException] = []

    def writer() -> None:
        assert process.stdin is not None
        try:
            for start in range(0, len(audio), chunk_frames):  # type: ignore[arg-type]
                block = audio[start : start + chunk_frames]  # type: ignore[index]
                process.stdin.write(block.tobytes(order="C"))
        except (BrokenPipeError, OSError) as exc:
            writer_error.append(exc)
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass

    thread: threading.Thread | None = None
    if audio is not None:
        thread = threading.Thread(target=writer, name="ffmpeg-stdin", daemon=True)
        thread.start()

    stdout_chunks: list[bytes] = []
    if capture_stdout and process.stdout is not None:
        while True:
            block = process.stdout.read(1024 * 1024)
            if not block:
                break
            stdout_chunks.append(block)
    stderr = process.stderr.read() if process.stderr is not None else b""
    return_code = process.wait()
    if process.stdout is not None:
        process.stdout.close()
    if process.stderr is not None:
        process.stderr.close()
    if process.stdin is not None:
        try:
            process.stdin.close()
        except OSError:
            pass
    if thread is not None:
        thread.join(timeout=5)

    if return_code != 0:
        detail = stderr.decode("utf-8", "replace").strip()
        raise ProcessingError(3020, f"FFmpeg filter failed (exit {return_code}): {detail[-1600:]}")
    if writer_error:
        detail = str(writer_error[0])
        if "Broken pipe" not in detail:
            raise ProcessingError(3021, f"failed while feeding FFmpeg: {detail}")
    return b"".join(stdout_chunks), stderr


def analyze_file(path: Path, frame_rate: float) -> LoudnessAnalysis:
    cmd = [
        _ffmpeg_binary(),
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-filter_threads",
        "1",
        "-loglevel",
        "info",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-af",
        _METADATA_FILTER,
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, check=False)
    except OSError as exc:
        raise ProcessingError(3020, f"cannot start FFmpeg: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise ProcessingError(
            3020, f"FFmpeg loudness analysis failed: {detail[-1600:]}"
        )
    return parse_loudness_output(completed.stdout, completed.stderr, frame_rate)


def analyze_array(
    audio: np.ndarray, sample_rate: int, channels: int, frame_rate: float
) -> LoudnessAnalysis:
    contiguous = np.ascontiguousarray(audio, dtype="<f4")
    cmd = [
        _ffmpeg_binary(),
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-filter_threads",
        "1",
        "-loglevel",
        "info",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-i",
        "pipe:0",
        "-af",
        _METADATA_FILTER,
        "-f",
        "null",
        "-",
    ]
    stdout, stderr = _run_stream(cmd, contiguous, capture_stdout=True)
    return parse_loudness_output(stdout, stderr, frame_rate)


def parse_loudness_output(
    metadata_bytes: bytes, stderr_bytes: bytes, frame_rate: float
) -> LoudnessAnalysis:
    text = metadata_bytes.decode("utf-8", "replace")
    windows: list[tuple[float, float | None, float | None, float | None]] = []
    current_time: float | None = None
    current_s: float | None = None
    current_tp: float | None = None
    current_sample_peak: float | None = None

    def flush() -> None:
        nonlocal current_time, current_s, current_tp, current_sample_peak
        if current_time is not None:
            windows.append(
                (current_time, current_s, current_tp, current_sample_peak)
            )
        current_time = None
        current_s = None
        current_tp = None
        current_sample_peak = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        frame_match = _FRAME_RE.match(line)
        if frame_match:
            flush()
            try:
                current_time = float(frame_match.group(1))
            except ValueError:
                current_time = None
            continue
        if line.startswith("lavfi.r128.S="):
            try:
                current_s = float(line.split("=", 1)[1])
            except ValueError:
                current_s = None
        elif line.startswith("lavfi.r128.true_peak="):
            try:
                current_tp = float(line.split("=", 1)[1])
            except ValueError:
                current_tp = None
        elif line.startswith("lavfi.astats.Overall.Peak_level="):
            try:
                current_sample_peak = float(line.split("=", 1)[1])
            except ValueError:
                current_sample_peak = None
    flush()

    stderr_text = stderr_bytes.decode("utf-8", "replace")
    integrated_matches = _INTEGRATED_RE.findall(stderr_text)
    integrated = (
        _parse_db(integrated_matches[-1], default=-120.0)
        if integrated_matches
        else _fallback_integrated(windows)
    )
    lra_matches = _LRA_RE.findall(stderr_text)
    lra = _parse_db(lra_matches[-1], default=0.0) if lra_matches else 0.0
    peak_matches = _TRUE_PEAK_RE.findall(stderr_text)
    if peak_matches:
        true_peak = _parse_db(peak_matches[-1], default=-120.0)
    else:
        valid_peaks = [
            _linear_to_db(tp)
            for _, _, tp, _ in windows
            if tp is not None and tp > 0
        ]
        true_peak = max(valid_peaks) if valid_peaks else -120.0

    short_term_values = [
        value
        for timestamp, value, _, _ in windows
        if value is not None and timestamp >= 2.9 - 1e-9
    ]
    if short_term_values:
        short_min = min(short_term_values)
        short_max = max(short_term_values)
        short_mean = float(np.mean(np.asarray(short_term_values, dtype=np.float64)))
    else:
        short_min = short_max = short_mean = None

    violations: list[dict] = []
    reported_times: set[float] = set()
    last_reported_peak = -math.inf
    for timestamp, _, linear_peak, sample_peak_db in windows:
        if linear_peak is None or linear_peak <= 0:
            continue
        peak_db = _linear_to_db(linear_peak)
        per_window_peak = (
            sample_peak_db
            if sample_peak_db is not None and math.isfinite(sample_peak_db)
            else None
        )
        if per_window_peak is not None and per_window_peak > -1.0:
            violations.append(
                {
                    "timecode": format_timecode(timestamp, frame_rate),
                    "window_start_seconds": round(timestamp, 3),
                    "true_peak_dbtp": round(per_window_peak, 3),
                    "detection": "per_window_sample_peak",
                    "ebur128_running_true_peak_dbtp": round(peak_db, 3),
                }
            )
            reported_times.add(timestamp)
            last_reported_peak = peak_db
        # ebur128's per-frame true_peak metadata is a running maximum. Its
        # increases also catch intersample peaks that a sample scan cannot see.
        elif (
            peak_db > -1.0
            and peak_db > last_reported_peak + 1e-6
            and timestamp not in reported_times
        ):
            violations.append(
                {
                    "timecode": format_timecode(timestamp, frame_rate),
                    "window_start_seconds": round(timestamp, 3),
                    "true_peak_dbtp": round(peak_db, 3),
                    "detection": "ebur128_new_maximum",
                }
            )
            reported_times.add(timestamp)
            last_reported_peak = peak_db

    duration = windows[-1][0] + 0.1 if windows else 0.0
    return LoudnessAnalysis(
        integrated_lufs=integrated,
        loudness_range_lu=lra,
        true_peak_dbtp=true_peak,
        short_term_min_lufs=short_min,
        short_term_max_lufs=short_max,
        short_term_mean_lufs=short_mean,
        short_term_window_count=len(short_term_values),
        true_peak_violations=violations,
        duration_seconds=duration,
    )


def _fallback_integrated(
    windows: Iterable[tuple[float, float | None, float | None, float | None]]
) -> float:
    values = [
        value
        for _, value, _, _ in windows
        if value is not None and value > -70.0
    ]
    return values[-1] if values else -70.0


def _linear_to_db(value: float) -> float:
    if value <= 0 or not math.isfinite(value):
        return -120.0
    return 20.0 * math.log10(value)


def format_timecode(seconds: float, frame_rate: float) -> str:
    fps = max(1, int(round(frame_rate)))
    total_frames = max(0, int(round(max(0.0, seconds) * frame_rate)))
    frames = total_frames % fps
    total_seconds = total_frames // fps
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"


def decode_audio(
    path: Path,
    *,
    stream_selector: str = "0:a:0",
    sample_rate: int,
    channels: int,
    duration_seconds: float | None = None,
) -> np.ndarray:
    cmd = [
        _ffmpeg_binary(),
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        stream_selector,
        "-vn",
        "-sn",
        "-dn",
    ]
    if duration_seconds is not None:
        cmd.extend(["-t", f"{duration_seconds:.9f}"])
    cmd.extend(
        [
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "pipe:1",
        ]
    )
    try:
        completed = subprocess.run(cmd, capture_output=True, check=False)
    except OSError as exc:
        raise ProcessingError(3011, f"cannot start FFmpeg decode: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise ProcessingError(
            3011, f"audio decode failed for {path}: {detail[-1600:]}"
        )
    byte_length = len(completed.stdout)
    frame_bytes = channels * 4
    if byte_length == 0 or byte_length % frame_bytes != 0:
        raise ProcessingError(
            3012,
            f"decoded sample data is not block aligned for {path}: "
            f"{byte_length} bytes, frame size {frame_bytes}",
        )
    audio = np.frombuffer(completed.stdout, dtype="<f4")
    if not np.all(np.isfinite(audio)):
        raise ProcessingError(3012, f"decoded audio contains NaN or infinity: {path}")
    return np.ascontiguousarray(audio.reshape(-1, channels), dtype=np.float32)


def run_loudnorm_pass(
    audio: np.ndarray,
    sample_rate: int,
    channels: int,
    *,
    target_lufs: float,
    true_peak_dbtp: float,
    measured: dict | None = None,
    capture_audio: bool,
) -> tuple[np.ndarray | None, dict]:
    contiguous = np.ascontiguousarray(audio, dtype="<f4")
    parts = [
        f"I={target_lufs}",
        f"TP={true_peak_dbtp}",
        "LRA=11",
        "print_format=json",
    ]
    if measured:
        parts.extend(
            [
                f"measured_I={measured['input_i']}",
                f"measured_TP={measured['input_tp']}",
                f"measured_LRA={measured['input_lra']}",
                f"measured_thresh={measured['input_thresh']}",
                f"offset={measured.get('target_offset', '0')}",
                "linear=true",
            ]
        )
    filter_expr = "loudnorm=" + ":".join(parts)
    cmd = [
        _ffmpeg_binary(),
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-filter_threads",
        "1",
        "-loglevel",
        "info",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-i",
        "pipe:0",
        "-af",
        filter_expr,
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
    ]
    if capture_audio:
        cmd.extend(["-f", "f32le", "pipe:1"])
    else:
        cmd.extend(["-f", "null", "-"])
    stdout, stderr = _run_stream(cmd, contiguous, capture_stdout=capture_audio)
    payload = parse_loudnorm_json(stderr)
    if not capture_audio:
        return None, payload
    frame_bytes = channels * 4
    if len(stdout) % frame_bytes != 0:
        raise ProcessingError(3022, "loudnorm output is not sample-block aligned")
    result = np.frombuffer(stdout, dtype="<f4")
    if not np.all(np.isfinite(result)):
        raise ProcessingError(3022, "loudnorm output contains NaN or infinity")
    return np.ascontiguousarray(result.reshape(-1, channels), dtype=np.float32), payload


def parse_loudnorm_json(stderr: bytes) -> dict:
    text = stderr.decode("utf-8", "replace")
    start = text.rfind("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ProcessingError(3022, "loudnorm did not return JSON statistics")
    import json

    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ProcessingError(3022, f"cannot parse loudnorm statistics: {exc}") from exc
    required = {
        "input_i",
        "input_tp",
        "input_lra",
        "input_thresh",
        "output_i",
        "output_tp",
    }
    if not required.issubset(payload):
        raise ProcessingError(3022, "loudnorm statistics are incomplete")
    return payload


def write_wav_24(path: Path, audio: np.ndarray, sample_rate: int, channels: int) -> None:
    contiguous = np.ascontiguousarray(audio, dtype="<f4")
    if contiguous.shape[1] != channels:
        raise ProcessingError(3030, "master channel count changed before WAV writing")
    cmd = [
        _ffmpeg_binary(),
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-i",
        "pipe:0",
        "-c:a",
        "pcm_s24le",
        "-f",
        "wav",
        str(path),
    ]
    _run_stream(cmd, contiguous, capture_stdout=False)
