from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf


SOFT_LIMIT_THRESHOLD = 0.95
NORMALIZE_TARGET_DBFS = -0.1
SILENCE_DBFS = -120.0


class AudioMixerError(Exception):
    """Base class for expected mixer failures."""

    code = "AUDIO_MIXER_ERROR"

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code


class ProjectError(AudioMixerError):
    code = "PROJECT_ERROR"


class EncodedOutputError(AudioMixerError):
    code = "FFMPEG_ENCODE_FAILED"

    def __init__(
        self,
        message: str,
        *,
        command: Sequence[str] = (),
        returncode: int | None = None,
        stderr: str = "",
    ):
        super().__init__(message)
        self.command = list(command)
        self.returncode = returncode
        self.stderr = stderr


@dataclass(frozen=True)
class OutputSpec:
    container: str
    bit_depth: int

    @property
    def extension(self) -> str:
        return f".{self.container}"

    @property
    def codec(self) -> str:
        if self.container == "flac":
            return "flac"
        return "pcm_s16le" if self.bit_depth == 16 else "pcm_s24le"

    @property
    def label(self) -> str:
        if self.container == "flac":
            return "flac"
        return f"wav_pcm{self.bit_depth}"


@dataclass
class Project:
    source_path: Path
    base_dir: Path
    sample_rate: int
    channels: int
    output: OutputSpec
    total_duration_sec: float
    total_samples: int
    clips: list[dict[str, Any]]
    output_stem: str
    raw: dict[str, Any] = field(repr=False)


@dataclass
class MixResult:
    audio: np.ndarray
    sample_rate: int
    channels: int
    total_samples: int
    peak_dbfs: float
    normalization_gain_db: float
    clip_reports: list[dict[str, Any]]
    logs: list[str]


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    """Convert seconds to samples with round-half-away-from-zero."""

    try:
        value = Decimal(str(seconds)) * Decimal(int(sample_rate))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("time value must be finite") from exc
    if not value.is_finite():
        raise ValueError("time value must be finite")
    return int(value.to_integral_value(rounding=ROUND_HALF_UP))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_number(
    value: Any, field_name: str, *, minimum: float | None = None
) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise ProjectError(f"{field_name} must be a finite number", code="INVALID_FIELD")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ProjectError(
            f"{field_name} must be >= {minimum}", code="INVALID_FIELD"
        )
    return number


def _require_positive_int(value: Any, field_name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ProjectError(
            f"{field_name} must be a positive integer", code="INVALID_FIELD"
        )
    return value


def _normalize_format_token(value: Any, field_name: str) -> tuple[str, int | None]:
    if not isinstance(value, str):
        raise ProjectError(f"{field_name} must be a string", code="INVALID_FORMAT")
    token = value.strip().lower().replace("-", "_").replace(" ", "_")
    if token in {"flac", "flac_lossless"}:
        return "flac", None
    if token in {"wav", "wave", "pcm"}:
        return "wav", None
    aliases = {
        "wav16": 16,
        "wav_16": 16,
        "wav_pcm16": 16,
        "wav_pcm_16": 16,
        "wav_pcm_s16le": 16,
        "wav_16_bit": 16,
        "pcm16": 16,
        "pcm_s16le": 16,
        "s16": 16,
        "wav24": 24,
        "wav_24": 24,
        "wav_pcm24": 24,
        "wav_pcm_24": 24,
        "wav_pcm_s24le": 24,
        "wav_24_bit": 24,
        "pcm24": 24,
        "pcm_s24le": 24,
        "s24": 24,
        "flac16": 16,
        "flac_16": 16,
        "flac24": 24,
        "flac_24": 24,
    }
    if token in aliases:
        return ("flac" if token.startswith("flac") else "wav"), aliases[token]
    raise ProjectError(
        f"unsupported output format {value!r}", code="INVALID_FORMAT"
    )


def _parse_output_format(raw: Any, project_raw: dict[str, Any]) -> OutputSpec:
    explicit_depth: int | None = None
    container: str

    if isinstance(raw, dict):
        token = raw.get("container", raw.get("format"))
        if token is None:
            codec = str(raw.get("codec", "")).lower()
            if "flac" in codec:
                container, token_depth = "flac", None
            elif "s24" in codec or "24" in codec:
                container, token_depth = "wav", 24
            elif "s16" in codec or "16" in codec:
                container, token_depth = "wav", 16
            else:
                raise ProjectError(
                    "output_format is missing container/format",
                    code="INVALID_FORMAT",
                )
        else:
            container, token_depth = _normalize_format_token(
                token, "output_format.container"
            )
        explicit_depth = token_depth
        depth_value = raw.get("bit_depth", raw.get("bits"))
        if depth_value is not None:
            if (
                not isinstance(depth_value, int)
                or isinstance(depth_value, bool)
                or depth_value not in {16, 24}
            ):
                raise ProjectError(
                    "output bit depth must be 16 or 24", code="INVALID_FORMAT"
                )
            explicit_depth = depth_value
    else:
        container, explicit_depth = _normalize_format_token(
            raw, "output_format"
        )

    if explicit_depth is None:
        top_depth = project_raw.get("output_bit_depth", project_raw.get("bit_depth"))
        if top_depth is not None:
            if (
                not isinstance(top_depth, int)
                or isinstance(top_depth, bool)
                or top_depth not in {16, 24}
            ):
                raise ProjectError(
                    "output_bit_depth must be 16 or 24", code="INVALID_FORMAT"
                )
            explicit_depth = top_depth

    if container == "wav":
        return OutputSpec("wav", explicit_depth or 16)
    return OutputSpec("flac", explicit_depth or 24)


def _project_output_stem(
    project_raw: dict[str, Any], source_path: Path, output: OutputSpec
) -> str:
    raw_name = project_raw.get(
        "output_name", project_raw.get("output_filename", source_path.stem)
    )
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ProjectError("output_name must be a non-empty string", code="INVALID_FIELD")
    name = Path(raw_name.strip()).name
    stem = Path(name).stem
    if not stem:
        raise ProjectError("output_name has no file stem", code="INVALID_FIELD")
    return stem


def load_project(path: str | os.PathLike[str]) -> Project:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise ProjectError(
            f"project file not found: {source_path}", code="PROJECT_NOT_FOUND"
        )
    try:
        with source_path.open("r", encoding="utf-8-sig") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ProjectError(
            f"invalid project JSON: line {exc.lineno}, column {exc.colno}",
            code="INVALID_JSON",
        ) from exc
    except UnicodeDecodeError as exc:
        raise ProjectError(
            f"invalid project JSON encoding: {exc}", code="INVALID_JSON"
        ) from exc
    except OSError as exc:
        raise ProjectError(
            f"cannot read project file: {exc}", code="PROJECT_IO_ERROR"
        ) from exc

    if not isinstance(raw, dict):
        raise ProjectError("project root must be a JSON object", code="INVALID_PROJECT")

    sample_rate = _require_positive_int(raw.get("sample_rate"), "sample_rate")
    channels = _require_positive_int(raw.get("channels"), "channels")
    if channels > 8:
        raise ProjectError("channels must be between 1 and 8", code="INVALID_FIELD")
    total_duration_sec = _require_number(
        raw.get("total_duration_sec"), "total_duration_sec", minimum=0.0
    )
    if total_duration_sec <= 0:
        raise ProjectError(
            "total_duration_sec must be greater than zero", code="INVALID_FIELD"
        )
    try:
        total_samples = seconds_to_samples(total_duration_sec, sample_rate)
    except (OverflowError, ValueError) as exc:
        raise ProjectError(
            f"total_duration_sec is outside the supported range: {exc}",
            code="INVALID_FIELD",
        ) from exc
    if total_samples <= 0:
        raise ProjectError(
            "total_duration_sec rounds to zero samples", code="INVALID_FIELD"
        )
    output = _parse_output_format(raw.get("output_format"), raw)
    clips = raw.get("clips")
    if not isinstance(clips, list):
        raise ProjectError("clips must be an array", code="INVALID_FIELD")
    output_stem = _project_output_stem(raw, source_path, output)

    return Project(
        source_path=source_path,
        base_dir=source_path.parent,
        sample_rate=sample_rate,
        channels=channels,
        output=output,
        total_duration_sec=total_duration_sec,
        total_samples=total_samples,
        clips=clips,
        output_stem=output_stem,
        raw=raw,
    )


def linear_resample(
    samples: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    """Linearly resample a channel-first or channel-last array."""

    if samples.ndim != 2:
        raise ValueError("samples must be two-dimensional")
    source_frames = int(samples.shape[0])
    if source_frames == 0:
        return samples.astype(np.float32, copy=True)
    if source_rate == target_rate:
        return samples.astype(np.float32, copy=True)
    target_frames = seconds_to_samples(source_frames / source_rate, target_rate)
    if target_frames <= 0:
        return np.empty((0, samples.shape[1]), dtype=np.float32)
    positions = np.arange(target_frames, dtype=np.float64) * (
        source_rate / target_rate
    )
    positions = np.minimum(positions, source_frames - 1)
    source_indices = np.arange(source_frames, dtype=np.float64)
    output = np.empty((target_frames, samples.shape[1]), dtype=np.float32)
    for channel in range(samples.shape[1]):
        output[:, channel] = np.interp(
            positions, source_indices, samples[:, channel]
        ).astype(np.float32)
    return output


def adapt_channels(samples: np.ndarray, target_channels: int) -> np.ndarray:
    source_channels = int(samples.shape[1])
    if source_channels == target_channels:
        return samples.astype(np.float32, copy=True)
    if source_channels > target_channels:
        return samples[:, :target_channels].astype(np.float32, copy=True)
    repeats = int(math.ceil(target_channels / source_channels))
    return np.tile(samples, (1, repeats))[:, :target_channels].astype(
        np.float32, copy=True
    )


def build_envelope(
    frame_count: int,
    fade_in_samples: int,
    fade_out_samples: int,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Build a multiplicative equal-power sine envelope bounded to [0, 1]."""

    envelope = np.ones(max(frame_count, 0), dtype=np.float64)
    if frame_count <= 0:
        return envelope, None

    fade_in = max(int(fade_in_samples), 0)
    fade_out = max(int(fade_out_samples), 0)
    warning: dict[str, Any] | None = None
    fade_total = fade_in + fade_out
    if fade_total > frame_count:
        ratio = frame_count / fade_total
        fade_in = int(math.floor(fade_in * ratio + 0.5))
        fade_out = int(math.floor(fade_out * ratio + 0.5))
        while fade_in + fade_out > frame_count:
            if fade_in >= fade_out and fade_in > 0:
                fade_in -= 1
            elif fade_out > 0:
                fade_out -= 1
            else:
                break
        warning = {
            "code": "FADE_SCALED",
            "message": (
                "fade-in and fade-out exceed clip duration; "
                "both were shortened proportionally"
            ),
        }

    if fade_in > 0:
        fade_in = min(fade_in, frame_count)
        positions = np.arange(fade_in, dtype=np.float64) / fade_in
        envelope[:fade_in] *= np.sin(0.5 * math.pi * positions)
    if fade_out > 0:
        fade_out = min(fade_out, frame_count)
        positions = np.arange(fade_out - 1, -1, -1, dtype=np.float64) / fade_out
        envelope[-fade_out:] *= np.sin(0.5 * math.pi * positions)

    np.clip(envelope, 0.0, 1.0, out=envelope)
    return envelope, warning


def peak_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return SILENCE_DBFS
    peak = float(np.max(np.abs(audio)))
    if not math.isfinite(peak) or peak <= 0.0:
        return SILENCE_DBFS
    return max(SILENCE_DBFS, 20.0 * math.log10(peak))


def _resolve_source(source: str, base_dir: Path) -> Path:
    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        source_path = base_dir / source_path
    return source_path.resolve()


def _new_clip_report(index: int, raw_clip: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        "index": index,
        "source": None,
        "start_sec": None,
        "duration_sec": None,
        "status": "processing",
        "statuses": [],
        "error_code": None,
        "error": None,
        "warning_codes": [],
        "warnings": [],
        "start_sample": None,
        "end_sample": None,
        "actual_duration_sec": 0.0,
    }
    if isinstance(raw_clip, dict):
        report["source"] = raw_clip.get("source")
        report["start_sec"] = raw_clip.get("start_sec")
        report["duration_sec"] = raw_clip.get("duration_sec")
    return report


def _skip_clip(
    report: dict[str, Any], code: str, message: str
) -> dict[str, Any]:
    report["status"] = "skipped"
    report["statuses"] = ["skipped"]
    report["error_code"] = code
    report["error"] = {"code": code, "message": message}
    report["warnings"] = []
    return report


def _finalize_status(report: dict[str, Any]) -> None:
    statuses: list[str] = []
    if report["status"] != "skipped":
        if report.get("_truncated"):
            statuses.append("truncated")
        else:
            statuses.append("success")
        if report["warnings"]:
            statuses.append("warning")
    else:
        statuses = ["skipped"]
    report["statuses"] = statuses
    report["status"] = statuses[0] if statuses[0] == "skipped" else (
        "truncated"
        if "truncated" in statuses
        else "warning"
        if "warning" in statuses
        else "success"
    )
    report["warning_codes"] = [
        warning["code"] for warning in report.get("warnings", [])
    ]
    report.pop("_truncated", None)


def _load_clip_audio(
    source_path: Path, project: Project
) -> tuple[np.ndarray, int]:
    if not source_path.is_file():
        raise FileNotFoundError(str(source_path))
    try:
        samples, source_rate = sf.read(
            str(source_path), dtype="float32", always_2d=True
        )
    except Exception as exc:
        raise ValueError(str(exc)) from exc
    if samples.ndim != 2 or samples.shape[1] == 0:
        raise ValueError("decoded audio has no channels")
    if samples.shape[0] == 0:
        raise ValueError("decoded audio has no frames")
    if not math.isfinite(float(source_rate)) or source_rate <= 0:
        raise ValueError("decoded audio has an invalid sample rate")
    if not np.all(np.isfinite(samples)):
        raise ValueError("decoded audio contains non-finite samples")
    resampled = linear_resample(samples, int(source_rate), project.sample_rate)
    return adapt_channels(resampled, project.channels), int(source_rate)


def _mix_clip(
    raw_clip: Any,
    index: int,
    project: Project,
    timeline: np.ndarray,
    activity: np.ndarray,
) -> dict[str, Any]:
    report = _new_clip_report(index, raw_clip)
    if not isinstance(raw_clip, dict):
        return _skip_clip(
            report, "INVALID_CLIP", "clip entry must be a JSON object"
        )

    source = raw_clip.get("source")
    if not isinstance(source, str) or not source.strip():
        return _skip_clip(
            report, "INVALID_SOURCE", "source must be a non-empty string"
        )
    try:
        start_sec = _require_number(raw_clip.get("start_sec"), "start_sec")
        duration_sec = _require_number(
            raw_clip.get("duration_sec"), "duration_sec"
        )
        gain_db = _require_number(raw_clip.get("gain_db", 0.0), "gain_db")
        fade_in_sec = _require_number(
            raw_clip.get("fade_in_sec", 0.0), "fade_in_sec"
        )
        fade_out_sec = _require_number(
            raw_clip.get("fade_out_sec", 0.0), "fade_out_sec"
        )
    except ProjectError as exc:
        return _skip_clip(report, exc.code, str(exc))

    report["start_sec"] = start_sec
    report["duration_sec"] = duration_sec
    report["gain_db"] = gain_db
    report["fade_in_sec"] = fade_in_sec
    report["fade_out_sec"] = fade_out_sec

    if start_sec < 0.0:
        return _skip_clip(
            report, "INVALID_START", "start_sec must not be negative"
        )
    if duration_sec <= 0.0:
        return _skip_clip(
            report, "INVALID_DURATION", "duration_sec must be greater than zero"
        )

    try:
        start_sample = seconds_to_samples(start_sec, project.sample_rate)
        requested_samples = seconds_to_samples(duration_sec, project.sample_rate)
    except (OverflowError, ValueError) as exc:
        return _skip_clip(
            report, "INVALID_TIME_RANGE", f"cannot convert clip times: {exc}"
        )
    if requested_samples <= 0:
        report["start_sample"] = start_sample
        return _skip_clip(
            report,
            "DURATION_BELOW_ONE_SAMPLE",
            "duration_sec rounds to zero samples",
        )
    requested_end = start_sample + requested_samples
    end_sample = min(requested_end, project.total_samples)
    report["start_sample"] = start_sample
    report["end_sample"] = end_sample
    report["requested_end_sample"] = requested_end

    if requested_end > project.total_samples:
        report["_truncated"] = True
        report["warnings"].append(
            {
                "code": "TIMELINE_TRUNCATED",
                "message": "clip exceeded total_duration_sec and was truncated",
            }
        )
    if start_sample >= project.total_samples:
        report["actual_duration_sec"] = 0.0
        report["_truncated"] = True
        _finalize_status(report)
        return report

    source_path = _resolve_source(source, project.base_dir)
    report["resolved_source"] = str(source_path)
    try:
        clip_audio, source_rate = _load_clip_audio(source_path, project)
    except FileNotFoundError:
        return _skip_clip(
            report, "SOURCE_NOT_FOUND", f"source file not found: {source_path}"
        )
    except (OSError, ValueError, RuntimeError) as exc:
        return _skip_clip(
            report, "SOURCE_DECODE_ERROR", f"cannot decode source: {exc}"
        )

    if source_rate != project.sample_rate:
        report["warnings"].append(
            {
                "code": "RESAMPLED",
                "message": (
                    f"linear resampling applied from {source_rate} Hz "
                    f"to {project.sample_rate} Hz"
                ),
            }
        )

    occupied_samples = end_sample - start_sample
    if occupied_samples <= 0:
        report["actual_duration_sec"] = 0.0
        report["_truncated"] = True
        _finalize_status(report)
        return report

    if clip_audio.shape[0] < occupied_samples:
        missing = occupied_samples - clip_audio.shape[0]
        clip_audio = np.pad(
            clip_audio, ((0, missing), (0, 0)), mode="constant"
        )
        report["warnings"].append(
            {
                "code": "SOURCE_SHORTER_THAN_CLIP",
                "message": f"{missing} missing samples were padded with silence",
            }
        )
    segment = clip_audio[:occupied_samples]

    if fade_in_sec < 0.0 or fade_out_sec < 0.0:
        report["warnings"].append(
            {
                "code": "NEGATIVE_FADE_CLAMPED",
                "message": "negative fade values were treated as zero",
            }
        )
    try:
        fade_in_samples = max(
            0, seconds_to_samples(fade_in_sec, project.sample_rate)
        )
        fade_out_samples = max(
            0, seconds_to_samples(fade_out_sec, project.sample_rate)
        )
    except (OverflowError, ValueError) as exc:
        return _skip_clip(
            report, "INVALID_FADE_RANGE", f"cannot convert fade times: {exc}"
        )
    envelope, fade_warning = build_envelope(
        occupied_samples, fade_in_samples, fade_out_samples
    )
    if fade_warning:
        report["warnings"].append(fade_warning)

    try:
        gain = float(10.0 ** (gain_db / 20.0))
    except OverflowError:
        gain = math.inf
    if not math.isfinite(gain) or gain < 0.0:
        return _skip_clip(
            report, "INVALID_GAIN", "gain_db produces a non-finite gain"
        )
    processed = segment.astype(np.float64, copy=False) * envelope[:, None] * gain
    timeline[start_sample:end_sample] += processed
    activity[start_sample:end_sample] += 1
    report["actual_duration_sec"] = round(
        occupied_samples / project.sample_rate, 3
    )
    _finalize_status(report)
    return report


def _apply_soft_limiter(
    audio: np.ndarray, activity: np.ndarray, logs: list[str]
) -> np.ndarray:
    limited = audio.copy()
    overlap = activity >= 2
    if np.any(overlap):
        values = limited[overlap]
        limited[overlap] = SOFT_LIMIT_THRESHOLD * np.tanh(
            values / SOFT_LIMIT_THRESHOLD
        )
        logs.append(
            "Applied tanh soft limiting at overlapping samples "
            f"(threshold {SOFT_LIMIT_THRESHOLD:.2f} full scale)."
        )
    else:
        logs.append("No overlapping samples; soft limiter was not engaged.")
    return limited


def _normalize_if_needed(
    audio: np.ndarray, logs: list[str]
) -> tuple[np.ndarray, float]:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1.0:
        logs.append("Final peak is within 0 dBFS; normalization not applied.")
        return audio, 0.0
    target = 10.0 ** (NORMALIZE_TARGET_DBFS / 20.0)
    linear_gain = target / peak
    gain_db = 20.0 * math.log10(linear_gain)
    logs.append(
        f"Final peak exceeded 0 dBFS; normalized to "
        f"{NORMALIZE_TARGET_DBFS:.1f} dBFS ({gain_db:.3f} dB)."
    )
    return audio * linear_gain, gain_db


def mix_project(project: Project) -> MixResult:
    timeline = np.zeros(
        (project.total_samples, project.channels), dtype=np.float64
    )
    activity = np.zeros(project.total_samples, dtype=np.int32)
    clip_reports: list[dict[str, Any]] = []
    logs: list[str] = [
        f"Loaded project with {len(project.clips)} clip(s), "
        f"{project.sample_rate} Hz, {project.channels} channel(s)."
    ]

    for index, raw_clip in enumerate(project.clips):
        report = _mix_clip(
            raw_clip, index, project, timeline, activity
        )
        clip_reports.append(report)
        if report["status"] == "skipped":
            logs.append(
                f"Clip {index} skipped: {report.get('error_code', 'UNKNOWN_ERROR')}."
            )
        elif report["status"] == "truncated":
            logs.append(f"Clip {index} truncated at the timeline boundary.")
        else:
            logs.append(
                f"Clip {index} mixed from sample {report['start_sample']} "
                f"to {report['end_sample']}."
            )
        if report["warnings"] and report["status"] != "skipped":
            for warning in report["warnings"]:
                logs.append(f"Clip {index} warning {warning['code']}: {warning['message']}")

    audio = _apply_soft_limiter(timeline, activity, logs)
    audio, normalization_gain = _normalize_if_needed(audio, logs)
    measured_peak = peak_dbfs(audio)
    logs.append(f"Rendered {project.total_samples} samples; peak {measured_peak:.2f} dBFS.")
    return MixResult(
        audio=audio.astype(np.float32),
        sample_rate=project.sample_rate,
        channels=project.channels,
        total_samples=project.total_samples,
        peak_dbfs=measured_peak,
        normalization_gain_db=normalization_gain,
        clip_reports=clip_reports,
        logs=logs,
    )


def _command_summary(command: Sequence[str]) -> str:
    def quote(part: str) -> str:
        if not part or any(character.isspace() for character in part):
            return json.dumps(part, ensure_ascii=False)
        return part

    return " ".join(quote(str(part)) for part in command)


def encode_with_ffmpeg(
    result: MixResult,
    output_path: Path,
    output_spec: OutputSpec,
    ffmpeg_path: str,
) -> dict[str, Any]:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with tempfile.TemporaryDirectory(prefix="audio_mixer_") as temp_dir:
        temp_root = Path(temp_dir)
        source_wav = temp_root / "timeline.wav"
        descriptor, encoded_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}.partial-",
            suffix=output_spec.extension,
            dir=str(output_path.parent),
        )
        os.close(descriptor)
        encoded_path = Path(encoded_name)
        sf.write(
            str(source_wav),
            result.audio,
            result.sample_rate,
            subtype="FLOAT",
            format="WAV",
        )
        command = [
            ffmpeg_path,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source_wav),
            "-map_metadata",
            "-1",
            "-ar",
            str(result.sample_rate),
            "-ac",
            str(result.channels),
            "-c:a",
            output_spec.codec,
        ]
        if output_spec.container == "flac":
            command.extend(["-sample_fmt", "s32"])
        command.append(str(encoded_path))

        try:
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
            except OSError as exc:
                raise EncodedOutputError(
                    f"cannot start FFmpeg: {exc}",
                    command=command,
                    stderr=str(exc),
                ) from exc

            summary = _command_summary(command)
            if completed.returncode != 0:
                raise EncodedOutputError(
                    "FFmpeg encoding failed",
                    command=command,
                    returncode=completed.returncode,
                    stderr=completed.stderr,
                )
            if not encoded_path.is_file() or encoded_path.stat().st_size == 0:
                raise EncodedOutputError(
                    "FFmpeg reported success but produced no output",
                    command=command,
                    returncode=completed.returncode,
                    stderr=completed.stderr,
                )
            os.replace(encoded_path, output_path)
        except Exception:
            if output_path.exists():
                output_path.unlink()
            if encoded_path.exists():
                encoded_path.unlink()
            raise
        finally:
            if encoded_path.exists():
                encoded_path.unlink()

    return {
        "executable": ffmpeg_path,
        "command": command,
        "command_summary": summary,
        "returncode": 0,
        "stderr": completed.stderr.strip(),
    }


def _read_actual_output(
    output_path: Path,
) -> tuple[float, float, int, int]:
    samples, sample_rate = sf.read(
        str(output_path), dtype="float32", always_2d=True
    )
    duration = samples.shape[0] / sample_rate if sample_rate > 0 else 0.0
    return duration, peak_dbfs(samples), int(samples.shape[0]), int(sample_rate)


def _report_path(output_path: Path) -> Path:
    return output_path.with_suffix(".report.json")


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _base_report(
    project: Project,
    output_path: Path,
    result: MixResult,
    encoding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "success",
        "output_file": output_path.name,
        "output_path": str(output_path),
        "report_file": _report_path(output_path).name,
        "output_format": project.output.label,
        "output_container": project.output.container,
        "output_bit_depth": project.output.bit_depth,
        "sample_rate": result.sample_rate,
        "channels": result.channels,
        "sample_count": result.total_samples,
        "duration_sec": round(
            result.total_samples / result.sample_rate, 3
        ),
        "actual_duration_sec": round(
            result.total_samples / result.sample_rate, 3
        ),
        "peak_dbfs": round(result.peak_dbfs, 3),
        "normalization_gain_db": round(result.normalization_gain_db, 6),
        "normalization_gain_linear": round(
            10.0 ** (result.normalization_gain_db / 20.0), 9
        ),
        "soft_limit_threshold": SOFT_LIMIT_THRESHOLD,
        "clips": result.clip_reports,
        "ffmpeg": encoding or {},
        "ffmpeg_command_summary": (encoding or {}).get("command_summary", ""),
        "logs": result.logs,
        "error": None,
    }


def render_project(
    project: Project,
    output_dir: str | os.PathLike[str],
    *,
    ffmpeg_path: str | None = None,
    mix_result: MixResult | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    output_directory = Path(output_dir).expanduser().resolve()
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AudioMixerError(
            f"cannot create output directory: {exc}", code="OUTPUT_IO_ERROR"
        ) from exc

    executable = ffmpeg_path or os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if not executable:
        raise AudioMixerError(
            "FFmpeg executable not found in PATH", code="FFMPEG_NOT_FOUND"
        )
    if not Path(executable).is_file():
        located = shutil.which(executable)
        if located is None:
            raise AudioMixerError(
                f"FFmpeg executable not found: {executable}",
                code="FFMPEG_NOT_FOUND",
            )
        executable = located

    result = mix_result or mix_project(project)
    output_path = output_directory / f"{project.output_stem}{project.output.extension}"
    report_path = _report_path(output_path)
    encoding: dict[str, Any] | None = None
    try:
        encoding = encode_with_ffmpeg(
            result, output_path, project.output, str(executable)
        )
        actual_duration, actual_peak, actual_count, actual_rate = (
            _read_actual_output(output_path)
        )
    except EncodedOutputError as exc:
        if output_path.exists():
            output_path.unlink()
        report = _base_report(project, output_path, result, encoding)
        report.update(
            {
                "status": "failed",
                "sample_count": 0,
                "duration_sec": 0.0,
                "actual_duration_sec": 0.0,
                "peak_dbfs": SILENCE_DBFS,
                "ffmpeg": {
                    "executable": str(executable),
                    "command": exc.command,
                    "command_summary": _command_summary(exc.command),
                    "returncode": exc.returncode,
                    "stderr": exc.stderr.strip(),
                },
                "ffmpeg_command_summary": _command_summary(exc.command),
                "error": {
                    "code": exc.code,
                    "message": str(exc),
                    "ffmpeg_returncode": exc.returncode,
                    "ffmpeg_stderr": exc.stderr.strip(),
                },
            }
        )
        write_report(report_path, report)
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        if output_path.exists():
            output_path.unlink()
        report = _base_report(project, output_path, result, encoding)
        report.update(
            {
                "status": "failed",
                "sample_count": 0,
                "duration_sec": 0.0,
                "actual_duration_sec": 0.0,
                "peak_dbfs": SILENCE_DBFS,
                "error": {
                    "code": "OUTPUT_DECODE_ERROR",
                    "message": f"cannot inspect encoded output: {exc}",
                },
            }
        )
        write_report(report_path, report)
        raise AudioMixerError(
            f"cannot inspect encoded output: {exc}", code="OUTPUT_DECODE_ERROR"
        ) from exc

    report = _base_report(project, output_path, result, encoding)
    report.update(
        {
            "sample_count": actual_count,
            "sample_rate": actual_rate,
            "duration_sec": round(actual_duration, 3),
            "actual_duration_sec": round(actual_duration, 3),
            "peak_dbfs": round(actual_peak, 3),
        }
    )
    report["logs"].append(
        f"Encoded output inspected: {actual_count} samples, "
        f"{actual_duration:.3f} seconds, {actual_peak:.2f} dBFS."
    )
    write_report(report_path, report)
    return output_path, report_path, report

