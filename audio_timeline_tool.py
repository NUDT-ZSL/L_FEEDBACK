#!/usr/bin/env python3
"""Offline audio timeline alignment and loop-preparation tool.

The tool reads a JSON timeline and deterministic local WAV inputs, then writes
48 kHz mono 16-bit PCM outputs, an aligned timeline, and a JSON processing
report.  NumPy is the only non-standard-library dependency.
"""

import argparse
import json
import math
import os
import struct
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


TARGET_SAMPLE_RATE = 48000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2
EVENT_SEARCH_SECONDS = 0.010
LOOP_FADE_SECONDS = 0.020
MIN_LOOP_SECONDS = 0.040
TARGET_RMS_DBFS = -20.0
MAX_PEAK_DBFS = -1.0
MAX_LOUDNESS_ATTENUATION_DB = 20.0
MIN_TIME_SCALE = 0.1
MAX_TIME_SCALE = 10.0
MAX_PEAK_PCM = int(math.floor(32767.0 * 10.0 ** (MAX_PEAK_DBFS / 20.0)))
MAX_PEAK_LINEAR = MAX_PEAK_PCM / 32768.0

TIMELINE_OUTPUT_NAME = "processed_timeline.json"
REPORT_OUTPUT_NAME = "report.json"
TOOL_VERSION = "1.0.0"


class TimelineError(ValueError):
    """The timeline is malformed or cannot be used safely."""


class OutputDirectoryError(OSError):
    """The requested output directory cannot be written."""


class AudioLoadFailure(Exception):
    """A recoverable per-audio input failure."""

    def __init__(
        self,
        code: str,
        message: str,
        source_sample_rate: Optional[int] = None,
        source_channels: Optional[int] = None,
        original_duration: Optional[float] = None,
        source_sample_width: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.source_sample_rate = source_sample_rate
        self.source_channels = source_channels
        self.original_duration = original_duration
        self.source_sample_width = source_sample_width


def _is_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _event_time(event: Any) -> float:
    if _is_number(event):
        return float(event)
    if isinstance(event, dict):
        for key in ("time", "timestamp"):
            if key in event:
                if not _is_number(event[key]):
                    raise TimelineError(
                        "event field {!r} must be a finite number".format(key)
                    )
                return float(event[key])
    raise TimelineError(
        "each event must be a number or an object containing a numeric 'time'"
    )


def _loop_bounds(loop: Any) -> Tuple[float, float]:
    if isinstance(loop, (list, tuple)):
        if len(loop) != 2:
            raise TimelineError("each loop array must contain [start, end]")
        start, end = loop
    elif isinstance(loop, dict):
        if "start" not in loop or "end" not in loop:
            raise TimelineError("each loop object must contain 'start' and 'end'")
        start, end = loop["start"], loop["end"]
    else:
        raise TimelineError(
            "each loop must be [start, end] or an object with start/end"
        )
    if not _is_number(start) or not _is_number(end):
        raise TimelineError("loop start and end must be finite numbers")
    return float(start), float(end)


def _safe_output_id(audio_id: Any) -> str:
    if not isinstance(audio_id, str) or not audio_id:
        raise TimelineError("each audio object must have a non-empty string 'id'")
    if (
        audio_id in (".", "..")
        or "/" in audio_id
        or "\\" in audio_id
        or "\x00" in audio_id
    ):
        raise TimelineError(
            "audio id {!r} is not a safe output file base name".format(audio_id)
        )
    return audio_id


def _load_timeline(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(
                handle,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError("non-finite JSON number {!r}".format(value))
                ),
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise TimelineError("cannot read timeline JSON: {}".format(exc))

    if not isinstance(data, dict):
        raise TimelineError("timeline root must be a JSON object")
    if "audio" not in data or not isinstance(data["audio"], list):
        raise TimelineError("timeline must contain an 'audio' array")

    seen_ids = set()
    for index, audio in enumerate(data["audio"]):
        if not isinstance(audio, dict):
            raise TimelineError("audio[{}] must be an object".format(index))
        audio_id = _safe_output_id(audio.get("id"))
        if audio_id in seen_ids:
            raise TimelineError("duplicate audio id {!r}".format(audio_id))
        seen_ids.add(audio_id)

        file_name = audio.get("file")
        if not isinstance(file_name, str) or not file_name:
            raise TimelineError(
                "audio {!r} must have a non-empty string 'file'".format(audio_id)
            )
        events = audio.get("events")
        if not isinstance(events, list):
            raise TimelineError(
                "audio {!r} must have an 'events' array".format(audio_id)
            )
        for event_index, event in enumerate(events):
            try:
                _event_time(event)
            except TimelineError as exc:
                raise TimelineError(
                    "audio {!r} event[{}]: {}".format(audio_id, event_index, exc)
                )

        loops = audio.get("loops")
        if not isinstance(loops, list):
            raise TimelineError(
                "audio {!r} must have a 'loops' array".format(audio_id)
            )
        for loop_index, loop in enumerate(loops):
            try:
                _loop_bounds(loop)
            except TimelineError as exc:
                raise TimelineError(
                    "audio {!r} loop[{}]: {}".format(audio_id, loop_index, exc)
                )
    return data


def _effective_time_scale(audio: Dict[str, Any]) -> Tuple[float, List[Dict[str, str]]]:
    raw = audio.get("time_scale", 1.0)
    diagnostics: List[Dict[str, str]] = []
    if (
        _is_number(raw)
        and MIN_TIME_SCALE <= float(raw) <= MAX_TIME_SCALE
    ):
        return float(raw), diagnostics

    diagnostics.append(
        {
            "code": "INVALID_TIME_SCALE",
            "message": (
                "time_scale {!r} is not finite or not within [{}, {}]; "
                "using 1.0"
            ).format(raw, MIN_TIME_SCALE, MAX_TIME_SCALE),
        }
    )
    return 1.0, diagnostics


def _resolve_audio_path(audio_dir: Path, file_name: str) -> Path:
    source = Path(file_name)
    if source.is_absolute():
        return source
    return audio_dir / source


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64)
        return data / 32768.0
    if sample_width == 3:
        octets = np.frombuffer(raw, dtype=np.uint8)
        if octets.size % 3:
            raise ValueError("truncated 24-bit PCM frame")
        octets = octets.reshape(-1, 3).astype(np.int32)
        values = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float64) / 8388608.0
    if sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64)
        return data / 2147483648.0
    raise ValueError("unsupported PCM sample width {}".format(sample_width))


def _looks_unsupported_wave_error(message: str) -> bool:
    lowered = message.lower()
    markers = (
        "unknown format",
        "unknown extended format",
        "bad sample width",
        "bad # channels",
    )
    return any(marker in lowered for marker in markers)


def _peek_wav_metadata(path: Path) -> Dict[str, Any]:
    """Read basic RIFF/WAVE metadata without decoding audio frames."""
    metadata: Dict[str, Any] = {
        "format_tag": None,
        "sample_rate": None,
        "channels": None,
        "sample_width": None,
        "original_duration": None,
    }
    try:
        with path.open("rb") as handle:
            header = handle.read(12)
            if (
                len(header) != 12
                or header[0:4] != b"RIFF"
                or header[8:12] != b"WAVE"
            ):
                return metadata
            fmt_chunk = None
            data_size = None
            while True:
                chunk_header = handle.read(8)
                if len(chunk_header) < 8:
                    break
                chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
                chunk_start = handle.tell()
                if chunk_id == b"fmt ":
                    fmt_chunk = handle.read(min(chunk_size, 64))
                elif chunk_id == b"data":
                    data_size = int(chunk_size)
                next_chunk = chunk_start + chunk_size + (chunk_size & 1)
                handle.seek(next_chunk)
            if fmt_chunk is None or len(fmt_chunk) < 16:
                return metadata
            (
                format_tag,
                channels,
                sample_rate,
                byte_rate,
                _block_align,
                bits_per_sample,
            ) = struct.unpack_from("<HHIIHH", fmt_chunk, 0)
            if format_tag == 0xFFFE and len(fmt_chunk) >= 26:
                format_tag = struct.unpack_from("<H", fmt_chunk, 24)[0]
            metadata.update(
                {
                    "format_tag": int(format_tag),
                    "sample_rate": int(sample_rate),
                    "channels": int(channels),
                    "sample_width": int(bits_per_sample + 7) // 8,
                }
            )
            if data_size is not None and byte_rate > 0:
                metadata["original_duration"] = float(data_size) / float(byte_rate)
            elif data_size is not None and sample_rate > 0 and channels > 0:
                frame_size = max(1, channels * ((bits_per_sample + 7) // 8))
                metadata["original_duration"] = (
                    float(data_size) / float(frame_size) / float(sample_rate)
                )
    except (OSError, ValueError, struct.error):
        return metadata
    return metadata


def _load_wav(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not path.is_file():
        raise AudioLoadFailure(
            "AUDIO_READ_ERROR", "audio file does not exist: {}".format(path)
        )

    peeked = _peek_wav_metadata(path)
    try:
        wave_reader = wave.open(str(path), "rb")
    except Exception as exc:
        message = "cannot decode WAV {}: {}".format(path, exc)
        unsupported = _looks_unsupported_wave_error(str(exc))
        tag = peeked["format_tag"]
        if tag not in (None, 1):
            unsupported = True
        code = "UNSUPPORTED_FORMAT" if unsupported else "AUDIO_READ_ERROR"
        raise AudioLoadFailure(
            code,
            message,
            peeked["sample_rate"],
            peeked["channels"],
            peeked["original_duration"],
            peeked["sample_width"],
        )

    try:
        channels = int(wave_reader.getnchannels())
        sample_rate = int(wave_reader.getframerate())
        sample_width = int(wave_reader.getsampwidth())
        frame_count = int(wave_reader.getnframes())
        compression = wave_reader.getcomptype()
        duration = (
            float(frame_count) / float(sample_rate)
            if sample_rate > 0 and frame_count >= 0
            else None
        )

        if compression != "NONE":
            raise AudioLoadFailure(
                "UNSUPPORTED_FORMAT",
                "compressed WAV is not supported: {}".format(compression),
                sample_rate,
                channels,
                duration,
                sample_width,
            )
        if channels != 1:
            raise AudioLoadFailure(
                "UNSUPPORTED_FORMAT",
                "only mono WAV is supported (found {} channels)".format(channels),
                sample_rate,
                channels,
                duration,
                sample_width,
            )
        if sample_width not in (1, 2, 3, 4):
            raise AudioLoadFailure(
                "UNSUPPORTED_FORMAT",
                "unsupported PCM sample width: {} bytes".format(sample_width),
                sample_rate,
                channels,
                duration,
                sample_width,
            )
        if sample_rate <= 0:
            raise AudioLoadFailure(
                "AUDIO_READ_ERROR",
                "invalid WAV sample rate: {}".format(sample_rate),
                sample_rate,
                channels,
                duration,
                sample_width,
            )

        raw = wave_reader.readframes(frame_count)
        expected_bytes = frame_count * sample_width * channels
        if len(raw) != expected_bytes:
            raise AudioLoadFailure(
                "AUDIO_READ_ERROR",
                "truncated WAV data: expected {} bytes, got {}".format(
                    expected_bytes, len(raw)
                ),
                sample_rate,
                channels,
                duration,
                sample_width,
            )
        samples = _decode_pcm(raw, sample_width)
        info = {
            "source_sample_rate": sample_rate,
            "source_channels": channels,
            "source_sample_width": sample_width,
            "original_duration": duration,
        }
        return samples, info
    except AudioLoadFailure:
        raise
    except Exception as exc:
        raise AudioLoadFailure(
            "AUDIO_READ_ERROR", "cannot decode WAV {}: {}".format(path, exc)
        )
    finally:
        wave_reader.close()


def _anti_alias_lowpass(
    samples: np.ndarray, source_rate: int, cutoff_hz: float
) -> np.ndarray:
    if samples.size == 0 or cutoff_hz <= 0 or cutoff_hz >= source_rate / 2.0:
        return samples

    taps = int(round(source_rate / cutoff_hz * 8.0))
    taps = max(31, taps | 1)
    taps = min(taps, 1023)
    half = taps // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    normalized_cutoff = cutoff_hz / float(source_rate)
    kernel = (
        2.0
        * normalized_cutoff
        * np.sinc(2.0 * normalized_cutoff * offsets)
        * np.hamming(taps)
    )
    kernel /= np.sum(kernel)
    padded = np.pad(samples, (half, half), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _resample_linear(
    samples: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    if source_rate == target_rate or samples.size == 0:
        return samples.copy()

    cutoff = 0.45 * min(float(source_rate), float(target_rate))
    filtered = _anti_alias_lowpass(samples, source_rate, cutoff)
    target_length = max(
        1, int(round(samples.size * float(target_rate) / float(source_rate)))
    )
    source_positions = np.arange(samples.size, dtype=np.float64)
    target_positions = (
        np.arange(target_length, dtype=np.float64)
        * float(source_rate)
        / float(target_rate)
    )
    return np.interp(target_positions, source_positions, filtered)


def _time_stretch(samples: np.ndarray, scale: float) -> np.ndarray:
    if samples.size == 0 or math.isclose(scale, 1.0):
        return samples.copy()
    target_length = max(1, int(round(samples.size * scale)))
    source_positions = np.arange(samples.size, dtype=np.float64)
    target_positions = np.arange(target_length, dtype=np.float64) / scale
    return np.interp(target_positions, source_positions, samples)


def _normalize_loudness(
    samples: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, str]]]:
    diagnostics: List[Dict[str, str]] = []
    if samples.size == 0:
        return samples.copy(), {
            "gain_db": 0.0,
            "rms_before": 0.0,
            "rms_after": 0.0,
            "peak_before": 0.0,
            "peak_after": 0.0,
        }, diagnostics

    rms_before = float(np.sqrt(np.mean(np.square(samples))))
    peak_before = float(np.max(np.abs(samples)))
    target_rms = 10.0 ** (TARGET_RMS_DBFS / 20.0)

    if rms_before > 1e-12:
        gain = target_rms / rms_before
    else:
        gain = 1.0

    requested_gain = gain
    limited = False
    if gain > 0.0:
        attenuation_db = -20.0 * math.log10(gain)
        if attenuation_db > MAX_LOUDNESS_ATTENUATION_DB + 1e-9:
            gain = 10.0 ** (-MAX_LOUDNESS_ATTENUATION_DB / 20.0)
            limited = True

    if peak_before > 1e-12:
        gain = min(gain, MAX_PEAK_LINEAR / peak_before)

    adjusted = np.clip(samples * gain, -1.0, 1.0)
    rms_after = float(np.sqrt(np.mean(np.square(adjusted))))
    peak_after = float(np.max(np.abs(adjusted))) if adjusted.size else 0.0
    gain_db = 20.0 * math.log10(gain) if gain > 0.0 else -math.inf

    if limited:
        diagnostics.append(
            {
                "code": "LOUDNESS_LIMITED",
                "message": (
                    "required attenuation {:.3f} dB exceeds {} dB; "
                    "limited attenuation to {} dB"
                ).format(
                    -20.0 * math.log10(requested_gain),
                    MAX_LOUDNESS_ATTENUATION_DB,
                    MAX_LOUDNESS_ATTENUATION_DB,
                ),
            }
        )

    return adjusted, {
        "gain_db": gain_db,
        "rms_before": rms_before,
        "rms_after": rms_after,
        "peak_before": peak_before,
        "peak_after": peak_after,
        "target_rms_dbfs": TARGET_RMS_DBFS,
    }, diagnostics


def _align_event(
    samples: np.ndarray,
    event_time: float,
    sample_rate: int,
) -> Tuple[float, int, str]:
    duration = samples.size / float(sample_rate)
    if event_time < 0.0 or event_time > duration:
        return event_time, -1, "EVENT_OUT_OF_RANGE"
    if samples.size == 0:
        return event_time, -1, "EVENT_OUT_OF_RANGE"

    left = max(
        0, int(math.floor((event_time - EVENT_SEARCH_SECONDS) * sample_rate))
    )
    right = min(
        samples.size - 1,
        int(math.ceil((event_time + EVENT_SEARCH_SECONDS) * sample_rate)),
    )
    if right < left:
        return event_time, -1, "EVENT_OUT_OF_RANGE"

    window = samples[left : right + 1]
    energy = np.square(window)
    peak_energy = float(np.max(energy))
    candidates = np.flatnonzero(energy >= peak_energy)
    target_sample = event_time * sample_rate
    if candidates.size > 1:
        absolute_samples = candidates + left
        nearest = int(np.argmin(np.abs(absolute_samples - target_sample)))
        aligned_sample = int(absolute_samples[nearest])
    else:
        aligned_sample = int(candidates[0] + left)
    return aligned_sample / float(sample_rate), aligned_sample, "OK"


def _set_event_result(
    event: Any,
    aligned_time: float,
    status: str,
) -> Dict[str, Any]:
    if isinstance(event, dict):
        result = event
        original_time = _event_time(event)
    else:
        original_time = float(event)
        result = {"time": original_time}
    result["aligned_time"] = float(aligned_time)
    result["status"] = status
    return result


def _make_loop(
    samples: np.ndarray,
    start: float,
    end: float,
    sample_rate: int,
) -> Tuple[Optional[np.ndarray], str, Optional[Dict[str, str]]]:
    duration = samples.size / float(sample_rate)
    if start < 0.0 or end > duration + 1e-12 or start > end:
        return (
            None,
            "SKIPPED",
            {
                "code": "LOOP_OUT_OF_RANGE",
                "message": "loop [{}, {}] is outside 0..{} seconds".format(
                    start, end, duration
                ),
            },
        )
    if end - start < MIN_LOOP_SECONDS:
        return (
            None,
            "LOOP_TOO_SHORT",
            {
                "code": "LOOP_TOO_SHORT",
                "message": "loop duration {:.6f}s is shorter than 0.040s".format(
                    end - start
                ),
            },
        )

    start_sample = max(0, int(round(start * sample_rate)))
    end_sample = min(samples.size, int(round(end * sample_rate)))
    segment = samples[start_sample:end_sample].copy()
    if segment.size == 0 or segment.size / float(sample_rate) < MIN_LOOP_SECONDS:
        return (
            None,
            "LOOP_TOO_SHORT",
            {
                "code": "LOOP_TOO_SHORT",
                "message": "loop has too few samples after sampling",
            },
        )

    fade_samples = min(
        int(round(LOOP_FADE_SECONDS * sample_rate)), segment.size // 2
    )
    if fade_samples > 0:
        phase = np.linspace(0.0, 1.0, fade_samples, endpoint=True)
        fade_in = np.sin(0.5 * np.pi * phase) ** 2
        fade_out = np.cos(0.5 * np.pi * phase) ** 2
        segment[:fade_samples] *= fade_in
        segment[-fade_samples:] *= fade_out
    return segment, "OK", None


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(
        np.rint(samples * 32768.0), -MAX_PEAK_PCM, MAX_PEAK_PCM
    ).astype("<i2")
    try:
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(TARGET_CHANNELS)
            writer.setsampwidth(TARGET_SAMPLE_WIDTH)
            writer.setframerate(sample_rate)
            writer.writeframes(pcm.tobytes())
    except Exception as exc:
        raise OSError("cannot write WAV {}: {}".format(path, exc))


def _write_json(path: Path, payload: Any) -> None:
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception as exc:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
        raise OSError("cannot write JSON {}: {}".format(path, exc))


def _prepare_output_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputDirectoryError(
            "cannot create output directory {}: {}".format(path, exc)
        )
    if not path.is_dir():
        raise OutputDirectoryError(
            "output path is not a directory: {}".format(path)
        )
    probe = path / ".audio_timeline_write_check"
    try:
        with probe.open("wb") as handle:
            handle.write(b"ok")
        probe.unlink()
    except OSError as exc:
        raise OutputDirectoryError(
            "output directory is not writable: {}: {}".format(path, exc)
        )


def _base_report_entry(
    audio: Dict[str, Any], effective_scale: float, diagnostics: List[Dict[str, str]]
) -> Dict[str, Any]:
    return {
        "id": audio["id"],
        "file": audio["file"],
        "processing_status": "SKIPPED",
        "source_sample_rate": None,
        "source_channels": None,
        "source_sample_width": None,
        "original_duration": None,
        "resampled_duration": None,
        "processed_duration": None,
        "time_scale": effective_scale,
        "event_alignment_offsets_ms": [],
        "events": [],
        "loops": [],
        "loudness": None,
        "diagnostics": diagnostics,
        "error": None,
    }


def _mark_all_skipped(
    audio: Dict[str, Any],
    report: Dict[str, Any],
    failure: Optional[AudioLoadFailure] = None,
) -> None:
    for event_index, event in enumerate(audio["events"]):
        original_time = _event_time(event)
        audio["events"][event_index] = _set_event_result(
            event, original_time, "SKIPPED"
        )
        report["events"].append(
            {
                "event_index": event_index,
                "time": original_time,
                "aligned_time": original_time,
                "offset_ms": 0.0,
                "status": "SKIPPED",
            }
        )
        report["event_alignment_offsets_ms"].append(0.0)

    for loop_index, loop in enumerate(audio["loops"]):
        start, end = _loop_bounds(loop)
        loop_result: Dict[str, Any]
        if isinstance(loop, dict):
            loop_result = loop
        else:
            loop_result = {"start": start, "end": end}
        loop_result["loop_file"] = None
        loop_result["loop_status"] = "SKIPPED"
        audio["loops"][loop_index] = loop_result
        report["loops"].append(
            {
                "loop_index": loop_index,
                "start": start,
                "end": end,
                "duration": max(0.0, end - start),
                "loop_file": None,
                "loop_status": "SKIPPED",
            }
        )

    if failure is not None:
        report["error"] = {"code": failure.code, "message": failure.message}
        report["diagnostics"].append(
            {"code": failure.code, "message": failure.message}
        )


def _process_audio(
    audio: Dict[str, Any],
    audio_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    effective_scale, diagnostics = _effective_time_scale(audio)
    report = _base_report_entry(audio, effective_scale, diagnostics)
    source_path = _resolve_audio_path(audio_dir, audio["file"])

    try:
        source_samples, source_info = _load_wav(source_path)
    except AudioLoadFailure as failure:
        report["source_sample_rate"] = failure.source_sample_rate
        report["source_channels"] = failure.source_channels
        report["source_sample_width"] = failure.source_sample_width
        report["original_duration"] = failure.original_duration
        _mark_all_skipped(audio, report, failure)
        return report

    report["source_sample_rate"] = source_info["source_sample_rate"]
    report["source_channels"] = source_info["source_channels"]
    report["source_sample_width"] = source_info["source_sample_width"]
    report["original_duration"] = source_info["original_duration"]

    resampled = _resample_linear(
        source_samples,
        source_info["source_sample_rate"],
        TARGET_SAMPLE_RATE,
    )
    report["resampled_duration"] = resampled.size / float(TARGET_SAMPLE_RATE)
    stretched = _time_stretch(resampled, effective_scale)
    processed = stretched
    processed_duration = processed.size / float(TARGET_SAMPLE_RATE)
    report["processed_duration"] = processed_duration

    for event_index, event in enumerate(audio["events"]):
        event_time = _event_time(event)
        aligned_time, aligned_sample, status = _align_event(
            processed, event_time, TARGET_SAMPLE_RATE
        )
        if status != "OK":
            report["diagnostics"].append(
                {
                    "code": "EVENT_OUT_OF_RANGE",
                    "message": (
                        "event {} at {:.6f}s is outside 0..{:.6f}s"
                    ).format(event_index, event_time, processed_duration),
                }
            )
        audio["events"][event_index] = _set_event_result(
            event, aligned_time, status
        )
        offset_ms = (aligned_time - event_time) * 1000.0
        if abs(offset_ms) < 1e-9:
            offset_ms = 0.0
        else:
            offset_ms = round(offset_ms, 6)
        report["event_alignment_offsets_ms"].append(offset_ms)
        report["events"].append(
            {
                "event_index": event_index,
                "time": event_time,
                "aligned_time": float(aligned_time),
                "aligned_sample": aligned_sample if aligned_sample >= 0 else None,
                "offset_ms": offset_ms,
                "status": status,
            }
        )

    normalized, loudness_info, loudness_diagnostics = _normalize_loudness(
        processed
    )
    report["loudness"] = loudness_info
    report["diagnostics"].extend(loudness_diagnostics)

    processed_name = "{}_processed.wav".format(audio["id"])
    _write_wav(output_dir / processed_name, normalized, TARGET_SAMPLE_RATE)
    report["processing_status"] = "OK"

    for loop_index, loop in enumerate(audio["loops"]):
        start, end = _loop_bounds(loop)
        segment, loop_status, loop_diagnostic = _make_loop(
            normalized, start, end, TARGET_SAMPLE_RATE
        )
        if loop_diagnostic is not None:
            report["diagnostics"].append(loop_diagnostic)

        loop_name: Optional[str] = None
        if loop_status == "OK" and segment is not None:
            loop_name = "{}_loop_{}.wav".format(audio["id"], loop_index)
            _write_wav(output_dir / loop_name, segment, TARGET_SAMPLE_RATE)

        if isinstance(loop, dict):
            loop_result = loop
        else:
            loop_result = {"start": start, "end": end}
        loop_result["loop_file"] = loop_name
        loop_result["loop_status"] = loop_status
        audio["loops"][loop_index] = loop_result
        report["loops"].append(
            {
                "loop_index": loop_index,
                "start": start,
                "end": end,
                "duration": max(0.0, end - start),
                "loop_file": loop_name,
                "loop_status": loop_status,
            }
        )

    return report


def run(
    timeline_path: Path,
    audio_dir: Path,
    output_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    timeline = _load_timeline(timeline_path)
    _prepare_output_directory(output_dir)

    audio_reports: List[Dict[str, Any]] = []
    for audio in timeline["audio"]:
        audio_reports.append(_process_audio(audio, audio_dir, output_dir))

    report = {
        "tool": "audio_timeline_tool",
        "version": TOOL_VERSION,
        "sample_rate": TARGET_SAMPLE_RATE,
        "channels": TARGET_CHANNELS,
        "sample_width_bits": TARGET_SAMPLE_WIDTH * 8,
        "target_loudness_lufs": TARGET_RMS_DBFS,
        "max_peak_dbfs": MAX_PEAK_DBFS,
        "event_search_ms_each_side": EVENT_SEARCH_SECONDS * 1000.0,
        "loop_fade_ms_each_side": LOOP_FADE_SECONDS * 1000.0,
        "audio": audio_reports,
    }

    _write_json(output_dir / TIMELINE_OUTPUT_NAME, timeline)
    _write_json(output_dir / REPORT_OUTPUT_NAME, report)
    return timeline, report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Align event peaks, prepare seamless loops, time-stretch, and "
            "normalize WAV audio from an animation timeline."
        )
    )
    parser.add_argument(
        "--timeline", required=True, type=Path, help="input timeline JSON"
    )
    parser.add_argument(
        "--audio-dir", required=True, type=Path, help="directory containing WAV inputs"
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="directory for generated files"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        timeline, report = run(
            args.timeline.expanduser(),
            args.audio_dir.expanduser(),
            args.output_dir.expanduser(),
        )
    except (TimelineError, OutputDirectoryError, OSError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print("error: unexpected processing failure: {}".format(exc), file=sys.stderr)
        return 1

    processed = sum(
        1 for item in report["audio"] if item["processing_status"] == "OK"
    )
    skipped = len(report["audio"]) - processed
    print(
        "Processed {} audio file(s); skipped {}. Wrote {} and {}.".format(
            processed,
            skipped,
            args.output_dir / TIMELINE_OUTPUT_NAME,
            args.output_dir / REPORT_OUTPUT_NAME,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
