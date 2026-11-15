from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

from . import SCHEMA_VERSION
from .dsp import (
    DEFAULT_TARGET_LUFS,
    FADE_SECONDS,
    GAIN_LIMIT_DB,
    SAMPLE_RATE,
    align_to_reference,
    db_to_linear,
    detect_clipped,
    detect_silent,
    estimate_offset_samples,
    linear_to_db,
    make_gain_envelope,
    peak_dbfs,
    scene_loudness_values,
    soft_limit,
)
from .io_utils import read_json, write_json

WARNING_ORDER = (
    "SILENT_TRACK",
    "CLIPPED_INPUT",
    "GAIN_CLAMPED",
    "LIMITER_ACTIVE",
)


class PipelineError(RuntimeError):
    pass


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def _read_audio(path: Path) -> Tuple[np.ndarray, int]:
    try:
        audio, sample_rate = sf.read(
            str(path), dtype="float64", always_2d=True
        )
    except Exception as exc:
        raise PipelineError(f"cannot read audio {path}: {exc}") from exc
    if sample_rate != SAMPLE_RATE:
        raise PipelineError(
            f"{path.name}: expected {SAMPLE_RATE} Hz, got {sample_rate} Hz"
        )
    if audio.shape[0] == 0:
        raise PipelineError(f"{path.name}: empty audio")
    return np.asarray(audio, dtype=np.float64), int(sample_rate)


def _to_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    if audio.shape[1] == 2:
        return audio
    mixed = np.mean(audio, axis=1, keepdims=True)
    return np.repeat(mixed, 2, axis=1)


def _validate_timeline(timeline: Dict[str, object]) -> None:
    if int(timeline.get("sample_rate", 0)) != SAMPLE_RATE:
        raise PipelineError(f"timeline sample_rate must be {SAMPLE_RATE}")
    tracks = timeline.get("tracks")
    scenes = timeline.get("scenes")
    if not isinstance(tracks, list) or not tracks:
        raise PipelineError("timeline tracks must be a non-empty list")
    if not isinstance(scenes, list) or not scenes:
        raise PipelineError("timeline scenes must be a non-empty list")
    ids = set()
    for track in tracks:
        if not isinstance(track, dict) or "id" not in track or "file" not in track:
            raise PipelineError("each track requires id and file")
        if track["id"] in ids:
            raise PipelineError(f"duplicate track id: {track['id']}")
        ids.add(track["id"])

    previous_end = -1
    for scene in scenes:
        if not isinstance(scene, dict):
            raise PipelineError("scenes must be objects")
        start = int(scene["start_sample"])
        end = int(scene["end_sample"])
        if start < 0 or end <= start:
            raise PipelineError("scene boundaries must satisfy 0 <= start < end")
        if start < previous_end:
            raise PipelineError("scenes must be ordered and non-overlapping")
        previous_end = end


def _warn_list(codes: Sequence[str]) -> List[str]:
    present = set(codes)
    return [code for code in WARNING_ORDER if code in present] + sorted(
        present.difference(WARNING_ORDER)
    )


def mix_group(
    timeline_path: Path,
    output_dir: Path,
    retry_count: int = 0,
) -> Dict[str, object]:
    """Align, measure, gain, mix, normalize and report one input group."""
    timeline_path = Path(timeline_path)
    output_dir = Path(output_dir)
    timeline = read_json(timeline_path)
    if not isinstance(timeline, dict):
        raise PipelineError("timeline root must be an object")
    _validate_timeline(timeline)

    timeline_dir = timeline_path.parent
    track_specs = timeline["tracks"]
    scene_specs = timeline["scenes"]
    target_lufs = float(timeline.get("target_lufs", DEFAULT_TARGET_LUFS))
    reference_file = str(timeline.get("reference_track", ""))
    if not reference_file:
        raise PipelineError("timeline.reference_track is required")

    reference_spec: Optional[Dict[str, object]] = None
    for spec in track_specs:
        if str(spec["file"]) == reference_file or str(spec["id"]) == reference_file:
            reference_spec = spec
            break
    if reference_spec is None:
        raise PipelineError(f"reference track not found: {reference_file}")

    reference_path = timeline_dir / str(reference_spec["file"])
    reference_audio, _ = _read_audio(reference_path)
    reference_audio = _to_stereo(reference_audio)
    if detect_silent(reference_audio):
        raise PipelineError(f"reference track is silent: {reference_file}")
    output_length = reference_audio.shape[0]

    scene_records = []
    for index, scene in enumerate(scene_specs):
        start = int(scene["start_sample"])
        end = int(scene["end_sample"])
        if end > output_length:
            raise PipelineError(
                f"scene {scene.get('id', index)} ends after reference recording"
            )
        scene_records.append(
            {
                "id": str(scene.get("id", f"scene_{index + 1:02d}")),
                "start_sample": start,
                "end_sample": end,
                "start_seconds": round(start / SAMPLE_RATE, 6),
                "end_seconds": round(end / SAMPLE_RATE, 6),
                "target_lufs": float(scene.get("target_lufs", target_lufs)),
            }
        )

    report_tracks: List[Dict[str, object]] = []
    aligned_tracks: List[np.ndarray] = []
    all_warnings: List[str] = []
    reference_id = str(reference_spec["id"])

    for spec in track_specs:
        track_id = str(spec["id"])
        track_file = str(spec["file"])
        audio, _ = _read_audio(timeline_dir / track_file)
        audio = _to_stereo(audio)
        silent = detect_silent(audio)
        clipped = detect_clipped(audio)
        warnings: List[str] = []
        if clipped:
            warnings.append("CLIPPED_INPUT")
            all_warnings.append("CLIPPED_INPUT")

        if silent:
            warnings.append("SILENT_TRACK")
            all_warnings.append("SILENT_TRACK")
            report_track: Dict[str, object] = {
                "id": track_id,
                "file": track_file,
                "status": "skipped",
                "is_reference": track_id == reference_id,
                "offset_samples": None,
                "offset_estimate_samples": None,
                "offset_error_samples": None,
                "offset_error_within_5_samples": None,
                "silent": True,
                "clipped": clipped,
                "clipped_input": clipped,
                "business_code": "SILENT_TRACK",
                "warnings": _warn_list(warnings),
            }
            report_tracks.append(report_track)
            continue

        if track_id == reference_id:
            estimate = 0
        else:
            estimate = estimate_offset_samples(reference_audio, audio)

        expected = spec.get("offset_samples")
        error = None if expected is None else abs(int(estimate) - int(expected))
        report_track = {
            "id": track_id,
            "file": track_file,
            "status": "ok",
            "is_reference": track_id == reference_id,
            "offset_samples": int(estimate),
            "offset_estimate_samples": int(estimate),
            "offset_error_samples": None if error is None else int(error),
            "offset_error_within_5_samples": None if error is None else bool(error <= 5),
            "silent": False,
            "clipped": clipped,
            "clipped_input": clipped,
            "business_code": "CLIPPED_INPUT" if clipped else None,
            "warnings": _warn_list(warnings),
        }
        report_tracks.append(report_track)
        aligned = align_to_reference(audio, estimate, output_length)
        aligned_tracks.append(aligned)

    if not aligned_tracks:
        raise PipelineError("no audible tracks remain after silent-track handling")

    aligned_mix = np.sum(aligned_tracks, axis=0)
    loudness = scene_loudness_values(
        aligned_mix, scene_records, sample_rate=SAMPLE_RATE
    )

    applied_gains_db: List[float] = []
    scene_report: List[Dict[str, object]] = []
    for scene, measured in zip(scene_records, loudness):
        target = float(scene["target_lufs"])
        required: Optional[float] = None
        applied = 0.0
        was_clamped = False
        if measured is not None and math.isfinite(measured):
            required = target - measured
            applied = float(np.clip(required, -GAIN_LIMIT_DB, GAIN_LIMIT_DB))
            was_clamped = bool(abs(required) > GAIN_LIMIT_DB + 1e-9)
        if was_clamped:
            all_warnings.append("GAIN_CLAMPED")
        applied_gains_db.append(applied)
        scene_report.append(
            {
                "id": scene["id"],
                "start_sample": scene["start_sample"],
                "end_sample": scene["end_sample"],
                "start_seconds": scene["start_seconds"],
                "end_seconds": scene["end_seconds"],
                "target_lufs": round(target, 4),
                "loudness_lufs": _round_or_none(measured),
                "required_gain_db": _round_or_none(required),
                "applied_gain_db": round(applied, 4),
                "gain_db": round(applied, 4),
                "gain_clamped": was_clamped,
                "business_code": "GAIN_CLAMPED" if was_clamped else None,
            }
        )

    boundaries = [int(scene["start_sample"]) for scene in scene_records[1:]]
    envelope = make_gain_envelope(
        applied_gains_db,
        boundaries,
        output_length,
        sample_rate=SAMPLE_RATE,
        fade_seconds=FADE_SECONDS,
    )
    output_mix = aligned_mix * envelope[:, None]
    pre_peak = float(np.max(np.abs(output_mix))) if output_mix.size else 0.0
    limiter_active = bool(pre_peak > 1.0)
    if limiter_active:
        all_warnings.append("LIMITER_ACTIVE")
        output_mix = soft_limit(output_mix)

    normalized_peak = db_to_linear(-1.0)
    limited_peak = float(np.max(np.abs(output_mix))) if output_mix.size else 0.0
    normalization_gain = (
        normalized_peak / limited_peak if limited_peak > 0.0 else 1.0
    )
    output_mix *= normalization_gain

    output_dir.mkdir(parents=True, exist_ok=True)
    output_wav = output_dir / "mix.wav"
    fd, temporary_name = tempfile.mkstemp(
        prefix=".mix.", suffix=".wav", dir=str(output_dir)
    )
    os.close(fd)
    try:
        sf.write(
            temporary_name,
            output_mix,
            SAMPLE_RATE,
            subtype="PCM_16",
            format="WAV",
        )
        os.replace(temporary_name, str(output_wav))
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise

    quantized_peak = float(np.max(np.abs(output_mix))) if output_mix.size else 0.0
    final_peak_dbfs = linear_to_db(quantized_peak)
    report: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "timeline": timeline_path.name,
            "seed": timeline.get("seed"),
        },
        "sample_rate": SAMPLE_RATE,
        "target_lufs": round(target_lufs, 4),
        "reference_track": reference_id,
        "retry_count": int(retry_count),
        "tracks": report_tracks,
        "scenes": scene_report,
        "warnings": _warn_list(all_warnings),
        "business_codes": _warn_list(all_warnings),
        "limiter_active": limiter_active,
        "pre_normalization_peak_dbfs": round(linear_to_db(pre_peak), 4),
        "normalization_gain_db": round(linear_to_db(normalization_gain), 4),
        "final_output_peak_dbfs": round(final_peak_dbfs, 4),
        "final_peak_dbfs": round(final_peak_dbfs, 4),
        "output": {
            "file": output_wav.name,
            "sample_rate": SAMPLE_RATE,
            "channels": 2,
            "sample_format": "PCM_16",
        },
    }
    write_json(output_dir / "report.json", report)
    return report
