"""End-to-end render orchestration and deterministic output serialization."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .chapters import ChapterResult, align_chapters
from .ducking import DuckingResult, apply_ducking
from .errors import BusinessError
from .loudness import NormalizationResult, normalize
from .manifest import Manifest, parse_manifest
from .timeline import FadeInfo, mix_tracks, render_timeline
from .wavio import wav_bytes


OUTPUT_WAV = "episode.wav"
OUTPUT_CHAPTERS = "chapters.json"
OUTPUT_REPORT = "report.json"


@dataclass
class RenderResult:
    manifest: Manifest
    track_arrays: dict[str, np.ndarray]
    mix: np.ndarray
    fades: list[FadeInfo]
    ducking: DuckingResult
    chapters: ChapterResult
    normalization: NormalizationResult
    wav: bytes
    chapters_json: bytes
    report_json: bytes

    @property
    def report(self) -> dict[str, Any]:
        return json.loads(self.report_json.decode("utf-8"))


def _stable_float(value: float, digits: int = 6) -> float | None:
    if not math.isfinite(value):
        return None
    rounded = round(float(value), digits)
    return 0.0 if rounded == 0.0 else rounded


def _peak(samples: np.ndarray) -> float:
    return float(np.max(np.abs(samples))) if samples.size else 0.0


def _peak_dbfs(samples: np.ndarray) -> float | None:
    peak = _peak(samples)
    return None if peak <= 0.0 else 20.0 * math.log10(peak)


def _json_bytes(value: Any) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def render_manifest(manifest_path: Path) -> RenderResult:
    """Validate and render without touching the output directory."""

    manifest = parse_manifest(manifest_path)
    timeline = render_timeline(manifest)
    ducking = apply_ducking(
        timeline.tracks,
        manifest.ducking,
        manifest.sample_rate,
    )
    mixture = mix_tracks(timeline.tracks)
    chapters = align_chapters(manifest, manifest.chapters, timeline.fades)
    normalization = normalize(
        mixture,
        manifest.sample_rate,
        manifest.loudness.target_lufs,
        manifest.loudness.max_true_peak_dbtp,
    )

    wav_payload = wav_bytes(
        normalization.samples,
        manifest.sample_rate,
        bit_depth=16,
    )
    chapter_payload = _build_chapter_json(manifest, chapters)
    report_payload = _build_report(
        manifest,
        timeline.tracks,
        timeline.fades,
        ducking,
        chapters,
        normalization,
    )
    return RenderResult(
        manifest=manifest,
        track_arrays=timeline.tracks,
        mix=mixture,
        fades=timeline.fades,
        ducking=ducking,
        chapters=chapters,
        normalization=normalization,
        wav=wav_payload,
        chapters_json=_json_bytes(chapter_payload),
        report_json=_json_bytes(report_payload),
    )


def _build_chapter_json(
    manifest: Manifest,
    chapters: ChapterResult,
) -> dict[str, Any]:
    spans = [
        {
            "index": span.index,
            "title": span.title,
            "start_sample": span.start_sample,
            "end_sample": span.end_sample,
            "duration_samples": span.duration_samples,
            "start_seconds": _stable_float(span.start_sample / manifest.sample_rate),
            "duration_seconds": _stable_float(span.duration_samples / manifest.sample_rate),
        }
        for span in chapters.spans
    ]
    return {
        "schema_version": 1,
        "sample_rate": manifest.sample_rate,
        "total_samples": manifest.total_samples,
        "chapter_count": len(spans),
        "chapters": spans,
        "alignment_adjustment_count": len(chapters.adjustments),
        "alignment_adjustments": chapters.adjustments,
        "merged_boundary_count": chapters.merged_count,
    }


def _build_report(
    manifest: Manifest,
    tracks: dict[str, np.ndarray],
    fades: list[FadeInfo],
    ducking: DuckingResult,
    chapters: ChapterResult,
    normalization: NormalizationResult,
) -> dict[str, Any]:
    final_scale = 10.0 ** (normalization.gain_db / 20.0)
    clip_counts = {
        track_name: sum(clip.track == track_name for clip in manifest.clips)
        for track_name in manifest.tracks
    }
    track_details = {
        name: {
            "clip_count": clip_counts[name],
            "peak_dbfs_before_normalization": _stable_float(_peak_dbfs(tracks[name]))
            if _peak_dbfs(tracks[name]) is not None
            else None,
            "peak_dbfs_after_normalization": _stable_float(
                (_peak_dbfs(tracks[name]) or 0.0) + normalization.gain_db
            )
            if _peak(tracks[name]) > 0.0
            else None,
        }
        for name in sorted(manifest.tracks)
    }
    track_peaks = {
        name: track_details[name]["peak_dbfs_before_normalization"]
        for name in track_details
    }
    track_peaks_final = {
        name: track_details[name]["peak_dbfs_after_normalization"]
        for name in track_details
    }
    fade_records = [
        {
            "id": fade.edit_id,
            "track": fade.track,
            "outgoing_clip": fade.outgoing_clip,
            "incoming_clip": fade.incoming_clip,
            "at_sample": fade.at_sample,
            "fade_start_sample": fade.fade_start,
            "fade_end_sample": fade.fade_end,
            "requested_samples": fade.requested_samples,
            "effective_samples": fade.effective_samples,
            "truncated": fade.truncated,
        }
        for fade in fades
    ]
    truncated_count = sum(fade.truncated for fade in fades)
    final_peak = normalization.final_true_peak_dbtp
    return {
        "schema_version": 1,
        "manifest_version": 1,
        "sample_rate": manifest.sample_rate,
        "channels": manifest.channels,
        "output_bit_depth": 16,
        "total_samples": manifest.total_samples,
        "duration_seconds": _stable_float(manifest.total_samples / manifest.sample_rate),
        "track_peaks": track_peaks,
        "track_peaks_after_normalization": track_peaks_final,
        "tracks": track_details,
        "crossfades": {
            "count": len(fades),
            "truncated_count": truncated_count,
            "transitions": fade_records,
        },
        "ducking": {
            "enabled": manifest.ducking.enabled,
            "trigger_track": manifest.ducking.trigger_track,
            "target_track": manifest.ducking.target_track,
            "threshold_dbfs": _stable_float(manifest.ducking.threshold_dbfs),
            "energy_window_ms": _stable_float(manifest.ducking.energy_window_ms),
            "energy_window_samples": ducking.energy_window_samples,
            "depth_db": _stable_float(manifest.ducking.depth_db),
            "attack_ms": _stable_float(manifest.ducking.attack_ms),
            "attack_samples": ducking.attack_samples,
            "release_ms": _stable_float(manifest.ducking.release_ms),
            "release_samples": ducking.release_samples,
            "trigger_count": ducking.trigger_count,
            "ducked_samples": ducking.ducked_samples,
            "max_reduction_db": _stable_float(ducking.max_reduction_db),
        },
        "chapters": {
            "count": len(chapters.spans),
            "alignment_adjustment_count": len(chapters.adjustments),
            "merged_boundary_count": chapters.merged_count,
        },
        "loudness": {
            "target_lufs": _stable_float(manifest.loudness.target_lufs),
            "max_true_peak_dbtp": _stable_float(
                manifest.loudness.max_true_peak_dbtp
            ),
            "input_integrated_lufs": _stable_float(normalization.input_lufs),
            "input_true_peak_dbtp": _stable_float(
                normalization.input_true_peak_dbtp
            ),
            "gain_db": _stable_float(normalization.gain_db),
            "peak_limited": normalization.peak_limited,
            "integrated_lufs": _stable_float(normalization.final_lufs),
            "true_peak_dbtp": _stable_float(final_peak),
        },
        "ducking_trigger_count": ducking.trigger_count,
        "chapter_alignment_adjustment_count": len(chapters.adjustments),
        "final_integrated_lufs": _stable_float(normalization.final_lufs),
        "final_true_peak_dbtp": _stable_float(final_peak),
        "output": {
            "wav": OUTPUT_WAV,
            "chapters": OUTPUT_CHAPTERS,
            "report": OUTPUT_REPORT,
            "sample_rate": manifest.sample_rate,
            "channels": manifest.channels,
            "bits_per_sample": 16,
            "data_bytes": manifest.total_samples * manifest.channels * 2,
        },
    }


def _atomic_write_all(output_dir: Path, files: dict[str, bytes]) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BusinessError(
            "E_PROCESSING", f"cannot create output directory {output_dir}: {exc}"
        ) from exc
    if not output_dir.is_dir():
        raise BusinessError("E_PROCESSING", f"output path is not a directory: {output_dir}")

    temporary: list[tuple[Path, Path]] = []
    try:
        for name in sorted(files):
            handle = tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{name}.",
                suffix=".tmp",
                dir=output_dir,
                delete=False,
            )
            temp_path = Path(handle.name)
            with handle:
                handle.write(files[name])
                handle.flush()
                os.fsync(handle.fileno())
            temporary.append((temp_path, output_dir / name))
        for temp_path, final_path in temporary:
            os.replace(temp_path, final_path)
    except OSError as exc:
        for temp_path, _ in temporary:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise BusinessError("E_PROCESSING", f"cannot write outputs: {exc}") from exc


def process_manifest(manifest_path: Path, output_dir: Path) -> RenderResult:
    """Render all outputs, then publish the three files atomically per file."""

    result = render_manifest(manifest_path)
    _atomic_write_all(
        output_dir,
        {
            OUTPUT_WAV: result.wav,
            OUTPUT_CHAPTERS: result.chapters_json,
            OUTPUT_REPORT: result.report_json,
        },
    )
    return result

