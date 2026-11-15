"""Absolute-timeline mixing and per-track contribution accounting."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from .audio import RAW_DTYPE, db_to_linear, linear_to_db, write_float_wav
from .manifest import Manifest
from .timeline import TimelineLayout, TrackLayout

MIX_BLOCK_FRAMES = 65_536


def _gain_for_block(
    layout: TrackLayout,
    absolute_start: int,
    frames: int,
) -> np.ndarray:
    base = db_to_linear(layout.spec.gain_db)
    gain = np.full(frames, base, dtype=np.float64)
    block_end = absolute_start + frames
    for interval in layout.duck_intervals:
        start = max(absolute_start, interval.start_frame)
        end = min(block_end, interval.end_frame)
        if end <= start:
            continue
        gain[start - absolute_start : end - absolute_start] *= db_to_linear(
            interval.gain_db
        )
    return gain


def mix_timeline(
    layout: TimelineLayout,
    cache_paths: dict[str, Path],
    output_float_wav: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest: Manifest = layout.manifest
    sample_rate = manifest.project.sample_rate
    total_frames = manifest.project.duration_frames
    accumulator = np.zeros((total_frames, 2), dtype=np.float64)
    track_reports: list[dict[str, Any]] = []
    contribution_energies: list[float] = []

    for track_layout in layout.tracks:
        track = track_layout.spec
        raw_path = cache_paths[track.track_id]
        cached = np.memmap(
            raw_path,
            dtype=RAW_DTYPE,
            mode="r",
            shape=(track_layout.resampled_frames, 2),
        )
        post_peak = 0.0
        post_energy = 0.0
        first_nonzero: int | None = None
        contribution_digest = hashlib.sha256()

        for relative_start in range(0, track_layout.resampled_frames, MIX_BLOCK_FRAMES):
            relative_end = min(
                relative_start + MIX_BLOCK_FRAMES, track_layout.resampled_frames
            )
            frames = relative_end - relative_start
            absolute_start = track_layout.start_frame + relative_start
            block = np.array(cached[relative_start:relative_end], dtype=np.float32, copy=True)
            if first_nonzero is None:
                nonzero = np.flatnonzero(np.max(np.abs(block), axis=1) > 0.0)
                if nonzero.size:
                    first_nonzero = absolute_start + int(nonzero[0])
            gains = _gain_for_block(track_layout, absolute_start, frames)
            contribution = block.astype(np.float64) * gains[:, None]
            accumulator[absolute_start : absolute_start + frames] += contribution
            post_peak = max(post_peak, float(np.max(np.abs(contribution))))
            post_energy += float(np.sum(contribution * contribution))
            contribution_digest.update(
                np.ascontiguousarray(contribution, dtype="<f4").tobytes()
            )

        del cached
        track_energy = post_energy
        contribution_energies.append(track_energy)
        rms = math.sqrt(track_energy / max(1, track_layout.resampled_frames * 2))
        segments = track_layout.gain_segments()
        track_reports.append(
            {
                "track_id": track.track_id,
                "role": track.role,
                "priority": track.priority,
                "source_path": str(track.path),
                "source_sample_rate": track.sample_rate,
                "source_channels": track.channels,
                "timeline_start_frame": track_layout.start_frame,
                "timeline_start_ms_requested": track_layout.requested_start_ms,
                "timeline_start_ms_resolved": (
                    track_layout.start_frame * 1000.0 / sample_rate
                ),
                "start_rounding_error_ms": track_layout.start_rounding_error_ms,
                "first_valid_sample": track_layout.start_frame,
                "first_nonzero_sample": first_nonzero,
                "first_nonzero_offset": (
                    first_nonzero - track_layout.start_frame
                    if first_nonzero is not None
                    else None
                ),
                "last_valid_sample_exclusive": track_layout.end_frame,
                "source_frames": track_layout.source_frames,
                "resampled_frames": track_layout.resampled_frames,
                "base_gain_db": track.gain_db,
                "duck_gain_db": (
                    min((interval.gain_db for interval in track_layout.duck_intervals), default=0.0)
                ),
                "duck_intervals": [
                    interval.as_dict(manifest) for interval in track_layout.duck_intervals
                ],
                "gain_segments": [segment.as_dict(manifest) for segment in segments],
                "post_gain_peak_dbfs": linear_to_db(post_peak),
                "post_gain_rms_dbfs": linear_to_db(rms),
                "post_gain_sum_squares": track_energy,
                "contribution_float32_sha256": contribution_digest.hexdigest(),
            }
        )

    total_track_energy = sum(contribution_energies)
    for report, energy in zip(track_reports, contribution_energies):
        ratio = energy / total_track_energy if total_track_energy > 0.0 else 0.0
        report["loudness_contribution"] = {
            "method": "post_gain_energy_share_before_global_master_gain",
            "sum_squares": energy,
            "share_linear": ratio,
            "share_db": linear_to_db(math.sqrt(ratio)) if ratio > 0.0 else None,
            "rms_dbfs": report["post_gain_rms_dbfs"],
        }

    mix_peak = float(np.max(np.abs(accumulator)))
    mix_energy = float(np.sum(accumulator * accumulator))
    write_float_wav(output_float_wav, accumulator, sample_rate)
    mix_stats = {
        "duration_frames": total_frames,
        "duration_ms": total_frames * 1000.0 / sample_rate,
        "sample_peak_dbfs": linear_to_db(mix_peak),
        "rms_dbfs": linear_to_db(
            math.sqrt(mix_energy / max(1, total_frames * 2))
        ),
        "track_energy_sum_squares": total_track_energy,
    }
    del accumulator
    return mix_stats, track_reports
