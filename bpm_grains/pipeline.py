"""End-to-end BPM-aligned granular rendering pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import BpmResult, CONFIDENCE_THRESHOLD, detect_bpm
from .artifacts import write_artifacts_atomically
from .granular import GrainCloud, synthesize_grain_cloud
from .mixing import (
    ORIGINAL_GAIN,
    PARTICLE_GAIN,
    PERCUSSION_GAIN,
    apply_soft_limiter,
    db_from_linear,
    prepare_percussion_loop,
    tile_to_length,
)
from .wav_io import read_pcm_wav, write_pcm16_wav

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderResult:
    output_path: Path
    report_path: Path
    mixing_plan_path: Path
    grains_path: Path
    bpm: float
    confidence: float
    beat_samples: int
    total_samples: int
    peak_dbfs: float
    grain_count: int
    limiter_applied: bool


def _ensure_stereo(source: np.ndarray) -> np.ndarray:
    if source.shape[1] == 2:
        return source
    return np.repeat(source, 2, axis=1)


def _octave_aligned_bpm(detected_bpm: float, master_bpm: float) -> float:
    """Resolve octave/subdivision ambiguity relative to the master tempo."""

    if detected_bpm <= 0.0 or master_bpm <= 0.0:
        return master_bpm
    candidates = [
        detected_bpm * (2.0**octave) for octave in range(-3, 4)
    ]
    aligned_bpm = min(
        candidates,
        key=lambda candidate: abs(math.log2(candidate / master_bpm)),
    )
    if abs(math.log2(aligned_bpm / master_bpm)) <= math.log2(1.02):
        return master_bpm
    return aligned_bpm


def _read_percussion(
    path: Path,
    target_beat_samples: int,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    percussion = read_pcm_wav(path)
    try:
        percussion_bpm = detect_bpm(
            percussion.samples,
            percussion.sample_rate,
            raise_on_low_confidence=False,
        )
    except Exception:
        LOGGER.warning("percussion BPM analysis failed; using start-grid alignment")
        percussion_bpm = BpmResult(
            bpm=0.0,
            confidence=0.0,
            beat_samples=target_beat_samples,
        )

    master_bpm = 60.0 * percussion.sample_rate / target_beat_samples
    grid_bpm = _octave_aligned_bpm(percussion_bpm.bpm, master_bpm)
    source_beat_samples = round(60.0 * percussion.sample_rate / grid_bpm)

    if percussion_bpm.confidence >= CONFIDENCE_THRESHOLD:
        LOGGER.info(
            "percussion aligned BPM %.3f (raw %.3f, confidence %.3f)",
            grid_bpm,
            percussion_bpm.bpm,
            percussion_bpm.confidence,
        )
    else:
        LOGGER.warning(
            "percussion BPM confidence %.3f is low; keeping source rate and "
            "aligning its first sample to the master downbeat",
            percussion_bpm.confidence,
        )

    aligned, loop_samples, ratio = prepare_percussion_loop(
        percussion.samples,
        target_beat_samples=target_beat_samples,
        source_beat_samples=source_beat_samples,
        confidence=percussion_bpm.confidence,
    )
    details = {
        "input_path": str(percussion.source_path),
        "start_sample": 0,
        "loop_samples": int(loop_samples),
        "detected_bpm": float(percussion_bpm.bpm),
        "grid_bpm": round(float(grid_bpm), 6),
        "source_beat_samples": int(source_beat_samples),
        "confidence": float(percussion_bpm.confidence),
        "tempo_ratio": round(float(ratio), 9),
        "gain": PERCUSSION_GAIN,
    }
    return aligned, loop_samples, details


def render(
    *,
    input_path: str | Path,
    output_path: str | Path,
    seed: int,
    percussion_path: str | Path | None = None,
) -> RenderResult:
    source = read_pcm_wav(input_path)
    bpm_result = detect_bpm(source.samples, source.sample_rate)
    beat_samples = bpm_result.beat_samples
    cloud_loop_samples = 4 * beat_samples

    LOGGER.info(
        "analyzed BPM %.3f (confidence %.3f), beat=%d samples",
        bpm_result.bpm,
        bpm_result.confidence,
        beat_samples,
    )

    grain_cloud: GrainCloud = synthesize_grain_cloud(
        source.samples,
        source.sample_rate,
        beat_samples,
        int(seed),
    )
    if grain_cloud.loop_samples != cloud_loop_samples:
        raise RuntimeError("grain cloud loop length is not exactly four beats")

    output_samples = max(
        cloud_loop_samples,
        int(np.ceil(source.frames / float(cloud_loop_samples)))
        * cloud_loop_samples,
    )

    original_loop = np.zeros(
        (output_samples, source.samples.shape[1]), dtype=np.float64
    )
    original_loop[: source.frames] = source.samples
    cloud_track = tile_to_length(grain_cloud.audio, output_samples)
    original_track = tile_to_length(original_loop, output_samples)
    original_track = _ensure_stereo(original_track)

    percussion_track: np.ndarray | None = None
    percussion_details: dict[str, Any] | None = None
    if percussion_path is not None:
        percussion_loop, percussion_loop_samples, percussion_details = (
            _read_percussion(Path(percussion_path), beat_samples)
        )
        percussion_track = tile_to_length(percussion_loop, output_samples)
        percussion_track = _ensure_stereo(percussion_track)
        LOGGER.info(
            "percussion aligned at sample 0 on a %d-sample bar grid",
            percussion_loop_samples,
        )

    mix = ORIGINAL_GAIN * original_track + PARTICLE_GAIN * cloud_track
    if percussion_track is not None:
        mix += PERCUSSION_GAIN * percussion_track

    limiter = apply_soft_limiter(mix)
    if limiter.applied:
        LOGGER.warning(
            "soft limiter applied: pre-limit peak %.3f dBFS, "
            "gain adjustment %.3f dB",
            db_from_linear(limiter.pre_peak),
            limiter.gain_adjustment_db,
        )

    output = Path(output_path)
    mixing_plan_path = output.parent / "mixing_plan.json"
    grains_path = output.parent / "grains.json"
    report_path = output.parent / "report.json"
    total_samples = int(output_samples)
    quantized = np.rint(np.clip(limiter.audio, -1.0, 1.0) * 32767.0)
    output_peak = (
        float(np.max(np.abs(quantized))) / 32768.0 if quantized.size else 0.0
    )
    peak_dbfs = round(db_from_linear(output_peak), 6)

    mixing_plan = {
        "bpm": bpm_result.bpm,
        "confidence": bpm_result.confidence,
        "beat_samples": beat_samples,
        "sample_rate": source.sample_rate,
        "seed": int(seed),
        "alignment_start_sample": 0,
        "cloud_loop_samples": int(cloud_loop_samples),
        "output_samples": total_samples,
        "tracks": {
            "original": {
                "start_sample": 0,
                "loop_samples": total_samples,
                "input_samples": source.frames,
                "gain": ORIGINAL_GAIN,
            },
            "grain_cloud": {
                "start_sample": 0,
                "loop_samples": int(cloud_loop_samples),
                "gain": PARTICLE_GAIN,
            },
            "percussion": percussion_details,
        },
    }
    grains_payload = {
        "count": len(grain_cloud.grains),
        "grain_length_samples": grain_cloud.grain_length_samples,
        "overlap_ratio": 0.50,
        "nominal_hop_samples": grain_cloud.nominal_hop_samples,
        "loop_samples": grain_cloud.loop_samples,
        "normalization_gain": grain_cloud.normalization_gain,
        "seed": int(seed),
        "grains": grain_cloud.grains,
    }
    report = {
        "total_samples": total_samples,
        "duration_seconds": round(total_samples / float(source.sample_rate), 9),
        "sample_rate": source.sample_rate,
        "bit_depth": 16,
        "channels": 2,
        "peak_level": round(float(output_peak), 9),
        "peak_dbfs": peak_dbfs,
        "peak_level_dbfs": peak_dbfs,
        "pre_limit_peak_dbfs": round(db_from_linear(limiter.pre_peak), 6),
        "bpm": bpm_result.bpm,
        "confidence": bpm_result.confidence,
        "beat_samples": beat_samples,
        "grain_count": len(grain_cloud.grains),
        "soft_limiting_applied": bool(limiter.applied),
        "soft_limited": bool(limiter.applied),
        "limiter_gain_adjustment_db": float(limiter.gain_adjustment_db),
        "limiter_gain_db": float(limiter.gain_adjustment_db),
        "seed": int(seed),
        "cloud_loop_samples": int(cloud_loop_samples),
        "alignment_start_sample": 0,
        "tracks": {
            "original_loop_samples": total_samples,
            "grain_cloud_loop_samples": int(cloud_loop_samples),
            "percussion_loop_samples": (
                percussion_details["loop_samples"]
                if percussion_details is not None
                else None
            ),
        },
    }

    write_artifacts_atomically(
        output,
        json_artifacts={
            "mixing_plan.json": mixing_plan,
            "grains.json": grains_payload,
            "report.json": report,
        },
        wav_writer=lambda path: write_pcm16_wav(
            path, limiter.audio, sample_rate=source.sample_rate, channels=2
        ),
    )

    LOGGER.info(
        "wrote %s (%d samples), peak %.3f dBFS, grains=%d, limiter=%s",
        output,
        total_samples,
        peak_dbfs,
        len(grain_cloud.grains),
        limiter.applied,
    )
    return RenderResult(
        output_path=output,
        report_path=report_path,
        mixing_plan_path=mixing_plan_path,
        grains_path=grains_path,
        bpm=bpm_result.bpm,
        confidence=bpm_result.confidence,
        beat_samples=beat_samples,
        total_samples=total_samples,
        peak_dbfs=peak_dbfs,
        grain_count=len(grain_cloud.grains),
        limiter_applied=bool(limiter.applied),
    )

