"""Per-track processing pipeline and parallel orchestration."""
from __future__ import annotations

import os
import shutil
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from .limiter import LIMIT_DBTP, apply_limiter
from .loudness import (
    find_silence_segments,
    measure_integrated_loudness,
    true_peak_db,
)
from .resample import ensure_ffmpeg, resample_to_48k
from .srt import SrtFormatError, adjust_timeline, read_srt, write_srt
from .wavio import (
    TARGET_SAMPLE_RATE,
    WavData,
    WavFormatError,
    read_wav,
    write_wav_24bit,
)

TARGET_LUFS = -23.0
LUFS_TOLERANCE = 0.5


@dataclass
class Job:
    wav_path: str
    srt_path: str
    base_name: str


class FatalConfigError(Exception):
    """Global configuration failure; the whole run must terminate."""


def discover_jobs(input_dir: str) -> list[Job]:
    jobs = []
    wav_files = sorted(
        name for name in os.listdir(input_dir)
        if name.lower().endswith(".wav")
    )
    for name in wav_files:
        stem = os.path.splitext(name)[0]
        srt_path = os.path.join(input_dir, stem + ".srt")
        jobs.append(Job(
            wav_path=os.path.join(input_dir, name),
            srt_path=srt_path,
            base_name=stem,
        ))
    return jobs


def _safe_float(value) -> float:
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        if np.isneginf(value):
            return -70.0
        if np.isposinf(value) or np.isnan(value):
            return None
        return round(float(value), 3)
    return value


def process_track(job: Job, output_dir: str, ffmpeg_bin: str,
                  work_root: str) -> dict:
    """Process one WAV/SRT pair inside an isolated temporary directory."""
    temp_dir = tempfile.mkdtemp(prefix=f"job_{job.base_name}_", dir=work_root)
    report = {
        "file_name": os.path.basename(job.wav_path),
        "status": "success",
        "errors": [],
        "temp_dir": None,
    }
    try:
        source = read_wav(job.wav_path)
        report["source"] = {
            "sample_rate_hz": source.sample_rate,
            "bit_depth": source.sample_width * 8,
            "channels": source.n_channels,
            "duration_seconds": round(source.duration_seconds, 3),
        }

        resampled_path = os.path.join(temp_dir, job.base_name + "_48k.wav")
        if source.sample_rate == TARGET_SAMPLE_RATE:
            resampled = source
        else:
            resampled = resample_to_48k(
                job.wav_path, resampled_path, ffmpeg_bin, TARGET_SAMPLE_RATE
            )
        report["resampling"] = {
            "input_sample_rate_hz": source.sample_rate,
            "output_sample_rate_hz": TARGET_SAMPLE_RATE,
            "performed": source.sample_rate != TARGET_SAMPLE_RATE,
            "algorithm": "ffmpeg_aresample_swr_256taps_antialiasing",
        }

        original_loudness = measure_integrated_loudness(
            resampled.samples, TARGET_SAMPLE_RATE
        )
        original_tp = true_peak_db(resampled.samples)
        if not np.isfinite(original_loudness):
            report["status"] = "error"
            report["errors"].append(
                "integrated loudness is not measurable (track is silent or "
                "too short); loudness normalization skipped"
            )
            gain_db = 0.0
        else:
            gain_db = TARGET_LUFS - original_loudness
        gain_linear = 10.0 ** (gain_db / 20.0)
        normalized = resampled.samples * gain_linear

        normalized_tp = true_peak_db(normalized)
        if normalized_tp > LIMIT_DBTP:
            normalized, limiter_stats = apply_limiter(
                normalized, TARGET_SAMPLE_RATE, LIMIT_DBTP
            )
        else:
            limiter_stats = {
                "limiter_applied": False,
                "trigger_count": 0,
                "max_gain_reduction_db": 0.0,
                "guard_passes": 0,
                "ceiling_db_tp": LIMIT_DBTP,
            }

        final_loudness = measure_integrated_loudness(
            normalized, TARGET_SAMPLE_RATE
        )
        final_tp = true_peak_db(normalized)

        out_wav = os.path.join(
            output_dir, job.base_name + "_normalized.wav"
        )
        write_wav_24bit(out_wav, normalized, TARGET_SAMPLE_RATE)

        report["loudness"] = {
            "original_integrated_lufs": _safe_float(original_loudness),
            "target_lufs": TARGET_LUFS,
            "applied_gain_db": round(float(gain_db), 3),
            "final_integrated_lufs": _safe_float(final_loudness),
            "loudness_error_lu": (
                round(abs(float(final_loudness) - TARGET_LUFS), 3)
                if np.isfinite(final_loudness) else None
            ),
            "within_tolerance": bool(
                np.isfinite(final_loudness)
                and abs(float(final_loudness) - TARGET_LUFS) <= LUFS_TOLERANCE
            ),
        }
        report["peak"] = {
            "original_true_peak_dbtp": _safe_float(original_tp),
            "normalized_pre_limiter_dbtp": _safe_float(normalized_tp),
            "final_true_peak_dbtp": _safe_float(final_tp),
            "ceiling_dbtp": LIMIT_DBTP,
        }
        report["limiter"] = limiter_stats

        _process_subtitles(job, resampled, output_dir, report)
    except (WavFormatError, SrtFormatError) as exc:
        report["status"] = "error"
        report["errors"].append(str(exc))
    except Exception as exc:  # noqa: BLE001 - record and keep temp evidence
        report["status"] = "error"
        report["errors"].append(
            f"unexpected failure: {exc.__class__.__name__}: {exc}"
        )
        report["traceback"] = traceback.format_exc()

    if report["status"] == "error":
        report["temp_dir"] = temp_dir
    else:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return report


def _process_subtitles(job: Job, audio: WavData, output_dir: str,
                       report: dict) -> None:
    if not os.path.exists(job.srt_path):
        report["subtitles"] = {"status": "missing", "adjustments": []}
        report["status"] = "error"
        report["errors"].append(
            f"matching SRT not found for {os.path.basename(job.wav_path)}"
        )
        return
    try:
        entries = read_srt(job.srt_path)
    except SrtFormatError as exc:
        report["subtitles"] = {"status": "error", "adjustments": []}
        report["status"] = "error"
        report["errors"].append(str(exc))
        return

    silence_segments = find_silence_segments(
        audio.samples, audio.sample_rate
    )
    adjustments = adjust_timeline(entries, silence_segments)
    out_srt = os.path.join(output_dir, job.base_name + "_normalized.srt")
    write_srt(out_srt, entries)
    report["subtitles"] = {
        "status": "ok",
        "entry_count": len(entries),
        "silence_segments": [
            {"start": round(s, 3), "end": round(e, 3)}
            for s, e in silence_segments
        ],
        "adjustment_count": len(adjustments),
        "adjustments": adjustments,
    }


def run_batch(input_dir: str, output_dir: str, parallelism: int = 2,
              work_root: str = None) -> dict:
    if not os.path.isdir(input_dir):
        raise FatalConfigError(f"input directory does not exist: {input_dir}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        probe_path = os.path.join(output_dir, ".write_probe")
        with open(probe_path, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(probe_path)
    except OSError as exc:
        raise FatalConfigError(
            f"output directory is not writable: {output_dir} ({exc})"
        ) from exc
    if parallelism < 1:
        raise FatalConfigError("parallelism must be >= 1")

    ffmpeg_bin = ensure_ffmpeg()
    jobs = discover_jobs(input_dir)
    work_root = work_root or tempfile.mkdtemp(prefix="audio_norm_")
    os.makedirs(work_root, exist_ok=True)

    ordered_reports = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        future_to_index = {
            executor.submit(
                process_track, job, output_dir, ffmpeg_bin, work_root
            ): index
            for index, job in enumerate(jobs)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            ordered_reports[index] = future.result()

    tracks = [r for r in ordered_reports if r is not None]
    any_failure = any(track["status"] == "error" for track in tracks)
    if any_failure:
        retained = sorted(
            os.path.join(work_root, name)
            for name in os.listdir(work_root)
            if os.path.isdir(os.path.join(work_root, name))
        )
    else:
        retained = []
        shutil.rmtree(work_root, ignore_errors=True)
    return {"tracks": tracks, "work_root": work_root,
            "retained_temp_dirs": retained}
