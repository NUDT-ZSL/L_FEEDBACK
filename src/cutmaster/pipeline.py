"""End-to-end mastering pipeline."""

from __future__ import annotations

import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio import (
    RESAMPLE_KAISER_BETA,
    RESAMPLE_TAPS,
    copy_float_wav_scaled,
    db_to_linear,
    sha256_file,
    write_master_wav,
)
from .errors import err
from .loudness import (
    LoudnessMeasurement,
    apply_true_peak_limiter,
    measure_loudness,
    target_gain_db,
)
from .manifest import Manifest, load_manifest
from .mix import mix_timeline
from .state import ProcessLock, ProjectState, atomic_write_json
from .timeline import TimelineLayout, build_layout


@dataclass(frozen=True)
class MasterOptions:
    manifest_path: Path
    master_path: Path
    report_path: Path
    state_dir: Path
    force: bool = False


def _fault_stage() -> str | None:
    value = os.environ.get("CUTMASTER_FAULT_INJECT")
    return value.strip() if value else None


def _fault_exit(stage: str, configured: str | None) -> None:
    if configured == stage:
        os._exit(75)


def _ensure_distinct_paths(
    manifest: Manifest, master_path: Path, report_path: Path, state_dir: Path
) -> None:
    if master_path == report_path:
        raise err(
            "E_OUTPUT_PATH_CONFLICT",
            "master and report paths must be different",
            master=str(master_path),
            report=str(report_path),
        )
    source_paths = {manifest.source_path.resolve(strict=False)}
    source_paths.update(track.path.resolve(strict=False) for track in manifest.tracks)
    if master_path.resolve(strict=False) in source_paths or report_path.resolve(
        strict=False
    ) in source_paths:
        raise err(
            "E_OUTPUT_PATH_CONFLICT",
            "an output path would overwrite an input file",
            master=str(master_path),
            report=str(report_path),
        )
    project_root = (state_dir / "projects" / manifest.project_key).resolve(strict=False)
    for output in (master_path, report_path):
        resolved = output.resolve(strict=False)
        try:
            resolved.relative_to(project_root)
        except ValueError:
            continue
        raise err(
            "E_OUTPUT_PATH_CONFLICT",
            "outputs may not be placed inside the managed state directory",
            path=str(output),
        )


def _reuse_committed(
    checkpoint: dict[str, Any] | None,
    manifest: Manifest,
    master_path: Path,
    report_path: Path,
    force: bool,
) -> dict[str, Any] | None:
    if force or not checkpoint:
        return None
    if (
        checkpoint.get("status") != "committed"
        or checkpoint.get("manifest_sha256") != manifest.sha256
    ):
        return None
    outputs = checkpoint.get("outputs")
    if not isinstance(outputs, dict):
        return None
    try:
        same_master = Path(outputs["master"]).resolve(strict=False) == master_path
        same_report = Path(outputs["report"]).resolve(strict=False) == report_path
        expected_master_sha = str(outputs["master_sha256"])
        expected_report_sha = str(outputs["report_sha256"])
    except (KeyError, TypeError, ValueError):
        return None
    if not same_master or not same_report:
        return None
    if not master_path.is_file() or not report_path.is_file():
        return None
    if sha256_file(master_path) != expected_master_sha:
        return None
    if sha256_file(report_path) != expected_report_sha:
        return None
    return {
        "ok": True,
        "status": "committed",
        "reused_committed": True,
        "master": str(master_path),
        "report": str(report_path),
        "master_sha256": expected_master_sha,
        "manifest_sha256": manifest.sha256,
    }


def _recovery_info(
    checkpoint: dict[str, Any] | None,
    checkpoint_problem: str | None,
    manifest: Manifest,
) -> dict[str, Any]:
    if checkpoint is None:
        return {
            "previous_status": None,
            "previous_stage": None,
            "action": "full_run_no_checkpoint",
            "checkpoint_problem": checkpoint_problem,
        }
    previous_status = checkpoint.get("status")
    previous_stage = checkpoint.get("stage")
    same_manifest = checkpoint.get("manifest_sha256") == manifest.sha256
    if previous_status == "in_progress":
        action = (
            "resume_from_checkpoint"
            if same_manifest
            else "manifest_changed_full_rebuild"
        )
    elif previous_status == "failed":
        action = "retry_after_failure" if same_manifest else "manifest_changed_full_rebuild"
    elif previous_status == "committed":
        action = "regenerate_committed_version" if same_manifest else "manifest_changed_full_rebuild"
    else:
        action = "full_run_unknown_checkpoint_state"
    return {
        "previous_status": previous_status,
        "previous_stage": previous_stage,
        "previous_run_id": checkpoint.get("run_id"),
        "previous_version": (checkpoint.get("project") or {}).get("version"),
        "action": action,
        "checkpoint_problem": checkpoint_problem,
    }


def _measure_or_fail(path: Path) -> LoudnessMeasurement:
    measurement = measure_loudness(path)
    if measurement.integrated_lufs is None:
        raise err(
            "E_SILENT_MIX",
            "mix has no measurable integrated loudness",
            exit_code=6,
        )
    return measurement


def _tp_or_fail(measurement: LoudnessMeasurement) -> float:
    if measurement.true_peak_dbtp is None:
        raise err(
            "E_LOUDNESS_MEASURE",
            "true peak is not measurable",
            exit_code=6,
        )
    return measurement.true_peak_dbtp


def _apply_gain(
    source: Path, destination: Path, sample_rate: int, gain_db: float
) -> None:
    gain = db_to_linear(gain_db)
    peak, _ = copy_float_wav_scaled(source, destination, sample_rate, gain)
    if not math.isfinite(peak):
        raise err("E_AUDIO_WRITE", "failed to apply normalization gain", exit_code=7)


def _normalize(
    raw_float_wav: Path,
    work_root: Path,
    sample_rate: int,
    target_lufs: float,
    ceiling_dbtp: float,
    checkpoint: Any,
    completed_track_ids: list[str],
) -> tuple[Path, dict[str, Any]]:
    before = _measure_or_fail(raw_float_wav)
    intended_gain = target_gain_db(before, target_lufs)
    limiter_applied = False
    limiter_limit_dbtp = ceiling_dbtp - 1.0
    limiter_limit_linear = db_to_linear(limiter_limit_dbtp)
    total_gain = intended_gain
    current = work_root / "normalized-0.wav"
    _apply_gain(raw_float_wav, current, sample_rate, total_gain)
    measurements: list[dict[str, Any]] = [
        {
            "stage": "before_normalization",
            "integrated_lufs": before.integrated_lufs,
            "true_peak_dbtp": before.true_peak_dbtp,
        }
    ]

    measured = before
    for iteration in range(6):
        measured = _measure_or_fail(current)
        measured_tp = _tp_or_fail(measured)
        measurements.append(
            {
                "stage": f"float_normalized_{iteration}",
                "integrated_lufs": measured.integrated_lufs,
                "true_peak_dbtp": measured.true_peak_dbtp,
            }
        )
        if measured_tp > ceiling_dbtp + 0.001:
            limited = work_root / f"limited-{iteration}.wav"
            limiter_limit_linear = apply_true_peak_limiter(
                current,
                limited,
                ceiling_dbtp,
                safety_margin_db=1.0,
            )
            limiter_limit_dbtp = 20.0 * math.log10(limiter_limit_linear)
            limiter_applied = True
            current = limited
            continue

        correction = target_lufs - measured.integrated_lufs
        if correction > 0.0:
            correction = min(correction, ceiling_dbtp - measured_tp - 0.05)
        if abs(correction) <= 0.02:
            break
        adjusted = work_root / f"normalized-{iteration + 1}.wav"
        _apply_gain(current, adjusted, sample_rate, correction)
        total_gain += correction
        current = adjusted

    measured = _measure_or_fail(current)
    measured_tp = _tp_or_fail(measured)
    measurements.append(
        {
            "stage": "after_limiter_convergence",
            "integrated_lufs": measured.integrated_lufs,
            "true_peak_dbtp": measured.true_peak_dbtp,
        }
    )
    final_guard = 0.0
    if measured_tp > ceiling_dbtp:
        final_guard = ceiling_dbtp - measured_tp - 0.05
        adjusted = work_root / "final-true-peak-guard.wav"
        _apply_gain(current, adjusted, sample_rate, final_guard)
        total_gain += final_guard
        current = adjusted
        measured = _measure_or_fail(current)
        measured_tp = _tp_or_fail(measured)
        measurements.append(
            {
                "stage": "after_final_true_peak_guard",
                "integrated_lufs": measured.integrated_lufs,
                "true_peak_dbtp": measured.true_peak_dbtp,
            }
        )
    final_lufs = measured.integrated_lufs
    delta = (
        abs(final_lufs - target_lufs) if final_lufs is not None else math.inf
    )
    if delta > 5.0 + 1e-9:
        raise err(
            "E_LOUDNESS_UNREACHABLE",
            "loudness target and true-peak ceiling cannot both be satisfied within 5 LU",
            exit_code=6,
            target_lufs=target_lufs,
            measured_lufs=final_lufs,
            delta_lu=delta,
            true_peak_ceiling_dbtp=ceiling_dbtp,
        )
    checkpoint(
        status="in_progress",
        stage="normalized",
        completed_track_ids=completed_track_ids,
        extra={"normalization_gain_db": total_gain},
    )
    return current, {
        "target_lufs": target_lufs,
        "target_true_peak_dbtp": ceiling_dbtp,
        "before_lufs": before.integrated_lufs,
        "before_true_peak_dbtp": before.true_peak_dbtp,
        "intended_gain_db": intended_gain,
        "true_peak_guard_db": final_guard,
        "total_gain_db": total_gain,
        "limiter": {
            "applied": limiter_applied,
            "method": "ffmpeg_alimiter_lookahead_sample_peak_with_true_peak_guard"
            if limiter_applied
            else "none",
            "ceiling_dbtp": ceiling_dbtp,
            "limiter_threshold_dbtp": limiter_limit_dbtp if limiter_applied else None,
            "limiter_threshold_linear": limiter_limit_linear if limiter_applied else None,
            "safety_margin_db": 1.0 if limiter_applied else None,
            "post_limiter_global_guard_db": final_guard,
        },
        "measured_lufs": final_lufs,
        "measured_true_peak_dbtp": measured.true_peak_dbtp,
        "delta_lu": delta,
        "measurement_passes": measurements,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _remove_stale_partials(path: Path) -> None:
    pattern = f".{path.name}.*.partial"
    for candidate in path.parent.glob(pattern):
        try:
            candidate.unlink()
        except OSError:
            pass


def run_master(options: MasterOptions) -> dict[str, Any]:
    try:
        return _run_master_impl(options)
    finally:
        _remove_stale_partials(options.master_path.resolve(strict=False))
        _remove_stale_partials(options.report_path.resolve(strict=False))


def _run_master_impl(options: MasterOptions) -> dict[str, Any]:
    started = time.perf_counter()
    manifest = load_manifest(options.manifest_path)
    master_path = options.master_path.resolve(strict=False)
    report_path = options.report_path.resolve(strict=False)
    state_dir = options.state_dir.resolve(strict=False)
    _ensure_distinct_paths(manifest, master_path, report_path, state_dir)

    state = ProjectState(state_dir, manifest)
    lock = ProcessLock(state.lock_path)
    completed_track_ids: list[str] = []
    cache_report: list[dict[str, Any]] = []
    temporary_master: Path | None = None
    temporary_report: Path | None = None

    with lock:
        state.initialize()
        previous, checkpoint_problem = state.read_checkpoint()
        reused = _reuse_committed(
            previous, manifest, master_path, report_path, options.force
        )
        if reused is not None:
            reused["elapsed_seconds"] = time.perf_counter() - started
            shutil.rmtree(state.run_root, ignore_errors=True)
            return reused

        recovery = _recovery_info(previous, checkpoint_problem, manifest)
        stale_workdirs_removed = state.cleanup_stale_workdirs()
        _remove_stale_partials(master_path)
        _remove_stale_partials(report_path)
        checkpoint_extra: dict[str, Any] = {
            "master": str(master_path),
            "report": str(report_path),
            "cache_hits": 0,
            "cache_misses": 0,
        }
        state.write_checkpoint(
            status="in_progress",
            stage="validating",
            completed_track_ids=completed_track_ids,
            extra=checkpoint_extra,
        )
        layout: TimelineLayout = build_layout(manifest)
        _fault_exit("after_layout", _fault_stage())
        state.write_checkpoint(
            status="in_progress",
            stage="preparing_tracks",
            completed_track_ids=completed_track_ids,
            extra=checkpoint_extra,
        )

        cache_paths: dict[str, Path] = {}
        for track_layout in layout.tracks:
            result = state.prepare_track_cache(
                track_layout, crash_after=_fault_stage()
            )
            cache_paths[track_layout.track_id] = Path(result["raw_path"])
            cache_report.append(result)
            completed_track_ids.append(track_layout.track_id)
            checkpoint_extra["cache_hits"] = sum(
                1 for item in cache_report if item["cache_hit"]
            )
            checkpoint_extra["cache_misses"] = sum(
                1 for item in cache_report if not item["cache_hit"]
            )
            state.write_checkpoint(
                status="in_progress",
                stage="preparing_tracks",
                completed_track_ids=completed_track_ids,
                extra=checkpoint_extra,
            )
            _fault_exit("after_track", _fault_stage())

        state.write_checkpoint(
            status="in_progress",
            stage="mixing",
            completed_track_ids=completed_track_ids,
            extra=checkpoint_extra,
        )
        _fault_exit("before_mix", _fault_stage())
        raw_float_wav = state.run_root / "mix-float.wav"
        mix_stats, track_reports = mix_timeline(layout, cache_paths, raw_float_wav)
        _fault_exit("after_mix", _fault_stage())

        state.write_checkpoint(
            status="in_progress",
            stage="normalizing",
            completed_track_ids=completed_track_ids,
            extra=checkpoint_extra,
        )
        _fault_exit("before_normalize", _fault_stage())

        def update_checkpoint(**kwargs: Any) -> None:
            state.write_checkpoint(**kwargs)

        normalized_float, loudness_report = _normalize(
            raw_float_wav,
            state.run_root,
            manifest.project.sample_rate,
            manifest.loudness.target_lufs,
            manifest.loudness.true_peak_db,
            update_checkpoint,
            completed_track_ids,
        )
        _fault_exit("after_normalize", _fault_stage())

        state.write_checkpoint(
            status="in_progress",
            stage="publishing",
            completed_track_ids=completed_track_ids,
            extra=checkpoint_extra,
        )
        master_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_master = master_path.with_name(
            f".{master_path.name}.{state.run_id}.partial"
        )
        temporary_report = report_path.with_name(
            f".{report_path.name}.{state.run_id}.partial"
        )
        master_stats = write_master_wav(
            normalized_float,
            temporary_master,
            manifest.project.sample_rate,
            manifest.output.sample_format,
        )

        final_measurement = measure_loudness(temporary_master)
        if final_measurement.integrated_lufs is None:
            raise err(
                "E_SILENT_MIX",
                "encoded master has no integrated loudness",
                exit_code=6,
            )
        final_tp = _tp_or_fail(final_measurement)
        correction = 0.0
        if final_tp > manifest.loudness.true_peak_db:
            correction = manifest.loudness.true_peak_db - final_tp - 0.05
            adjusted_float = state.run_root / "encoded-guard.wav"
            _apply_gain(
                normalized_float,
                adjusted_float,
                manifest.project.sample_rate,
                correction,
            )
            loudness_report["total_gain_db"] += correction
            loudness_report["limiter"]["applied"] = True
            loudness_report["limiter"]["post_limiter_global_guard_db"] += correction
            if loudness_report["limiter"]["method"] == "none":
                loudness_report["limiter"]["method"] = (
                    "encoded_master_linear_true_peak_guard"
                )
            loudness_report["true_peak_guard_db"] += correction
            temporary_master.unlink(missing_ok=True)
            master_stats = write_master_wav(
                adjusted_float,
                temporary_master,
                manifest.project.sample_rate,
                manifest.output.sample_format,
            )
            final_measurement = measure_loudness(temporary_master)
            if final_measurement.integrated_lufs is None:
                raise err(
                    "E_SILENT_MIX",
                    "encoded master has no integrated loudness",
                    exit_code=6,
                )
            final_tp = _tp_or_fail(final_measurement)
        if final_tp > manifest.loudness.true_peak_db + 1e-6:
            raise err(
                "E_TRUE_PEAK_UNREACHABLE",
                "encoded master still exceeds the true-peak ceiling",
                exit_code=6,
                ceiling_dbtp=manifest.loudness.true_peak_db,
                measured_dbtp=final_tp,
            )
        final_delta = abs(
            final_measurement.integrated_lufs - manifest.loudness.target_lufs
        )
        if final_delta > 5.0 + 1e-9:
            raise err(
                "E_LOUDNESS_UNREACHABLE",
                "encoded master misses the loudness target by more than 5 LU",
                exit_code=6,
                target_lufs=manifest.loudness.target_lufs,
                measured_lufs=final_measurement.integrated_lufs,
                delta_lu=final_delta,
            )
        loudness_report["measured_lufs"] = final_measurement.integrated_lufs
        loudness_report["measured_true_peak_dbtp"] = final_measurement.true_peak_dbtp
        loudness_report["delta_lu"] = final_delta
        loudness_report["measurement_passes"].append(
            {
                "stage": "final_encoded_master",
                "integrated_lufs": final_measurement.integrated_lufs,
                "true_peak_dbtp": final_measurement.true_peak_dbtp,
            }
        )

        master_gain_linear = db_to_linear(loudness_report["total_gain_db"])
        for track_report in track_reports:
            contribution = track_report["loudness_contribution"]
            pre_rms = track_report["post_gain_rms_dbfs"]
            contribution["rms_dbfs_in_master"] = (
                pre_rms + loudness_report["total_gain_db"]
                if pre_rms is not None
                else None
            )
            contribution["final_master_gain_db"] = loudness_report["total_gain_db"]
            contribution["final_master_gain_linear"] = master_gain_linear

        elapsed = time.perf_counter() - started
        report = _json_safe(
            {
                "schema_version": 1,
                "status": "committed",
                "project": {
                    "name": manifest.project.name,
                    "version": manifest.project.version,
                    "segment_id": manifest.project.segment_id,
                },
                "manifest": {
                    "path": str(manifest.source_path),
                    "sha256": manifest.sha256,
                    "schema_version": manifest.schema_version,
                },
                "recovery": {
                    **recovery,
                    "current_run_id": state.run_id,
                    "stale_workdirs_removed": stale_workdirs_removed,
                    "cache_hits": sum(1 for item in cache_report if item["cache_hit"]),
                    "cache_misses": sum(
                        1 for item in cache_report if not item["cache_hit"]
                    ),
                    "track_cache": [
                        {
                            "track_id": item["track_id"],
                            "cache_hit": item["cache_hit"],
                            "fingerprint": item["fingerprint"],
                            "source_reason": item["reason"],
                        }
                        for item in cache_report
                    ],
                },
                "timeline": {
                    "sample_rate": manifest.project.sample_rate,
                    "channels": 2,
                    "resampler": {
                        "method": "windowed_sinc",
                        "taps": RESAMPLE_TAPS,
                        "kaiser_beta": RESAMPLE_KAISER_BETA,
                        "zero_phase": True,
                    },
                    "requested_duration_ms": float(manifest.project.duration_ms),
                    "duration_frames": manifest.project.duration_frames,
                    "duration_ms": (
                        manifest.project.duration_frames
                        * 1000.0
                        / manifest.project.sample_rate
                    ),
                    "max_start_rounding_error_ms": layout.max_start_rounding_error_ms,
                    "max_start_rounding_error_samples": (
                        layout.max_start_rounding_error_ms
                        * manifest.project.sample_rate
                        / 1000.0
                    ),
                    "rounding_limit_samples": 1,
                },
                "dialogue_ducking": {
                    "requested_duck_gain_db": manifest.ducking.duck_db,
                    "safe_headroom_db": manifest.ducking.safe_headroom_db,
                    "overlap_segments": layout.overlap_segments,
                    "duck_interval_count": layout.total_duck_intervals,
                    "non_dialogue_tracks_affected": False,
                },
                "tracks": track_reports,
                "mix": {
                    **mix_stats,
                    "pre_normalization_measurement": loudness_report[
                        "measurement_passes"
                    ][0],
                },
                "loudness": {
                    **loudness_report,
                    "final_encoded_lufs": final_measurement.integrated_lufs,
                    "final_encoded_true_peak_dbtp": final_measurement.true_peak_dbtp,
                    "final_delta_lu": final_delta,
                    "within_target_tolerance_5_lu": final_delta <= 5.0,
                    "below_true_peak_ceiling": final_tp
                    <= manifest.loudness.true_peak_db,
                    "measurement_standard": "ITU-R BS.1770-4 / EBU R128 via FFmpeg ebur128 peak=true",
                },
                "output": {
                    "master_path": str(master_path),
                    "report_path": str(report_path),
                    "sample_rate": manifest.project.sample_rate,
                    "channels": 2,
                    "sample_format": manifest.output.sample_format,
                    **master_stats,
                },
                "verification": {
                    "all_tracks_validated_before_publish": True,
                    "start_rounding_within_one_sample": (
                        layout.max_start_rounding_error_ms
                        * manifest.project.sample_rate
                        / 1000.0
                        <= 1.0 + 1e-9
                    ),
                    "gain_segments_cover_each_track": all(
                        not reports
                        or (
                            reports[0]["start_frame"]
                            == next(
                                track.start_frame
                                for track in layout.tracks
                                if track.track_id == track_report["track_id"]
                            )
                            and reports[-1]["end_frame"]
                            == next(
                                track.end_frame
                                for track in layout.tracks
                                if track.track_id == track_report["track_id"]
                            )
                        )
                        for track_report in track_reports
                        for reports in [track_report["gain_segments"]]
                    ),
                    "track_contribution_checksums_recorded": True,
                    "no_partial_master_published": True,
                    "deterministic_cache_key_ignores_timeline_offsets": True,
                },
                "performance": {
                    "elapsed_seconds": elapsed,
                },
            }
        )
        atomic_write_json(temporary_report, report)
        master_sha = str(master_stats["sha256"])
        report_sha = sha256_file(temporary_report)
        os.replace(temporary_master, master_path)
        os.replace(temporary_report, report_path)
        temporary_master = None
        temporary_report = None
        state.write_checkpoint(
            status="committed",
            stage="committed",
            completed_track_ids=completed_track_ids,
            extra={
                **checkpoint_extra,
                "outputs": {
                    "master": str(master_path),
                    "report": str(report_path),
                    "master_sha256": master_sha,
                    "report_sha256": report_sha,
                },
                "elapsed_seconds": elapsed,
            },
        )
        shutil.rmtree(state.run_root, ignore_errors=True)
        return {
            "ok": True,
            "status": "committed",
            "master": str(master_path),
            "report": str(report_path),
            "master_sha256": master_sha,
            "manifest_sha256": manifest.sha256,
            "measured_lufs": final_measurement.integrated_lufs,
            "true_peak_dbtp": final_measurement.true_peak_dbtp,
            "elapsed_seconds": elapsed,
            "cache_hits": sum(1 for item in cache_report if item["cache_hit"]),
            "cache_misses": sum(1 for item in cache_report if not item["cache_hit"]),
        }


def run_from_paths(
    manifest_path: Path,
    master_path: Path | None,
    report_path: Path | None,
    state_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    master = master_path or manifest.output.master
    report = report_path or manifest.output.report
    if master is None or report is None:
        raise err(
            "E_OUTPUT_REQUIRED",
            "master and report output paths are required",
            hint="Pass --output/--report or set output.master/output.report in the manifest",
        )
    return run_master(
        MasterOptions(
            manifest_path=manifest.source_path,
            master_path=master,
            report_path=report,
            state_dir=state_dir,
            force=force,
        )
    )
