from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from . import __version__
from .errors import BusinessError, ProcessingError
from .ffmpeg_tools import (
    analyze_array,
    analyze_file,
    decode_audio,
    format_timecode,
    write_wav_24,
)
from .manifest import load_manifest
from .mastering import (
    LOUDNESS_TOLERANCE_LU,
    MAX_TRUE_PEAK_DBTP,
    TARGET_LUFS,
    normalize_master,
    verification,
)
from .media import ensure_ffmpeg_tools, probe_all_media
from .models import Manifest, MediaInfo
from .recovery import ProgressStore
from .sync import (
    DEFAULT_SEARCH_RANGE_MS,
    DEFAULT_SYNC_WINDOW_SECONDS,
    dry_run_sync,
    estimate_sync_for_output,
    shift_audio,
)
from .timeline import plan_timeline, synthesize_timeline
from .wav import probe_wav


def _now() -> float:
    return time.perf_counter()


def _ms(start: float, end: float | None = None) -> float:
    return round(((end if end is not None else time.perf_counter()) - start) * 1000.0, 3)


def _language_analysis_record(
    manifest: Manifest,
    media: MediaInfo,
    language_id: str,
    analysis: dict,
    used_segments: list[int],
) -> dict:
    info = media.wavs[language_id]
    language = manifest.language_by_id[language_id]
    integrated = float(analysis["integrated_lufs"])
    return {
        "id": language_id,
        "path": str(language.path),
        "sample_rate": info.sample_rate,
        "channels": info.channels,
        "bit_depth": info.bits_per_sample,
        "frame_count": info.data_frames,
        "duration_seconds": round(info.data_frames / info.sample_rate, 6),
        "block_aligned": info.block_aligned,
        "silence_risk": integrated < -40.0,
        "loudness": analysis,
        "used_in_segment_indices": used_segments,
    }


def _read_and_analyze_language(
    manifest: Manifest, language_id: str, frame_rate: float
) -> tuple[dict, int]:
    language = manifest.language_by_id[language_id]
    info = probe_wav(language.path, validate_alignment=True)
    audio = decode_audio(
        language.path,
        sample_rate=manifest.audio.sample_rate,
        channels=manifest.audio.channels,
    )
    if audio.shape[0] != info.data_frames:
        raise ProcessingError(
            3012,
            f"decoded sample count {audio.shape[0]} does not match WAV data frame "
            f"count {info.data_frames}: {language.path}",
        )
    analysis = analyze_array(
        audio,
        manifest.audio.sample_rate,
        manifest.audio.channels,
        frame_rate,
    )
    return analysis.to_report(), info.data_frames


def _read_language_audio(manifest: Manifest, language_id: str) -> np.ndarray:
    info = probe_wav(
        manifest.language_by_id[language_id].path, validate_alignment=True
    )
    audio = decode_audio(
        manifest.language_by_id[language_id].path,
        sample_rate=manifest.audio.sample_rate,
        channels=manifest.audio.channels,
    )
    if audio.shape[0] != info.data_frames:
        raise ProcessingError(
            3012,
            f"decoded sample count changed while reading {language_id}",
        )
    return audio


def _build_report(
    manifest: Manifest,
    media: MediaInfo,
    plan,
    segment_reports: list[dict],
    analyses: dict[str, dict],
    sync_report: dict,
    normalization: dict,
    final_analysis,
    output_path: Path,
    sample_count: int,
    store: ProgressStore,
    timings: dict,
    status: str,
) -> dict:
    final_verification = verification(
        final_analysis,
        target_lufs=TARGET_LUFS,
        tolerance_lu=LOUDNESS_TOLERANCE_LU,
        max_true_peak_dbtp=MAX_TRUE_PEAK_DBTP,
    )
    used: dict[str, list[int]] = {}
    for segment in plan.segments:
        used.setdefault(segment.language_id, []).append(segment.index)
    languages = [
        _language_analysis_record(
            manifest,
            media,
            language.language_id,
            analyses[language.language_id],
            used.get(language.language_id, []),
        )
        for language in manifest.languages
    ]
    normalization = dict(normalization)
    normalization["after"] = final_analysis.to_report()
    normalization["gain_db"] = round(
        final_analysis.integrated_lufs
        - float(normalization["before"]["integrated_lufs"]),
        3,
    )
    normalization["effective_gain_db"] = normalization["gain_db"]
    normalization["verified_on_final_wav"] = True
    return {
        "report_schema_version": "1.0",
        "tool": {"name": "broadcast-master", "version": __version__},
        "status": status,
        "manifest": {
            "path": str(manifest.manifest_path),
            "frame_rate": manifest.frame_rate,
            "duration_frames": plan.duration_frames,
            "first_frame": 0,
            "first_frame_timecode": "00:00:00:00",
            "audio": {
                "sample_rate": manifest.audio.sample_rate,
                "channels": manifest.audio.channels,
                "bit_depth": manifest.audio.bit_depth,
            },
            "crossfade_ms": manifest.crossfade_ms,
        },
        "timeline": {
            "duration_frames": plan.duration_frames,
            "duration_seconds": round(
                sample_count / manifest.audio.sample_rate, 6
            ),
            "total_samples": sample_count,
            "requested_crossfade_samples": plan.crossfade_samples,
            "effective_switches": [
                {
                    "switch_index": item.original_index,
                    "frame": item.frame,
                    "language": item.language_id,
                    "sample": item.sample,
                    "implicit_initial": item.implicit,
                    "timecode": format_timecode(
                        item.frame / manifest.frame_rate, manifest.frame_rate
                    ),
                }
                for item in plan.effective_switches
            ],
        },
        "languages": languages,
        "segments": segment_reports,
        "fallbacks": plan.fallbacks,
        "sync": sync_report,
        "normalization": normalization,
        "verification": final_verification,
        "output": {
            "path": str(output_path),
            "single_audio_stream": True,
            "sample_rate": manifest.audio.sample_rate,
            "channels": manifest.audio.channels,
            "bit_depth": 24,
            "sample_count": sample_count,
            "duration_seconds": round(
                sample_count / manifest.audio.sample_rate, 6
            ),
            "aligned_to_manifest_first_frame": True,
        },
        "media": {
            "video_path": str(media.video.path),
            "video_duration_seconds": media.video.duration_seconds,
            "audio_stream_count": len(media.video.audio_streams),
            "reference_stream_index": media.video.selected_reference_index,
            "audio_streams": media.video.audio_streams,
        },
        "safety": {
            "all_samples_finite": True,
            "output_peak_at_or_below_minus_one_dbtp": final_verification[
                "true_peak_within_limit"
            ],
            "loudness_in_tolerance": final_verification[
                "integrated_within_tolerance"
            ],
            "no_output_samples_above_full_scale": final_analysis.true_peak_dbtp
            <= 0.0,
        },
        "recovery": store.report(),
        "stage_timings_ms": timings,
    }


def process_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    temp_root: str | Path | None = None,
) -> tuple[dict, dict]:
    total_start = _now()
    parse_start = _now()
    manifest = load_manifest(manifest_path)
    parse_ms = _ms(parse_start)
    if manifest.audio.bit_depth != 24:
        raise BusinessError(1006, "audio.bit_depth must be 24")
    output = Path(output_path).expanduser().resolve()
    report = Path(report_path).expanduser().resolve()
    if output == report:
        raise BusinessError(1000, "--output and --report must be different paths")
    ensure_ffmpeg_tools()

    store = ProgressStore(manifest, temp_root=temp_root)
    store.begin()
    temporary_output: Path | None = None
    temporary_report: Path | None = None
    timings: dict[str, float] = {"parse_manifest_ms": parse_ms}
    try:
        probe_start = _now()
        media = probe_all_media(manifest)
        timings["probe_media_ms"] = _ms(probe_start)

        analysis_start = _now()
        analyses: dict[str, dict] = {}
        for language in manifest.languages:
            cached = store.completed_analysis(language.language_id, language.path)
            if cached is not None:
                analyses[language.language_id] = cached
                store.note_skipped(language.language_id)
                continue
            store.set_stage(f"analyzing_language:{language.language_id}")
            analysis, _ = _read_and_analyze_language(
                manifest, language.language_id, manifest.frame_rate
            )
            analyses[language.language_id] = analysis
            store.mark_language(language.language_id, language.path, analysis)
        timings["language_analysis_ms"] = _ms(analysis_start)

        plan_start = _now()
        plan = plan_timeline(manifest, media)
        timings["timeline_plan_ms"] = _ms(plan_start)

        decode_start = _now()
        used_languages = sorted({segment.language_id for segment in plan.segments})
        language_audio = {
            language_id: _read_language_audio(manifest, language_id)
            for language_id in used_languages
        }
        timings["language_decode_ms"] = _ms(decode_start)

        store.set_stage("synthesizing")
        synth_start = _now()
        synthesized, segment_reports = synthesize_timeline(
            manifest, plan, language_audio
        )
        timings["synthesis_ms"] = _ms(synth_start)

        store.set_stage("sync_estimation")
        sync_start = _now()
        sync_report = estimate_sync_for_output(
            synthesized,
            manifest.audio.sample_rate,
            media.video,
            window_seconds=DEFAULT_SYNC_WINDOW_SECONDS,
        )
        if sync_report["status"] == "compensated":
            shift_samples = int(sync_report["applied_shift_samples"])
            if shift_samples:
                synthesized = shift_audio(synthesized, shift_samples)
        timings["sync_ms"] = _ms(sync_start)

        store.set_stage("normalizing")
        normalize_start = _now()
        mastered, normalization = normalize_master(
            synthesized,
            manifest.audio.sample_rate,
            manifest.audio.channels,
            manifest.frame_rate,
        )
        mastered = np.ascontiguousarray(mastered, dtype=np.float32)
        if mastered.shape[0] != plan.total_samples:
            raise ProcessingError(
                3052,
                f"master sample count {mastered.shape[0]} does not match timeline "
                f"{plan.total_samples}",
            )
        timings["normalization_ms"] = _ms(normalize_start)

        store.set_stage("writing_outputs")
        write_start = _now()
        output.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        nonce = f"{os.getpid()}-{time.time_ns()}"
        temporary_output = output.parent / f".{output.name}.{nonce}.tmp"
        temporary_report = report.parent / f".{report.name}.{nonce}.tmp"
        try:
            write_wav_24(
                temporary_output,
                mastered,
                manifest.audio.sample_rate,
                manifest.audio.channels,
            )
            output_info = probe_wav(temporary_output, validate_alignment=True)
            if (
                output_info.sample_rate != manifest.audio.sample_rate
                or output_info.channels != manifest.audio.channels
                or output_info.bits_per_sample != 24
                or output_info.data_frames != plan.total_samples
            ):
                raise ProcessingError(
                    3053, "written WAV format does not match the delivery manifest"
                )
            final_analysis = analyze_file(temporary_output, manifest.frame_rate)
            final_check = verification(
                final_analysis,
                target_lufs=TARGET_LUFS,
                tolerance_lu=LOUDNESS_TOLERANCE_LU,
                max_true_peak_dbtp=MAX_TRUE_PEAK_DBTP,
            )
            silent_master = normalization.get("method") == "silent_master"
            if not final_check["passed"] and not silent_master:
                raise ProcessingError(
                    3054,
                    "final 24-bit WAV failed loudness or true-peak verification",
                    final_check,
                )
            status = "ok" if final_check["passed"] else "warning_silent_master"
            timings["write_and_verify_ms"] = _ms(write_start)
            timings["total_ms"] = round(_ms(total_start), 3)
            report_payload = _build_report(
                manifest,
                media,
                plan,
                segment_reports,
                analyses,
                sync_report,
                normalization,
                final_analysis,
                output,
                plan.total_samples,
                store,
                timings,
                status,
            )
            with temporary_report.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    report_payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=False,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_output, output)
            temporary_output = None
            os.replace(temporary_report, report)
            temporary_report = None
        finally:
            if temporary_output is not None:
                temporary_output.unlink(missing_ok=True)
            if temporary_report is not None:
                temporary_report.unlink(missing_ok=True)

        summary = {
            "status": status,
            "output": str(output),
            "report": str(report),
            "integrated_lufs": final_analysis.integrated_lufs,
            "true_peak_dbtp": final_analysis.true_peak_dbtp,
            "sync_status": sync_report["status"],
            "fallback_count": len(plan.fallbacks),
        }
        store.succeed()
        return summary, report_payload
    except BusinessError:
        store.abort_business()
        raise
    except ProcessingError as exc:
        store.fail(exc)
        raise
    except Exception as exc:
        failure = ProcessingError(3999, f"unexpected processing failure: {exc}")
        store.fail(failure)
        raise failure from exc


def dry_run_manifest(
    manifest_path: str | Path,
    *,
    sync_window_seconds: float = DEFAULT_SYNC_WINDOW_SECONDS,
) -> dict:
    total_start = _now()
    parse_start = _now()
    manifest = load_manifest(manifest_path)
    parse_ms = _ms(parse_start)
    ensure_ffmpeg_tools()

    probe_start = _now()
    media = probe_all_media(manifest)
    probe_ms = _ms(probe_start)

    plan_start = _now()
    plan = plan_timeline(manifest, media)
    plan_ms = _ms(plan_start)

    sync_start = _now()
    sync_report = dry_run_sync(
        manifest, media.video, plan, window_seconds=sync_window_seconds
    )
    sync_ms = _ms(sync_start)
    total_ms = round(_ms(total_start), 3)
    return {
        "mode": "dry-run",
        "status": "ok",
        "manifest": {
            "path": str(manifest.manifest_path),
            "frame_rate": manifest.frame_rate,
            "duration_frames": plan.duration_frames,
            "crossfade_ms": manifest.crossfade_ms,
            "audio": {
                "sample_rate": manifest.audio.sample_rate,
                "channels": manifest.audio.channels,
                "bit_depth": manifest.audio.bit_depth,
            },
        },
        "languages": [
            {
                "id": language.language_id,
                "path": str(language.path),
                "frame_count": media.wavs[language.language_id].data_frames,
                "duration_seconds": round(
                    media.wavs[language.language_id].data_frames
                    / media.wavs[language.language_id].sample_rate,
                    6,
                ),
                "sample_rate": media.wavs[language.language_id].sample_rate,
                "channels": media.wavs[language.language_id].channels,
                "bit_depth": media.wavs[language.language_id].bits_per_sample,
            }
            for language in manifest.languages
        ],
        "segments": [
            segment.to_report(manifest.audio.sample_rate, manifest.frame_rate)
            for segment in plan.segments
        ],
        "fallbacks": plan.fallbacks,
        "timeline": {
            "duration_frames": plan.duration_frames,
            "total_samples": plan.total_samples,
            "duration_seconds": round(
                plan.total_samples / manifest.audio.sample_rate, 6
            ),
            "requested_crossfade_samples": plan.crossfade_samples,
            "aligned_to_manifest_first_frame": True,
        },
        "sync": sync_report,
        "media": {
            "video_path": str(media.video.path),
            "audio_stream_count": len(media.video.audio_streams),
            "reference_stream_index": media.video.selected_reference_index,
        },
        "stage_timings_ms": {
            "parse_manifest_ms": parse_ms,
            "probe_media_ms": probe_ms,
            "simulate_timeline_ms": plan_ms,
            "sync_estimate_ms": sync_ms,
            "total_ms": total_ms,
        },
    }

