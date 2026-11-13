"""End-to-end orchestration of the offline audio packaging pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .assets import AssetError, AssetStore, FailureInjector, LoadedClip
from .config import Config
from .cue_index import build_index_entries, reconcile, render_cue_sheet
from .errors import ErrorCode, Issue, IssueCollector
from .hashing import atomic_write_bytes, atomic_write_json
from .manifest import Manifest, load_manifest
from .renderer import build_language_track, total_frames, write_pcm16_wav
from .scheduler import LanguageSchedule, schedule_all
from .snapshots import (
    block_asset_hash,
    evaluate_reuse,
    load_snapshot,
    load_track,
    revalidate_offsets,
    save_snapshot,
)


@dataclass
class PipelineResult:
    ok: bool
    report: dict
    manifest: Manifest | None = None
    issues: list[Issue] = field(default_factory=list)


def _issue_from_asset_error(error: AssetError) -> Issue:
    return Issue(
        code=error.code,
        message=str(error),
        path=error.path,
        detail={"retryable": error.retryable},
    )


def run_pipeline(
    manifest_path: str,
    assets_dir: str,
    work_dir: str,
    output_dir: str,
    config: Config | None = None,
    injector: FailureInjector | None = None,
    stop_after_lang: str | None = None,
) -> PipelineResult:
    config = config or Config()
    collector = IssueCollector()
    blocks_dir = os.path.join(work_dir, "blocks")
    os.makedirs(blocks_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    manifest = load_manifest(manifest_path, config)
    if not manifest.valid:
        report = _failure_report(manifest, config, manifest.issues)
        atomic_write_json(os.path.join(output_dir, "report.json"), report)
        return PipelineResult(
            False, report, manifest, manifest.issues.issues
        )

    # Collect every (voice_key, lang) pair required by voiced events.
    pairs: set[tuple[str, str]] = set()
    voiced_events = [event for event in manifest.events if event.voice_key]
    for event in voiced_events:
        for lang in event.langs:
            pairs.add((event.voice_key, lang))

    store = AssetStore(config, injector)
    path_map = store.required_paths(assets_dir, pairs)
    loaded, load_failures = store.load_many(list(path_map))
    for error in load_failures:
        collector.add(
            error.code,
            str(error),
            path=error.path,
            detail={"retryable": error.retryable},
        )
    if collector:
        report = _failure_report(manifest, config, collector)
        atomic_write_json(os.path.join(output_dir, "report.json"), report)
        return PipelineResult(False, report, manifest, collector.issues)

    # Organize clips per language and verify every referenced pair resolved.
    clips_by_lang: dict[str, dict[str, LoadedClip]] = {}
    for path, (voice_key, lang) in path_map.items():
        clips_by_lang.setdefault(lang, {})[voice_key] = loaded[path]

    schedules = schedule_all(manifest, clips_by_lang, config)
    # A language whose voiced events are all dropped produces no track; it is
    # reported rather than emitting a zero-frame WAV and empty index.
    empty_langs = [
        lang for lang, schedule in schedules.items()
        if not schedule.placements
    ]
    active_schedules = {
        lang: schedule for lang, schedule in schedules.items()
        if schedule.placements
    }

    block_summaries: list[dict] = []
    index_by_lang: dict[str, list[dict]] = {}
    wav_info_by_lang: dict[str, dict] = {}
    normalization_by_lang: dict[str, dict] = {}
    reused_langs: list[str] = []
    recomputed_langs: list[str] = []

    for lang in sorted(active_schedules):
        schedule = active_schedules[lang]
        block_dir = os.path.join(blocks_dir, lang)
        clips_for_lang = clips_by_lang[lang]
        clip_hash_by_source = {
            clip.source_name: clip.content_hash for clip in clips_for_lang.values()
        }
        asset_hash = block_asset_hash(schedule, clip_hash_by_source)

        decision = evaluate_reuse(
            block_dir, config, manifest.manifest_hash, asset_hash, schedule
        )

        if decision.reusable and decision.snapshot is not None:
            snapshot = decision.snapshot
            track = load_track(block_dir, snapshot)
            current_total = total_frames(schedule)
            violations = revalidate_offsets(snapshot, schedule, current_total)
            if violations:
                # Stale offsets invalidate reuse; recompute this block only.
                for violation in violations:
                    collector.add(
                        ErrorCode.E_SNAPSHOT,
                        violation["message"],
                        lang=lang,
                        detail=violation["detail"],
                    )
                decision = type(decision)(False, "offset_revalidation_failed", snapshot)

        if decision.reusable and decision.snapshot is not None:
            snapshot = decision.snapshot
            reused_langs.append(lang)
            entries = [dict(entry) for entry in snapshot.index_entries]
            wav_info = dict(snapshot.wav_info)
            normalization_by_lang[lang] = dict(snapshot.normalization)
            block_summaries.append(
                {
                    "lang": lang,
                    "status": "reused",
                    "reason": decision.reason,
                    "total_frames": snapshot.total_frames,
                }
            )
        else:
            clips_by_event = {
                placement.event_id: clips_for_lang[placement.voice_key]
                for placement in schedule.placements
            }
            track, normalization = build_language_track(
                schedule, clips_by_event, config
            )
            wav_info = _preview_wav_info(track, config, lang)
            entries = build_index_entries(schedule, config)
            save_snapshot(
                block_dir,
                config,
                manifest.manifest_hash,
                asset_hash,
                schedule,
                track,
                normalization,
                wav_info,
                entries,
            )
            normalization_by_lang[lang] = normalization.to_dict()
            recomputed_langs.append(lang)
            block_summaries.append(
                {
                    "lang": lang,
                    "status": "recomputed",
                    "reason": decision.reason,
                    "total_frames": int(track.shape[0]),
                }
            )

        index_by_lang[lang] = entries
        wav_info_by_lang[lang] = wav_info

        if stop_after_lang is not None and lang == stop_after_lang:
            # Simulate an interruption: blocks so far are durable on disk.
            report = _interrupted_report(
                manifest,
                config,
                block_summaries,
                reused_langs,
                recomputed_langs,
                stop_after_lang,
            )
            atomic_write_json(os.path.join(output_dir, "report.json"), report)
            return PipelineResult(False, report, manifest, collector.issues)

    # Final assembly: encode reusable/recomputed tracks to PCM16 WAVs, write
    # CUE sheets and index JSON, then reconcile each language independently.
    outputs: list[dict] = []
    reconciliation_errors: list[dict] = []
    for lang in sorted(active_schedules):
        schedule = active_schedules[lang]
        block_dir = os.path.join(blocks_dir, lang)
        snapshot = load_snapshot(block_dir)
        if snapshot is None:
            collector.add(
                ErrorCode.E_SNAPSHOT,
                "snapshot disappeared before final assembly",
                lang=lang,
            )
            continue
        track = load_track(block_dir, snapshot)

        wav_name = f"track_{lang}.wav"
        cue_name = f"track_{lang}.cue"
        index_name = f"index_{lang}.json"
        wav_path = os.path.join(output_dir, wav_name)
        cue_path = os.path.join(output_dir, cue_name)
        index_path = os.path.join(output_dir, index_name)

        wav_info = write_pcm16_wav(wav_path, track, config)
        entries = build_index_entries(schedule, config)
        cue_text = render_cue_sheet(schedule, wav_name, config)
        atomic_write_bytes(cue_path, cue_text.encode("utf-8"))
        atomic_write_json(
            index_path,
            {
                "lang": lang,
                "wav": wav_name,
                "sample_rate": config.sample_rate,
                "total_frames": wav_info["frames"],
                "entries": entries,
            },
        )

        errors = reconcile(schedule, entries, wav_info, config)
        for error in errors:
            reconciliation_errors.append({"lang": lang, **error.to_dict()})

        outputs.append(
            {
                "lang": lang,
                "wav": wav_name,
                "cue": cue_name,
                "index": index_name,
                "frames": wav_info["frames"],
                "duration_ms": wav_info["duration_ms"],
                "retained": len(schedule.placements),
                "dropped": len(schedule.dropped),
            }
        )

    for violation in reconciliation_errors:
        collector.add(
            ErrorCode.E_SNAPSHOT,
            violation["message"],
            lang=violation["lang"],
            detail=violation["detail"],
        )

    report = _success_report(
        manifest=manifest,
        config=config,
        schedules=schedules,
        empty_langs=empty_langs,
        outputs=outputs,
        block_summaries=block_summaries,
        reused_langs=reused_langs,
        recomputed_langs=recomputed_langs,
        normalization_by_lang=normalization_by_lang,
        reconciliation_errors=reconciliation_errors,
        injector=injector,
    )
    atomic_write_json(os.path.join(output_dir, "report.json"), report)
    ok = not collector and not reconciliation_errors
    return PipelineResult(ok, report, manifest, collector.issues)


def _preview_wav_info(track, config: Config, lang: str) -> dict:
    """Header-equivalent metadata derived without writing the final WAV yet."""
    frames = int(track.shape[0])
    return {
        "path": f"track_{lang}.wav",
        "frames": frames,
        "sample_rate": config.sample_rate,
        "channels": 1,
        "subtype": "PCM_16",
        "duration_ms": round(frames / config.ms_frame, 3),
    }


def _dropped_report(schedules: dict[str, LanguageSchedule]) -> list[dict]:
    dropped = []
    for lang in sorted(schedules):
        for item in schedules[lang].dropped:
            payload = item.to_dict()
            payload["lang"] = lang
            dropped.append(payload)
    return dropped


def _shift_report(schedules: dict[str, LanguageSchedule]) -> list[dict]:
    shifts = []
    for lang in sorted(schedules):
        for placement in schedules[lang].placements:
            if placement.shift_samples > 0:
                shifts.append(
                    {
                        "lang": lang,
                        "event_id": placement.event_id,
                        "shift_samples": placement.shift_samples,
                        "shift_ms": round(placement.shift_ms, 3),
                        "natural_start_sample": placement.natural_start,
                        "final_start_sample": placement.start,
                    }
                )
    return shifts


def _base_report(manifest: Manifest, config: Config) -> dict:
    return {
        "schema_version": 1,
        "snapshot_version": config.snapshot_version,
        "config_hash": config.content_hash(),
        "manifest_hash": manifest.manifest_hash,
        "sample_rate": config.sample_rate,
        "target_loudness_db": config.target_loudness_db,
    }


def _failure_report(
    manifest: Manifest, config: Config, collector: IssueCollector
) -> dict:
    report = _base_report(manifest, config)
    report.update(
        {
            "status": "failed",
            "errors": collector.to_list(),
            "error_codes": collector.codes,
        }
    )
    return report


def _interrupted_report(
    manifest: Manifest,
    config: Config,
    block_summaries: list[dict],
    reused_langs: list[str],
    recomputed_langs: list[str],
    stop_after_lang: str,
) -> dict:
    report = _base_report(manifest, config)
    report.update(
        {
            "status": "interrupted",
            "stopped_after_lang": stop_after_lang,
            "blocks": block_summaries,
            "reused_langs": reused_langs,
            "recomputed_langs": recomputed_langs,
        }
    )
    return report


def _success_report(
    *,
    manifest: Manifest,
    config: Config,
    schedules: dict[str, LanguageSchedule],
    empty_langs: list[str],
    outputs: list[dict],
    block_summaries: list[dict],
    reused_langs: list[str],
    recomputed_langs: list[str],
    normalization_by_lang: dict[str, dict],
    reconciliation_errors: list[dict],
    injector: FailureInjector | None,
) -> dict:
    loudness_alerts = []
    for lang in sorted(normalization_by_lang):
        info = normalization_by_lang[lang]
        if info.get("gain_clamped") or info.get("out_of_target_intervals"):
            loudness_alerts.append(
                {
                    "lang": lang,
                    "gain_clamped": info.get("gain_clamped", False),
                    "measured_loudness_db": info.get("measured_loudness_db"),
                    "requested_gain_db": info.get("requested_gain_db"),
                    "applied_gain_db": info.get("applied_gain_db"),
                    "out_of_target_intervals": info.get(
                        "out_of_target_intervals", []
                    ),
                }
            )

    report = _base_report(manifest, config)
    report.update(
        {
            "status": "ok" if not reconciliation_errors else "ok_with_warnings",
            "outputs": outputs,
            "blocks": block_summaries,
            "empty_langs": empty_langs,
            "reused_langs": reused_langs,
            "recomputed_langs": recomputed_langs,
            "dropped_events": _dropped_report(schedules),
            "shifts": _shift_report(schedules),
            "loudness": normalization_by_lang,
            "loudness_alerts": loudness_alerts,
            "reconciliation_errors": reconciliation_errors,
            "asset_attempts": dict(injector.attempts) if injector else {},
            "event_count": len(manifest.events),
        }
    )
    return report
