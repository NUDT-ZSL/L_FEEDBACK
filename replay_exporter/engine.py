"""Chunked render engine with checkpointed resume and traceable gain reports."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from . import checkpoint as cp
from .audioio import (
    SAMPLE_RATE,
    StereoMasterWriter,
    WavData,
)
from .checkpoint import build_checkpoint, load_checkpoint, write_checkpoint_atomic
from .dsp import (
    compute_segment_gain,
    db_to_linear,
    linear_to_db,
    measure_clip,
    render_block,
)
from .errors import ErrorCode, ExporterError
from .locks import PLAN_CACHE, export_lock
from .model import SegmentGainReport, Snapshot, TimelinePlan
from .planner import build_plan, load_segment_inputs, load_sources, verify_length_invariant
from .snapshot import sha256_file
from .snapshot import (
    SNAPSHOT_VERSION,
    build_snapshot,
    load_snapshot,
    reconcile_snapshots,
    sha256_bytes,
    write_snapshot_atomic,
)

MASTER_FILE = "master.wav"
GAIN_REPORT_FILE = "gain_report.json"
ERROR_REPORT_FILE = "error_report.json"


def _write_error(output_dir: Optional[str], error: ExporterError) -> None:
    if not output_dir:
        return
    try:
        os.makedirs(output_dir, exist_ok=True)
        target = os.path.join(output_dir, ERROR_REPORT_FILE)
        temporary = target + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(error.to_dict(), handle, sort_keys=True, indent=2)
        os.replace(temporary, target)
    except OSError:
        pass


def parse_and_plan(timeline_path: str) -> Tuple[TimelinePlan, Dict[int, Tuple[WavData, WavData]]]:
    absolute = os.path.abspath(timeline_path)

    def loader():
        inputs, timeline_json = load_segment_inputs(timeline_path)
        sources = load_sources(inputs)
        plan = build_plan(inputs, timeline_json)
        verify_length_invariant(plan)
        return plan, sources

    # Content-sensitive key: timeline + metadata + WAV file hashes. Any input
    # edit changes the key, so a stale plan can never be served, while repeated
    # exports of unchanged inputs reuse the locked, read-only cached entry.
    inputs, _ = load_segment_inputs(timeline_path)
    key_parts = [absolute, sha256_file(absolute)]
    for item in inputs:
        key_parts.append(sha256_file(item.voice_wav))
        key_parts.append(sha256_file(item.ambience_wav))
    timeline_key = "|".join(key_parts)
    return PLAN_CACHE.get_or_load(timeline_key, loader)


def calculate_gains(
    plan: TimelinePlan,
    sources: Dict[int, Tuple[WavData, WavData]],
) -> Tuple[Dict[int, float], Dict[int, Optional[float]], List[SegmentGainReport]]:
    """Per-segment loudness/peak gain, boundary smoothing chain and report rows."""
    gains: Dict[int, float] = {}
    previous_gains: Dict[int, Optional[float]] = {}
    reports: List[SegmentGainReport] = []
    previous_gain: Optional[float] = None

    for segment in plan.segments:
        voice, ambience = sources[segment.index]
        peak_voice, rms_voice = measure_clip(voice.frames)
        peak_ambience, rms_ambience = measure_clip(ambience.frames)
        # Conservative mix estimate: peaks may add, RMS energy is additive.
        mix_peak = min(1.0, peak_voice + peak_ambience)
        mix_rms = min(1.0, (rms_voice ** 2 + rms_ambience ** 2) ** 0.5)
        applied_db, loudness_db, peak_db = compute_segment_gain(
            mix_peak,
            mix_rms,
            segment.target_peak_dbfs,
            plan.loudness_target_dbfs,
        )
        applied_linear = db_to_linear(applied_db)
        gains[segment.index] = applied_linear
        previous_gains[segment.index] = previous_gain
        previous_gain = applied_linear

        ceiling_linear = db_to_linear(plan.limiter_ceiling_dbfs)
        limiter_active = mix_peak * applied_linear > ceiling_linear
        reports.append(
            SegmentGainReport(
                segment_index=segment.index,
                measured_peak_dbfs=round(linear_to_db(mix_peak), 4),
                measured_rms_dbfs=round(linear_to_db(mix_rms), 4),
                target_peak_dbfs=segment.target_peak_dbfs,
                loudness_gain_db=round(loudness_db, 4),
                peak_gain_db=round(peak_db, 4),
                applied_gain_db=round(applied_db, 4),
                smoothed=previous_gain is not None and plan.gain_smoothing_ms > 0,
                limiter_active=limiter_active,
                limiter_ceiling_dbfs=plan.limiter_ceiling_dbfs,
            )
        )
    return gains, previous_gains, reports


def prepare_snapshot(
    output_dir: str,
    plan: TimelinePlan,
    timeline_path: str,
) -> Tuple[Snapshot, List[dict]]:
    new_snapshot = build_snapshot(plan, timeline_path)
    old_snapshot = load_snapshot(output_dir)
    reconciled, deprecated = reconcile_snapshots(old_snapshot, new_snapshot)
    write_snapshot_atomic(output_dir, reconciled)
    return reconciled, deprecated


def _resume_state(
    output_dir: str,
    snapshot: Snapshot,
    master_path: str,
    writer: StereoMasterWriter,
) -> Tuple[List[int], Dict[str, str], int, List[dict]]:
    """Validate a prior checkpoint against the snapshot and actual PCM bytes."""
    checkpoint = load_checkpoint(output_dir)
    if checkpoint is None:
        return [], {}, 0, []
    if checkpoint.get("version") != snapshot.version:
        raise ExporterError(
            ErrorCode.SNAPSHOT_VERSION_MISMATCH,
            "checkpoint version does not match snapshot; start a fresh export",
            detail={
                "checkpoint_version": checkpoint.get("version"),
                "snapshot_version": snapshot.version,
            },
        )

    block_by_index = {block["index"]: block for block in snapshot.blocks}
    hashes: Dict[str, str] = {}
    reusable: List[int] = []
    rebuilt: List[dict] = []
    for raw_index in checkpoint.get("completed_blocks", []):
        block = block_by_index.get(raw_index)
        if block is None:
            rebuilt.append({"block_index": raw_index, "reason": "not_in_snapshot"})
            continue
        expected = checkpoint["block_output_hashes"].get(str(raw_index))
        if not expected:
            rebuilt.append({"block_index": raw_index, "reason": "missing_output_hash"})
            continue
        ok = cp.verify_persisted_block(
            master_path=master_path,
            block_index=raw_index,
            block_frames=plan_block_frames(snapshot),
            frame_count=block["frame_count"],
            expected_hash=expected,
            read_block=writer.read_block,
            sha256_bytes=sha256_bytes,
        )
        if ok:
            reusable.append(raw_index)
            hashes[str(raw_index)] = expected
        else:
            rebuilt.append({"block_index": raw_index, "reason": "half_written_or_hash_mismatch"})
    return reusable, hashes, len(reusable), rebuilt


def plan_block_frames(snapshot: Snapshot) -> int:
    return int(snapshot.plan["block_frames"])


def render_master(
    timeline_path: str,
    output_dir: str,
    *,
    checkpoint_every: int = 5,
    stop_after_blocks: Optional[int] = None,
) -> Dict[str, Any]:
    """Full export entry point; safe to re-invoke for crash recovery."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        with export_lock(output_dir):
            return _render_locked(
                timeline_path,
                output_dir,
                checkpoint_every=checkpoint_every,
                stop_after_blocks=stop_after_blocks,
            )
    except ExporterError as exc:
        _write_error(output_dir, exc)
        raise


def _render_locked(
    timeline_path: str,
    output_dir: str,
    *,
    checkpoint_every: int,
    stop_after_blocks: Optional[int],
) -> Dict[str, Any]:
    plan, sources = parse_and_plan(timeline_path)
    gains, previous_gains, gain_reports = calculate_gains(plan, sources)
    snapshot, deprecated = prepare_snapshot(output_dir, plan, timeline_path)

    master_path = os.path.join(output_dir, MASTER_FILE)
    fresh_master = not os.path.exists(master_path)
    # Open without recreating a truncated container: reconciliation must see
    # the torn file so its blocks are correctly classified for recomputation.
    writer = StereoMasterWriter(
        master_path, plan.total_frames, create_if_missing=False
    )

    completed, output_hashes, resumed_count, rebuilt = _resume_state(
        output_dir, snapshot, master_path, writer
    )
    writer.initialize_container()
    completed_set = set(completed)
    block_frames = plan.block_frames

    rendered_this_run = 0
    for block in snapshot.blocks:
        index = block["index"]
        if index in completed_set:
            continue
        pcm = render_block(plan, index, sources, gains, previous_gains)
        if len(pcm) != block["frame_count"] * 4:
            raise ExporterError(
                ErrorCode.BLOCK_HASH_MISMATCH,
                f"block {index}: rendered {len(pcm)} bytes, planned {block['frame_count'] * 4}",
                detail={"block_index": index},
            )
        writer.write_block(index, block_frames, pcm)
        output_hashes[str(index)] = sha256_bytes(pcm)
        completed_set.add(index)
        rendered_this_run += 1

        if rendered_this_run % max(1, checkpoint_every) == 0:
            write_checkpoint_atomic(
                output_dir,
                build_checkpoint(
                    version=snapshot.version,
                    completed_blocks=sorted(completed_set),
                    master_data_bytes=writer.data_length_bytes(),
                    block_output_hashes=output_hashes,
                    master_path=master_path,
                    extra={"total_blocks": len(snapshot.blocks)},
                ),
            )

        if stop_after_blocks is not None and rendered_this_run >= stop_after_blocks:
            write_checkpoint_atomic(
                output_dir,
                build_checkpoint(
                    version=snapshot.version,
                    completed_blocks=sorted(completed_set),
                    master_data_bytes=writer.data_length_bytes(),
                    block_output_hashes=output_hashes,
                    master_path=master_path,
                    extra={
                        "total_blocks": len(snapshot.blocks),
                        "interrupted": True,
                    },
                ),
            )
            return {
                "ok": True,
                "interrupted": True,
                "output_dir": os.path.abspath(output_dir),
                "master_path": master_path,
                "completed_blocks": sorted(completed_set),
                "resumed_blocks": resumed_count,
                "rebuilt_blocks": rebuilt,
                "deprecated_blocks": deprecated,
            }

    # Final checkpoint and per-segment gain traceability report.
    write_checkpoint_atomic(
        output_dir,
        build_checkpoint(
            version=snapshot.version,
            completed_blocks=sorted(completed_set),
            master_data_bytes=writer.data_length_bytes(),
            block_output_hashes=output_hashes,
            master_path=master_path,
            extra={"total_blocks": len(snapshot.blocks), "completed": True},
        ),
    )
    with open(os.path.join(output_dir, GAIN_REPORT_FILE), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "version": snapshot.version,
                "sample_rate": SAMPLE_RATE,
                "limiter_ceiling_dbfs": plan.limiter_ceiling_dbfs,
                "segments": [asdict(row) for row in gain_reports],
            },
            handle,
            sort_keys=True,
            indent=2,
        )

    return {
        "ok": True,
        "interrupted": False,
        "output_dir": os.path.abspath(output_dir),
        "master_path": master_path,
        "snapshot_version": snapshot.version,
        "total_blocks": len(snapshot.blocks),
        "completed_blocks": sorted(completed_set),
        "resumed_blocks": resumed_count,
        "rebuilt_blocks": rebuilt,
        "deprecated_blocks": deprecated,
        "fresh_master": fresh_master,
        "gain_report": os.path.join(output_dir, GAIN_REPORT_FILE),
    }
