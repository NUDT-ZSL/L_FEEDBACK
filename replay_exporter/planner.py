"""Timeline planner: input validation plus deterministic frame layout.

Layout rules (all frame math uses a single 48 kHz master clock):

* Segments occupy strictly disjoint master intervals. The gap declared between
  two segments is rendered as digital silence, so adjacent segments never
  overlap at the sample level.
* Each segment owns a fade-in head and fade-out tail. The boundary descriptor
  joins the left fade-out, the (possibly empty) gap and the right fade-in into
  one crossfade region, which the validator checks for energy continuity.
* ``sum(segment frames) + sum(gap frames)`` equals the master duration exactly.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from .audioio import (
    SAMPLE_RATE,
    WavData,
    assert_master_sample_rate,
    read_mono_wav,
)
from .dsp import frames_to_ms, ms_to_frames
from .errors import ErrorCode, ExporterError
from .model import (
    CrossfadeRegion,
    PlannedSegment,
    SegmentInput,
    TimelinePlan,
)

BLOCK_MS = 100


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise ExporterError(
            ErrorCode.FILE_NOT_FOUND,
            f"JSON file not found: {path}",
            detail={"path": path},
        )
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ExporterError(
            ErrorCode.INVALID_JSON,
            f"{path}: malformed JSON at line {exc.lineno} column {exc.colno}",
            offset_ms=None,
            detail={"path": path, "reason": exc.msg},
        ) from exc


def _require_number(payload: Dict[str, Any], key: str, segment_index: int) -> float:
    if key not in payload:
        raise ExporterError(
            ErrorCode.MISSING_METADATA_FIELD,
            f"segment {segment_index}: metadata field '{key}' is required",
            segment_index=segment_index,
            detail={"field": key},
        )
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExporterError(
            ErrorCode.MISSING_METADATA_FIELD,
            f"segment {segment_index}: metadata field '{key}' must be a number",
            segment_index=segment_index,
            detail={"field": key, "value": value},
        )
    return float(value)


def load_segment_inputs(
    timeline_path: str,
) -> Tuple[List[SegmentInput], Dict[str, Any]]:
    """Parse the master timeline JSON and every segment metadata JSON."""
    timeline = _load_json(timeline_path)
    base_dir = os.path.dirname(os.path.abspath(timeline_path))
    segment_entries = timeline.get("segments")
    if not isinstance(segment_entries, list) or not segment_entries:
        raise ExporterError(
            ErrorCode.INVALID_TIMELINE,
            "timeline JSON must contain a non-empty 'segments' array",
            detail={"path": timeline_path},
        )

    gaps = timeline.get("gaps_ms", [])
    if not isinstance(gaps, list):
        raise ExporterError(
            ErrorCode.INVALID_TIMELINE,
            "'gaps_ms' must be an array of per-boundary gap lengths",
            detail={"path": timeline_path},
        )
    if len(gaps) not in (0, len(segment_entries) - 1):
        raise ExporterError(
            ErrorCode.INVALID_TIMELINE,
            f"expected {len(segment_entries) - 1} gap entries, got {len(gaps)}",
            detail={"path": timeline_path},
        )

    inputs: List[SegmentInput] = []
    for index, entry in enumerate(segment_entries):
        if not isinstance(entry, dict) or "metadata" not in entry:
            raise ExporterError(
                ErrorCode.INVALID_TIMELINE,
                f"segment {index}: timeline entry must reference a metadata JSON",
                segment_index=index,
            )
        metadata_path = entry["metadata"]
        if not os.path.isabs(metadata_path):
            metadata_path = os.path.join(base_dir, metadata_path)
        metadata = _load_json(metadata_path)

        duration_ms = _require_number(metadata, "duration_ms", index)
        if duration_ms < 0:
            raise ExporterError(
                ErrorCode.NEGATIVE_DURATION,
                f"segment {index}: negative duration {duration_ms} ms",
                segment_index=index,
                offset_ms=duration_ms,
            )
        fade_ms = _require_number(metadata, "fade_ms", index)
        if fade_ms < 0:
            raise ExporterError(
                ErrorCode.NEGATIVE_DURATION,
                f"segment {index}: negative fade duration {fade_ms} ms",
                segment_index=index,
                offset_ms=fade_ms,
            )
        target_peak = _require_number(metadata, "target_peak_dbfs", index)

        if "dialogue_start_ms" not in metadata or "dialogue_end_ms" not in metadata:
            raise ExporterError(
                ErrorCode.MISSING_DIALOGUE_RANGE,
                f"segment {index}: dialogue_start_ms/dialogue_end_ms are required",
                segment_index=index,
                detail={"metadata": metadata_path},
            )
        dialogue_start = float(metadata["dialogue_start_ms"])
        dialogue_end = float(metadata["dialogue_end_ms"])
        if dialogue_start < 0 or dialogue_end < 0:
            raise ExporterError(
                ErrorCode.INVALID_DIALOGUE_RANGE,
                f"segment {index}: dialogue offsets must be non-negative",
                segment_index=index,
                offset_ms=min(dialogue_start, dialogue_end),
            )
        if dialogue_end < dialogue_start:
            raise ExporterError(
                ErrorCode.INVALID_DIALOGUE_RANGE,
                f"segment {index}: dialogue end {dialogue_end} ms precedes start "
                f"{dialogue_start} ms",
                segment_index=index,
                offset_ms=dialogue_end,
            )
        if dialogue_end > duration_ms:
            raise ExporterError(
                ErrorCode.INVALID_DIALOGUE_RANGE,
                f"segment {index}: dialogue end {dialogue_end} ms exceeds segment "
                f"duration {duration_ms} ms",
                segment_index=index,
                offset_ms=dialogue_end,
            )

        voice_wav = metadata.get("voice_wav")
        ambience_wav = metadata.get("ambience_wav")
        for field_name, value in (("voice_wav", voice_wav), ("ambience_wav", ambience_wav)):
            if not isinstance(value, str) or not value:
                raise ExporterError(
                    ErrorCode.MISSING_METADATA_FIELD,
                    f"segment {index}: '{field_name}' must name a WAV file",
                    segment_index=index,
                    detail={"field": field_name},
                )
            if not os.path.isabs(value):
                set_path = os.path.join(base_dir, value)
                if field_name == "voice_wav":
                    voice_wav = set_path
                else:
                    ambience_wav = set_path

        gap_after_ms = float(gaps[index]) if index < len(gaps) else 0.0
        if gap_after_ms < 0:
            raise ExporterError(
                ErrorCode.NEGATIVE_GAP,
                f"gap after segment {index} is negative: {gap_after_ms} ms",
                segment_index=index,
                offset_ms=gap_after_ms,
            )

        inputs.append(
            SegmentInput(
                index=index,
                event_type=str(metadata.get("event_type", entry.get("event_type", "unknown"))),
                voice_wav=voice_wav,
                ambience_wav=ambience_wav,
                duration_ms=duration_ms,
                dialogue_start_ms=dialogue_start,
                dialogue_end_ms=dialogue_end,
                target_peak_dbfs=target_peak,
                fade_ms=fade_ms,
                gap_after_ms=gap_after_ms,
            )
        )
    return inputs, timeline


def load_sources(inputs: List[SegmentInput]) -> Dict[int, Tuple[WavData, WavData]]:
    """Read both WAV tracks per segment and enforce master-clock compatibility."""
    sources: Dict[int, Tuple[WavData, WavData]] = {}
    for item in inputs:
        voice = read_mono_wav(item.voice_wav, item.index)
        ambience = read_mono_wav(item.ambience_wav, item.index)
        assert_master_sample_rate(voice, ambience, item.index)
        sources[item.index] = (voice, ambience)
    return sources


def build_plan(
    inputs: List[SegmentInput],
    timeline: Dict[str, Any],
    block_ms: int = BLOCK_MS,
) -> TimelinePlan:
    """Convert validated inputs into frame-accurate segment/crossfade layout."""
    sample_rate = SAMPLE_RATE
    block_frames = ms_to_frames(block_ms, sample_rate)

    loudness_target = float(timeline.get("loudness_target_dbfs", -20.0))
    duck = timeline.get("ducking", {})
    duck_gain_db = float(duck.get("gain_db", -12.0))
    duck_attack_ms = float(duck.get("attack_ms", 40.0))
    duck_release_ms = float(duck.get("release_ms", 80.0))
    smoothing_ms = float(timeline.get("gain_smoothing_ms", 50.0))
    if duck_attack_ms < 0 or duck_release_ms < 0 or smoothing_ms < 0:
        raise ExporterError(
            ErrorCode.INVALID_TIMELINE,
            "ducking attack/release and gain smoothing must be non-negative",
            detail={
                "attack_ms": duck_attack_ms,
                "release_ms": duck_release_ms,
                "smoothing_ms": smoothing_ms,
            },
        )

    planned: List[PlannedSegment] = []
    cursor = 0
    for item in inputs:
        frames = ms_to_frames(item.duration_ms, sample_rate)
        if frames <= 0:
            raise ExporterError(
                ErrorCode.NEGATIVE_DURATION,
                f"segment {item.index}: duration resolves to zero frames",
                segment_index=item.index,
                offset_ms=item.duration_ms,
            )
        fade_frames = ms_to_frames(item.fade_ms, sample_rate)
        if 2 * fade_frames > frames:
            raise ExporterError(
                ErrorCode.FADE_LONGER_THAN_SEGMENT,
                f"segment {item.index}: fade in+out ({2 * item.fade_ms} ms) exceeds "
                f"segment duration ({item.duration_ms} ms)",
                segment_index=item.index,
                offset_ms=item.fade_ms,
                detail={"fade_ms": item.fade_ms, "duration_ms": item.duration_ms},
            )
        duck_start = cursor + ms_to_frames(item.dialogue_start_ms, sample_rate)
        duck_end = cursor + ms_to_frames(item.dialogue_end_ms, sample_rate)
        planned.append(
            PlannedSegment(
                index=item.index,
                event_type=item.event_type,
                voice_wav=item.voice_wav,
                ambience_wav=item.ambience_wav,
                start_frame=cursor,
                end_frame=cursor + frames,
                fade_in_frames=fade_frames,
                fade_out_frames=fade_frames,
                duck_start_frame=duck_start,
                duck_end_frame=duck_end,
                dialogue_start_ms=item.dialogue_start_ms,
                dialogue_end_ms=item.dialogue_end_ms,
                target_peak_dbfs=item.target_peak_dbfs,
                fade_ms=item.fade_ms,
                start_ms=frames_to_ms(cursor, sample_rate),
                end_ms=frames_to_ms(cursor + frames, sample_rate),
            )
        )
        cursor += frames
        if item.index < len(inputs) - 1:
            cursor += ms_to_frames(item.gap_after_ms, sample_rate)

    total_frames = cursor

    # Ordering conflict check: intervals must be strictly increasing/disjoint.
    for previous, current in zip(planned, planned[1:]):
        if current.start_frame < previous.end_frame:
            raise ExporterError(
                ErrorCode.SEGMENT_ORDER_CONFLICT,
                f"segment {current.index} starts at frame {current.start_frame} before "
                f"segment {previous.index} ends at frame {previous.end_frame}",
                segment_index=current.index,
                offset_ms=frames_to_ms(current.start_frame, sample_rate),
            )

    crossfades: List[CrossfadeRegion] = []
    for boundary_index, (left, right) in enumerate(zip(planned, planned[1:])):
        fade_region_start = left.end_frame - left.fade_out_frames
        fade_region_end = right.start_frame + right.fade_in_frames
        crossfades.append(
            CrossfadeRegion(
                boundary_index=boundary_index,
                left_segment=left.index,
                right_segment=right.index,
                start_frame=fade_region_start,
                end_frame=fade_region_end,
                fade_frames=max(left.fade_out_frames, right.fade_in_frames),
                gap_frames=right.start_frame - left.end_frame,
                start_ms=frames_to_ms(fade_region_start, sample_rate),
                end_ms=frames_to_ms(fade_region_end, sample_rate),
            )
        )

    return TimelinePlan(
        sample_rate=sample_rate,
        block_frames=block_frames,
        total_frames=total_frames,
        total_ms=frames_to_ms(total_frames, sample_rate),
        loudness_target_dbfs=loudness_target,
        duck_gain_db=duck_gain_db,
        duck_attack_ms=duck_attack_ms,
        duck_release_ms=duck_release_ms,
        gain_smoothing_ms=smoothing_ms,
        limiter_ceiling_dbfs=float(
            timeline.get("limiter_ceiling_dbfs", -1.0)
        ),
        segments=planned,
        crossfades=crossfades,
        inputs=inputs,
    )


def verify_length_invariant(plan: TimelinePlan) -> None:
    """Assert the disjoint-interval length identity used by the renderer."""
    segment_total = sum(segment.frame_count for segment in plan.segments)
    gap_total = sum(crossfade.gap_frames for crossfade in plan.crossfades)
    if segment_total + gap_total != plan.total_frames:
        raise ExporterError(
            ErrorCode.INVALID_TIMELINE,
            "segment frames plus gap frames do not match total timeline length",
            detail={
                "segment_frames": segment_total,
                "gap_frames": gap_total,
                "total_frames": plan.total_frames,
            },
        )
