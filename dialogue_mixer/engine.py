"""Project execution engine.

The engine is intentionally deterministic: no wall-clock timestamps are written
to logs, and all timeline decisions are made on integer sample indices.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from . import __version__
from .audio_io import (
    AudioDecodeError,
    decode_audio,
    decode_wav_float32,
    write_wav_float32,
)
from .config import (
    ClipConfig,
    DialogueConfig,
    Issue,
    Project,
    ProjectFatalError,
    find_project_log_path,
    parse_project,
)
from .envelope import (
    DuckEvent,
    build_track_envelopes,
    db_to_linear,
    envelope_stats,
    find_interval_overlaps,
)
from .wsola import WSOLAParams, wsola_stretch


EXIT_SUCCESS = 0
EXIT_FATAL = 2
EXIT_PARTIAL = 3


@dataclass
class RunResult:
    exit_code: int
    status: str
    log_path: Path
    audio_path: Optional[Path] = None
    index_path: Optional[Path] = None
    error_message: str = ""


class Diagnostics:
    def __init__(self, issues: Iterable[Issue] = ()) -> None:
        self.issues: List[Issue] = list(issues)

    def error(self, code: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.issues.append(Issue(code, "error", message, context or {}))

    def warning(
        self, code: str, message: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        self.issues.append(Issue(code, "warning", message, context or {}))

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    def errors(self) -> List[Dict[str, Any]]:
        return [issue.to_dict() for issue in self.issues if issue.severity == "error"]

    def warnings(self) -> List[Dict[str, Any]]:
        return [issue.to_dict() for issue in self.issues if issue.severity == "warning"]

    def error_codes(self) -> List[str]:
        return sorted(
            {
                issue.code
                for issue in self.issues
                if issue.severity == "error"
            }
        )


@dataclass
class StretchTask:
    dialogue: DialogueConfig
    clip: ClipConfig
    start_sample: int
    end_sample: int

    @property
    def target_samples(self) -> int:
        return self.end_sample - self.start_sample


@dataclass
class RenderSegment:
    clip_id: str
    track_id: str
    start_sample: int
    samples: np.ndarray
    clip_gain: float
    source_kind: str
    source_file: str
    source_duration_samples: int
    dialogue_id: Optional[str] = None
    stretch_record: Optional[Dict[str, Any]] = None
    gain_envelope: Optional[np.ndarray] = None
    crossfade_in: List[Dict[str, Any]] = field(default_factory=list)
    crossfade_out: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.gain_envelope is None:
            self.gain_envelope = np.ones(self.samples.size, dtype=np.float64)

    @property
    def end_sample(self) -> int:
        return int(self.start_sample + self.samples.size)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _db_or_none(value: float) -> Optional[float]:
    if value <= 0.0 or not math.isfinite(value):
        return None
    return float(20.0 * np.log10(value))


def _deduplicate(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _default_wsola_params(sample_rate: int) -> WSOLAParams:
    frame_size = 2048 if sample_rate >= 16000 else 512
    return WSOLAParams(
        frame_size=frame_size,
        synthesis_hop=frame_size // 4,
        search_radius=max(32, min(frame_size // 4, int(round(sample_rate * 0.01)))),
    )


def _equal_power_fades(length: int) -> Tuple[np.ndarray, np.ndarray]:
    if length <= 0:
        return np.zeros(0), np.zeros(0)
    if length == 1:
        return np.array([0.0]), np.array([1.0])
    progress = np.arange(length, dtype=np.float64) / float(length - 1)
    fade_out = np.cos(progress * np.pi / 2.0)
    fade_in = np.sin(progress * np.pi / 2.0)
    return fade_out, fade_in


class MixEngine:
    def __init__(
        self,
        project: Project,
        parse_issues: Iterable[Issue],
        log_path: Path,
        audio_override: Optional[Path] = None,
        index_override: Optional[Path] = None,
    ) -> None:
        self.project = project
        self.diag = Diagnostics(parse_issues)
        self.log_path = Path(log_path).expanduser().resolve()
        self.audio_path = (
            Path(audio_override).expanduser().resolve()
            if audio_override is not None
            else project.output.audio_path
        )
        self.index_path = (
            Path(index_override).expanduser().resolve()
            if index_override is not None
            else project.output.index_path
        )
        self.sample_rate = project.sample_rate
        self.total_samples = project.total_samples
        self.stages: List[Dict[str, Any]] = []
        self.stretch_tasks: List[StretchTask] = []
        self.consumed_clip_ids: Set[str] = set()
        self.dropped_dialogue_ids: Set[str] = set()
        self.segments: List[RenderSegment] = []
        self._decoded: Dict[str, Optional[np.ndarray]] = {}
        self._decode_attempted: Set[str] = set()
        self.stretch_records: List[Dict[str, Any]] = []
        self.crossfade_records: List[Dict[str, Any]] = []
        self.overlap_records: List[Dict[str, Any]] = []
        self.duck_event_records: List[Dict[str, Any]] = []
        self.duck_events: List[DuckEvent] = []
        self.duck_envelopes: Dict[str, np.ndarray] = {}
        self.duck_envelope_summary: Dict[str, Dict[str, float]] = {}
        self.loaded_clip_ids: List[str] = []
        self.failed_clip_ids: List[str] = []
        self.normalization: Dict[str, Any] = {}
        self.index_payload: Dict[str, Any] = {}
        self.audio_written = False
        self.index_written = False
        self.output_peak_linear = 0.0
        self.output_sample_count = 0
        self.fatal_override: Optional[int] = None

    def add_stage(
        self,
        name: str,
        status: str,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.stages.append(
            {"stage": name, "status": status, "summary": summary or {}}
        )

    def run(self) -> RunResult:
        try:
            return self._run_impl()
        except Exception as exc:  # Keep partial outputs and always try to log.
            self.diag.error(
                "INTERNAL_ERROR",
                "unhandled engine error: {0}".format(exc),
                {"exception_type": type(exc).__name__},
            )
            self.fatal_override = EXIT_FATAL
            return self._finish()

    def _run_impl(self) -> RunResult:
        if not self._validate_output_paths():
            self.fatal_override = EXIT_FATAL
            return self._finish()

        self._prepare_stretch_tasks()
        self._render_segments()
        self._apply_crossfades()
        self._prepare_ducking()
        mix = self._mix_and_normalize()
        if not self._write_audio(mix):
            self.fatal_override = EXIT_FATAL
            return self._finish()
        if not self._write_index():
            self.fatal_override = EXIT_FATAL
            return self._finish()
        return self._finish()

    def _validate_output_paths(self) -> bool:
        paths = [self.audio_path, self.index_path, self.log_path]
        for left_index, left in enumerate(paths):
            for right in paths[left_index + 1 :]:
                if left == right:
                    self.diag.error(
                        "OUTPUT_PATH_CONFLICT",
                        "audio, index and log outputs must be distinct paths",
                        {"path": str(left)},
                    )
                    self.add_stage("validate_outputs", "failed", {"paths": len(paths)})
                    return False
        self.add_stage(
            "validate_outputs",
            "ok",
            {
                "audio": str(self.audio_path),
                "index": str(self.index_path),
                "log": str(self.log_path),
            },
        )
        return True

    def _clip_by_id(self) -> Dict[str, ClipConfig]:
        return {clip.id: clip for clip in self.project.clips}

    def _dialogue_bounds(self, dialogue: DialogueConfig) -> Tuple[int, int]:
        return (
            int(round(dialogue.start_seconds * self.sample_rate)),
            int(round(dialogue.end_seconds * self.sample_rate)),
        )

    def _prepare_stretch_tasks(self) -> None:
        clips = self._clip_by_id()
        tasks: List[StretchTask] = []
        requested_count = sum(
            1
            for item in self.project.dialogues
            if item.stretch_ratio is not None
        )
        for dialogue in self.project.dialogues:
            if dialogue.stretch_ratio is None:
                continue
            if not dialogue.voice_clip_id:
                self.diag.error(
                    "MISSING_VOICE_CLIP",
                    "dialogue requests time stretching without voice_clip_id",
                    {"dialogue_id": dialogue.id},
                )
                continue
            clip = clips.get(dialogue.voice_clip_id)
            if clip is None:
                self.diag.error(
                    "MISSING_VOICE_CLIP",
                    "dialogue references a voice clip that does not exist",
                    {
                        "dialogue_id": dialogue.id,
                        "voice_clip_id": dialogue.voice_clip_id,
                    },
                )
                continue
            track = self.project.tracks.get(clip.track_id)
            if track is None:
                self.diag.error(
                    "MISSING_TRACK",
                    "voice clip references a track that does not exist",
                    {
                        "dialogue_id": dialogue.id,
                        "clip_id": clip.id,
                        "track_id": clip.track_id,
                    },
                )
                continue
            if track.kind != "dialogue":
                self.diag.error(
                    "INVALID_VOICE_TRACK",
                    "time-stretched dialogue audio must be on a dialogue-kind track",
                    {
                        "dialogue_id": dialogue.id,
                        "clip_id": clip.id,
                        "track_id": clip.track_id,
                        "track_kind": track.kind,
                    },
                )
                continue
            start_sample, end_sample = self._dialogue_bounds(dialogue)
            if end_sample <= start_sample:
                self.diag.error(
                    "INVALID_DIALOGUE_INTERVAL",
                    "dialogue interval is shorter than one sample",
                    {"dialogue_id": dialogue.id},
                )
                continue
            tasks.append(
                StretchTask(
                    dialogue=dialogue,
                    clip=clip,
                    start_sample=start_sample,
                    end_sample=end_sample,
                )
            )

        grouped: Dict[str, List[StretchTask]] = {}
        for task in tasks:
            grouped.setdefault(task.clip.id, []).append(task)

        kept_tasks: List[StretchTask] = []
        for clip_id, group in grouped.items():
            ordered = sorted(
                group,
                key=lambda item: (
                    item.start_sample,
                    item.end_sample,
                    item.dialogue.id,
                ),
            )
            kept_for_clip: List[StretchTask] = []
            for task in ordered:
                conflicting = [
                    previous
                    for previous in kept_for_clip
                    if task.start_sample < previous.end_sample
                    and previous.start_sample < task.end_sample
                ]
                if conflicting:
                    previous = conflicting[0]
                    self.diag.error(
                        "UNMERGEABLE_OVERLAP",
                        "overlapping stretched uses of one voice clip cannot be merged; later dialogue skipped",
                        {
                            "clip_id": clip_id,
                            "dialogue_id": task.dialogue.id,
                            "conflicts_with": previous.dialogue.id,
                        },
                    )
                    self.dropped_dialogue_ids.add(task.dialogue.id)
                    overlap_start = max(task.start_sample, previous.start_sample)
                    overlap_end = min(task.end_sample, previous.end_sample)
                    self.overlap_records.append(
                        {
                            "dialogue_a": previous.dialogue.id,
                            "dialogue_b": task.dialogue.id,
                            "start_sample": int(overlap_start),
                            "end_sample": int(overlap_end),
                            "duration_samples": int(overlap_end - overlap_start),
                            "mergeable": False,
                            "merge_strategy": "not_applied_later_dialogue_skipped",
                            "reason": "same_voice_clip_overlapping_stretch",
                        }
                    )
                else:
                    kept_for_clip.append(task)
            kept_tasks.extend(kept_for_clip)
            if kept_for_clip:
                self.consumed_clip_ids.add(clip_id)

        self.stretch_tasks = sorted(
            kept_tasks,
            key=lambda item: (
                item.start_sample,
                item.end_sample,
                item.dialogue.id,
            ),
        )
        self.add_stage(
            "prepare_time_stretch",
            "partial" if self.dropped_dialogue_ids else "ok",
            {
                "requested": requested_count,
                "accepted": len(self.stretch_tasks),
                "dropped_unmergeable": len(self.dropped_dialogue_ids),
            },
        )

    def _load_clip(self, clip: ClipConfig) -> Optional[np.ndarray]:
        if clip.id in self._decode_attempted:
            return self._decoded.get(clip.id)
        self._decode_attempted.add(clip.id)
        if not clip.path.is_file():
            self.diag.error(
                "MISSING_AUDIO_FILE",
                "audio file does not exist",
                {"clip_id": clip.id, "path": str(clip.path)},
            )
            self.failed_clip_ids.append(clip.id)
            self._decoded[clip.id] = None
            return None
        try:
            samples = decode_audio(clip.path, self.sample_rate)
        except AudioDecodeError as exc:
            self.diag.error(
                "AUDIO_DECODE_ERROR",
                "failed to decode audio: {0}".format(exc),
                {"clip_id": clip.id, "path": str(clip.path)},
            )
            self.failed_clip_ids.append(clip.id)
            self._decoded[clip.id] = None
            return None
        self._decoded[clip.id] = samples
        self.loaded_clip_ids.append(clip.id)
        return samples

    def _render_segments(self) -> None:
        for clip in self.project.clips:
            if clip.id in self.consumed_clip_ids:
                continue
            if clip.track_id not in self.project.tracks:
                self.diag.error(
                    "MISSING_TRACK",
                    "clip references a track that does not exist; clip skipped",
                    {"clip_id": clip.id, "track_id": clip.track_id},
                )
                continue
            samples = self._load_clip(clip)
            if samples is None:
                continue
            start_sample = int(round(clip.start_seconds * self.sample_rate))
            if start_sample >= self.total_samples:
                self.diag.warning(
                    "CLIP_OUTSIDE_TIMELINE",
                    "clip starts outside the project duration and was skipped",
                    {
                        "clip_id": clip.id,
                        "start_sample": start_sample,
                        "total_samples": self.total_samples,
                    },
                )
                continue
            source_duration = int(samples.size)
            available = self.total_samples - start_sample
            render_count = min(source_duration, available)
            if render_count <= 0:
                continue
            if render_count < source_duration:
                self.diag.warning(
                    "CLIP_TRUNCATED",
                    "clip was truncated at the project timeline end",
                    {
                        "clip_id": clip.id,
                        "source_samples": source_duration,
                        "rendered_samples": render_count,
                    },
                )
            self.segments.append(
                RenderSegment(
                    clip_id=clip.id,
                    track_id=clip.track_id,
                    start_sample=start_sample,
                    samples=np.asarray(samples[:render_count], dtype=np.float64),
                    clip_gain=clip.gain,
                    source_kind="original",
                    source_file=clip.file_name,
                    source_duration_samples=source_duration,
                )
            )

        params = _default_wsola_params(self.sample_rate)
        for task in self.stretch_tasks:
            samples = self._load_clip(task.clip)
            if samples is None:
                continue
            source_duration = int(samples.size)
            requested_ratio = float(task.dialogue.stretch_ratio or 1.0)
            target_samples = task.target_samples
            expected_target = int(round(source_duration * requested_ratio))
            requested_alignment_error = abs(expected_target - target_samples)
            if requested_alignment_error > 1:
                self.diag.warning(
                    "STRETCH_RATIO_ADJUSTED",
                    "dialogue interval overrides the requested stretch ratio",
                    {
                        "dialogue_id": task.dialogue.id,
                        "requested_ratio": requested_ratio,
                        "expected_samples": expected_target,
                        "interval_samples": target_samples,
                        "difference_samples": int(requested_alignment_error),
                    },
                )

            nominal_source_start = int(
                round(task.clip.start_seconds * self.sample_rate)
            )
            if abs(nominal_source_start - task.start_sample) > 1:
                self.diag.warning(
                    "VOICE_CLIP_START_OVERRIDDEN",
                    "dialogue interval start places the stretched voice segment",
                    {
                        "dialogue_id": task.dialogue.id,
                        "clip_id": task.clip.id,
                        "clip_start_sample": nominal_source_start,
                        "dialogue_start_sample": task.start_sample,
                    },
                )

            algorithm = "none"
            transform_applied = target_samples != source_duration
            if transform_applied:
                try:
                    stretched = wsola_stretch(
                        samples,
                        target_samples,
                        self.sample_rate,
                        params=params,
                    )
                except (ValueError, FloatingPointError) as exc:
                    self.diag.error(
                        "TIME_STRETCH_FAILED",
                        "WSOLA time stretch failed; dialogue audio skipped",
                        {"dialogue_id": task.dialogue.id, "error": str(exc)},
                    )
                    continue
                algorithm = "wsola"
            else:
                stretched = np.asarray(samples, dtype=np.float64).copy()
            if stretched.size != target_samples:
                self.diag.error(
                    "TIME_STRETCH_LENGTH_MISMATCH",
                    "time-stretched audio length does not match the dialogue interval",
                    {
                        "dialogue_id": task.dialogue.id,
                        "expected_samples": target_samples,
                        "actual_samples": int(stretched.size),
                    },
                )
                continue

            actual_ratio = (
                float(target_samples) / float(source_duration)
                if source_duration
                else 0.0
            )
            record = {
                "dialogue_id": task.dialogue.id,
                "clip_id": task.clip.id,
                "requested_ratio": requested_ratio,
                "actual_ratio": actual_ratio,
                "source_samples": source_duration,
                "target_samples": target_samples,
                "source_seconds": source_duration / self.sample_rate,
                "target_seconds": target_samples / self.sample_rate,
                "algorithm": algorithm,
                "parameters": params.to_dict() if transform_applied else None,
                "transform_applied": transform_applied,
                "alignment_error_samples": 0,
            }
            self.stretch_records.append(record)
            self.segments.append(
                RenderSegment(
                    clip_id=task.clip.id,
                    track_id=task.clip.track_id,
                    start_sample=task.start_sample,
                    samples=stretched,
                    clip_gain=task.clip.gain,
                    source_kind="stretched" if transform_applied else "original",
                    source_file=task.clip.file_name,
                    source_duration_samples=source_duration,
                    dialogue_id=task.dialogue.id,
                    stretch_record=record,
                )
            )

        self.add_stage(
            "render_segments",
            "partial" if self.failed_clip_ids else "ok",
            {
                "clips_total": len(self.project.clips),
                "clips_loaded": len(self.loaded_clip_ids),
                "clips_failed": len(self.failed_clip_ids),
                "segments_rendered": len(self.segments),
                "stretched_segments": len(self.stretch_records),
            },
        )

    def _apply_crossfades(self) -> None:
        by_clip: Dict[str, List[RenderSegment]] = {}
        for segment in self.segments:
            by_clip.setdefault(segment.clip_id, []).append(segment)
        regions_by_track: Dict[str, List[Tuple[int, int, str, str]]] = {}
        applied = 0
        for transition in self.project.crossfades:
            loaded: List[RenderSegment] = []
            for clip_id in (transition.from_clip_id, transition.to_clip_id):
                matches = by_clip.get(clip_id, [])
                if len(matches) != 1:
                    self.diag.error(
                        "CROSSFADE_CLIP_MISSING",
                        "crossfade requires exactly one rendered clip on each side",
                        {
                            "from_clip_id": transition.from_clip_id,
                            "to_clip_id": transition.to_clip_id,
                            "clip_id": clip_id,
                            "matches": len(matches),
                        },
                    )
                    loaded = []
                    break
                loaded.append(matches[0])
            if len(loaded) != 2:
                continue
            outgoing, incoming = loaded
            if outgoing.track_id != incoming.track_id:
                self.diag.error(
                    "CROSSFADE_TRACK_MISMATCH",
                    "crossfade clips must be on the same track",
                    {
                        "from_clip_id": outgoing.clip_id,
                        "to_clip_id": incoming.clip_id,
                    },
                )
                continue
            track = self.project.tracks[outgoing.track_id]
            if track.kind != "music":
                self.diag.error(
                    "CROSSFADE_NOT_MUSIC",
                    "crossfade is only supported between music clips",
                    {
                        "track_id": track.id,
                        "track_kind": track.kind,
                    },
                )
                continue
            fade_samples = int(
                round(transition.duration_seconds * self.sample_rate)
            )
            if fade_samples <= 0:
                self.diag.error(
                    "INVALID_CROSSFADE",
                    "crossfade duration rounds to zero samples",
                    {
                        "from_clip_id": outgoing.clip_id,
                        "to_clip_id": incoming.clip_id,
                    },
                )
                continue
            region_start = incoming.start_sample
            region_end = region_start + fade_samples
            if (
                region_start < outgoing.start_sample
                or region_end > outgoing.end_sample
                or region_end > incoming.end_sample
            ):
                self.diag.error(
                    "CROSSFADE_REGION_INVALID",
                    "crossfade region must overlap the outgoing tail and incoming head",
                    {
                        "from_clip_id": outgoing.clip_id,
                        "to_clip_id": incoming.clip_id,
                        "region_start_sample": region_start,
                        "region_end_sample": region_end,
                        "outgoing_end_sample": outgoing.end_sample,
                        "incoming_end_sample": incoming.end_sample,
                    },
                )
                continue

            conflict = None
            for previous in regions_by_track.get(outgoing.track_id, []):
                if region_start < previous[1] and previous[0] < region_end:
                    conflict = previous
                    break
            if conflict is not None:
                self.diag.error(
                    "UNMERGEABLE_OVERLAP",
                    "crossfade regions overlap ambiguously; later crossfade skipped",
                    {
                        "from_clip_id": outgoing.clip_id,
                        "to_clip_id": incoming.clip_id,
                        "conflicts_with": {
                            "from_clip_id": conflict[2],
                            "to_clip_id": conflict[3],
                        },
                    },
                )
                continue

            fade_out, fade_in = _equal_power_fades(fade_samples)
            outgoing_offset = region_start - outgoing.start_sample
            outgoing.gain_envelope[
                outgoing_offset : outgoing_offset + fade_samples
            ] *= fade_out
            incoming.gain_envelope[:fade_samples] *= fade_in
            metadata = {
                "from_clip_id": outgoing.clip_id,
                "to_clip_id": incoming.clip_id,
                "track_id": outgoing.track_id,
                "start_sample": int(region_start),
                "end_sample": int(region_end),
                "duration_samples": int(fade_samples),
                "curve": "equal_power",
                "fade_out": "cos(pi/2 * p)",
                "fade_in": "sin(pi/2 * p)",
                "independent_of_ducking": True,
            }
            outgoing.crossfade_out.append(dict(metadata))
            incoming.crossfade_in.append(dict(metadata))
            regions_by_track.setdefault(outgoing.track_id, []).append(
                (
                    region_start,
                    region_end,
                    outgoing.clip_id,
                    incoming.clip_id,
                )
            )
            self.crossfade_records.append(metadata)
            applied += 1

        self.add_stage(
            "crossfades",
            "partial"
            if len(self.crossfade_records) < len(self.project.crossfades)
            else "ok",
            {
                "requested": len(self.project.crossfades),
                "applied": applied,
                "curve": "equal_power",
                "ducking_independent": True,
            },
        )

    def _prepare_ducking(self) -> None:
        kept_dialogues = [
            dialogue
            for dialogue in self.project.dialogues
            if dialogue.id not in self.dropped_dialogue_ids
        ]
        dialogue_by_id = {dialogue.id: dialogue for dialogue in kept_dialogues}
        intervals = []
        for dialogue in kept_dialogues:
            start_sample, end_sample = self._dialogue_bounds(dialogue)
            intervals.append((dialogue.id, start_sample, end_sample))

        for overlap in find_interval_overlaps(intervals):
            left = dialogue_by_id[str(overlap["dialogue_a"])]
            right = dialogue_by_id[str(overlap["dialogue_b"])]
            common = sorted(
                set(left.duck_tracks).intersection(set(right.duck_tracks))
            )
            self.overlap_records.append(
                {
                    "dialogue_a": overlap["dialogue_a"],
                    "dialogue_b": overlap["dialogue_b"],
                    "start_sample": overlap["start_sample"],
                    "end_sample": overlap["end_sample"],
                    "duration_samples": overlap["duration_samples"],
                    "mergeable": True,
                    "merge_strategy": "per_sample_minimum",
                    "common_duck_tracks": common,
                }
            )

        if not self.project.ducking.enabled:
            self.add_stage(
                "ducking",
                "skipped",
                {
                    "enabled": False,
                    "event_count": 0,
                    "overlap_count": len(self.overlap_records),
                },
            )
            return

        events: List[DuckEvent] = []
        for dialogue in kept_dialogues:
            start_sample, end_sample = self._dialogue_bounds(dialogue)
            targets = _deduplicate(dialogue.duck_tracks)
            for target in targets:
                if target not in self.project.tracks:
                    self.diag.error(
                        "MISSING_DUCK_TARGET",
                        "dialogue duck target track does not exist; target skipped",
                        {
                            "dialogue_id": dialogue.id,
                            "target_track": target,
                        },
                    )
                    continue
                depth_db = (
                    dialogue.depth_db
                    if dialogue.depth_db is not None
                    else self.project.ducking.depth_db
                )
                attack_seconds = (
                    dialogue.attack_seconds
                    if dialogue.attack_seconds is not None
                    else self.project.ducking.attack_seconds
                )
                release_seconds = (
                    dialogue.release_seconds
                    if dialogue.release_seconds is not None
                    else self.project.ducking.release_seconds
                )
                attack_curve = (
                    dialogue.attack_curve
                    if dialogue.attack_curve is not None
                    else self.project.ducking.attack_curve
                )
                release_curve = (
                    dialogue.release_curve
                    if dialogue.release_curve is not None
                    else self.project.ducking.release_curve
                )
                event = DuckEvent(
                    dialogue_id=dialogue.id,
                    target_track=target,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    attack_samples=int(round(attack_seconds * self.sample_rate)),
                    release_samples=int(round(release_seconds * self.sample_rate)),
                    depth_db=float(depth_db),
                    attack_curve=str(attack_curve),
                    release_curve=str(release_curve),
                )
                events.append(event)
                self.duck_event_records.append(
                    {
                        "dialogue_id": dialogue.id,
                        "target_track": target,
                        "start_sample": int(start_sample),
                        "end_sample": int(end_sample),
                        "start_seconds": start_sample / self.sample_rate,
                        "end_seconds": end_sample / self.sample_rate,
                        "attack_samples": int(event.attack_samples),
                        "release_samples": int(event.release_samples),
                        "attack_seconds": event.attack_samples / self.sample_rate,
                        "release_seconds": event.release_samples / self.sample_rate,
                        "depth_db": float(event.depth_db),
                        "attack_curve": event.attack_curve,
                        "release_curve": event.release_curve,
                    }
                )

        target_names: List[str] = []
        for track_id in self.project.tracks:
            if any(event.target_track == track_id for event in events):
                target_names.append(track_id)
        self.duck_events = events
        self.duck_envelopes = build_track_envelopes(
            events, target_names, self.total_samples
        )
        for track_name, envelope in self.duck_envelopes.items():
            self.duck_envelope_summary[track_name] = envelope_stats(envelope)

        curves = set()
        for event in events:
            curves.add(event.attack_curve)
            curves.add(event.release_curve)
        self.add_stage(
            "ducking",
            "partial" if self.diag.has_errors else "ok",
            {
                "enabled": True,
                "event_count": len(events),
                "dialogue_count": len(kept_dialogues),
                "overlap_count": len(self.overlap_records),
                "merge_rule": "per_sample_minimum",
                "attack_release_curves": sorted(curves),
            },
        )

    def _mix_and_normalize(self) -> np.ndarray:
        track_buffers = {
            track_id: np.zeros(self.total_samples, dtype=np.float64)
            for track_id in self.project.tracks
        }
        for segment in self.segments:
            if segment.track_id not in track_buffers:
                continue
            end_sample = min(self.total_samples, segment.end_sample)
            usable = end_sample - segment.start_sample
            if usable <= 0:
                continue
            segment_gain = (
                segment.samples[:usable]
                * float(segment.clip_gain)
                * segment.gain_envelope[:usable]
            )
            track_buffers[segment.track_id][
                segment.start_sample:end_sample
            ] += segment_gain

        mix = np.zeros(self.total_samples, dtype=np.float64)
        for track_id, buffer in track_buffers.items():
            track_gain = float(self.project.tracks[track_id].gain)
            envelope = self.duck_envelopes.get(track_id)
            if envelope is None:
                mix += buffer * track_gain
            else:
                mix += buffer * track_gain * envelope

        if not self.segments:
            self.diag.warning(
                "EMPTY_MIX",
                "no audio segments were rendered; output is silence",
            )

        peak_before = float(np.max(np.abs(mix))) if mix.size else 0.0
        ceiling = min(1.0, db_to_linear(self.project.normalization.peak_ceiling_dbfs))
        factor = 1.0
        normalization_applied = False
        if peak_before > ceiling and peak_before > 0.0:
            factor = ceiling / peak_before
            normalization_applied = True
        normalized = mix * factor
        clip_count = int(np.count_nonzero(np.abs(normalized) > 1.0))
        normalized = np.clip(normalized, -1.0, 1.0)
        peak_after = float(np.max(np.abs(normalized))) if normalized.size else 0.0
        self.normalization = {
            "mode": "prevent_clipping",
            "peak_ceiling_dbfs": float(self.project.normalization.peak_ceiling_dbfs),
            "peak_ceiling_linear": float(ceiling),
            "peak_before_linear": peak_before,
            "peak_before_dbfs": _db_or_none(peak_before),
            "normalization_factor": float(factor),
            "normalization_factor_db": float(20.0 * np.log10(factor))
            if factor > 0.0
            else None,
            "applied": normalization_applied,
            "clipped_samples_after_scaling": clip_count,
            "peak_after_linear": peak_after,
            "peak_after_dbfs": _db_or_none(peak_after),
        }
        self.add_stage(
            "mix_and_normalize",
            "ok",
            {
                "rendered_segments": len(self.segments),
                "tracks_mixed": len(track_buffers),
                "ducking_applied_per_track": sorted(self.duck_envelopes),
                "peak_before_dbfs": self.normalization["peak_before_dbfs"],
                "peak_after_dbfs": self.normalization["peak_after_dbfs"],
                "normalization_factor": factor,
                "normalization_applied": normalization_applied,
            },
        )
        return normalized

    def _write_audio(self, mix: np.ndarray) -> bool:
        try:
            write_wav_float32(self.audio_path, mix, self.sample_rate)
            self.audio_written = True
        except OSError as exc:
            self.diag.error(
                "AUDIO_WRITE_ERROR",
                "failed to write mixed audio: {0}".format(exc),
                {"path": str(self.audio_path)},
            )
            self.add_stage("write_audio", "failed", {"path": str(self.audio_path)})
            return False

        try:
            written, written_rate = decode_wav_float32(self.audio_path)
        except (OSError, AudioDecodeError) as exc:
            self.diag.error(
                "OUTPUT_VERIFICATION_FAILED",
                "could not verify the written mix: {0}".format(exc),
                {"path": str(self.audio_path)},
            )
            self.add_stage("write_audio", "failed", {"path": str(self.audio_path)})
            return False

        self.output_sample_count = int(written.size)
        self.output_peak_linear = (
            float(np.max(np.abs(written))) if written.size else 0.0
        )
        peak_limit = float(self.normalization["peak_ceiling_linear"]) + 1e-7
        verification_ok = (
            written_rate == self.sample_rate
            and written.size == self.total_samples
            and self.output_peak_linear <= peak_limit
        )
        if not verification_ok:
            self.diag.error(
                "OUTPUT_VERIFICATION_FAILED",
                "written WAV failed sample-count, sample-rate or peak verification",
                {
                    "expected_sample_rate": self.sample_rate,
                    "actual_sample_rate": written_rate,
                    "expected_samples": self.total_samples,
                    "actual_samples": int(written.size),
                    "peak_linear": self.output_peak_linear,
                    "peak_limit_linear": peak_limit,
                },
            )
        self.add_stage(
            "write_audio",
            "ok" if verification_ok else "failed",
            {
                "path": str(self.audio_path),
                "sample_rate": written_rate,
                "samples": int(written.size),
                "peak_linear": self.output_peak_linear,
                "peak_dbfs": _db_or_none(self.output_peak_linear),
                "peak_within_ceiling": self.output_peak_linear <= peak_limit,
            },
        )
        return verification_ok

    def _write_index(self) -> bool:
        track_order = {track_id: i for i, track_id in enumerate(self.project.tracks)}
        ordered = sorted(
            self.segments,
            key=lambda segment: (
                track_order.get(segment.track_id, 10**9),
                segment.start_sample,
                segment.dialogue_id or "",
                segment.clip_id,
            ),
        )
        track_segments: Dict[str, List[Dict[str, Any]]] = {
            track_id: [] for track_id in self.project.tracks
        }
        max_alignment_error = 0.0
        for number, segment in enumerate(ordered, start=1):
            segment_id = "seg_{0:04d}".format(number)
            duration_samples = int(segment.samples.size)
            end_sample = int(segment.start_sample + duration_samples)
            actual_start_seconds = segment.start_sample / self.sample_rate
            actual_duration_seconds = duration_samples / self.sample_rate
            actual_end_seconds = end_sample / self.sample_rate
            alignment_error = max(
                abs(actual_start_seconds * self.sample_rate - segment.start_sample),
                abs(
                    actual_duration_seconds * self.sample_rate
                    - duration_samples
                ),
                abs(actual_end_seconds * self.sample_rate - end_sample),
            )
            max_alignment_error = max(max_alignment_error, float(alignment_error))
            track = self.project.tracks[segment.track_id]
            record = {
                "segment_id": segment_id,
                "clip_id": segment.clip_id,
                "track_id": segment.track_id,
                "dialogue_id": segment.dialogue_id,
                "source_kind": segment.source_kind,
                "source_file": segment.source_file,
                "source_duration_samples": int(segment.source_duration_samples),
                "start_sample": int(segment.start_sample),
                "end_sample": end_sample,
                "duration_samples": duration_samples,
                "actual_start_seconds": actual_start_seconds,
                "actual_duration_seconds": actual_duration_seconds,
                "actual_end_seconds": actual_end_seconds,
                "sample_rate": self.sample_rate,
                "clip_gain": float(segment.clip_gain),
                "track_gain": float(track.gain),
                "effective_static_gain": float(segment.clip_gain * track.gain),
                "gain_envelope_minimum": float(np.min(segment.gain_envelope))
                if segment.gain_envelope.size
                else 1.0,
                "crossfade_in": segment.crossfade_in,
                "crossfade_out": segment.crossfade_out,
                "stretch": segment.stretch_record,
                "timeline_alignment_error_samples": float(alignment_error),
            }
            track_segments[segment.track_id].append(record)

        max_alignment_error = max(max_alignment_error, 0.0)
        if max_alignment_error > 1.0:
            self.diag.error(
                "INDEX_ALIGNMENT_ERROR",
                "track index exceeds the one-sample timeline tolerance",
                {"max_alignment_error_samples": max_alignment_error},
            )
        self.index_payload = {
            "schema_version": 1,
            "project": {
                "name": self.project.name,
                "file": str(self.project.source_path),
            },
            "sample_rate": self.sample_rate,
            "channels": 1,
            "sample_format": "float32",
            "total_samples": self.total_samples,
            "total_seconds": self.total_samples / self.sample_rate,
            "tracks": [
                {
                    "id": track_id,
                    "kind": track.kind,
                    "gain": float(track.gain),
                    "segments": track_segments[track_id],
                }
                for track_id, track in self.project.tracks.items()
            ],
            "alignment": {
                "timebase": "integer_samples",
                "tolerance_samples": 1,
                "max_alignment_error_samples": float(max_alignment_error),
                "within_tolerance": max_alignment_error <= 1.0,
            },
        }
        try:
            _atomic_write_json(self.index_path, self.index_payload)
            self.index_written = True
        except (OSError, ValueError) as exc:
            self.diag.error(
                "INDEX_WRITE_ERROR",
                "failed to write track index: {0}".format(exc),
                {"path": str(self.index_path)},
            )
            self.add_stage("write_index", "failed", {"path": str(self.index_path)})
            return False
        self.add_stage(
            "write_index",
            "ok",
            {
                "path": str(self.index_path),
                "segments": len(ordered),
                "max_alignment_error_samples": float(max_alignment_error),
                "within_tolerance": max_alignment_error <= 1.0,
            },
        )
        return True

    def _build_log(self, exit_code: int, status: str) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "tool": {
                "name": "dialogue-mixer",
                "version": __version__,
            },
            "project": {
                "name": self.project.name,
                "file": str(self.project.source_path),
                "sample_rate": self.sample_rate,
                "duration_seconds": self.project.duration_seconds,
                "total_samples": self.total_samples,
            },
            "status": status,
            "exit_code": int(exit_code),
            "counts": {
                "tracks": len(self.project.tracks),
                "clips": len(self.project.clips),
                "dialogues": len(self.project.dialogues),
                "rendered_segments": len(self.segments),
                "time_stretched_segments": len(self.stretch_records),
                "crossfades_applied": len(self.crossfade_records),
                "duck_events": len(self.duck_events),
                "overlap_events": len(self.overlap_records),
                "business_errors": len(self.diag.errors()),
                "warnings": len(self.diag.warnings()),
            },
            "stages": self.stages,
            "time_stretch": {
                "algorithm": "wsola",
                "segments": self.stretch_records,
            },
            "ducking": {
                "enabled": self.project.ducking.enabled,
                "merge_rule": "per_sample_minimum",
                "event_count": len(self.duck_events),
                "events": self.duck_event_records,
                "overlaps": self.overlap_records,
                "per_track_envelopes": self.duck_envelope_summary,
            },
            "crossfades": {
                "count": len(self.crossfade_records),
                "transitions": self.crossfade_records,
                "independent_of_ducking": True,
            },
            "normalization": self.normalization,
            "outputs": {
                "audio": str(self.audio_path),
                "audio_written": self.audio_written,
                "audio_samples": self.output_sample_count,
                "output_peak_linear": self.output_peak_linear,
                "output_peak_dbfs": _db_or_none(self.output_peak_linear),
                "index": str(self.index_path),
                "index_written": self.index_written,
                "index_max_alignment_error_samples": self.index_payload.get(
                    "alignment", {}
                ).get("max_alignment_error_samples"),
                "log": str(self.log_path),
            },
            "errors": self.diag.errors(),
            "warnings": self.diag.warnings(),
            "business_error_codes": self.diag.error_codes(),
        }

    def _finish(self) -> RunResult:
        if self.fatal_override is not None:
            exit_code = self.fatal_override
            status = "failed"
        elif self.diag.has_errors:
            exit_code = EXIT_PARTIAL
            status = "partial"
        else:
            exit_code = EXIT_SUCCESS
            status = "success"
        payload = self._build_log(exit_code, status)
        error_message = ""
        try:
            _atomic_write_json(self.log_path, payload)
        except (OSError, ValueError) as exc:
            error_message = "failed to write log: {0}".format(exc)
            exit_code = EXIT_FATAL
            status = "failed"
        return RunResult(
            exit_code=exit_code,
            status=status,
            log_path=self.log_path,
            audio_path=self.audio_path if self.audio_written else None,
            index_path=self.index_path if self.index_written else None,
            error_message=error_message,
        )


def _write_parse_failure_log(
    log_path: Path,
    project_path: Path,
    error: ProjectFatalError,
) -> str:
    payload = {
        "schema_version": 1,
        "tool": {"name": "dialogue-mixer", "version": __version__},
        "project": {"name": project_path.stem, "file": str(project_path)},
        "status": "failed",
        "exit_code": EXIT_FATAL,
        "counts": {
            "tracks": 0,
            "clips": 0,
            "dialogues": 0,
            "rendered_segments": 0,
            "time_stretched_segments": 0,
            "crossfades_applied": 0,
            "duck_events": 0,
            "overlap_events": 0,
            "business_errors": 1,
            "warnings": 0,
        },
        "stages": [
            {
                "stage": "parse_project",
                "status": "failed",
                "summary": {"path": str(project_path)},
            }
        ],
        "time_stretch": {"algorithm": "wsola", "segments": []},
        "ducking": {
            "enabled": False,
            "merge_rule": "per_sample_minimum",
            "event_count": 0,
            "events": [],
            "overlaps": [],
            "per_track_envelopes": {},
        },
        "crossfades": {
            "count": 0,
            "transitions": [],
            "independent_of_ducking": True,
        },
        "normalization": {},
        "outputs": {"log": str(log_path)},
        "errors": [
            {
                "code": error.code,
                "severity": "error",
                "message": error.message,
                "context": error.context,
            }
        ],
        "warnings": [],
        "business_error_codes": [error.code],
    }
    try:
        _atomic_write_json(log_path, payload)
        return ""
    except (OSError, ValueError) as exc:
        return "failed to write log: {0}".format(exc)


def run_project(
    project_path: Path,
    audio_override: Optional[Path] = None,
    index_override: Optional[Path] = None,
    log_override: Optional[Path] = None,
) -> RunResult:
    """Run a project and always attempt to produce a structured JSON log."""

    project_path = Path(project_path).expanduser().resolve()
    if log_override is not None:
        log_path = Path(log_override).expanduser().resolve()
    else:
        log_path = find_project_log_path(project_path)

    try:
        project, parse_issues = parse_project(project_path)
    except ProjectFatalError as exc:
        error_message = _write_parse_failure_log(log_path, project_path, exc)
        return RunResult(
            exit_code=EXIT_FATAL,
            status="failed",
            log_path=log_path,
            error_message=error_message,
        )

    if log_override is None:
        log_path = project.output.log_path
    engine = MixEngine(
        project=project,
        parse_issues=parse_issues,
        log_path=log_path,
        audio_override=audio_override,
        index_override=index_override,
    )
    engine.add_stage(
        "parse_project",
        "ok",
        {
            "path": str(project.source_path),
            "tracks": len(project.tracks),
            "clips": len(project.clips),
            "dialogues": len(project.dialogues),
            "parse_errors": sum(
                1 for issue in parse_issues if issue.severity == "error"
            ),
            "parse_warnings": sum(
                1 for issue in parse_issues if issue.severity == "warning"
            ),
        },
    )
    return engine.run()

