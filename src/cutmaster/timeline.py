"""Absolute-frame layout and deterministic dialogue ducking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .audio import inspect_source
from .errors import err
from .manifest import Manifest, TrackSpec
from .timing import frames_to_ms, ms_to_frames, resampled_frame_count


@dataclass
class DuckInterval:
    start_frame: int
    end_frame: int
    gain_db: float
    winner_ids: tuple[str, ...]

    def as_dict(self, manifest: Manifest) -> dict[str, object]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_ms": frames_to_ms(self.start_frame, manifest.project.sample_rate),
            "end_ms": frames_to_ms(self.end_frame, manifest.project.sample_rate),
            "applied_gain_db": self.gain_db,
            "winner_track_ids": list(self.winner_ids),
        }


@dataclass
class GainSegment:
    start_frame: int
    end_frame: int
    base_gain_db: float
    duck_gain_db: float
    winners: tuple[str, ...] = ()

    @property
    def total_gain_db(self) -> float:
        return self.base_gain_db + self.duck_gain_db

    def as_dict(self, manifest: Manifest) -> dict[str, object]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_ms": frames_to_ms(self.start_frame, manifest.project.sample_rate),
            "end_ms": frames_to_ms(self.end_frame, manifest.project.sample_rate),
            "base_gain_db": self.base_gain_db,
            "duck_gain_db": self.duck_gain_db,
            "total_gain_db": self.total_gain_db,
            "winner_track_ids": list(self.winners),
        }


@dataclass
class TrackLayout:
    spec: TrackSpec
    start_frame: int
    source_frames: int
    resampled_frames: int
    end_frame: int
    requested_start_ms: float
    start_rounding_error_ms: float
    duck_intervals: list[DuckInterval] = field(default_factory=list)

    @property
    def track_id(self) -> str:
        return self.spec.track_id

    def gain_segments(self) -> list[GainSegment]:
        if self.end_frame <= self.start_frame:
            return []
        breaks = {self.start_frame, self.end_frame}
        for interval in self.duck_intervals:
            breaks.add(max(self.start_frame, interval.start_frame))
            breaks.add(min(self.end_frame, interval.end_frame))
        ordered = sorted(breaks)
        segments: list[GainSegment] = []
        for start, end in zip(ordered, ordered[1:]):
            if end <= start:
                continue
            duck = 0.0
            winners: tuple[str, ...] = ()
            for interval in self.duck_intervals:
                if interval.start_frame <= start and end <= interval.end_frame:
                    duck = interval.gain_db
                    winners = interval.winner_ids
                    break
            segment = GainSegment(
                start_frame=start,
                end_frame=end,
                base_gain_db=self.spec.gain_db,
                duck_gain_db=duck,
                winners=winners,
            )
            if (
                segments
                and segments[-1].end_frame == segment.start_frame
                and segments[-1].base_gain_db == segment.base_gain_db
                and segments[-1].duck_gain_db == segment.duck_gain_db
                and segments[-1].winners == segment.winners
            ):
                segments[-1].end_frame = segment.end_frame
            else:
                segments.append(segment)
        return segments


@dataclass
class TimelineLayout:
    manifest: Manifest
    tracks: list[TrackLayout]
    overlap_segments: list[dict[str, object]]
    total_duck_intervals: int
    max_start_rounding_error_ms: float

    def by_id(self) -> dict[str, TrackLayout]:
        return {track.track_id: track for track in self.tracks}


def _merge_segments(
    segments: Iterable[tuple[int, int, float, tuple[str, ...]]]
) -> tuple[list[tuple[int, int, float, tuple[str, ...]]], int]:
    merged: list[tuple[int, int, float, tuple[str, ...]]] = []
    count = 0
    for start, end, gain, winners in segments:
        if end <= start:
            continue
        count += 1
        if (
            merged
            and merged[-1][1] == start
            and merged[-1][2] == gain
            and merged[-1][3] == winners
        ):
            merged[-1] = (merged[-1][0], end, gain, winners)
        else:
            merged.append((start, end, gain, winners))
    return merged, count


def build_layout(manifest: Manifest) -> TimelineLayout:
    project = manifest.project
    duration = project.duration_frames
    layouts: list[TrackLayout] = []

    for track in manifest.tracks:
        info = inspect_source(track.path, track.sample_rate, track.channels)
        start_frame = ms_to_frames(track.start_ms, project.sample_rate)
        if start_frame >= duration:
            raise err(
                "E_START_OUT_OF_RANGE",
                "track start lies outside the project timeline",
                exit_code=4,
                track_id=track.track_id,
                start_frame=start_frame,
                duration_frames=duration,
            )
        frames = resampled_frame_count(
            info.frames, track.sample_rate, project.sample_rate
        )
        end_frame = start_frame + frames
        if end_frame > duration:
            raise err(
                "E_TRACK_OVERRUN",
                "track extends beyond the project timeline",
                exit_code=4,
                track_id=track.track_id,
                start_frame=start_frame,
                end_frame=end_frame,
                duration_frames=duration,
            )
        requested_ms = float(track.start_ms)
        resolved_ms = frames_to_ms(start_frame, project.sample_rate)
        layouts.append(
            TrackLayout(
                spec=track,
                start_frame=start_frame,
                source_frames=info.frames,
                resampled_frames=frames,
                end_frame=end_frame,
                requested_start_ms=requested_ms,
                start_rounding_error_ms=abs(resolved_ms - requested_ms),
            )
        )

    dialogue = [layout for layout in layouts if layout.spec.is_dialogue]
    boundaries = sorted(
        {bound for layout in dialogue for bound in (layout.start_frame, layout.end_frame)}
    )
    duck_segments_by_track: dict[str, list[tuple[int, int, float, tuple[str, ...]]]] = {
        layout.track_id: [] for layout in dialogue
    }
    overlap_segments: list[dict[str, object]] = []

    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        active = [
            layout
            for layout in dialogue
            if layout.start_frame <= start and layout.end_frame >= end
        ]
        if len(active) < 2:
            continue
        highest = max(layout.spec.priority for layout in active)
        winners = [layout for layout in active if layout.spec.priority == highest]
        if len(winners) != 1:
            raise err(
                "E_DUCK_UNSATISFIABLE",
                "overlapping dialogue tracks share the highest priority",
                exit_code=5,
                start_frame=start,
                end_frame=end,
                track_ids=[layout.track_id for layout in winners],
                priority=highest,
            )
        winner = winners[0]
        losers = [layout for layout in active if layout is not winner]
        winner_ids = (winner.track_id,)
        for layout in losers:
            duck_segments_by_track[layout.track_id].append(
                (
                    start,
                    end,
                    manifest.ducking.duck_db,
                    winner_ids,
                )
            )
        overlap_segments.append(
            {
                "start_frame": start,
                "end_frame": end,
                "start_ms": frames_to_ms(start, project.sample_rate),
                "end_ms": frames_to_ms(end, project.sample_rate),
                "active_track_ids": [layout.track_id for layout in active],
                "winner_track_id": winner.track_id,
                "ducked_track_ids": [layout.track_id for layout in losers],
                "duck_gain_db": manifest.ducking.duck_db,
            }
        )

    total_intervals = 0
    for layout in dialogue:
        merged, count = _merge_segments(duck_segments_by_track[layout.track_id])
        layout.duck_intervals = [
            DuckInterval(
                start_frame=start,
                end_frame=end,
                gain_db=gain,
                winner_ids=winners,
            )
            for start, end, gain, winners in merged
        ]
        total_intervals += count

    max_error_ms = max(
        (layout.start_rounding_error_ms for layout in layouts), default=0.0
    )
    max_error_samples = max_error_ms * project.sample_rate / 1000.0
    if max_error_samples > 1.0 + 1e-9:
        raise err(
            "E_ALIGNMENT_PRECISION",
            "millisecond-to-sample rounding exceeded one sample",
            exit_code=4,
            max_error_samples=max_error_samples,
        )
    return TimelineLayout(
        manifest=manifest,
        tracks=layouts,
        overlap_segments=overlap_segments,
        total_duck_intervals=total_intervals,
        max_start_rounding_error_ms=max_error_ms,
    )
