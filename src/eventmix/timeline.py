"""Timeline schema parsing, anchor fitting, and channel layout validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .errors import ConfigError
from .wav import SPEAKER_MASKS

_STANDARD_ORDER = ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"]
_LEFT_ORDER = ["FL", "BL", "SL", "FLC", "TFL"]
_RIGHT_ORDER = ["FR", "BR", "SR", "FRC", "TFR"]
_MASK_TO_SPEAKER = {value: key for key, value in SPEAKER_MASKS.items()}


@dataclass(frozen=True)
class Anchor:
    source_seconds: float
    program_seconds: float


@dataclass(frozen=True)
class AnchorError:
    source_seconds: float
    expected_program_seconds: float
    predicted_program_seconds: float
    error_seconds: float
    error_ms: float
    error_frames: float


@dataclass(frozen=True)
class AlignmentFit:
    offset_seconds: float
    drift_factor: float
    drift_ppm: float
    anchor_errors: tuple[AnchorError, ...]
    max_abs_error_ms: float
    rms_error_ms: float

    def program_to_source_seconds(self, program_seconds: np.ndarray | float):
        return (np.asarray(program_seconds) - self.offset_seconds) / self.drift_factor

    def source_to_program_seconds(self, source_seconds: float) -> float:
        return self.offset_seconds + self.drift_factor * source_seconds


@dataclass(frozen=True)
class SourceRoute:
    channel: int
    gain: float = 1.0


@dataclass(frozen=True)
class ChannelSlot:
    index: int
    name: str
    track_id: str
    speaker: str
    mask_bit: int
    sources: tuple[SourceRoute, ...] | None


@dataclass(frozen=True)
class TrackSpec:
    id: str
    name: str
    role: str
    path: Path
    anchors: tuple[Anchor, ...]
    alignment: AlignmentFit
    slots: tuple[int, ...]


@dataclass(frozen=True)
class CueSpec:
    id: str
    name: str
    program_seconds: float
    frame: int
    wav_cue_id: int


@dataclass(frozen=True)
class ProgramSpec:
    sample_rate: int
    duration_seconds: float
    total_frames: int
    bit_depth: int
    channels: int
    channel_mask: int


@dataclass(frozen=True)
class Policy:
    on_track_error: str = "silence"
    silence_threshold_dbfs: float = -80.0
    min_silence_seconds: float = 0.35
    max_anchor_error_ms: float = 10.0


@dataclass(frozen=True)
class ProjectSpec:
    timeline_path: Path
    event: Mapping[str, Any]
    program: ProgramSpec
    tracks: Mapping[str, TrackSpec]
    channel_slots: tuple[ChannelSlot, ...]
    cues: tuple[CueSpec, ...]
    policy: Policy
    raw: Mapping[str, Any]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_object(path: Path | str, *, what: str = "JSON") -> dict[str, Any]:
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except FileNotFoundError as exc:
        raise ConfigError(f"{what} file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid {what} at {path}:{exc.lineno}:{exc.colno}: {exc.msg}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read {what} file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{what} root must be a JSON object")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{field} must be finite")
    return result


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field} must be an integer")
    return value


def _round_frames(seconds: float, sample_rate: int) -> int:
    return int(math.floor(seconds * sample_rate + 0.5))


def fit_alignment(anchors: Iterable[Anchor], target_sample_rate: int) -> AlignmentFit:
    anchor_list = sorted(anchors, key=lambda item: item.source_seconds)
    if not anchor_list:
        raise ConfigError("track must provide at least one anchor")
    x = np.asarray([item.source_seconds for item in anchor_list], dtype=np.float64)
    y = np.asarray([item.program_seconds for item in anchor_list], dtype=np.float64)
    if np.any(np.diff(x) <= 0):
        raise ConfigError("anchor source times must be strictly increasing")
    if len(anchor_list) == 1:
        drift = 1.0
        offset = float(y[0] - x[0])
    else:
        xm = float(np.mean(x))
        ym = float(np.mean(y))
        denominator = float(np.sum((x - xm) ** 2))
        if denominator <= 0:
            raise ConfigError("anchor source times do not span a positive interval")
        drift = float(np.sum((x - xm) * (y - ym)) / denominator)
        offset = float(ym - drift * xm)
    if not math.isfinite(offset) or not math.isfinite(drift):
        raise ConfigError("anchor fit produced a non-finite parameter")
    if drift <= 0:
        raise ConfigError("anchor fit produced a non-positive drift factor")
    if abs(drift - 1.0) > 0.1:
        raise ConfigError(
            f"anchor drift factor {drift:.9f} is outside the supported ±10% range"
        )

    errors: list[AnchorError] = []
    for anchor in anchor_list:
        predicted = offset + drift * anchor.source_seconds
        error_seconds = predicted - anchor.program_seconds
        errors.append(
            AnchorError(
                source_seconds=anchor.source_seconds,
                expected_program_seconds=anchor.program_seconds,
                predicted_program_seconds=predicted,
                error_seconds=error_seconds,
                error_ms=error_seconds * 1000.0,
                error_frames=error_seconds * target_sample_rate,
            )
        )
    error_ms = np.asarray([item.error_ms for item in errors], dtype=np.float64)
    return AlignmentFit(
        offset_seconds=offset,
        drift_factor=drift,
        drift_ppm=(drift - 1.0) * 1_000_000.0,
        anchor_errors=tuple(errors),
        max_abs_error_ms=float(np.max(np.abs(error_ms))),
        rms_error_ms=float(np.sqrt(np.mean(error_ms**2))),
    )


def _parse_anchors(raw: Any, track_id: str) -> tuple[Anchor, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"tracks[{track_id}].anchors must be a non-empty array")
    anchors: list[Anchor] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ConfigError(f"tracks[{track_id}].anchors[{index}] must be an object")
        source_value = item.get("source_seconds", item.get("track_seconds"))
        program_value = item.get("program_seconds")
        if source_value is None or program_value is None:
            raise ConfigError(
                f"tracks[{track_id}].anchors[{index}] needs source_seconds "
                "(or track_seconds) and program_seconds"
            )
        source_seconds = _number(source_value, f"tracks[{track_id}].anchors[{index}].source_seconds")
        program_seconds = _number(program_value, f"tracks[{track_id}].anchors[{index}].program_seconds")
        if source_seconds < 0:
            raise ConfigError("anchor source_seconds cannot be negative")
        if program_seconds < 0:
            raise ConfigError("anchor program_seconds cannot be negative")
        anchors.append(Anchor(source_seconds, program_seconds))
    return tuple(anchors)


def _parse_sources(raw: Any, field: str) -> tuple[SourceRoute, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{field} must be a non-empty array when present")
    routes: list[SourceRoute] = []
    for index, item in enumerate(raw):
        if isinstance(item, int) and not isinstance(item, bool):
            channel = item
            gain = 1.0
        elif isinstance(item, dict):
            channel = _integer(item.get("channel"), f"{field}[{index}].channel")
            gain = _number(item.get("gain", 1.0), f"{field}[{index}].gain")
        else:
            raise ConfigError(f"{field}[{index}] must be an integer or object")
        if channel < 0:
            raise ConfigError(f"{field}[{index}].channel cannot be negative")
        if not math.isfinite(gain):
            raise ConfigError(f"{field}[{index}].gain must be finite")
        routes.append(SourceRoute(channel=channel, gain=gain))
    return tuple(routes)


def _suffix_speaker(name: str) -> str | None:
    token = name.replace("-", ".").replace("_", ".").split(".")[-1].strip().upper()
    if token in SPEAKER_MASKS:
        return token
    if token in {"L", "LEFT"}:
        return "L"
    if token in {"R", "RIGHT"}:
        return "R"
    if token in {"C", "CENTER", "CENTRE", "MONO"}:
        return "FC"
    if token == "LFE":
        return "LFE"
    return None


def _resolve_speakers(entries: list[dict[str, Any]]) -> tuple[list[str], int]:
    speakers: list[str] = []
    left_index = 0
    right_index = 0
    for index, entry in enumerate(entries):
        explicit = entry.get("speaker")
        if explicit is not None:
            if not isinstance(explicit, str):
                raise ConfigError(f"channel_layout[{index}].speaker must be a string")
            speaker = explicit.upper()
            if speaker not in SPEAKER_MASKS:
                raise ConfigError(
                    f"channel_layout[{index}].speaker {explicit!r} is not a WAVE speaker id"
                )
        else:
            inferred = _suffix_speaker(str(entry["name"]))
            if inferred == "L":
                if left_index >= len(_LEFT_ORDER):
                    raise ConfigError("too many left channel groups to infer a speaker mask")
                speaker = _LEFT_ORDER[left_index]
                left_index += 1
            elif inferred == "R":
                if right_index >= len(_RIGHT_ORDER):
                    raise ConfigError("too many right channel groups to infer a speaker mask")
                speaker = _RIGHT_ORDER[right_index]
                right_index += 1
            elif inferred is not None:
                speaker = inferred
            elif index < len(_STANDARD_ORDER):
                speaker = _STANDARD_ORDER[index]
            else:
                raise ConfigError(
                    f"cannot infer speaker id for channel_layout[{index}]; "
                    "set the speaker field explicitly"
                )
        speakers.append(speaker)
    mask = 0
    for speaker in speakers:
        mask |= SPEAKER_MASKS[speaker]
    return speakers, mask


def _parse_channel_layout(
    raw: Any,
    tracks: Mapping[str, TrackSpec],
) -> tuple[tuple[ChannelSlot, ...], dict[str, tuple[int, ...]], int]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError("channel_layout must be a non-empty array")
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ConfigError(f"channel_layout[{index}] must be an object")
        name = item.get("name")
        track_id = item.get("track", item.get("track_id"))
        if not isinstance(name, str) or not name:
            raise ConfigError(f"channel_layout[{index}].name must be a non-empty string")
        if not isinstance(track_id, str) or track_id not in tracks:
            raise ConfigError(
                f"channel_layout[{index}] references unknown track {track_id!r}"
            )
        entries.append(item)
    speakers, mask = _resolve_speakers(entries)
    slots: list[ChannelSlot] = []
    track_slots: dict[str, list[int]] = {track_id: [] for track_id in tracks}
    for index, (entry, speaker) in enumerate(zip(entries, speakers)):
        track_id = str(entry.get("track", entry.get("track_id")))
        routes = _parse_sources(entry.get("sources"), f"channel_layout[{index}].sources")
        slot = ChannelSlot(
            index=index,
            name=str(entry["name"]),
            track_id=track_id,
            speaker=speaker,
            mask_bit=SPEAKER_MASKS[speaker],
            sources=routes,
        )
        slots.append(slot)
        track_slots[track_id].append(index)
    for track_id, indices in track_slots.items():
        if not indices:
            raise ConfigError(f"track {track_id!r} has no destination channels")
    if len({slot.mask_bit for slot in slots}) != len(slots):
        raise ConfigError("channel_layout assigns the same WAVE speaker bit more than once")
    return tuple(slots), {track_id: tuple(v) for track_id, v in track_slots.items()}, mask


def _parse_policy(raw: Any) -> Policy:
    if raw is None:
        return Policy()
    if not isinstance(raw, dict):
        raise ConfigError("policy must be an object")
    on_track_error = raw.get("on_track_error", "silence")
    if on_track_error not in {"silence", "fail"}:
        raise ConfigError("policy.on_track_error must be 'silence' or 'fail'")
    threshold = _number(
        raw.get("silence_threshold_dbfs", -80.0), "policy.silence_threshold_dbfs"
    )
    if threshold > 0:
        raise ConfigError("policy.silence_threshold_dbfs must be <= 0")
    min_silence = _number(
        raw.get("min_silence_seconds", 0.35), "policy.min_silence_seconds"
    )
    max_anchor_error = _number(
        raw.get("max_anchor_error_ms", 10.0), "policy.max_anchor_error_ms"
    )
    if min_silence < 0 or max_anchor_error < 0:
        raise ConfigError("policy durations and tolerances cannot be negative")
    return Policy(
        on_track_error=on_track_error,
        silence_threshold_dbfs=threshold,
        min_silence_seconds=min_silence,
        max_anchor_error_ms=max_anchor_error,
    )


def parse_project(path: Path | str) -> ProjectSpec:
    timeline_path = Path(path).resolve()
    raw = load_json_object(timeline_path, what="timeline")
    schema = raw.get("schema_version")
    if schema not in (None, "eventmix-timeline/v1"):
        raise ConfigError(f"unsupported timeline schema_version: {schema!r}")

    program_raw = raw.get("program")
    if not isinstance(program_raw, dict):
        raise ConfigError("program must be an object")
    sample_rate = _integer(program_raw.get("sample_rate"), "program.sample_rate")
    if sample_rate < 8000 or sample_rate > 768000:
        raise ConfigError("program.sample_rate must be between 8000 and 768000")
    bit_depth = _integer(program_raw.get("bit_depth", 24), "program.bit_depth")
    if bit_depth not in (16, 24, 32):
        raise ConfigError("program.bit_depth must be 16, 24, or 32")
    duration_seconds = _number(
        program_raw.get("duration_seconds"), "program.duration_seconds"
    )
    if duration_seconds <= 0:
        raise ConfigError("program.duration_seconds must be positive")
    total_frames = _round_frames(duration_seconds, sample_rate)
    if total_frames < 1:
        raise ConfigError("program duration rounds to zero frames")

    tracks_raw = raw.get("tracks")
    if not isinstance(tracks_raw, list) or not tracks_raw:
        raise ConfigError("tracks must be a non-empty array")
    partial_tracks: dict[str, Any] = {}
    anchors_by_track: dict[str, tuple[Anchor, ...]] = {}
    for index, item in enumerate(tracks_raw):
        if not isinstance(item, dict):
            raise ConfigError(f"tracks[{index}] must be an object")
        track_id = item.get("id")
        if not isinstance(track_id, str) or not track_id:
            raise ConfigError(f"tracks[{index}].id must be a non-empty string")
        if track_id in partial_tracks:
            raise ConfigError(f"duplicate track id: {track_id}")
        rel_path = item.get("path")
        if not isinstance(rel_path, str) or not rel_path:
            raise ConfigError(f"tracks[{track_id}].path must be a non-empty string")
        input_path = (timeline_path.parent / rel_path).resolve()
        anchors = _parse_anchors(item.get("anchors"), track_id)
        partial_tracks[track_id] = (item, input_path, anchors)
        anchors_by_track[track_id] = anchors

    tracks: dict[str, TrackSpec] = {}
    for track_id, (item, input_path, anchors) in partial_tracks.items():
        fit = fit_alignment(anchors, sample_rate)
        tracks[track_id] = TrackSpec(
            id=track_id,
            name=str(item.get("name", track_id)),
            role=str(item.get("role", "commentary")),
            path=input_path,
            anchors=anchors,
            alignment=fit,
            slots=(),
        )

    slots, slot_map, inferred_mask = _parse_channel_layout(
        raw.get("channel_layout"), tracks
    )
    if inferred_mask == 0:
        raise ConfigError("channel_layout derived an empty WAVE channel mask")
    override_mask = program_raw.get("channel_mask")
    if override_mask is not None:
        channel_mask = _integer(override_mask, "program.channel_mask")
        if channel_mask <= 0 or channel_mask > 0x3FFFF:
            raise ConfigError("program.channel_mask must fit in 18 speaker bits")
        if channel_mask.bit_count() != len(slots):
            raise ConfigError(
                "program.channel_mask bit count must match channel_layout length"
            )
    else:
        channel_mask = inferred_mask
    if channel_mask != inferred_mask:
        raise ConfigError(
            "program.channel_mask does not match the speaker layout derived from "
            f"channel_layout (derived 0x{inferred_mask:x}, got 0x{channel_mask:x})"
        )

    updated_tracks: dict[str, TrackSpec] = {}
    for track_id, track in tracks.items():
        updated_tracks[track_id] = TrackSpec(
            id=track.id,
            name=track.name,
            role=track.role,
            path=track.path,
            anchors=track.anchors,
            alignment=track.alignment,
            slots=slot_map[track_id],
        )

    policy = _parse_policy(raw.get("policy"))
    cues_raw = raw.get("cues", [])
    if not isinstance(cues_raw, list):
        raise ConfigError("cues must be an array")
    cue_drafts: list[tuple[int, CueSpec]] = []
    seen_cue_ids: set[str] = set()
    for index, item in enumerate(cues_raw):
        if not isinstance(item, dict):
            raise ConfigError(f"cues[{index}] must be an object")
        cue_id = item.get("id", f"cue-{index + 1}")
        if not isinstance(cue_id, str) or not cue_id:
            raise ConfigError(f"cues[{index}].id must be a non-empty string")
        if cue_id in seen_cue_ids:
            raise ConfigError(f"duplicate cue id: {cue_id}")
        seen_cue_ids.add(cue_id)
        program_time = item.get("program_seconds", item.get("time_seconds"))
        if program_time is None:
            raise ConfigError(f"cues[{index}] needs program_seconds")
        cue_seconds = _number(program_time, f"cues[{index}].program_seconds")
        if cue_seconds < 0 or cue_seconds > total_frames / sample_rate + 0.5 / sample_rate:
            raise ConfigError(f"cue {cue_id!r} lies outside the program duration")
        frame = min(total_frames, max(0, _round_frames(cue_seconds, sample_rate)))
        cue_drafts.append(
            (
                index,
                CueSpec(
                    id=cue_id,
                    name=str(item.get("name", cue_id)),
                    program_seconds=frame / sample_rate,
                    frame=frame,
                    wav_cue_id=0,
                ),
            )
        )
    cue_drafts.sort(key=lambda pair: (pair[1].frame, pair[0]))
    cues = tuple(
        CueSpec(
            id=cue.id,
            name=cue.name,
            program_seconds=cue.program_seconds,
            frame=cue.frame,
            wav_cue_id=index + 1,
        )
        for index, (_, cue) in enumerate(cue_drafts)
    )

    event_raw = raw.get("event", {})
    if not isinstance(event_raw, dict):
        raise ConfigError("event must be an object")
    return ProjectSpec(
        timeline_path=timeline_path,
        event=event_raw,
        program=ProgramSpec(
            sample_rate=sample_rate,
            duration_seconds=total_frames / sample_rate,
            total_frames=total_frames,
            bit_depth=bit_depth,
            channels=len(slots),
            channel_mask=channel_mask,
        ),
        tracks=updated_tracks,
        channel_slots=slots,
        cues=cues,
        policy=policy,
        raw=raw,
    )


def alignment_to_dict(fit: AlignmentFit) -> dict[str, Any]:
    return {
        "offset_seconds": fit.offset_seconds,
        "drift_factor": fit.drift_factor,
        "drift_ppm": fit.drift_ppm,
        "max_abs_anchor_error_ms": fit.max_abs_error_ms,
        "rms_anchor_error_ms": fit.rms_error_ms,
        "anchor_errors": [
            {
                "source_seconds": item.source_seconds,
                "expected_program_seconds": item.expected_program_seconds,
                "predicted_program_seconds": item.predicted_program_seconds,
                "error_seconds": item.error_seconds,
                "error_ms": item.error_ms,
                "error_frames": item.error_frames,
            }
            for item in fit.anchor_errors
        ],
    }


def speaker_names_for_mask(mask: int) -> list[str]:
    names: list[str] = []
    for bit, name in _MASK_TO_SPEAKER.items():
        if mask & bit:
            names.append(name)
    return names

