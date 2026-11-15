"""Manifest parsing and semantic validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import BusinessError
from .wavio import read_wav


@dataclass(frozen=True)
class Track:
    name: str
    role: str
    asset: Path
    gain_db: float


@dataclass(frozen=True)
class Clip:
    clip_id: str
    track: str
    asset: Path
    source_in: int
    source_out: int
    timeline_in: int
    timeline_out: int
    gain_db: float
    order: int

    @property
    def source_duration(self) -> int:
        return self.source_out - self.source_in

    @property
    def timeline_duration(self) -> int:
        return self.timeline_out - self.timeline_in


@dataclass(frozen=True)
class EditPoint:
    edit_id: str
    track: str
    outgoing_clip: str
    incoming_clip: str
    at_sample: int
    fade_samples: int
    order: int


@dataclass(frozen=True)
class Chapter:
    title: str
    start_sample: int


@dataclass(frozen=True)
class DuckingConfig:
    enabled: bool
    trigger_track: str
    target_track: str
    threshold_dbfs: float
    energy_window_ms: float
    depth_db: float
    attack_ms: float
    release_ms: float


@dataclass(frozen=True)
class LoudnessConfig:
    target_lufs: float
    max_true_peak_dbtp: float


@dataclass
class Manifest:
    path: Path
    sample_rate: int
    channels: int
    bit_depth: int
    total_samples: int
    tracks: dict[str, Track]
    clips: list[Clip]
    edit_points: list[EditPoint]
    chapters: list[Chapter]
    ducking: DuckingConfig
    loudness: LoudnessConfig
    assets: dict[Path, Any] = field(default_factory=dict)


def _schema(message: str) -> BusinessError:
    return BusinessError("E_SCHEMA", message)


def _clip_error(message: str) -> BusinessError:
    return BusinessError("E_CLIP_RANGE", message)


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _schema(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise _schema(f"{label} must be an array")
    return value


def _int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _schema(f"{label} must be an integer sample count")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _schema(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise _schema(f"{label} must be finite")
    return result


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _schema(f"{label} must be a non-empty string")
    return value


def _resolve_asset(manifest_dir: Path, value: Any, label: str) -> Path:
    raw = Path(_string(value, label))
    if not raw.is_absolute():
        raw = manifest_dir / raw
    return raw.resolve()


def parse_manifest(path: Path) -> Manifest:
    """Parse and fully validate a manifest, including all referenced assets."""

    path = path.resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BusinessError("E_ASSET_MISSING", f"manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise _schema(f"invalid manifest JSON at character {exc.pos}: {exc.msg}") from exc
    except OSError as exc:
        raise BusinessError("E_ASSET_MISSING", f"cannot read manifest: {path}: {exc}") from exc

    root = _require_dict(document, "manifest")
    version = _int(root.get("version"), "version")
    if version != 1:
        raise _schema(f"unsupported manifest version: {version}")
    sample_rate = _int(root.get("sample_rate"), "sample_rate")
    channels = _int(root.get("channels"), "channels")
    bit_depth = _int(root.get("bit_depth"), "bit_depth")
    total_samples = _int(root.get("total_samples"), "total_samples")
    if sample_rate != 48000:
        raise BusinessError(
            "E_FORMAT_UNSUPPORTED", f"manifest sample_rate must be 48000, got {sample_rate}"
        )
    if channels not in (1, 2):
        raise BusinessError(
            "E_FORMAT_UNSUPPORTED", f"manifest channels must be 1 or 2, got {channels}"
        )
    if bit_depth != 24:
        raise BusinessError(
            "E_FORMAT_UNSUPPORTED", f"manifest bit_depth must be 24, got {bit_depth}"
        )
    if total_samples <= 0:
        raise _clip_error("total_samples must be positive")

    manifest_dir = path.parent
    tracks: dict[str, Track] = {}
    for index, item in enumerate(_require_list(root.get("tracks"), "tracks")):
        obj = _require_dict(item, f"tracks[{index}]")
        name = _string(obj.get("name"), f"tracks[{index}].name")
        if name in tracks:
            raise _schema(f"duplicate track name: {name}")
        role = _string(obj.get("role"), f"tracks[{index}].role")
        if role not in ("host", "guest", "music"):
            raise _schema(f"track role must be host, guest, or music: {role}")
        asset = _resolve_asset(manifest_dir, obj.get("asset"), f"tracks[{index}].asset")
        gain_db = _number(obj.get("gain_db", 0.0), f"tracks[{index}].gain_db")
        tracks[name] = Track(name, role, asset, gain_db)
    if not tracks:
        raise _schema("tracks must not be empty")
    roles = {track.role for track in tracks.values()}
    if roles != {"host", "guest", "music"}:
        raise _schema("manifest must define exactly one host, guest, and music role")
    if sum(track.role == "host" for track in tracks.values()) != 1:
        raise _schema("manifest must define exactly one host track")
    if sum(track.role == "guest" for track in tracks.values()) != 1:
        raise _schema("manifest must define exactly one guest track")
    if sum(track.role == "music" for track in tracks.values()) != 1:
        raise _schema("manifest must define exactly one music track")

    clips: list[Clip] = []
    clip_by_id: dict[str, Clip] = {}
    for index, item in enumerate(_require_list(root.get("clips"), "clips")):
        obj = _require_dict(item, f"clips[{index}]")
        clip_id = _string(obj.get("id"), f"clips[{index}].id")
        if clip_id in clip_by_id:
            raise _schema(f"duplicate clip id: {clip_id}")
        track_name = _string(obj.get("track"), f"clips[{index}].track")
        if track_name not in tracks:
            raise _schema(f"clip {clip_id} references unknown track {track_name}")
        track = tracks[track_name]
        asset_value = obj.get("asset", str(track.asset))
        asset = _resolve_asset(manifest_dir, asset_value, f"clips[{index}].asset")
        if asset != track.asset:
            raise _schema(
                f"clip {clip_id} asset must match its track asset: {track.asset}"
            )
        source_in = _int(obj.get("source_in"), f"clips[{index}].source_in")
        source_out = _int(obj.get("source_out"), f"clips[{index}].source_out")
        timeline_in = _int(obj.get("timeline_in"), f"clips[{index}].timeline_in")
        timeline_out = _int(obj.get("timeline_out"), f"clips[{index}].timeline_out")
        gain_db = _number(obj.get("gain_db", 0.0), f"clips[{index}].gain_db")
        if source_in < 0 or source_out <= source_in:
            raise _clip_error(
                f"clip {clip_id} has invalid source range [{source_in}, {source_out})"
            )
        if timeline_in < 0 or timeline_out <= timeline_in:
            raise _clip_error(
                f"clip {clip_id} has invalid timeline range [{timeline_in}, {timeline_out})"
            )
        if timeline_out > total_samples:
            raise _clip_error(
                f"clip {clip_id} timeline_out {timeline_out} exceeds total_samples {total_samples}"
            )
        if source_out - source_in != timeline_out - timeline_in:
            raise _clip_error(
                f"clip {clip_id} source duration does not equal timeline duration"
            )
        clip = Clip(
            clip_id,
            track_name,
            asset,
            source_in,
            source_out,
            timeline_in,
            timeline_out,
            gain_db,
            index,
        )
        clips.append(clip)
        clip_by_id[clip_id] = clip
    if not clips:
        raise _schema("clips must not be empty")

    processing = _require_dict(root.get("processing"), "processing")
    default_fade = _int(processing.get("fade_samples", 0), "processing.fade_samples")
    if default_fade < 0:
        raise _clip_error("processing.fade_samples must not be negative")

    edit_points: list[EditPoint] = []
    edit_ids: set[str] = set()
    for index, item in enumerate(_require_list(root.get("edit_points", []), "edit_points")):
        obj = _require_dict(item, f"edit_points[{index}]")
        edit_id = _string(obj.get("id"), f"edit_points[{index}].id")
        if edit_id in edit_ids:
            raise _schema(f"duplicate edit point id: {edit_id}")
        track_name = _string(obj.get("track"), f"edit_points[{index}].track")
        if track_name not in tracks:
            raise _schema(f"edit point {edit_id} references unknown track {track_name}")
        outgoing_id = _string(
            obj.get("outgoing_clip"), f"edit_points[{index}].outgoing_clip"
        )
        incoming_id = _string(
            obj.get("incoming_clip"), f"edit_points[{index}].incoming_clip"
        )
        if outgoing_id not in clip_by_id or incoming_id not in clip_by_id:
            raise _schema(f"edit point {edit_id} references an unknown clip")
        outgoing = clip_by_id[outgoing_id]
        incoming = clip_by_id[incoming_id]
        if outgoing.track != track_name or incoming.track != track_name:
            raise _schema(f"edit point {edit_id} clips must belong to track {track_name}")
        if outgoing.clip_id == incoming.clip_id:
            raise _schema(f"edit point {edit_id} must join two different clips")
        at_sample = _int(obj.get("at_sample"), f"edit_points[{index}].at_sample")
        if at_sample <= 0 or at_sample >= total_samples:
            raise _clip_error(
                f"edit point {edit_id} at_sample {at_sample} is outside the episode interior"
            )
        fade_samples = _int(
            obj.get("fade_samples", default_fade),
            f"edit_points[{index}].fade_samples",
        )
        if fade_samples <= 0:
            raise _clip_error(f"edit point {edit_id} fade_samples must be positive")
        if outgoing.timeline_out < at_sample or incoming.timeline_out <= at_sample:
            raise _clip_error(
                f"edit point {edit_id} clips do not both cover the fade endpoint {at_sample}"
            )
        available = at_sample - max(
            outgoing.timeline_in,
            incoming.timeline_in,
            at_sample - fade_samples,
        )
        if available <= 0:
            raise _clip_error(f"edit point {edit_id} has no usable crossfade overlap")
        edit_points.append(
            EditPoint(
                edit_id,
                track_name,
                outgoing_id,
                incoming_id,
                at_sample,
                fade_samples,
                index,
            )
        )
        edit_ids.add(edit_id)

    chapters: list[Chapter] = []
    chapter_items = _require_list(root.get("chapters"), "chapters")
    if not chapter_items:
        raise BusinessError("E_CHAPTER_RANGE", "chapters must not be empty")
    previous = -1
    for index, item in enumerate(chapter_items):
        obj = _require_dict(item, f"chapters[{index}]")
        title = _string(obj.get("title"), f"chapters[{index}].title")
        start = _int(obj.get("start_sample"), f"chapters[{index}].start_sample")
        if start < 0 or start >= total_samples:
            raise BusinessError(
                "E_CHAPTER_RANGE",
                f"chapter start {start} must be in [0, total_samples)",
            )
        if start <= previous:
            raise BusinessError(
                "E_CHAPTER_RANGE", "chapter starts must be strictly increasing"
            )
        chapters.append(Chapter(title, start))
        previous = start
    if chapters[0].start_sample != 0:
        raise BusinessError("E_CHAPTER_RANGE", "the first chapter must start at sample 0")

    duck_obj = _require_dict(processing.get("ducking", {}), "processing.ducking")
    ducking = DuckingConfig(
        enabled=bool(duck_obj.get("enabled", True)),
        trigger_track=_string(
            duck_obj.get("trigger_track", "host"), "processing.ducking.trigger_track"
        ),
        target_track=_string(
            duck_obj.get("target_track", "music"), "processing.ducking.target_track"
        ),
        threshold_dbfs=_number(
            duck_obj.get("threshold_dbfs", -30.0),
            "processing.ducking.threshold_dbfs",
        ),
        energy_window_ms=_number(
            duck_obj.get("energy_window_ms", 20.0),
            "processing.ducking.energy_window_ms",
        ),
        depth_db=_number(
            duck_obj.get("depth_db", 12.0), "processing.ducking.depth_db"
        ),
        attack_ms=_number(
            duck_obj.get("attack_ms", 10.0), "processing.ducking.attack_ms"
        ),
        release_ms=_number(
            duck_obj.get("release_ms", 250.0), "processing.ducking.release_ms"
        ),
    )
    if ducking.trigger_track not in tracks or ducking.target_track not in tracks:
        raise _schema("ducking references an unknown trigger or target track")
    if tracks[ducking.trigger_track].role != "host":
        raise _schema("ducking trigger track must have role host")
    if tracks[ducking.target_track].role != "music":
        raise _schema("ducking target track must have role music")
    if ducking.energy_window_ms <= 0:
        raise _schema("ducking.energy_window_ms must be positive")
    if ducking.depth_db < 0 or ducking.depth_db > 96:
        raise _schema("ducking.depth_db must be in [0, 96]")
    if ducking.attack_ms < 0 or ducking.release_ms < 0:
        raise _schema("ducking attack/release times must not be negative")

    loudness_obj = _require_dict(
        processing.get("loudness", {}), "processing.loudness"
    )
    loudness = LoudnessConfig(
        target_lufs=_number(
            loudness_obj.get("target_lufs", -16.0), "processing.loudness.target_lufs"
        ),
        max_true_peak_dbtp=_number(
            loudness_obj.get("max_true_peak_dbtp", -1.0),
            "processing.loudness.max_true_peak_dbtp",
        ),
    )
    if loudness.max_true_peak_dbtp > 0:
        raise _schema("processing.loudness.max_true_peak_dbtp must not exceed 0 dBTP")

    assets: dict[Path, Any] = {}
    for track in tracks.values():
        if track.asset in assets:
            _, info = assets[track.asset]
        else:
            if not track.asset.exists():
                raise BusinessError(
                    "E_ASSET_MISSING", f"asset not found: {track.asset}"
                )
            samples, info = read_wav(track.asset)
            assets[track.asset] = (samples, info)
        if info.sample_rate != sample_rate or info.channels != channels:
            raise BusinessError(
                "E_FORMAT_UNSUPPORTED",
                f"asset format mismatch for {track.asset}: "
                f"{info.sample_rate} Hz/{info.channels} ch, expected "
                f"{sample_rate} Hz/{channels} ch",
            )

    for clip in clips:
        _, info = assets[clip.asset]
        if clip.source_out > info.frames:
            raise _clip_error(
                f"clip {clip.clip_id} source_out {clip.source_out} exceeds "
                f"asset length {info.frames}: {clip.asset}"
            )

    return Manifest(
        path=path,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=bit_depth,
        total_samples=total_samples,
        tracks=tracks,
        clips=clips,
        edit_points=edit_points,
        chapters=chapters,
        ducking=ducking,
        loudness=loudness,
        assets=assets,
    )

