from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .model import ClipSpec, EventSpec, ParsedTimeline, TrackConfig


class TimelineFatalError(Exception):
    pass


def _first(mapping: Dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _number(
    value: Any,
    field: str,
    *,
    allow_zero: bool = True,
    errors: Optional[List[str]] = None,
) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        if errors is not None:
            errors.append(f"{field} must be a finite number")
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        if errors is not None:
            errors.append(f"{field} must be a finite number")
        return None
    if not math.isfinite(parsed):
        if errors is not None:
            errors.append(f"{field} must be a finite number")
        return None
    if not allow_zero and parsed == 0.0:
        if errors is not None:
            errors.append(f"{field} must not be zero")
        return None
    return parsed


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return False


def _resolve_path(raw_path: str, input_dir: Path, timeline_dir: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    input_candidate = (input_dir / candidate).resolve()
    if input_candidate.exists():
        return input_candidate
    timeline_candidate = (timeline_dir / candidate).resolve()
    if timeline_candidate.exists():
        return timeline_candidate
    return input_candidate


def _extract_manifest(
    document: Any,
) -> Tuple[Any, Optional[str], Optional[float], str, Any, Any]:
    if isinstance(document, list):
        return [], None, None, "", [], document
    if not isinstance(document, dict):
        raise TimelineFatalError("timeline JSON root must be an object or an event array")
    events = _first(document, ("events", "timeline", "event_list"))
    if events is None:
        raise TimelineFatalError("timeline JSON is missing an 'events' array")
    if not isinstance(events, list):
        raise TimelineFatalError("'events' must be an array")
    assets = _first(document, ("assets", "clips", "asset_list", "asset_manifest_data"), [])
    external_manifest = _first(
        document, ("asset_manifest", "assets_file", "assets_path", "manifest")
    )
    duration = _first(
        document,
        ("duration_ms", "timeline_duration_ms", "max_time_ms", "project_duration_ms"),
    )
    title = str(_first(document, ("title", "project", "name"), "game_audio_mix"))
    tracks = _first(document, ("tracks", "track_states", "track_config"), [])
    return assets, external_manifest, duration, title, tracks, events


def _parse_tracks(raw_tracks: Any) -> Dict[str, TrackConfig]:
    configs: Dict[str, TrackConfig] = {}
    if isinstance(raw_tracks, dict):
        items = [
            dict(value, id=key) if isinstance(value, dict) else {"id": key, "mute": value}
            for key, value in raw_tracks.items()
        ]
    elif isinstance(raw_tracks, list):
        items = raw_tracks
    elif raw_tracks in (None, ""):
        items = []
    else:
        raise TimelineFatalError("'tracks' must be an array or object")

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        track_id = _first(item, ("id", "track_id", "name"))
        if track_id is None or not str(track_id).strip():
            track_id = f"track_{index + 1}"
        track_id = str(track_id).strip()
        mute = _bool(_first(item, ("mute", "muted", "initial_mute"), False))
        solo = _bool(_first(item, ("solo", "soloed", "initial_solo"), False))
        parsed_gain = _number(
            _first(item, ("gain_db", "gain"), 0.0),
            f"track '{track_id}' gain_db",
        )
        gain_db = 0.0 if parsed_gain is None else parsed_gain
        configs[track_id] = TrackConfig(
            id=track_id,
            name=str(_first(item, ("name", "label"), track_id)),
            initial_mute=mute,
            initial_solo=solo,
            gain_db=gain_db,
        )
    return configs


def _parse_assets(
    raw_assets: Any,
    input_dir: Path,
    timeline_dir: Path,
    basic_errors: List[str],
) -> Dict[str, ClipSpec]:
    specs: Dict[str, ClipSpec] = {}
    if raw_assets in (None, ""):
        return specs
    if isinstance(raw_assets, dict):
        items = []
        for asset_id, value in raw_assets.items():
            if isinstance(value, str):
                items.append({"id": asset_id, "path": value})
            elif isinstance(value, dict):
                item = dict(value)
                item.setdefault("id", asset_id)
                items.append(item)
            else:
                basic_errors.append(f"asset '{asset_id}' has an unsupported definition")
    elif isinstance(raw_assets, list):
        items = raw_assets
    else:
        raise TimelineFatalError("asset manifest must be an array or object")

    for index, item in enumerate(items):
        if isinstance(item, str):
            item = {"id": item, "path": item}
        if not isinstance(item, dict):
            basic_errors.append(f"asset entry {index} is not an object")
            continue
        raw_id = _first(item, ("id", "clip_id", "name"))
        raw_path = _first(
            item,
            ("path", "file", "filename", "source", "asset", "url"),
        )
        if raw_path is None:
            basic_errors.append(f"asset entry {index} has no file path")
            continue
        asset_id = str(raw_id if raw_id is not None else raw_path).strip()
        if not asset_id:
            asset_id = str(raw_path)
        path = _resolve_path(str(raw_path), input_dir, timeline_dir)
        offset = _number(
            _first(item, ("offset_ms", "source_offset_ms", "clip_offset_ms"), 0.0),
            f"asset '{asset_id}' offset_ms",
        )
        duration_value = _first(
            item, ("duration_ms", "clip_duration_ms", "length_ms")
        )
        duration = (
            _number(duration_value, f"asset '{asset_id}' duration_ms", allow_zero=False)
            if duration_value is not None
            else None
        )
        if asset_id in specs:
            basic_errors.append(f"duplicate asset id '{asset_id}' ignored")
            continue
        specs[asset_id] = ClipSpec(
            id=asset_id,
            path=path,
            offset_ms=max(0.0, offset or 0.0),
            duration_ms=duration,
        )
    return specs


def load_timeline(
    timeline_path: Path,
    input_dir: Path,
) -> ParsedTimeline:
    timeline_path = timeline_path.expanduser().resolve()
    try:
        with timeline_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except OSError as exc:
        raise TimelineFatalError(f"cannot read timeline file '{timeline_path}': {exc}")
    except json.JSONDecodeError as exc:
        raise TimelineFatalError(
            f"cannot parse timeline JSON '{timeline_path}': "
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
        )

    (
        raw_assets,
        external_manifest,
        raw_duration,
        title,
        raw_tracks,
        raw_events,
    ) = _extract_manifest(document)
    warnings: List[str] = []
    errors: List[Dict[str, Any]] = []
    track_configs = _parse_tracks(raw_tracks)

    if external_manifest:
        manifest_path = _resolve_path(
            str(external_manifest), input_dir, timeline_path.parent
        )
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest_document = json.load(handle)
            if isinstance(manifest_document, dict):
                raw_assets = _first(
                    manifest_document,
                    ("assets", "clips", "asset_list"),
                    manifest_document,
                )
                manifest_tracks = _first(
                    manifest_document, ("tracks", "track_states"), None
                )
                if manifest_tracks is not None and not track_configs:
                    track_configs = _parse_tracks(manifest_tracks)
            else:
                raw_assets = manifest_document
        except (OSError, json.JSONDecodeError) as exc:
            raise TimelineFatalError(
                f"cannot parse asset manifest '{manifest_path}': {exc}"
            )

    clip_specs = _parse_assets(raw_assets, input_dir, timeline_path.parent, warnings)
    duration_ms: Optional[float] = None
    if raw_duration is not None:
        duration_ms = _number(
            raw_duration, "timeline duration_ms", allow_zero=False, errors=warnings
        )

    events: List[EventSpec] = []
    seen_ids = set()
    for index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, dict):
            errors.append(
                {
                    "event_index": index,
                    "event_id": None,
                    "reason": "event entry is not an object",
                }
            )
            continue
        local_errors: List[str] = []
        event_id_raw = _first(raw_event, ("id", "event_id", "name"))
        event_id = (
            str(event_id_raw).strip()
            if event_id_raw is not None and str(event_id_raw).strip()
            else f"event_{index + 1}"
        )
        if event_id in seen_ids:
            new_id = f"{event_id}_{index + 1}"
            warnings.append(f"duplicate event id '{event_id}' renamed to '{new_id}'")
            event_id = new_id
        seen_ids.add(event_id)

        event_type_raw = _first(raw_event, ("type", "event_type", "kind"))
        event_type = str(event_type_raw).strip() if event_type_raw is not None else ""
        if not event_type:
            local_errors.append("missing event type")

        time_raw = _first(
            raw_event,
            ("time_ms", "timestamp_ms", "start_ms", "position_ms", "time"),
        )
        time_ms = _number(time_raw, "time_ms", errors=local_errors)
        if time_ms is not None and time_ms < 0.0:
            local_errors.append("time_ms must be non-negative")

        track_raw = _first(raw_event, ("track", "track_id", "bus"))
        track_id = str(track_raw).strip() if track_raw is not None else ""
        if not track_id:
            local_errors.append("missing track")

        asset_raw = _first(
            raw_event,
            ("asset", "asset_id", "clip", "clip_id", "source", "file"),
        )
        inline_clip: Optional[Dict[str, Any]] = None
        if isinstance(asset_raw, dict):
            inline_clip = asset_raw
            asset_raw = _first(inline_clip, ("id", "clip_id", "asset_id", "path", "file"))
        asset_id = str(asset_raw).strip() if asset_raw is not None else ""
        if not asset_id:
            local_errors.append("missing clip/asset reference")

        gain_db = _number(_first(raw_event, ("gain_db", "gain"), 0.0), "gain_db")
        if gain_db is None:
            local_errors.append("gain_db must be a finite number")

        offset_value = _first(
            raw_event,
            ("offset_ms", "source_offset_ms", "clip_offset_ms"),
        )
        if offset_value is None and inline_clip is not None:
            offset_value = _first(
                inline_clip, ("offset_ms", "source_offset_ms", "clip_offset_ms")
            )
        offset_ms = (
            _number(offset_value, "offset_ms", errors=local_errors)
            if offset_value is not None
            else None
        )
        if offset_ms is not None and offset_ms < 0.0:
            local_errors.append("offset_ms must be non-negative")

        duration_value = _first(
            raw_event, ("duration_ms", "clip_duration_ms", "length_ms")
        )
        if duration_value is None and inline_clip is not None:
            duration_value = _first(
                inline_clip, ("duration_ms", "clip_duration_ms", "length_ms")
            )
        event_duration = (
            _number(duration_value, "duration_ms", allow_zero=False, errors=local_errors)
            if duration_value is not None
            else None
        )

        if (
            inline_clip is not None
            and asset_id
            and asset_id not in clip_specs
        ):
            inline_path = _first(
                inline_clip, ("path", "file", "filename", "source")
            )
            if inline_path:
                clip_specs[asset_id] = ClipSpec(
                    id=asset_id,
                    path=_resolve_path(
                        str(inline_path), input_dir, timeline_path.parent
                    ),
                    offset_ms=max(0.0, offset_ms or 0.0),
                    duration_ms=event_duration,
                )

        if local_errors:
            errors.append(
                {
                    "event_index": index,
                    "event_id": event_id,
                    "reason": "; ".join(local_errors),
                }
            )
            continue

        if track_id not in track_configs:
            track_configs[track_id] = TrackConfig(
                id=track_id, name=track_id, initial_mute=False, initial_solo=False
            )
        events.append(
            EventSpec(
                id=event_id,
                event_type=event_type,
                time_ms=float(time_ms),
                track_id=track_id,
                asset_id=asset_id,
                gain_db=float(gain_db if gain_db is not None else 0.0),
                offset_ms=offset_ms,
                duration_ms=event_duration,
                source_index=index,
            )
        )

    return ParsedTimeline(
        track_configs=list(track_configs.values()),
        clip_specs=clip_specs,
        events=events,
        timeline_duration_ms=duration_ms,
        title=title,
        parse_errors=errors,
        parse_warnings=warnings,
        asset_manifest=str(external_manifest) if external_manifest else None,
        timeline_path=timeline_path,
    )
