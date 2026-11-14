"""JSON project parsing, schema validation and canonical hashing."""

import hashlib
import json
import os

from .errors import (
    E_SAMPLE_RATE,
    E_SCHEMA,
    E_UNKNOWN_REF,
    ProjectValidationError,
)
from .models import Channel, ExportOptions, Project, RawEvent, Speaker

SUPPORTED_LAYOUTS = ("mono", "stereo")
KNOWN_EVENT_TYPES = ("start", "end", "ptt", "switch", "block", "unblock")


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _require(obj, key, path, errors, expected_type=None):
    if not isinstance(obj, dict) or key not in obj:
        errors.append(
            {
                "code": E_SCHEMA,
                "message": f"missing required field '{key}'",
                "path": f"{path}.{key}" if path else key,
            }
        )
        return False
    if expected_type is not None and not isinstance(obj[key], expected_type):
        errors.append(
            {
                "code": E_SCHEMA,
                "message": f"field '{key}' has wrong type",
                "path": f"{path}.{key}" if path else key,
            }
        )
        return False
    return True


def _normalize_number(value, default):
    return value if _is_number(value) else default


def _normalize_string(value, default=""):
    return value if isinstance(value, str) else default


def _normalize_bool(value, default=False):
    if isinstance(value, bool):
        return value
    return default


def canonical_project(raw):
    """Return a deterministic canonical representation and its sha256."""
    canonical = json.dumps(
        raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), canonical


def load_project(path):
    if not os.path.isfile(path):
        raise ProjectValidationError(
            E_SCHEMA, f"project file not found: {path}", path=path
        )
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ProjectValidationError(
            E_SCHEMA, f"invalid JSON: {exc.msg}", path=path
        ) from exc
    except OSError as exc:
        raise ProjectValidationError(
            E_SCHEMA, f"cannot read project: {exc}", path=path
        ) from exc
    return parse_project(raw)


def parse_project(raw):
    errors = []
    if not isinstance(raw, dict):
        raise ProjectValidationError(
            E_SCHEMA, "project root must be a JSON object", path="$"
        )

    sample_rate = raw.get("sample_rate")
    if not _is_int(sample_rate) or not (8000 <= sample_rate <= 192000):
        raise ProjectValidationError(
            E_SAMPLE_RATE,
            "sample_rate must be an integer between 8000 and 192000",
            path="sample_rate",
        )

    layout = raw.get("channel_layout", "mono")
    if layout not in SUPPORTED_LAYOUTS:
        errors.append(
            {
                "code": E_SCHEMA,
                "message": f"channel_layout must be one of {SUPPORTED_LAYOUTS}",
                "path": "channel_layout",
            }
        )

    tick_rate = raw.get("tick_rate", 1000)
    if not _is_int(tick_rate) or tick_rate <= 0:
        errors.append(
            {
                "code": E_SCHEMA,
                "message": "tick_rate must be a positive integer",
                "path": "tick_rate",
            }
        )

    channels = _parse_channels(raw, errors)
    speakers = _parse_speakers(raw, channels, errors)
    events = _parse_events(raw, channels, speakers, errors)
    export_options = _parse_export(raw, errors)

    if errors:
        raise _AggregateValidation(errors)

    canonical_hash, _ = canonical_project(raw)
    return Project(
        sample_rate=sample_rate,
        channels=channels,
        speakers=speakers,
        events=events,
        export=export_options,
        tick_rate=tick_rate,
        raw=raw,
        canonical_hash=canonical_hash,
    )


class _AggregateValidation(ProjectValidationError):
    def __init__(self, errors):
        super().__init__(E_SCHEMA, "project validation failed", path="$")
        self.errors = errors

    def to_obj(self):
        return {"code": E_SCHEMA, "message": "project validation failed", "errors": self.errors}


def _parse_channels(raw, errors):
    channels = {}
    if not _require(raw, "channels", "$", errors, list):
        return channels
    seen = set()
    for pos, item in enumerate(raw["channels"]):
        path = f"channels[{pos}]"
        if not isinstance(item, dict):
            errors.append({"code": E_SCHEMA, "message": "channel must be an object", "path": path})
            continue
        cid = item.get("id")
        if not isinstance(cid, str) or not cid:
            errors.append({"code": E_SCHEMA, "message": "channel id must be a non-empty string", "path": f"{path}.id"})
            continue
        if cid in seen:
            errors.append({"code": E_SCHEMA, "message": f"duplicate channel id '{cid}'", "path": f"{path}.id"})
            continue
        seen.add(cid)
        gain_db = item.get("gain_db", 0.0)
        if not _is_number(gain_db):
            errors.append({"code": E_SCHEMA, "message": "gain_db must be a number", "path": f"{path}.gain_db"})
            gain_db = 0.0
        channels[cid] = Channel(
            channel_id=cid,
            gain_db=float(gain_db),
            muted=_normalize_bool(item.get("muted", False), False),
        )
    return channels


def _parse_speakers(raw, channels, errors):
    speakers = {}
    if not _require(raw, "speakers", "$", errors, list):
        return speakers
    seen = set()
    for pos, item in enumerate(raw["speakers"]):
        path = f"speakers[{pos}]"
        if not isinstance(item, dict):
            errors.append({"code": E_SCHEMA, "message": "speaker must be an object", "path": path})
            continue
        sid = item.get("id")
        if not isinstance(sid, str) or not sid:
            errors.append({"code": E_SCHEMA, "message": "speaker id must be a non-empty string", "path": f"{path}.id"})
            continue
        if sid in seen:
            errors.append({"code": E_SCHEMA, "message": f"duplicate speaker id '{sid}'", "path": f"{path}.id"})
            continue
        seen.add(sid)

        bound = item.get("bound_channel")
        if not isinstance(bound, str) or bound not in channels:
            errors.append(
                {
                    "code": E_UNKNOWN_REF,
                    "message": f"unknown bound_channel '{bound}'",
                    "path": f"{path}.bound_channel",
                }
            )
            bound = bound if isinstance(bound, str) and bound else next(iter(channels), "__invalid__")

        seed = item.get("seed", 0)
        if not _is_int(seed):
            errors.append({"code": E_SCHEMA, "message": "seed must be an integer", "path": f"{path}.seed"})
            seed = 0
        base_frequency = _normalize_number(item.get("base_frequency", 120.0), 120.0)
        if base_frequency <= 0:
            errors.append({"code": E_SCHEMA, "message": "base_frequency must be positive", "path": f"{path}.base_frequency"})
            base_frequency = 120.0
        harmonic_ratio = _normalize_number(item.get("harmonic_ratio", 0.25), 0.25)
        if not (0.0 <= harmonic_ratio <= 1.0):
            errors.append({"code": E_SCHEMA, "message": "harmonic_ratio must be in [0,1]", "path": f"{path}.harmonic_ratio"})
            harmonic_ratio = min(1.0, max(0.0, harmonic_ratio))
        pitch = _normalize_number(item.get("pitch_semitones", 0.0), 0.0)
        source_ms = _normalize_number(item.get("source_duration_ms", 1000.0), 1000.0)
        if source_ms <= 0:
            errors.append({"code": E_SCHEMA, "message": "source_duration_ms must be positive", "path": f"{path}.source_duration_ms"})
            source_ms = 1000.0
        pan = _normalize_number(item.get("pan", 0.0), 0.0)
        pan = min(1.0, max(-1.0, pan))

        speakers[sid] = Speaker(
            speaker_id=sid,
            seed=int(seed),
            base_frequency=float(base_frequency),
            harmonic_ratio=float(harmonic_ratio),
            pitch_semitones=float(pitch),
            source_duration_ms=float(source_ms),
            bound_channel=bound,
            pan=float(pan),
        )
    return speakers


def _parse_events(raw, channels, speakers, errors):
    events = []
    if "events" not in raw:
        errors.append({"code": E_SCHEMA, "message": "missing required field 'events'", "path": "events"})
        return events
    if not isinstance(raw["events"], list):
        errors.append({"code": E_SCHEMA, "message": "events must be a list", "path": "events"})
        return events

    last_tick = None
    for index, item in enumerate(raw["events"]):
        path = f"events[{index}]"
        if not isinstance(item, dict):
            errors.append({"code": E_SCHEMA, "message": "event must be an object", "path": path, "index": index})
            continue
        etype = item.get("type")
        if etype not in KNOWN_EVENT_TYPES:
            errors.append(
                {
                    "code": E_SCHEMA,
                    "message": f"unknown event type '{etype}'",
                    "path": f"{path}.type",
                    "index": index,
                }
            )
            continue
        tick = item.get("tick")
        if not _is_int(tick) or tick < 0:
            errors.append(
                {
                    "code": E_SCHEMA,
                    "message": "tick must be a non-negative integer",
                    "path": f"{path}.tick",
                    "index": index,
                }
            )
            continue
        if last_tick is not None and tick < last_tick:
            errors.append(
                {
                    "code": "E_TICK_ORDER",
                    "message": f"tick {tick} is earlier than previous tick {last_tick}",
                    "path": f"{path}.tick",
                    "index": index,
                }
            )
        last_tick = tick

        sid = item.get("speaker")
        if not isinstance(sid, str) or sid not in speakers:
            errors.append(
                {
                    "code": E_UNKNOWN_REF,
                    "message": f"unknown speaker '{sid}'",
                    "path": f"{path}.speaker",
                    "index": index,
                }
            )
            continue

        channel = item.get("channel")
        if channel is not None:
            if not isinstance(channel, str) or channel not in channels:
                errors.append(
                    {
                        "code": E_UNKNOWN_REF,
                        "message": f"unknown channel '{channel}'",
                        "path": f"{path}.channel",
                        "index": index,
                    }
                )
                continue
        events.append(
            RawEvent(
                index=index,
                tick=tick,
                type=etype,
                speaker=sid,
                channel=channel,
            )
        )
    return events


def _parse_export(raw, errors):
    export_raw = raw.get("export", {})
    if not isinstance(export_raw, dict):
        errors.append({"code": E_SCHEMA, "message": "export must be an object", "path": "export"})
        export_raw = {}
    output_wav = _normalize_string(export_raw.get("output_wav", "replay.wav"), "replay.wav")
    manifest_path = _normalize_string(export_raw.get("manifest", "manifest.json"), "manifest.json")
    work_dir = _normalize_string(export_raw.get("work_dir", "render_work"), "render_work")
    chunk_samples = export_raw.get("chunk_samples", 8000)
    if not _is_int(chunk_samples) or chunk_samples <= 0:
        errors.append(
            {
                "code": E_SCHEMA,
                "message": "chunk_samples must be a positive integer",
                "path": "export.chunk_samples",
            }
        )
        chunk_samples = 8000
    intro_ms = _normalize_number(export_raw.get("intro_ms", 300.0), 300.0)
    outro_ms = _normalize_number(export_raw.get("outro_ms", 400.0), 400.0)
    if intro_ms < 0 or outro_ms < 0:
        errors.append(
            {
                "code": E_SCHEMA,
                "message": "intro_ms/outro_ms must be non-negative",
                "path": "export",
            }
        )
        intro_ms = max(0.0, intro_ms)
        outro_ms = max(0.0, outro_ms)
    enable_prompts = _normalize_bool(export_raw.get("enable_prompts", True), True)
    return ExportOptions(
        output_wav=output_wav,
        manifest_path=manifest_path,
        work_dir=work_dir,
        chunk_samples=int(chunk_samples),
        intro_ms=float(intro_ms),
        outro_ms=float(outro_ms),
        enable_prompts=enable_prompts,
    )
