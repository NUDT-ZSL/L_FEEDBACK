"""JSON configuration parsing and validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .errors import MixerError


MIN_RATE = 8000
MAX_RATE = 192000
MAX_DURATION_SECONDS = 3600.0


@dataclass(frozen=True)
class CompressorConfig:
    threshold_db: float = -18.0
    ratio: float = 4.0
    knee_db: float = 6.0
    attack_ms: float = 10.0
    release_ms: float = 100.0


@dataclass(frozen=True)
class LimiterConfig:
    ceiling_dbfs: float = -1.0
    lookahead_ms: float = 5.0
    release_ms: float = 80.0


@dataclass(frozen=True)
class TrackConfig:
    track_id: str
    input_path: Path
    offset_ms: float
    gain_db: float
    compressor: Optional[CompressorConfig]
    limiter: Optional[LimiterConfig]
    channel_map: Optional[tuple]


@dataclass(frozen=True)
class OutputConfig:
    path: Path
    sample_rate: int
    channels: int
    duration_seconds: Optional[float]
    subtype: str
    limiter_placement: str
    bus_limiter: LimiterConfig
    report_path: Path


@dataclass(frozen=True)
class MixConfig:
    source_path: Path
    output: OutputConfig
    tracks: tuple
    workers: int
    chunk_frames: int


def _fail(code, message, details=None, exit_code=2):
    raise MixerError(code, message, details, exit_code)


def _mapping(value, field):
    if value is None:
        return {}
    if not isinstance(value, dict):
        _fail("E_CONFIG_INVALID", "{} must be an object".format(field), {"field": field})
    return value


def _number(value, field, code="E_CONFIG_INVALID"):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(code, "{} must be a finite number".format(field), {"field": field})
    result = float(value)
    if not math.isfinite(result):
        _fail(code, "{} must be a finite number".format(field), {"field": field})
    return result


def _required(mapping, names, field):
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    _fail(
        "E_CONFIG_MISSING_FIELD",
        "missing required field {}".format(field),
        {"field": field},
    )


def _integer(value, field, code="E_CONFIG_INVALID"):
    number = _number(value, field, code)
    if not number.is_integer():
        _fail(code, "{} must be an integer".format(field), {"field": field, "value": value})
    return int(number)


def _resolve(base, value):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _parse_compressor(raw, track_id):
    if raw is None or raw is False:
        return None
    cfg = _mapping(raw, "tracks[{}].compressor".format(track_id))
    threshold = _number(
        cfg.get("threshold_db", cfg.get("threshold", -18.0)),
        "tracks[{}].compressor.threshold_db".format(track_id),
        "E_INVALID_COMPRESSOR",
    )
    ratio = _number(
        cfg.get("ratio", 4.0),
        "tracks[{}].compressor.ratio".format(track_id),
        "E_INVALID_COMPRESSOR",
    )
    knee_type = str(cfg.get("knee_type", "soft")).lower()
    if knee_type not in {"soft", "hard"}:
        _fail(
            "E_INVALID_COMPRESSOR",
            "compressor knee_type must be soft or hard",
            {"track": track_id, "field": "knee_type", "value": knee_type},
        )
    knee_raw = cfg.get("knee_db", cfg.get("knee", 6.0))
    if knee_type == "hard":
        knee_raw = 0.0
    knee = _number(
        knee_raw,
        "tracks[{}].compressor.knee_db".format(track_id),
        "E_INVALID_COMPRESSOR",
    )
    attack = _number(
        cfg.get("attack_ms", cfg.get("attack", 10.0)),
        "tracks[{}].compressor.attack_ms".format(track_id),
        "E_INVALID_COMPRESSOR",
    )
    release = _number(
        cfg.get("release_ms", cfg.get("release", 100.0)),
        "tracks[{}].compressor.release_ms".format(track_id),
        "E_INVALID_COMPRESSOR",
    )
    if not -120.0 <= threshold <= 0.0:
        _fail(
            "E_INVALID_COMPRESSOR",
            "compressor threshold_db must be in [-120, 0]",
            {"track": track_id, "field": "threshold_db", "value": threshold},
        )
    if not 1.0 <= ratio <= 100.0:
        _fail(
            "E_INVALID_COMPRESSOR",
            "compressor ratio must be in [1, 100]",
            {"track": track_id, "field": "ratio", "value": ratio},
        )
    if not 0.0 <= knee <= 48.0:
        _fail(
            "E_INVALID_COMPRESSOR",
            "compressor knee_db must be in [0, 48]",
            {"track": track_id, "field": "knee_db", "value": knee},
        )
    if not 0.01 <= attack <= 5000.0:
        _fail(
            "E_INVALID_COMPRESSOR",
            "compressor attack_ms must be in [0.01, 5000]",
            {"track": track_id, "field": "attack_ms", "value": attack},
        )
    if not 0.01 <= release <= 10000.0:
        _fail(
            "E_INVALID_COMPRESSOR",
            "compressor release_ms must be in [0.01, 10000]",
            {"track": track_id, "field": "release_ms", "value": release},
        )
    return CompressorConfig(threshold, ratio, knee, attack, release)


def _parse_limiter(raw, field, default_ceiling=-1.0):
    cfg = _mapping(raw, field)
    ceiling = _number(
        cfg.get("ceiling_dbfs", cfg.get("ceiling", default_ceiling)),
        field + ".ceiling_dbfs",
        "E_INVALID_LIMITER",
    )
    lookahead = _number(
        cfg.get("lookahead_ms", cfg.get("lookahead", 5.0)),
        field + ".lookahead_ms",
        "E_INVALID_LIMITER",
    )
    release = _number(
        cfg.get("release_ms", cfg.get("release", 80.0)),
        field + ".release_ms",
        "E_INVALID_LIMITER",
    )
    if not -60.0 <= ceiling <= 0.0:
        _fail(
            "E_INVALID_LIMITER",
            "limiter ceiling_dbfs must be in [-60, 0]",
            {"field": field + ".ceiling_dbfs", "value": ceiling},
        )
    if not 0.1 <= lookahead <= 50.0:
        _fail(
            "E_INVALID_LIMITER",
            "limiter lookahead_ms must be in [0.1, 50]",
            {"field": field + ".lookahead_ms", "value": lookahead},
        )
    if not 0.1 <= release <= 10000.0:
        _fail(
            "E_INVALID_LIMITER",
            "limiter release_ms must be in [0.1, 10000]",
            {"field": field + ".release_ms", "value": release},
        )
    return LimiterConfig(ceiling, lookahead, release)


def load_config(path, output_override=None, report_override=None, workers_override=None,
                chunk_frames_override=None):
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        _fail(
            "E_CONFIG_MISSING",
            "configuration file does not exist",
            {"path": str(config_path)},
            exit_code=2,
        )
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        _fail(
            "E_CONFIG_INVALID_JSON",
            "configuration is not valid JSON",
            {"path": str(config_path), "line": exc.lineno, "column": exc.colno},
            exit_code=2,
        )
    except OSError as exc:
        _fail(
            "E_CONFIG_READ",
            "unable to read configuration",
            {"path": str(config_path), "reason": str(exc)},
            exit_code=2,
        )
    if not isinstance(data, dict):
        _fail("E_CONFIG_INVALID", "top-level configuration must be an object", exit_code=2)

    base = config_path.parent
    output_raw = data.get("output", {})
    if isinstance(output_raw, str):
        output_map = {"path": output_raw}
    else:
        output_map = _mapping(output_raw, "output")
    output_path_raw = (
        output_map.get("path")
        or output_map.get("file")
        or output_map.get("output_path")
        or data.get("output_path")
    )
    if output_override:
        output_path_raw = output_override
    if output_path_raw is None:
        _fail(
            "E_CONFIG_MISSING_FIELD",
            "missing required field output.path",
            {"field": "output.path"},
        )
    output_path = (
        _resolve(Path.cwd(), output_path_raw)
        if output_override
        else _resolve(base, output_path_raw)
    )

    rate_raw = output_map.get(
        "sample_rate",
        data.get("output_sample_rate", data.get("sample_rate")),
    )
    if rate_raw is None:
        _fail(
            "E_CONFIG_MISSING_FIELD",
            "missing required field output.sample_rate",
            {"field": "output.sample_rate"},
        )
    output_rate = _integer(rate_raw, "output.sample_rate", "E_UNSUPPORTED_RATE")
    if not MIN_RATE <= output_rate <= MAX_RATE:
        _fail(
            "E_UNSUPPORTED_RATE",
            "output sample rate must be in [8000, 192000]",
            {"sample_rate": output_rate},
        )

    channels = _integer(
        output_map.get("channels", output_map.get("num_channels", 2)),
        "output.channels",
    )
    if not 1 <= channels <= 16:
        _fail("E_INVALID_OUTPUT", "output channels must be in [1, 16]", {"channels": channels})

    duration_seconds = None
    if "duration_seconds" in output_map or "duration_s" in output_map:
        duration_seconds = _number(
            output_map.get("duration_seconds", output_map.get("duration_s")),
            "output.duration_seconds",
        )
    elif "duration_ms" in output_map:
        duration_seconds = (
            _number(output_map["duration_ms"], "output.duration_ms") / 1000.0
        )
    elif "duration_seconds" in data or "duration_s" in data:
        duration_seconds = _number(
            data.get("duration_seconds", data.get("duration_s")),
            "duration_seconds",
        )
    elif "output_duration_ms" in data:
        duration_seconds = _number(data["output_duration_ms"], "output_duration_ms") / 1000.0
    elif "duration_ms" in data:
        duration_seconds = _number(data["duration_ms"], "duration_ms") / 1000.0
    if duration_seconds is not None and not 0.0 < duration_seconds <= MAX_DURATION_SECONDS:
        _fail(
            "E_INVALID_OUTPUT",
            "output duration must be in (0, 3600] seconds",
            {"duration_seconds": duration_seconds},
        )

    subtype = str(output_map.get("subtype", "PCM_32")).upper()
    if subtype not in {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}:
        _fail(
            "E_INVALID_OUTPUT",
            "unsupported output subtype",
            {"subtype": subtype},
        )
    if output_path.suffix.lower() != ".wav":
        _fail(
            "E_INVALID_OUTPUT",
            "output path must use the .wav extension",
            {"path": str(output_path)},
        )

    placement = str(
        output_map.get(
            "limiter_placement",
            output_map.get(
                "limiter_mode",
                output_map.get(
                    "limiter_scope",
                    data.get(
                        "limiter_placement",
                        data.get("limiter_mode", data.get("limiter_scope", "bus")),
                    ),
                ),
            ),
        )
    ).strip().lower().replace("-", "_").replace(" ", "_")
    if placement in {"track", "tracks", "per_track"}:
        placement = "per_track"
    elif placement in {"master", "mix_bus"}:
        placement = "bus"
    if placement not in {"bus", "per_track", "both"}:
        _fail(
            "E_INVALID_OUTPUT",
            "limiter_placement must be bus, per_track, or both",
            {"limiter_placement": placement},
        )

    bus_cfg = data.get("bus")
    bus_map = _mapping(bus_cfg, "bus")
    bus_limiter_raw = (
        bus_map.get("limiter")
        or data.get("bus_limiter")
        or data.get("master_limiter")
        or data.get("limiter")
        or output_map.get("limiter")
        or {}
    )
    bus_limiter = _parse_limiter(bus_limiter_raw, "bus.limiter", -1.0)

    report_raw = (
        output_map.get("report_path")
        or data.get("report_path")
        or data.get("report")
    )
    if report_override:
        report_raw = report_override
    if isinstance(report_raw, dict):
        report_raw = report_raw.get("path")
    if report_raw:
        report_path = (
            _resolve(Path.cwd(), report_raw)
            if report_override
            else _resolve(base, report_raw)
        )
    else:
        report_path = output_path.with_suffix(".report.json")
    if report_path.suffix.lower() != ".json":
        _fail(
            "E_INVALID_OUTPUT",
            "report path must use the .json extension",
            {"path": str(report_path)},
        )

    tracks_raw = data.get("tracks")
    if not isinstance(tracks_raw, list) or not tracks_raw:
        _fail(
            "E_CONFIG_MISSING_FIELD",
            "tracks must be a non-empty array",
            {"field": "tracks"},
        )
    tracks = []
    seen_ids = set()
    for index, raw in enumerate(tracks_raw):
        item = _mapping(raw, "tracks[{}]".format(index))
        track_id = str(item.get("id", item.get("name", "track_{:03d}".format(index + 1))))
        if not track_id:
            _fail("E_INVALID_TRACK", "track id must not be empty", {"index": index})
        if track_id in seen_ids:
            _fail(
                "E_INVALID_TRACK",
                "track id must be unique",
                {"track": track_id},
            )
        seen_ids.add(track_id)
        input_raw = item.get("input", item.get("input_path", item.get("path")))
        if input_raw is None:
            _fail(
                "E_CONFIG_MISSING_FIELD",
                "missing required field tracks[].input",
                {"track": track_id, "field": "input"},
            )
        input_path = _resolve(base, input_raw)
        offset_ms = _number(
            item.get(
                "offset_ms",
                item.get(
                    "start_ms",
                    item.get(
                        "start_offset_ms",
                        item.get("start_time_ms", item.get("offset", 0.0)),
                    ),
                ),
            ),
            "tracks[{}].offset_ms".format(track_id),
        )
        gain_db = _number(
            item.get("gain_db", item.get("gain", item.get("volume_db", 0.0))),
            "tracks[{}].gain_db".format(track_id),
        )
        if not -96.0 <= gain_db <= 24.0:
            _fail(
                "E_INVALID_TRACK",
                "track gain_db must be in [-96, 24]",
                {"track": track_id, "gain_db": gain_db},
            )
        if abs(offset_ms) > MAX_DURATION_SECONDS * 1000.0 + 1000.0:
            _fail(
                "E_INVALID_TRACK",
                "track offset_ms is outside the supported range",
                {"track": track_id, "offset_ms": offset_ms},
            )
        compressor = _parse_compressor(
            item.get("compressor", item.get("compression")),
            track_id,
        )
        limiter_raw = item.get("limiter")
        limiter = None
        if limiter_raw is not None and limiter_raw is not False:
            limiter = _parse_limiter(
                limiter_raw,
                "tracks[{}].limiter".format(track_id),
                bus_limiter.ceiling_dbfs,
            )
        elif placement in {"per_track", "both"}:
            limiter = LimiterConfig(
                ceiling_dbfs=bus_limiter.ceiling_dbfs,
                lookahead_ms=bus_limiter.lookahead_ms,
                release_ms=bus_limiter.release_ms,
            )
        channel_map = None
        if "channel_map" in item:
            raw_map = item["channel_map"]
            if not isinstance(raw_map, list) or len(raw_map) != channels:
                _fail(
                    "E_INVALID_TRACK",
                    "track channel_map must contain exactly output.channels entries",
                    {"track": track_id, "output_channels": channels},
                )
            parsed_map = []
            for output_channel, source_channel in enumerate(raw_map):
                if source_channel is None:
                    parsed_map.append(None)
                else:
                    index = _integer(
                        source_channel,
                        "tracks[{}].channel_map[{}]".format(track_id, output_channel),
                        "E_INVALID_TRACK",
                    )
                    if index < 0:
                        _fail(
                            "E_INVALID_TRACK",
                            "channel_map indices must be non-negative",
                            {"track": track_id, "index": index},
                        )
                    parsed_map.append(index)
            channel_map = tuple(parsed_map)
        tracks.append(
            TrackConfig(
                track_id=track_id,
                input_path=input_path,
                offset_ms=offset_ms,
                gain_db=gain_db,
                compressor=compressor,
                limiter=limiter,
                channel_map=channel_map,
            )
        )

    processing = _mapping(data.get("processing"), "processing")
    workers_raw = workers_override
    if workers_raw is None:
        workers_raw = processing.get(
            "workers",
            processing.get(
                "threads",
                processing.get(
                    "max_workers",
                    data.get(
                        "workers",
                        data.get("threads", data.get("max_workers", 0)),
                    ),
                ),
            ),
        )
    workers = _integer(workers_raw, "processing.workers")
    if workers == 0:
        import os

        workers = os.cpu_count() or 1
    if not 1 <= workers <= 256:
        _fail("E_CONFIG_INVALID", "workers must be in [1, 256]", {"workers": workers})

    chunk_raw = chunk_frames_override
    if chunk_raw is None:
        chunk_raw = processing.get(
            "chunk_frames",
            processing.get(
                "block_frames",
                data.get("chunk_frames", data.get("block_frames", 32768)),
            ),
        )
    chunk_frames = _integer(chunk_raw, "processing.chunk_frames")
    if not 1024 <= chunk_frames <= 1048576:
        _fail(
            "E_CONFIG_INVALID",
            "chunk_frames must be in [1024, 1048576]",
            {"chunk_frames": chunk_frames},
        )

    output = OutputConfig(
        path=output_path,
        sample_rate=output_rate,
        channels=channels,
        duration_seconds=duration_seconds,
        subtype=subtype,
        limiter_placement=placement,
        bus_limiter=bus_limiter,
        report_path=report_path,
    )
    return MixConfig(
        source_path=config_path,
        output=output,
        tracks=tuple(tracks),
        workers=workers,
        chunk_frames=chunk_frames,
    )
