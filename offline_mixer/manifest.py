"""Manifest parsing and fail-fast parameter validation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .errors import ManifestError, ParameterError


SUPPORTED_SAMPLE_RATE = 48_000
SUPPORTED_CHANNEL_COUNTS = {2: "stereo", 6: "5.1"}
ROLE_ALIASES = {
    "dialog": "dialogue",
    "dialogue": "dialogue",
    "speech": "dialogue",
    "voice": "dialogue",
    "对白": "dialogue",
    "music": "music",
    "bgm": "music",
    "音乐": "music",
    "sfx": "sfx",
    "effect": "sfx",
    "effects": "sfx",
    "sound_effect": "sfx",
    "音效": "sfx",
}


@dataclass(frozen=True)
class NoiseGateParams:
    window_ms: float = 20.0
    hop_ms: float = 10.0
    threshold_db: float = -45.0
    attenuation_db: float = 18.0
    snr_margin_db: float = 6.0
    noise_estimate_window_ms: float = 500.0
    noise_percentile: float = 20.0
    attack_ms: float = 5.0
    release_ms: float = 30.0


@dataclass(frozen=True)
class LoudnessParams:
    window_ms: float = 400.0
    tolerance_lu: float = 1.0
    attack_ms: float = 30.0
    release_ms: float = 200.0
    max_gain_db: float = 30.0


@dataclass(frozen=True)
class DuckingParams:
    depth_db: float = 9.0
    attack_ms: float = 5.0
    release_ms: float = 300.0
    vad_threshold_db: float = -40.0
    vad_snr_db: float = 6.0
    vad_window_ms: float = 10.0
    min_gap_ms: float = 50.0


@dataclass(frozen=True)
class ChannelMap:
    matrix: tuple[tuple[float, ...], ...] | None = None
    method: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class SegmentSpec:
    id: str
    start_s: float
    end_s: float
    index: int


@dataclass(frozen=True)
class TrackSpec:
    id: str
    path: Path
    role: str
    target_lufs: float
    gain_db: float
    declared_channels: int | None
    noise_gate: NoiseGateParams
    loudness: LoudnessParams
    ducking: DuckingParams
    channel_map: ChannelMap
    index: int


@dataclass(frozen=True)
class MixManifest:
    source_path: Path
    output_channels: int
    channel_layout: str
    target_lufs: float
    duration_s: float | None
    noise_gate: NoiseGateParams
    loudness: LoudnessParams
    ducking: DuckingParams
    tracks: tuple[TrackSpec, ...]
    segments: tuple[SegmentSpec, ...]
    raw: dict[str, Any] = field(repr=False, compare=False)


def load_manifest(path: str | Path) -> MixManifest:
    manifest_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError("MIX1001", f"manifest not found: {manifest_path}") from exc
    except OSError as exc:
        raise ManifestError("MIX1001", f"cannot read manifest: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(
            "MIX1001",
            f"manifest is not valid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
        ) from exc
    if not isinstance(raw, dict):
        raise ManifestError("MIX1002", "manifest root must be a JSON object")
    return parse_manifest(raw, manifest_path)


def parse_manifest(raw: dict[str, Any], source_path: Path) -> MixManifest:
    sample_rate = _number(raw.get("sample_rate", SUPPORTED_SAMPLE_RATE), "sample_rate")
    if sample_rate != SUPPORTED_SAMPLE_RATE:
        raise ParameterError(
            "MIX1003",
            f"sample_rate must be {SUPPORTED_SAMPLE_RATE} Hz for this tool",
        )

    output_section = raw.get("output")
    if output_section is None:
        output_section = {}
    if not isinstance(output_section, dict):
        raise ManifestError("MIX1002", "output must be an object")

    layout_value = _first_present(
        raw, ("channel_layout", "output_layout", "layout")
    )
    layout_from_output = _first_present(
        output_section, ("channel_layout", "layout")
    )
    if layout_value is not None and layout_from_output is not None:
        if _normalise_layout(layout_value) != _normalise_layout(layout_from_output):
            raise ParameterError("MIX1003", "conflicting output channel layouts")
    layout_value = layout_from_output if layout_from_output is not None else layout_value

    channels_value = _first_present(
        raw, ("output_channels", "channels")
    )
    channels_from_output = _first_present(
        output_section, ("channels", "output_channels")
    )
    if channels_value is not None and channels_from_output is not None:
        if _integer(channels_value, "output_channels") != _integer(
            channels_from_output, "output_channels"
        ):
            raise ParameterError("MIX1003", "conflicting output channel counts")
    channels_value = (
        channels_from_output if channels_from_output is not None else channels_value
    )

    if layout_value is not None:
        layout = _normalise_layout(layout_value)
        derived_channels = 2 if layout == "stereo" else 6
        if channels_value is not None:
            channels = _integer(channels_value, "output_channels")
            if channels != derived_channels:
                raise ParameterError(
                    "MIX1003",
                    f"output layout {layout} requires {derived_channels} channels, got {channels}",
                )
        else:
            channels = derived_channels
    elif channels_value is not None:
        channels = _integer(channels_value, "output_channels")
        if channels not in SUPPORTED_CHANNEL_COUNTS:
            raise ParameterError(
                "MIX1003",
                "output_channels must be 2 (stereo) or 6 (5.1)",
            )
        layout = SUPPORTED_CHANNEL_COUNTS[channels]
    else:
        channels = 2
        layout = "stereo"

    target_lufs = _number(
        raw.get(
            "target_lufs",
            raw.get("dialogue_target_lufs", raw.get("loudness_target_lufs", -18.0)),
        ),
        "target_lufs",
    )
    _range(target_lufs, -60.0, -5.0, "target_lufs")

    raw_duration = raw.get("duration_s")
    duration_s = None
    if raw_duration is not None:
        duration_s = _number(raw_duration, "duration_s")
        _range(duration_s, 0.01, 21_600.0, "duration_s")

    global_gate = _object_or_empty(
        raw.get("noise_gate", raw.get("gate")), "noise_gate"
    )
    global_gate = _with_flat_aliases(
        global_gate,
        raw,
        {
            "gate_window_ms": "window_ms",
            "gate_hop_ms": "hop_ms",
            "gate_threshold_db": "threshold_db",
            "snr_margin_db": "snr_margin_db",
            "noise_estimate_window_ms": "noise_estimate_window_ms",
        },
    )
    noise_gate = _parse_noise_gate(
        global_gate,
        "noise_gate",
    )
    global_loudness = _object_or_empty(raw.get("loudness"), "loudness")
    global_loudness = _with_flat_aliases(
        global_loudness,
        raw,
        {
            "loudness_window_ms": "window_ms",
            "short_term_window_ms": "window_ms",
            "loudness_attack_ms": "attack_ms",
            "loudness_release_ms": "release_ms",
        },
    )
    loudness = _parse_loudness(
        global_loudness,
        "loudness",
    )
    global_ducking = _object_or_empty(raw.get("ducking"), "ducking")
    global_ducking = _with_flat_aliases(
        global_ducking,
        raw,
        {
            "duck_depth_db": "depth_db",
            "duck_release_ms": "release_ms",
            "duck_attack_ms": "attack_ms",
            "vad_threshold_db": "vad_threshold_db",
        },
    )
    ducking = _parse_ducking(
        global_ducking,
        "ducking",
    )

    tracks_raw = raw.get("tracks", raw.get("audio_tracks", raw.get("inputs")))
    if not isinstance(tracks_raw, list) or not tracks_raw:
        raise ManifestError("MIX1002", "tracks must be a non-empty JSON array")

    tracks: list[TrackSpec] = []
    track_ids: set[str] = set()
    for index, item in enumerate(tracks_raw):
        if not isinstance(item, dict):
            raise ManifestError("MIX1002", f"tracks[{index}] must be an object")
        track = _parse_track(
            item,
            index=index,
            source_dir=source_path.parent,
            output_channels=channels,
            default_target_lufs=target_lufs,
            default_noise_gate=noise_gate,
            default_loudness=loudness,
            default_ducking=ducking,
        )
        if track.id in track_ids:
            raise ParameterError("MIX1003", f"duplicate track id: {track.id}")
        track_ids.add(track.id)
        tracks.append(track)

    segments = _parse_segments(
        raw.get("segments", raw.get("ranges")),
        duration_s,
    )
    return MixManifest(
        source_path=source_path,
        output_channels=channels,
        channel_layout=layout,
        target_lufs=target_lufs,
        duration_s=duration_s,
        noise_gate=noise_gate,
        loudness=loudness,
        ducking=ducking,
        tracks=tuple(tracks),
        segments=tuple(segments),
        raw=raw,
    )


def _parse_track(
    item: dict[str, Any],
    *,
    index: int,
    source_dir: Path,
    output_channels: int,
    default_target_lufs: float,
    default_noise_gate: NoiseGateParams,
    default_loudness: LoudnessParams,
    default_ducking: DuckingParams,
) -> TrackSpec:
    prefix = f"tracks[{index}]"
    role_raw = item.get("role", item.get("type", item.get("kind")))
    if not isinstance(role_raw, str):
        raise ManifestError("MIX1002", f"{prefix}.role must be a string")
    role = ROLE_ALIASES.get(role_raw.strip().lower())
    if role is None:
        raise ParameterError(
            "MIX1003",
            f"{prefix}.role must be dialogue, music, or sfx (got {role_raw!r})",
        )

    path_raw = item.get("path", item.get("file", item.get("wav_path")))
    if not isinstance(path_raw, str) or not path_raw.strip():
        raise ManifestError("MIX1002", f"{prefix}.path must be a non-empty string")
    track_path = Path(path_raw).expanduser()
    if not track_path.is_absolute():
        track_path = (source_dir / track_path).resolve()

    default_id = f"{role}_{index}"
    track_id_raw = item.get("id", item.get("name", default_id))
    if not isinstance(track_id_raw, str) or not track_id_raw.strip():
        raise ManifestError("MIX1002", f"{prefix}.id must be a non-empty string")
    track_id = track_id_raw.strip()
    if len(track_id) > 128:
        raise ParameterError("MIX1003", f"{prefix}.id is too long")

    target = _number(
        item.get(
            "target_lufs",
            item.get(
                "target_loudness_lufs",
                item.get("target_loudness", default_target_lufs),
            ),
        ),
        f"{prefix}.target_lufs",
    )
    _range(target, -60.0, -5.0, f"{prefix}.target_lufs")

    gain_db = _number(item.get("gain_db", 0.0), f"{prefix}.gain_db")
    _range(gain_db, -60.0, 24.0, f"{prefix}.gain_db")

    declared_channels_raw = _first_present(
        item, ("input_channels", "source_channels", "channels")
    )
    declared_channels = None
    if declared_channels_raw is not None:
        declared_channels = _integer(declared_channels_raw, f"{prefix}.input_channels")
        _range(declared_channels, 1, 16, f"{prefix}.input_channels")

    gate_override = _object_or_empty(
        item.get("noise_gate", item.get("gate")), f"{prefix}.noise_gate"
    )
    gate_override = _with_flat_aliases(
        gate_override,
        item,
        {
            "gate_window_ms": "window_ms",
            "gate_hop_ms": "hop_ms",
            "gate_threshold_db": "threshold_db",
            "snr_margin_db": "snr_margin_db",
        },
    )
    noise_gate = _parse_noise_gate(
        gate_override,
        f"{prefix}.noise_gate",
        base=default_noise_gate,
    )

    loudness_override = _object_or_empty(
        item.get("loudness"), f"{prefix}.loudness"
    )
    loudness_override = _with_flat_aliases(
        loudness_override,
        item,
        {
            "loudness_window_ms": "window_ms",
            "short_term_window_ms": "window_ms",
            "loudness_attack_ms": "attack_ms",
            "loudness_release_ms": "release_ms",
        },
    )
    loudness = _parse_loudness(
        loudness_override,
        f"{prefix}.loudness",
        base=default_loudness,
    )

    ducking_override = _object_or_empty(
        item.get("ducking"), f"{prefix}.ducking"
    )
    ducking_override = _with_flat_aliases(
        ducking_override,
        item,
        {
            "duck_depth_db": "depth_db",
            "duck_release_ms": "release_ms",
            "duck_attack_ms": "attack_ms",
            "vad_threshold_db": "vad_threshold_db",
        },
    )
    ducking = _parse_ducking(
        ducking_override,
        f"{prefix}.ducking",
        base=default_ducking,
    )

    channel_map = _parse_channel_map(
        item,
        prefix=prefix,
        output_channels=output_channels,
        declared_channels=declared_channels,
    )

    return TrackSpec(
        id=track_id,
        path=track_path,
        role=role,
        target_lufs=target,
        gain_db=gain_db,
        declared_channels=declared_channels,
        noise_gate=noise_gate,
        loudness=loudness,
        ducking=ducking,
        channel_map=channel_map,
        index=index,
    )


def _parse_noise_gate(
    values: dict[str, Any],
    prefix: str,
    *,
    base: NoiseGateParams | None = None,
) -> NoiseGateParams:
    base = base or NoiseGateParams()
    window_ms = _number(
        values.get("window_ms", values.get("window_length_ms", base.window_ms)),
        f"{prefix}.window_ms",
    )
    _range(window_ms, 5.0, 1_000.0, f"{prefix}.window_ms")
    hop_ms = _number(values.get("hop_ms", base.hop_ms), f"{prefix}.hop_ms")
    _range(hop_ms, 1.0, window_ms, f"{prefix}.hop_ms")
    threshold_db = _number(
        values.get("threshold_db", values.get("gate_threshold_db", base.threshold_db)),
        f"{prefix}.threshold_db",
    )
    _range(threshold_db, -100.0, -5.0, f"{prefix}.threshold_db")
    attenuation_db = _number(
        values.get(
            "attenuation_db",
            values.get("max_attenuation_db", base.attenuation_db),
        ),
        f"{prefix}.attenuation_db",
    )
    _range(attenuation_db, 0.0, 60.0, f"{prefix}.attenuation_db")
    snr_margin_db = _number(
        values.get("snr_margin_db", base.snr_margin_db), f"{prefix}.snr_margin_db"
    )
    _range(snr_margin_db, 0.0, 24.0, f"{prefix}.snr_margin_db")
    estimate_window_ms = _number(
        values.get(
            "noise_estimate_window_ms",
            values.get("estimate_window_ms", base.noise_estimate_window_ms),
        ),
        f"{prefix}.noise_estimate_window_ms",
    )
    _range(
        estimate_window_ms,
        window_ms,
        10_000.0,
        f"{prefix}.noise_estimate_window_ms",
    )
    percentile = _number(
        values.get("noise_percentile", base.noise_percentile),
        f"{prefix}.noise_percentile",
    )
    _range(percentile, 0.0, 50.0, f"{prefix}.noise_percentile")
    attack_ms = _number(values.get("attack_ms", base.attack_ms), f"{prefix}.attack_ms")
    _range(attack_ms, 0.0, 1_000.0, f"{prefix}.attack_ms")
    release_ms = _number(
        values.get("release_ms", base.release_ms), f"{prefix}.release_ms"
    )
    _range(release_ms, 0.0, 5_000.0, f"{prefix}.release_ms")
    return NoiseGateParams(
        window_ms=window_ms,
        hop_ms=hop_ms,
        threshold_db=threshold_db,
        attenuation_db=attenuation_db,
        snr_margin_db=snr_margin_db,
        noise_estimate_window_ms=estimate_window_ms,
        noise_percentile=percentile,
        attack_ms=attack_ms,
        release_ms=release_ms,
    )


def _parse_loudness(
    values: dict[str, Any],
    prefix: str,
    *,
    base: LoudnessParams | None = None,
) -> LoudnessParams:
    base = base or LoudnessParams()
    window_ms = _number(
        values.get("window_ms", values.get("short_term_window_ms", base.window_ms)),
        f"{prefix}.window_ms",
    )
    if abs(window_ms - 400.0) > 1e-9:
        raise ParameterError(
            "MIX1003",
            f"{prefix}.window_ms must be exactly 400 ms (BS.1770 short-term)",
        )
    tolerance_lu = _number(
        values.get("tolerance_lu", base.tolerance_lu), f"{prefix}.tolerance_lu"
    )
    _range(tolerance_lu, 0.0, 1.0, f"{prefix}.tolerance_lu")
    attack_ms = _number(values.get("attack_ms", base.attack_ms), f"{prefix}.attack_ms")
    _range(attack_ms, 0.0, 500.0, f"{prefix}.attack_ms")
    release_ms = _number(
        values.get("release_ms", base.release_ms), f"{prefix}.release_ms"
    )
    _range(release_ms, 0.0, 5_000.0, f"{prefix}.release_ms")
    max_gain_db = _number(
        values.get("max_gain_db", base.max_gain_db), f"{prefix}.max_gain_db"
    )
    _range(max_gain_db, 0.0, 40.0, f"{prefix}.max_gain_db")
    return LoudnessParams(
        window_ms=window_ms,
        tolerance_lu=tolerance_lu,
        attack_ms=attack_ms,
        release_ms=release_ms,
        max_gain_db=max_gain_db,
    )


def _parse_ducking(
    values: dict[str, Any],
    prefix: str,
    *,
    base: DuckingParams | None = None,
) -> DuckingParams:
    base = base or DuckingParams()
    depth_db = _number(
        values.get("depth_db", values.get("duck_db", base.depth_db)),
        f"{prefix}.depth_db",
    )
    _range(depth_db, 0.0, 48.0, f"{prefix}.depth_db")
    attack_ms = _number(values.get("attack_ms", base.attack_ms), f"{prefix}.attack_ms")
    _range(attack_ms, 0.0, 500.0, f"{prefix}.attack_ms")
    release_ms = _number(
        values.get("release_ms", base.release_ms), f"{prefix}.release_ms"
    )
    _range(release_ms, 0.0, 10_000.0, f"{prefix}.release_ms")
    vad_threshold_db = _number(
        values.get(
            "vad_threshold_db",
            values.get("activity_threshold_db", base.vad_threshold_db),
        ),
        f"{prefix}.vad_threshold_db",
    )
    _range(vad_threshold_db, -100.0, -5.0, f"{prefix}.vad_threshold_db")
    vad_snr_db = _number(
        values.get("vad_snr_db", base.vad_snr_db), f"{prefix}.vad_snr_db"
    )
    _range(vad_snr_db, 0.0, 40.0, f"{prefix}.vad_snr_db")
    vad_window_ms = _number(
        values.get("vad_window_ms", base.vad_window_ms),
        f"{prefix}.vad_window_ms",
    )
    _range(vad_window_ms, 1.0, 100.0, f"{prefix}.vad_window_ms")
    min_gap_ms = _number(
        values.get("min_gap_ms", base.min_gap_ms), f"{prefix}.min_gap_ms"
    )
    _range(min_gap_ms, 0.0, 1_000.0, f"{prefix}.min_gap_ms")
    return DuckingParams(
        depth_db=depth_db,
        attack_ms=attack_ms,
        release_ms=release_ms,
        vad_threshold_db=vad_threshold_db,
        vad_snr_db=vad_snr_db,
        vad_window_ms=vad_window_ms,
        min_gap_ms=min_gap_ms,
    )


def _parse_channel_map(
    item: dict[str, Any],
    *,
    prefix: str,
    output_channels: int,
    declared_channels: int | None,
) -> ChannelMap:
    candidate: Any = None
    for key in (
        "channel_map",
        "mapping",
        "mapping_matrix",
        "channel_mapping",
    ):
        if key in item:
            candidate = item[key]
            break
    if candidate is None and "upmix_matrix" in item:
        candidate = item["upmix_matrix"]
    if candidate is None and "downmix_matrix" in item:
        candidate = item["downmix_matrix"]
    if candidate is None and "conversion" in item:
        conversion = item["conversion"]
        if isinstance(conversion, dict):
            candidate = conversion.get("matrix", conversion.get("channel_map"))
            if candidate is None and "upmix" in conversion:
                candidate = conversion["upmix"]
            if candidate is None and "downmix" in conversion:
                candidate = conversion["downmix"]
        elif isinstance(conversion, list):
            candidate = conversion

    if candidate is None:
        return ChannelMap()

    description: str | None = None
    method: str | None = None
    matrix_raw: Any = candidate
    if isinstance(candidate, dict):
        matrix_raw = candidate.get("matrix", candidate.get("coefficients"))
        description_raw = candidate.get(
            "description", candidate.get("name", candidate.get("label"))
        )
        if description_raw is not None:
            if not isinstance(description_raw, str):
                raise ParameterError(
                    "MIX1003", f"{prefix}.channel_map.description must be a string"
                )
            description = description_raw
        method_raw = candidate.get(
            "method", candidate.get("mode", candidate.get("direction"))
        )
        if method_raw is not None:
            if not isinstance(method_raw, str):
                raise ParameterError(
                    "MIX1003", f"{prefix}.channel_map.method must be a string"
                )
            method = _normalise_mapping_method(method_raw, prefix)

    method_from_track = item.get("mapping_method", item.get("conversion_method"))
    if method_from_track is not None:
        if not isinstance(method_from_track, str):
            raise ParameterError(
                "MIX1003", f"{prefix}.mapping_method must be a string"
            )
        track_method = _normalise_mapping_method(method_from_track, prefix)
        if method is not None and method != track_method:
            raise ParameterError(
                "MIX1003", f"{prefix} declares conflicting mapping methods"
            )
        method = track_method

    if not isinstance(matrix_raw, list) or not matrix_raw:
        raise ParameterError(
            "MIX1003", f"{prefix}.channel_map must contain a non-empty matrix"
        )
    matrix: list[tuple[float, ...]] = []
    for row_index, row in enumerate(matrix_raw):
        if not isinstance(row, list) or not row:
            raise ParameterError(
                "MIX1003",
                f"{prefix}.channel_map matrix row {row_index} must be a non-empty array",
            )
        numeric_row = tuple(
            _number(value, f"{prefix}.channel_map[{row_index}]") for value in row
        )
        if not all(math.isfinite(value) for value in numeric_row):
            raise ParameterError(
                "MIX1003", f"{prefix}.channel_map matrix must be finite"
            )
        matrix.append(numeric_row)
    widths = {len(row) for row in matrix}
    if len(widths) != 1:
        raise ParameterError(
            "MIX1003", f"{prefix}.channel_map matrix rows must have equal length"
        )
    in_channels = widths.pop()
    if len(matrix) != output_channels:
        raise ParameterError(
            "MIX1003",
            f"{prefix}.channel_map has {len(matrix)} rows; "
            f"output has {output_channels} channels",
        )
    if declared_channels is not None and in_channels != declared_channels:
        raise ParameterError(
            "MIX1003",
            f"{prefix}.channel_map has {in_channels} columns; "
            f"declared input_channels is {declared_channels}",
        )
    if declared_channels is not None:
        expected_method = (
            "identity"
            if declared_channels == output_channels
            else "upmix"
            if declared_channels < output_channels
            else "downmix"
        )
        if method is not None and method != expected_method:
            raise ParameterError(
                "MIX1003",
                f"{prefix}.mapping_method {method} conflicts with "
                f"{declared_channels}->{output_channels} conversion",
            )
        if method is None:
            method = expected_method
    elif method is None:
        method = "explicit"

    return ChannelMap(
        matrix=tuple(matrix),
        method=method,
        description=description,
    )


def _parse_segments(value: Any, duration_s: float | None) -> tuple[SegmentSpec, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ManifestError("MIX1002", "segments must be an array")
    if not value:
        return ()
    segments: list[SegmentSpec] = []
    ids: set[str] = set()
    for index, item in enumerate(value):
        if isinstance(item, dict):
            start_raw = _first_present(item, ("start_s", "start", "start_seconds"))
            end_raw = _first_present(item, ("end_s", "end", "end_seconds"))
            id_raw = item.get("id", item.get("name", f"segment_{index:03d}"))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            start_raw, end_raw = item
            id_raw = f"segment_{index:03d}"
        else:
            raise ManifestError(
                "MIX1002",
                f"segments[{index}] must be an object or a [start, end] pair",
            )
        if not isinstance(id_raw, str) or not id_raw.strip():
            raise ParameterError(
                "MIX1003", f"segments[{index}].id must be a non-empty string"
            )
        segment_id = id_raw.strip()
        if len(segment_id) > 128:
            raise ParameterError("MIX1003", f"segments[{index}].id is too long")
        if segment_id in ids:
            raise ParameterError(
                "MIX1003", f"duplicate segment id: {segment_id}"
            )
        ids.add(segment_id)
        if start_raw is None or end_raw is None:
            raise ManifestError(
                "MIX1002", f"segments[{index}] requires start and end times"
            )
        start_s = _number(start_raw, f"segments[{index}].start_s")
        end_s = _number(end_raw, f"segments[{index}].end_s")
        if start_s < 0 or end_s <= start_s:
            raise ParameterError(
                "MIX1003",
                f"segments[{index}] must satisfy 0 <= start_s < end_s",
            )
        _range(start_s, 0.0, 86_400.0, f"segments[{index}].start_s")
        _range(end_s, 0.0, 86_400.0, f"segments[{index}].end_s")
        if duration_s is not None and end_s > duration_s + 1e-9:
            raise ParameterError(
                "MIX1003",
                f"segments[{index}].end_s ({end_s}) exceeds duration_s ({duration_s})",
            )
        segments.append(
            SegmentSpec(
                id=segment_id,
                start_s=start_s,
                end_s=end_s,
                index=index,
            )
        )
    return tuple(segments)


def _object_or_empty(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ManifestError("MIX1002", f"{name} must be an object")
    return value


def _first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _with_flat_aliases(
    section: dict[str, Any],
    source: dict[str, Any],
    aliases: dict[str, str],
) -> dict[str, Any]:
    if not section:
        result: dict[str, Any] = {}
    else:
        result = dict(section)
    for source_key, destination_key in aliases.items():
        if destination_key not in result and source_key in source:
            result[destination_key] = source[source_key]
    return result


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterError("MIX1003", f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ParameterError("MIX1003", f"{name} must be finite")
    return result


def _integer(value: Any, name: str) -> int:
    number = _number(value, name)
    if not number.is_integer():
        raise ParameterError("MIX1003", f"{name} must be an integer")
    return int(number)


def _range(value: float, minimum: float, maximum: float, name: str) -> None:
    if value < minimum or value > maximum:
        raise ParameterError(
            "MIX1003",
            f"{name}={value:g} is outside the allowed range "
            f"[{minimum:g}, {maximum:g}]",
        )


def _normalise_layout(value: Any) -> str:
    if not isinstance(value, str):
        raise ParameterError("MIX1003", "channel_layout must be a string")
    normalized = value.strip().lower().replace(" ", "")
    if normalized in {"stereo", "2.0", "2ch", "twochannel"}:
        return "stereo"
    if normalized in {"5.1", "surround", "5.1surround", "6ch"}:
        return "5.1"
    raise ParameterError(
        "MIX1003", "channel_layout must be stereo (2.0) or 5.1"
    )


def _normalise_mapping_method(value: str, prefix: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "up": "upmix",
        "upmix": "upmix",
        "down": "downmix",
        "downmix": "downmix",
        "identity": "identity",
        "pass_through": "identity",
        "passthrough": "identity",
        "explicit": "explicit",
        "matrix": "explicit",
    }
    if normalized not in aliases:
        raise ParameterError(
            "MIX1003",
            f"{prefix}.mapping_method must be upmix, downmix, identity, or explicit",
        )
    return aliases[normalized]


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    cleaned = cleaned or "segment"
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
    if cleaned.upper() in reserved:
        cleaned = f"_{cleaned}"
    return cleaned
