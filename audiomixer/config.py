"""Scene configuration model, JSON loading and semantic validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ErrorCode, MixerError


SUPPORTED_SAMPLE_WIDTHS = (2, 3, 4)
SUPPORTED_CHANNELS = (1, 2)


@dataclass(frozen=True)
class FadeSpec:
    """Fade applied to one event.

    Durations are seconds. ``curve`` is ``linear`` or ``equal_power``.
    ``crossfade_with`` optionally names the previous event on the same track.
    """

    fade_in: float = 0.0
    fade_out: float = 0.0
    curve: str = "linear"
    crossfade: float = 0.0
    crossfade_curve: str = "equal_power"


@dataclass(frozen=True)
class EventSpec:
    event_id: str
    start: float
    duration: float
    frequency: float
    amplitude: float
    pan: float = 0.0
    asset: str | None = None
    sample_rate: int | None = None
    channels: int = 1
    sample_width: int = 2
    fades: FadeSpec = field(default_factory=FadeSpec)


@dataclass(frozen=True)
class TrackSpec:
    track_id: str
    gain_db: float = 0.0
    muted: bool = False
    events: tuple[EventSpec, ...] = ()


@dataclass(frozen=True)
class LoudnessSpec:
    target_lufs: float = -23.0
    peak_limit_dbfs: float = -1.0
    # When the peak ceiling forces a gain below the target gain, accept it only
    # if the loudness shortfall stays within this tolerance (LU); otherwise the
    # scene fails with E_PEAK_LIMIT.
    peak_shortfall_tolerance_lu: float = 0.5
    true_peak: bool = False


@dataclass(frozen=True)
class SceneConfig:
    scene_id: str
    project_sample_rate: int
    duration: float
    output_path: str
    report_path: str
    asset_dir: str
    tracks: tuple[TrackSpec, ...]
    loudness: LoudnessSpec
    output_sample_width: int = 4
    generate_assets: bool = True


def _require_number(obj: dict, key: str, scene_id: str) -> float:
    if key not in obj or isinstance(obj[key], bool) or not isinstance(obj[key], (int, float)):
        raise MixerError(
            ErrorCode.INVALID_CONFIG,
            f"scene {scene_id!r}: field {key!r} must be a number",
            {"field": key},
        )
    return float(obj[key])


def _parse_fades(raw: dict | None, scene_id: str, event_id: str) -> FadeSpec:
    if not raw:
        return FadeSpec()
    curve = raw.get("curve", "linear")
    cross_curve = raw.get("crossfade_curve", "equal_power")
    for name, value in (("curve", curve), ("crossfade_curve", cross_curve)):
        if value not in ("linear", "equal_power"):
            raise MixerError(
                ErrorCode.INVALID_CONFIG,
                f"scene {scene_id!r} event {event_id!r}: unsupported fade curve {value!r}",
                {"field": name, "value": value},
            )
    return FadeSpec(
        fade_in=float(raw.get("fade_in", 0.0)),
        fade_out=float(raw.get("fade_out", 0.0)),
        curve=curve,
        crossfade=float(raw.get("crossfade", 0.0)),
        crossfade_curve=cross_curve,
    )


def _parse_event(raw: dict, scene_id: str) -> EventSpec:
    for key in ("id", "start", "duration", "frequency", "amplitude"):
        if key not in raw:
            raise MixerError(
                ErrorCode.INVALID_CONFIG,
                f"scene {scene_id!r}: event missing required field {key!r}",
                {"field": key},
            )
    event_id = str(raw["id"])
    pan = float(raw.get("pan", 0.0))
    if not -1.0 <= pan <= 1.0:
        raise MixerError(
            ErrorCode.OUT_OF_RANGE,
            f"scene {scene_id!r} event {event_id!r}: pan={pan} outside [-1, 1]",
            {"event_id": event_id, "pan": pan},
        )
    channels = int(raw.get("channels", 1))
    sample_width = int(raw.get("sample_width", 2))
    if channels not in SUPPORTED_CHANNELS:
        raise MixerError(
            ErrorCode.UNSUPPORTED_FORMAT,
            f"scene {scene_id!r} event {event_id!r}: {channels} channels are not supported",
            {"event_id": event_id, "channels": channels},
        )
    if sample_width not in SUPPORTED_SAMPLE_WIDTHS:
        raise MixerError(
            ErrorCode.UNSUPPORTED_FORMAT,
            f"scene {scene_id!r} event {event_id!r}: sample width {sample_width} bytes unsupported",
            {"event_id": event_id, "sample_width": sample_width},
        )
    sample_rate = raw.get("sample_rate")
    return EventSpec(
        event_id=event_id,
        start=_require_number(raw, "start", scene_id),
        duration=_require_number(raw, "duration", scene_id),
        frequency=_require_number(raw, "frequency", scene_id),
        amplitude=_require_number(raw, "amplitude", scene_id),
        pan=pan,
        asset=raw.get("asset"),
        sample_rate=int(sample_rate) if sample_rate is not None else None,
        channels=channels,
        sample_width=sample_width,
        fades=_parse_fades(raw.get("fades"), scene_id, event_id),
    )


def _parse_track(raw: dict, scene_id: str) -> TrackSpec:
    if "id" not in raw:
        raise MixerError(ErrorCode.INVALID_CONFIG, f"scene {scene_id!r}: track missing 'id'")
    events = tuple(_parse_event(event_raw, scene_id) for event_raw in raw.get("events", []))
    return TrackSpec(
        track_id=str(raw["id"]),
        gain_db=float(raw.get("gain_db", 0.0)),
        muted=bool(raw.get("muted", False)),
        events=events,
    )


def parse_scene(raw: dict) -> SceneConfig:
    scene_id = str(raw.get("id", "<unknown>"))
    for key in ("project_sample_rate", "duration", "output", "report", "asset_dir"):
        if key not in raw:
            raise MixerError(
                ErrorCode.INVALID_CONFIG,
                f"scene {scene_id!r}: missing required field {key!r}",
                {"field": key},
            )
    project_sr = int(raw["project_sample_rate"])
    if project_sr <= 0:
        raise MixerError(
            ErrorCode.INVALID_CONFIG,
            f"scene {scene_id!r}: project_sample_rate must be positive",
        )
    duration = float(raw["duration"])
    if duration <= 0:
        raise MixerError(
            ErrorCode.INVALID_CONFIG, f"scene {scene_id!r}: duration must be positive"
        )
    loud_raw = raw.get("loudness", {})
    loudness = LoudnessSpec(
        target_lufs=float(loud_raw.get("target_lufs", -23.0)),
        peak_limit_dbfs=float(loud_raw.get("peak_limit_dbfs", -1.0)),
        peak_shortfall_tolerance_lu=float(
            loud_raw.get("peak_shortfall_tolerance_lu", 0.5)
        ),
        true_peak=bool(loud_raw.get("true_peak", False)),
    )
    output_width = int(raw.get("output_sample_width", 4))
    if output_width not in SUPPORTED_SAMPLE_WIDTHS:
        raise MixerError(
            ErrorCode.UNSUPPORTED_FORMAT,
            f"scene {scene_id!r}: output sample width {output_width} bytes unsupported",
        )
    tracks = tuple(_parse_track(track_raw, scene_id) for track_raw in raw.get("tracks", []))
    generate_assets = bool(raw.get("generate", True))
    scene = SceneConfig(
        scene_id=scene_id,
        project_sample_rate=project_sr,
        duration=duration,
        output_path=str(raw["output"]),
        report_path=str(raw["report"]),
        asset_dir=str(raw["asset_dir"]),
        tracks=tracks,
        loudness=loudness,
        output_sample_width=output_width,
        generate_assets=generate_assets,
    )
    validate_timeline(scene)
    return scene


def validate_timeline(scene: SceneConfig) -> None:
    """Reject events that cross the project end or whose fades exceed their length."""
    sr = scene.project_sample_rate
    for track in scene.tracks:
        for event in track.events:
            if event.start < 0.0:
                raise MixerError(
                    ErrorCode.OUT_OF_RANGE,
                    f"event {event.event_id!r}: negative start time {event.start}",
                    {"event_id": event.event_id, "start": event.start},
                )
            end = event.start + event.duration
            if end > scene.duration + 1.0 / sr:
                raise MixerError(
                    ErrorCode.OUT_OF_RANGE,
                    f"event {event.event_id!r} ends at {end:.6f}s, beyond project duration "
                    f"{scene.duration:.6f}s",
                    {
                        "event_id": event.event_id,
                        "event_end": end,
                        "project_duration": scene.duration,
                    },
                )
            fades = event.fades
            if fades.fade_in < 0 or fades.fade_out < 0 or fades.crossfade < 0:
                raise MixerError(
                    ErrorCode.OUT_OF_RANGE,
                    f"event {event.event_id!r}: fade durations must be non-negative",
                    {"event_id": event.event_id},
                )
            if fades.fade_in + fades.fade_out > event.duration + 1e-9:
                raise MixerError(
                    ErrorCode.OUT_OF_RANGE,
                    f"event {event.event_id!r}: fade-in+fade-out ({fades.fade_in}+{fades.fade_out}) "
                    f"exceeds event duration {event.duration}",
                    {"event_id": event.event_id},
                )
            if fades.crossfade > event.duration + 1e-9:
                raise MixerError(
                    ErrorCode.OUT_OF_RANGE,
                    f"event {event.event_id!r}: crossfade {fades.crossfade} exceeds duration "
                    f"{event.duration}",
                    {"event_id": event.event_id},
                )


def load_scene(path: str | Path) -> SceneConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise MixerError(
            ErrorCode.MISSING_ASSET,
            f"scene configuration not found: {config_path}",
            {"path": str(config_path)},
        )
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MixerError(
            ErrorCode.INVALID_CONFIG,
            f"invalid JSON in {config_path}: {exc.msg} at line {exc.lineno}",
            {"path": str(config_path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    return parse_scene(raw)


def load_batch(path: str | Path) -> list[SceneConfig]:
    config_path = Path(path)
    if not config_path.is_file():
        raise MixerError(
            ErrorCode.MISSING_ASSET,
            f"batch configuration not found: {config_path}",
            {"path": str(config_path)},
        )
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MixerError(
            ErrorCode.INVALID_CONFIG,
            f"invalid JSON in {config_path}: {exc.msg} at line {exc.lineno}",
            {"path": str(config_path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    scenes_raw = raw.get("scenes") if isinstance(raw, dict) else raw
    if not isinstance(scenes_raw, list) or not scenes_raw:
        raise MixerError(
            ErrorCode.INVALID_CONFIG,
            "batch config must contain a non-empty 'scenes' array",
            {"path": str(config_path)},
        )
    return [parse_scene(item) for item in scenes_raw]
