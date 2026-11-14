"""Scene data model, YAML parsing and deterministic validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .curves import Curve
from .errors import (
    E_BUS_CONFIG,
    E_CURVE,
    E_INVALID_LAYOUT,
    E_INVALID_VALUE,
    E_MISSING_FIELD,
    E_SAMPLE_RATE,
    E_TRAJECTORY,
    E_EVENT_OVERLAP,
    SceneError,
)
from .spatial import LAYOUT_CHANNELS, LAYOUT_MONO, LAYOUT_STEREO, LAYOUT_51
from .trajectory import ListenerTrajectory, TrajectoryKey

VALID_LAYOUTS = (LAYOUT_MONO, LAYOUT_STEREO, LAYOUT_51)


@dataclass(frozen=True)
class EventSpec:
    event_id: str
    time: float
    source: str
    position: tuple[float, float, float]
    voice: str | None
    seed_offset: int
    bus: str
    gain_curve: Curve
    lowpass_curve: Curve


@dataclass(frozen=True)
class BusSpec:
    name: str
    gain_db: float
    limiter: bool
    delay_seconds: float
    delay_feedback: float
    route_to: str | None


@dataclass
class Scene:
    sample_rate: int
    layout: str
    global_seed: int
    block_size: int
    duration: float | None
    min_distance: float
    events: list[EventSpec] = field(default_factory=list)
    buses: dict[str, BusSpec] = field(default_factory=dict)
    bus_order: list[str] = field(default_factory=list)
    trajectory: ListenerTrajectory | None = None
    scene_dir: Path = field(default_factory=lambda: Path("."))

    @property
    def channels(self) -> int:
        return LAYOUT_CHANNELS[self.layout]


def load_scene(path: str | Path) -> Scene:
    scene_path = Path(path)
    try:
        raw = yaml.safe_load(scene_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SceneError(E_MISSING_FIELD, f"scene file not found: {scene_path}") from exc
    except yaml.YAMLError as exc:
        raise SceneError(E_INVALID_VALUE, f"invalid YAML syntax: {exc}") from exc
    if not isinstance(raw, dict):
        raise SceneError(
            E_MISSING_FIELD, "scene root must be a mapping of parameters"
        )
    return _build_scene(raw, scene_path.resolve().parent)


def _require(mapping: dict, key: str, path: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise SceneError(E_MISSING_FIELD, f"missing required field '{key}'", path)
    return mapping[key]


def _build_scene(raw: dict, scene_dir: Path) -> Scene:
    sample_rate = _require(raw, "sample_rate", "$")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise SceneError(
            E_SAMPLE_RATE, f"sample_rate must be a positive integer, got {sample_rate!r}", "$"
        )

    layout = str(_require(raw, "layout", "$"))
    if layout not in VALID_LAYOUTS:
        raise SceneError(
            E_INVALID_LAYOUT,
            f"layout must be one of {list(VALID_LAYOUTS)}, got {layout!r}",
            "$.layout",
        )

    global_seed = int(raw.get("global_seed", 0))
    block_size = int(raw.get("block_size", 4096))
    if block_size <= 0:
        raise SceneError(E_INVALID_VALUE, "block_size must be positive", "$.block_size")
    min_distance = float(raw.get("min_distance", 1.0))
    if min_distance <= 0.0:
        raise SceneError(
            E_INVALID_VALUE, "min_distance must be > 0", "$.min_distance"
        )
    duration = raw.get("duration")
    if duration is not None:
        duration = float(duration)
        if duration < 0.0:
            raise SceneError(E_INVALID_VALUE, "duration must be >= 0", "$.duration")

    trajectory = _parse_trajectory(raw.get("trajectory", []))
    bus_input = raw.get("buses")
    if not bus_input:
        bus_input = [
            {"name": "master", "gain_db": 0.0, "limiter": True}
        ]
    buses, bus_order = _parse_buses(bus_input)
    events = _parse_events(raw.get("events", []), buses)
    _check_overlap(events)

    return Scene(
        sample_rate=sample_rate,
        layout=layout,
        global_seed=global_seed,
        block_size=block_size,
        duration=duration,
        min_distance=min_distance,
        events=events,
        buses=buses,
        bus_order=bus_order,
        trajectory=trajectory,
        scene_dir=scene_dir,
    )


def _parse_curve(value: Any, path: str, default: float) -> Curve:
    if value is None:
        return Curve.constant(default)
    if isinstance(value, (int, float)):
        return Curve.constant(float(value))
    if not isinstance(value, list) or not value:
        raise SceneError(E_CURVE, f"curve at {path} must be a non-empty list", path)
    points: list[tuple[float, float]] = []
    last_t = -np.inf
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, dict) or "t" not in item or "value" not in item:
            raise SceneError(
                E_CURVE, f"curve point must define 't' and 'value'", item_path
            )
        t_point = float(item["t"])
        v_point = float(item["value"])
        if not np.isfinite(t_point) or not np.isfinite(v_point):
            raise SceneError(E_CURVE, "curve values must be finite", item_path)
        if t_point < last_t:
            raise SceneError(
                E_CURVE, "curve time points must be monotonic increasing", item_path
            )
        last_t = t_point
        points.append((t_point, v_point))
    if points[0][0] != 0.0 or points[-1][0] != 1.0:
        raise SceneError(
            E_CURVE, "curve must span normalised time from t=0.0 to t=1.0", path
        )
    return Curve(tuple(points))


def _parse_trajectory(value: Any) -> ListenerTrajectory:
    if not value:
        value = [
            {"time": 0.0, "position": [0.0, 0.0, 0.0], "yaw": 0.0}
        ]
    if not isinstance(value, list) or not value:
        raise SceneError(E_TRAJECTORY, "trajectory must be a non-empty list", "$.trajectory")
    keys: list[TrajectoryKey] = []
    last_time = -np.inf
    for index, item in enumerate(value):
        path = f"$.trajectory[{index}]"
        if not isinstance(item, dict):
            raise SceneError(E_TRAJECTORY, "trajectory key must be a mapping", path)
        time_s = float(_require(item, "time", path))
        position = _require(item, "position", path)
        if not isinstance(position, list) or len(position) != 3:
            raise SceneError(E_TRAJECTORY, "position must be [x, y, z]", f"{path}.position")
        yaw = float(item.get("yaw", 0.0))
        if time_s < last_time:
            raise SceneError(E_TRAJECTORY, "trajectory times must be monotonic increasing", path)
        last_time = time_s
        keys.append(
            TrajectoryKey(time_s, tuple(float(v) for v in position), yaw)
        )
    return ListenerTrajectory(keys)


def _parse_buses(value: Any) -> tuple[dict[str, BusSpec], list[str]]:
    if value is None:
        value = []
    if not isinstance(value, list):
        raise SceneError(E_BUS_CONFIG, "buses must be a list", "$.buses")
    buses: dict[str, BusSpec] = {}
    order: list[str] = []
    for index, item in enumerate(value):
        path = f"$.buses[{index}]"
        if not isinstance(item, dict):
            raise SceneError(E_BUS_CONFIG, "bus must be a mapping", path)
        name = str(_require(item, "name", path))
        if name in buses:
            raise SceneError(E_BUS_CONFIG, f"duplicate bus name {name!r}", path)
        delay_seconds = float(item.get("delay_seconds", 0.0))
        delay_feedback = float(item.get("delay_feedback", 0.0))
        if delay_seconds < 0.0:
            raise SceneError(E_BUS_CONFIG, "delay_seconds must be >= 0", path)
        if not 0.0 <= delay_feedback < 1.0:
            raise SceneError(
                E_BUS_CONFIG, "delay_feedback must be in [0, 1)", path
            )
        bus = BusSpec(
            name=name,
            gain_db=float(item.get("gain_db", 0.0)),
            limiter=bool(item.get("limiter", False)),
            delay_seconds=delay_seconds,
            delay_feedback=delay_feedback,
            route_to=item.get("route_to"),
        )
        buses[name] = bus
        order.append(name)

    # Validate routes and detect cycles, then produce a dependency order.
    for bus in buses.values():
        if bus.route_to is not None and bus.route_to not in buses:
            raise SceneError(
                E_BUS_CONFIG,
                f"bus {bus.name!r} routes to unknown bus {bus.route_to!r}",
                f"$.buses[{bus.name}]",
            )
    resolved = _topological_order(buses, order)
    return buses, resolved


def _topological_order(buses: dict[str, BusSpec], order: list[str]) -> list[str]:
    visited: dict[str, int] = {}
    result: list[str] = []

    def visit(name: str, stack: list[str]) -> None:
        state = visited.get(name, 0)
        if state == 1:
            cycle = " -> ".join(stack + [name])
            raise SceneError(E_BUS_CONFIG, f"bus routing cycle detected: {cycle}")
        if state == 2:
            return
        visited[name] = 1
        target = buses[name].route_to
        if target is not None:
            visit(target, stack + [name])
        visited[name] = 2
        result.append(name)

    for name in order:
        visit(name, [])
    return result


def _parse_events(value: Any, buses: dict[str, BusSpec]) -> list[EventSpec]:
    if value is None:
        value = []
    if not isinstance(value, list):
        raise SceneError(E_MISSING_FIELD, "events must be a list", "$.events")
    events: list[EventSpec] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(value):
        path = f"$.events[{index}]"
        if not isinstance(item, dict):
            raise SceneError(E_MISSING_FIELD, "event must be a mapping", path)
        event_id = str(_require(item, "id", path))
        if event_id in seen_ids:
            raise SceneError(E_MISSING_FIELD, f"duplicate event id {event_id!r}", path)
        seen_ids.add(event_id)
        time_s = float(_require(item, "time", path))
        if time_s < 0.0:
            raise SceneError(E_INVALID_VALUE, "event time must be >= 0", path)
        source = str(_require(item, "source", path))
        position = _require(item, "position", path)
        if not isinstance(position, list) or len(position) != 3:
            raise SceneError(E_INVALID_VALUE, "position must be [x, y, z]", f"{path}.position")
        bus_name = str(item.get("bus", "master"))
        if bus_name not in buses:
            raise SceneError(
                E_BUS_CONFIG,
                f"event {event_id!r} routes to unknown bus {bus_name!r}",
                path,
            )
        gain_curve = _parse_curve(item.get("gain_curve"), f"{path}.gain_curve", 0.0)
        lowpass_curve = _parse_curve(
            item.get("lowpass_curve"), f"{path}.lowpass_curve", 20000.0
        )
        events.append(
            EventSpec(
                event_id=event_id,
                time=time_s,
                source=source,
                position=tuple(float(v) for v in position),
                voice=item.get("voice"),
                seed_offset=int(item.get("seed_offset", 0)),
                bus=bus_name,
                gain_curve=gain_curve,
                lowpass_curve=lowpass_curve,
            )
        )
    events.sort(key=lambda event: event.time)
    return events


def _check_overlap(events: list[EventSpec]) -> None:
    """Events sharing the same non-null ``voice`` may not start at the same time.

    Source durations are only discoverable while streaming each WAV, so the
    static scene check enforces the deterministic scheduling rule that one
    voice cannot host two coincident events; the renderer keeps voice state
    independent for any later-arriving events.
    """
    by_voice: dict[str, list[EventSpec]] = {}
    for event in events:
        if event.voice is not None:
            by_voice.setdefault(event.voice, []).append(event)
    for voice, grouped in by_voice.items():
        grouped.sort(key=lambda event: event.time)
        for previous, current in zip(grouped, grouped[1:]):
            if current.time <= previous.time:
                raise SceneError(
                    E_EVENT_OVERLAP,
                    (
                        f"events {previous.event_id!r} and {current.event_id!r} "
                        f"start at the same time on shared voice {voice!r}"
                    ),
                    f"$.events[voice={voice}]",
                )
