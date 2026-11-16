"""Input parsing and validation for charts, events and WAV layer tracks."""

from dataclasses import dataclass, field
import json
import math
import os

import numpy as np
import soundfile as sf

from .errors import (
    BadWavError,
    ChartError,
    EventError,
    GainError,
    SampleRateError,
    UnknownLayerError,
)

REQUIRED_SAMPLE_RATE = 44100
DEFAULT_CROSSFADE_MS = 10.0
VALID_ACTIONS = ("enable", "disable")


@dataclass
class LayerSpec:
    name: str
    file: str
    samples: np.ndarray  # float32 mono, range [-1, 1)


@dataclass
class EventSpec:
    time_ms: float
    layer: str
    action: str
    gain: float = 1.0


@dataclass
class Chart:
    bpm: float
    total_duration_ms: float
    sample_rate: int
    crossfade_ms: float
    loop_bars: int | None
    beats_per_bar: int | None
    layers: dict
    events: list = field(default_factory=list)

    @property
    def loop_enabled(self):
        return self.loop_bars is not None and self.beats_per_bar is not None


def _load_json(path, missing_code_class):
    if not os.path.isfile(path):
        raise missing_code_class(f"required JSON file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise missing_code_class(
            f"invalid JSON in {os.path.basename(path)}: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})",
            detail={"file": os.path.basename(path)},
        ) from exc
    except OSError as exc:
        raise missing_code_class(f"cannot read {path}: {exc}") from exc


def _finite_number(value, field_name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _event_time_ms(raw_event, index):
    if "time_ms" in raw_event:
        value = raw_event["time_ms"]
        if not _finite_number(value, "time_ms"):
            raise EventError(
                f"event #{index}: time_ms must be a finite number",
                detail={"event_index": index},
            )
        return float(value)
    if "time_s" in raw_event:
        value = raw_event["time_s"]
        if not _finite_number(value, "time_s"):
            raise EventError(
                f"event #{index}: time_s must be a finite number",
                detail={"event_index": index},
            )
        return float(value) * 1000.0
    raise EventError(
        f"event #{index}: missing 'time_ms' (or 'time_s')",
        detail={"event_index": index},
    )


def _load_layer_wav(path, layer_name):
    if not os.path.isfile(path):
        raise BadWavError(
            f"layer '{layer_name}' WAV not found: {path}",
            detail={"layer": layer_name},
        )
    try:
        info = sf.info(path)
    except (sf.LibsndfileError, RuntimeError, OSError, ValueError) as exc:
        raise BadWavError(
            f"illegal or unreadable WAV header for layer '{layer_name}': "
            f"{exc}",
            detail={"layer": layer_name},
        ) from exc
    if info.samplerate != REQUIRED_SAMPLE_RATE:
        raise SampleRateError(
            f"layer '{layer_name}' sample rate is {info.samplerate} Hz, "
            f"required {REQUIRED_SAMPLE_RATE} Hz",
            detail={"layer": layer_name, "samplerate": info.samplerate},
        )
    if info.channels != 1:
        raise BadWavError(
            f"layer '{layer_name}' must be mono, found {info.channels} "
            f"channel(s)",
            detail={"layer": layer_name, "channels": info.channels},
        )
    if info.subtype != "PCM_16":
        raise BadWavError(
            f"layer '{layer_name}' must be 16-bit PCM, found subtype "
            f"{info.subtype}",
            detail={"layer": layer_name, "subtype": info.subtype},
        )
    try:
        samples, _ = sf.read(path, dtype="float32", always_2d=False)
    except (sf.LibsndfileError, RuntimeError, OSError, ValueError) as exc:
        raise BadWavError(
            f"failed decoding layer '{layer_name}': {exc}",
            detail={"layer": layer_name},
        ) from exc
    if samples.ndim != 1:
        raise BadWavError(
            f"layer '{layer_name}' decoded as multi-channel data",
            detail={"layer": layer_name},
        )
    return np.ascontiguousarray(samples, dtype=np.float32)


def _parse_chart(raw):
    if not isinstance(raw, dict):
        raise ChartError("chart JSON root must be an object")
    bpm = raw.get("bpm")
    if not _finite_number(bpm, "bpm") or float(bpm) <= 0:
        raise ChartError("chart 'bpm' must be a positive finite number")
    bpm = float(bpm)

    duration_ms = raw.get("total_duration_ms", raw.get("duration_ms"))
    if not _finite_number(duration_ms, "total_duration_ms") or float(duration_ms) < 0:
        raise ChartError(
            "chart 'total_duration_ms' must be a non-negative finite number"
        )
    duration_ms = float(duration_ms)

    crossfade_ms = float(raw.get("crossfade_ms", DEFAULT_CROSSFADE_MS))
    if not math.isfinite(crossfade_ms) or crossfade_ms < 0:
        raise ChartError("'crossfade_ms' must be a non-negative finite number")

    loop_bars = raw.get("loop_bars")
    beats_per_bar = raw.get("beats_per_bar")
    if (loop_bars is None) != (beats_per_bar is None):
        raise ChartError(
            "'loop_bars' and 'beats_per_bar' must be declared together"
        )
    if loop_bars is not None:
        if not isinstance(loop_bars, int) or isinstance(loop_bars, bool):
            raise ChartError("'loop_bars' must be a positive integer")
        if not isinstance(beats_per_bar, int) or isinstance(beats_per_bar, bool):
            raise ChartError("'beats_per_bar' must be a positive integer")
        if loop_bars <= 0 or beats_per_bar <= 0:
            raise ChartError("'loop_bars' and 'beats_per_bar' must be positive")

    raw_layers = raw.get("layers")
    if not isinstance(raw_layers, list) or len(raw_layers) < 3:
        raise ChartError("chart must declare at least three layers")
    layers = {}
    for entry in raw_layers:
        if not isinstance(entry, dict):
            raise ChartError("each layer entry must be an object")
        name = entry.get("name")
        file_name = entry.get("file")
        if not isinstance(name, str) or not name:
            raise ChartError("each layer requires a non-empty 'name'")
        if not isinstance(file_name, str) or not file_name:
            raise ChartError(f"layer '{name}' requires a 'file' path")
        if name in layers:
            raise ChartError(f"duplicate layer name '{name}'")
        layers[name] = file_name
    return bpm, duration_ms, crossfade_ms, loop_bars, beats_per_bar, layers


def _parse_events(raw, known_layers):
    if not isinstance(raw, dict) or not isinstance(raw.get("events"), list):
        raise EventError("events JSON must contain an 'events' array")
    events = []
    for index, raw_event in enumerate(raw["events"]):
        if not isinstance(raw_event, dict):
            raise EventError(
                f"event #{index} must be an object",
                detail={"event_index": index},
            )
        time_ms = _event_time_ms(raw_event, index)
        if time_ms < 0:
            raise EventError(
                f"event #{index}: time must be non-negative",
                detail={"event_index": index, "time_ms": time_ms},
            )
        layer = raw_event.get("layer")
        if not isinstance(layer, str) or not layer:
            raise EventError(
                f"event #{index}: 'layer' must be a non-empty string",
                detail={"event_index": index},
            )
        if layer not in known_layers:
            raise UnknownLayerError(
                f"event #{index} references undefined layer '{layer}'",
                detail={"event_index": index, "layer": layer},
            )
        action = raw_event.get("action")
        if action not in VALID_ACTIONS:
            raise EventError(
                f"event #{index}: action must be one of {list(VALID_ACTIONS)}",
                detail={"event_index": index, "action": action},
            )
        gain = 1.0
        if "gain" in raw_event and raw_event["gain"] is not None:
            raw_gain = raw_event["gain"]
            if isinstance(raw_gain, bool) or not isinstance(
                raw_gain, (int, float)
            ) or not math.isfinite(float(raw_gain)):
                raise GainError(
                    f"event #{index}: gain must be a finite number",
                    detail={"event_index": index, "gain": raw_gain},
                )
            gain = float(raw_gain)
        events.append(
            EventSpec(
                time_ms=time_ms, layer=layer, action=action, gain=gain
            )
        )
    return events


def load_inputs(input_dir):
    """Load and validate every input file under ``input_dir``."""
    chart_path = os.path.join(input_dir, "chart.json")
    events_path = os.path.join(input_dir, "events.json")
    raw_chart = _load_json(chart_path, ChartError)
    (
        bpm,
        duration_ms,
        crossfade_ms,
        loop_bars,
        beats_per_bar,
        layer_files,
    ) = _parse_chart(raw_chart)

    layers = {}
    for name, file_name in layer_files.items():
        wav_path = os.path.join(input_dir, file_name)
        layers[name] = LayerSpec(
            name=name,
            file=file_name,
            samples=_load_layer_wav(wav_path, name),
        )

    raw_events = _load_json(events_path, EventError)
    events = _parse_events(raw_events, set(layers))

    return Chart(
        bpm=bpm,
        total_duration_ms=duration_ms,
        sample_rate=REQUIRED_SAMPLE_RATE,
        crossfade_ms=crossfade_ms,
        loop_bars=loop_bars,
        beats_per_bar=beats_per_bar,
        layers=layers,
        events=events,
    )
