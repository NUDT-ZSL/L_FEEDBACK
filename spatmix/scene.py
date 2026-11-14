"""Scene description parsing and validation.

Validation failures raise :class:`PipelineError` with one of the
structured error codes ``E_SCHEMA`` / ``E_TRACK_ORDER`` / ``E_RANGE``.
"""

import json
import math

from . import SCENE_VERSION
from .errors import PipelineError

WAVEFORMS = ("sine", "noise", "impulse")
LAYOUTS = ("2.0", "5.1")
AXES = ("x", "y", "z")

DEFAULT_CONFIG = {
    "speed_of_sound": 343.0,
    "reference_distance": 1.0,
    "max_audible_distance": 200.0,
    "air_absorption": 0.004,
    "doppler_min": 0.5,
    "doppler_max": 2.0,
    "doppler_smoothing": 0.03,
    "min_reflection_gain": 1e-3,
    "snapshot_interval": 0,
}


def _is_num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _req(obj, key, ctx):
    if not isinstance(obj, dict) or key not in obj:
        raise PipelineError("E_SCHEMA", f"{ctx}: missing required field '{key}'")
    return obj[key]


def _num(value, ctx, lo=None, hi=None, allow_equal_lo=True):
    if not _is_num(value):
        raise PipelineError("E_SCHEMA", f"{ctx}: expected a number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise PipelineError("E_RANGE", f"{ctx}: value must be finite, got {value!r}")
    if lo is not None and (value < lo or (not allow_equal_lo and value <= lo)):
        raise PipelineError("E_RANGE", f"{ctx}: value {value} below minimum {lo}")
    if hi is not None and value > hi:
        raise PipelineError("E_RANGE", f"{ctx}: value {value} above maximum {hi}")
    return value


def _int(value, ctx, lo=None, hi=None):
    if not isinstance(value, int) or isinstance(value, bool):
        raise PipelineError("E_SCHEMA", f"{ctx}: expected an integer")
    if lo is not None and value < lo:
        raise PipelineError("E_RANGE", f"{ctx}: value {value} below minimum {lo}")
    if hi is not None and value > hi:
        raise PipelineError("E_RANGE", f"{ctx}: value {value} above maximum {hi}")
    return value


def _vec3(value, ctx):
    if not isinstance(value, list) or len(value) != 3:
        raise PipelineError("E_SCHEMA", f"{ctx}: expected a 3-element array")
    return [_num(v, f"{ctx}[{i}]") for i, v in enumerate(value)]


def _str(value, ctx):
    if not isinstance(value, str) or not value:
        raise PipelineError("E_SCHEMA", f"{ctx}: expected a non-empty string")
    return value


def _parse_keyframes(raw, ctx):
    if not isinstance(raw, list) or len(raw) < 1:
        raise PipelineError("E_SCHEMA", f"{ctx}: expected a non-empty array of keyframes")
    frames = []
    prev_t = None
    for i, item in enumerate(raw):
        kctx = f"{ctx}[{i}]"
        t = _num(_req(item, "t", kctx), f"{kctx}.t", lo=0.0)
        pos = _vec3(_req(item, "pos", kctx), f"{kctx}.pos")
        heading = _num(item.get("heading", 0.0), f"{kctx}.heading")
        if prev_t is not None and t <= prev_t:
            raise PipelineError(
                "E_TRACK_ORDER",
                f"{kctx}.t: timestamps must be strictly increasing "
                f"({t} follows {prev_t})",
            )
        prev_t = t
        frames.append({"t": t, "pos": pos, "heading": heading})
    return frames


def _parse_source(raw, idx, sample_rate):
    ctx = f"sources[{idx}]"
    src_id = _str(_req(raw, "id", ctx), f"{ctx}.id")
    waveform = _str(_req(raw, "waveform", ctx), f"{ctx}.waveform")
    if waveform not in WAVEFORMS:
        raise PipelineError(
            "E_SCHEMA", f"{ctx}.waveform: must be one of {WAVEFORMS}, got {waveform!r}"
        )
    frequency = _num(
        _req(raw, "frequency", ctx), f"{ctx}.frequency",
        lo=0.0, allow_equal_lo=False, hi=sample_rate / 2.0,
    )
    duration = _num(_req(raw, "duration", ctx), f"{ctx}.duration",
                    lo=0.0, allow_equal_lo=False)
    pos = _vec3(_req(raw, "pos", ctx), f"{ctx}.pos")
    amplitude = _num(raw.get("amplitude", 1.0), f"{ctx}.amplitude", lo=0.0)
    material = raw.get("material", "default")
    material = _str(material, f"{ctx}.material")
    return {
        "id": src_id,
        "waveform": waveform,
        "frequency": frequency,
        "duration": duration,
        "pos": pos,
        "amplitude": amplitude,
        "material": material,
    }


def _parse_obstacle(raw, idx):
    ctx = f"obstacles[{idx}]"
    omin = _vec3(_req(raw, "min", ctx), f"{ctx}.min")
    omax = _vec3(_req(raw, "max", ctx), f"{ctx}.max")
    for axis in range(3):
        if omin[axis] >= omax[axis]:
            raise PipelineError(
                "E_RANGE",
                f"{ctx}: min[{axis}] ({omin[axis]}) must be < max[{axis}] ({omax[axis]})",
            )
    occlusion = _num(raw.get("occlusion", 0.8), f"{ctx}.occlusion", lo=0.0, hi=1.0)
    return {
        "id": str(raw.get("id", f"obstacle_{idx}")),
        "min": omin,
        "max": omax,
        "occlusion": occlusion,
    }


def _parse_reflector(raw, idx):
    ctx = f"reflectors[{idx}]"
    axis = _str(_req(raw, "axis", ctx), f"{ctx}.axis")
    if axis not in AXES:
        raise PipelineError("E_SCHEMA", f"{ctx}.axis: must be one of {AXES}")
    value = _num(_req(raw, "value", ctx), f"{ctx}.value")
    absorption = _num(raw.get("absorption", 0.3), f"{ctx}.absorption",
                      lo=0.0, hi=1.0, allow_equal_lo=True)
    if absorption >= 1.0:
        raise PipelineError("E_RANGE", f"{ctx}.absorption: must be < 1.0")
    return {
        "id": str(raw.get("id", f"reflector_{idx}")),
        "axis": AXES.index(axis),
        "value": value,
        "absorption": absorption,
    }


def _parse_config(raw):
    if raw is None:
        return dict(DEFAULT_CONFIG)
    if not isinstance(raw, dict):
        raise PipelineError("E_SCHEMA", "config: expected an object")
    cfg = dict(DEFAULT_CONFIG)
    for key, value in raw.items():
        if key not in DEFAULT_CONFIG:
            raise PipelineError("E_SCHEMA", f"config: unknown key '{key}'")
        cfg[key] = value
    cfg["speed_of_sound"] = _num(cfg["speed_of_sound"], "config.speed_of_sound",
                                 lo=0.0, allow_equal_lo=False)
    cfg["reference_distance"] = _num(cfg["reference_distance"], "config.reference_distance",
                                     lo=0.0, allow_equal_lo=False)
    cfg["max_audible_distance"] = _num(cfg["max_audible_distance"],
                                       "config.max_audible_distance",
                                       lo=0.0, allow_equal_lo=False)
    cfg["air_absorption"] = _num(cfg["air_absorption"], "config.air_absorption", lo=0.0)
    cfg["doppler_min"] = _num(cfg["doppler_min"], "config.doppler_min",
                              lo=0.0, allow_equal_lo=False)
    cfg["doppler_max"] = _num(cfg["doppler_max"], "config.doppler_max",
                              lo=0.0, allow_equal_lo=False)
    if cfg["doppler_max"] < cfg["doppler_min"]:
        raise PipelineError("E_RANGE", "config.doppler_max must be >= config.doppler_min")
    cfg["doppler_smoothing"] = _num(cfg["doppler_smoothing"], "config.doppler_smoothing",
                                    lo=0.0)
    cfg["min_reflection_gain"] = _num(cfg["min_reflection_gain"],
                                      "config.min_reflection_gain",
                                      lo=0.0, allow_equal_lo=False)
    cfg["snapshot_interval"] = _int(cfg["snapshot_interval"], "config.snapshot_interval",
                                    lo=0)
    return cfg


def parse_scene_dict(data):
    """Validate a scene dictionary and return a normalized scene."""
    if not isinstance(data, dict):
        raise PipelineError("E_SCHEMA", "scene root must be a JSON object")
    version = _req(data, "version", "scene")
    if version != SCENE_VERSION:
        raise PipelineError(
            "E_SCHEMA", f"scene.version: unsupported version {version!r} "
            f"(expected {SCENE_VERSION})"
        )
    sample_rate = _num(_req(data, "sample_rate", "scene"), "scene.sample_rate",
                       lo=8000.0, hi=384000.0)
    if float(sample_rate) != int(sample_rate):
        raise PipelineError("E_RANGE", "scene.sample_rate: must be an integer rate")
    sample_rate = int(sample_rate)
    duration = _num(_req(data, "duration", "scene"), "scene.duration",
                    lo=0.0, allow_equal_lo=False)
    frame_size = _int(_req(data, "frame_size", "scene"), "scene.frame_size",
                      lo=16, hi=16384)
    layout = _str(_req(data, "layout", "scene"), "scene.layout")
    if layout not in LAYOUTS:
        raise PipelineError("E_SCHEMA", f"scene.layout: must be one of {LAYOUTS}")

    listener = _req(data, "listener", "scene")
    keyframes = _parse_keyframes(_req(listener, "keyframes", "listener"),
                                 "listener.keyframes")

    raw_sources = _req(data, "sources", "scene")
    if not isinstance(raw_sources, list) or len(raw_sources) < 1:
        raise PipelineError("E_SCHEMA", "sources: expected a non-empty array")
    sources = []
    seen_ids = set()
    for i, raw in enumerate(raw_sources):
        src = _parse_source(raw, i, float(sample_rate))
        if src["id"] in seen_ids:
            raise PipelineError("E_SCHEMA", f"sources[{i}].id: duplicate id {src['id']!r}")
        seen_ids.add(src["id"])
        sources.append(src)

    obstacles = [_parse_obstacle(o, i)
                 for i, o in enumerate(data.get("obstacles", []) or [])]
    reflectors = [_parse_reflector(r, i)
                  for i, r in enumerate(data.get("reflectors", []) or [])]

    materials = {}
    raw_materials = data.get("materials", {}) or {}
    if not isinstance(raw_materials, dict):
        raise PipelineError("E_SCHEMA", "materials: expected an object")
    for label, props in raw_materials.items():
        mctx = f"materials.{label}"
        materials[label] = {
            "air_absorption": _num(_req(props, "air_absorption", mctx),
                                   f"{mctx}.air_absorption", lo=0.0),
        }

    config = _parse_config(data.get("config"))

    return {
        "version": SCENE_VERSION,
        "sample_rate": sample_rate,
        "duration": duration,
        "frame_size": frame_size,
        "layout": layout,
        "listener": {"keyframes": keyframes},
        "sources": sources,
        "obstacles": obstacles,
        "reflectors": reflectors,
        "materials": materials,
        "config": config,
    }


def load_scene(path):
    """Load and validate a scene JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise PipelineError("E_SCHEMA", f"{path}: invalid JSON: {exc}") from exc
    except OSError as exc:
        raise PipelineError("E_IO", f"{path}: {exc}") from exc
    return parse_scene_dict(data)
