"""Scene JSON parsing and validation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import soundfile as sf

from .errors import SceneValidationError


@dataclass(frozen=True)
class AmbienceSpec:
    file: Path
    loop_start: float
    loop_end: float
    gain: float


@dataclass(frozen=True)
class EffectEventSpec:
    time: float
    file: Path
    gain: float
    name: str


@dataclass(frozen=True)
class MusicLayerSpec:
    file: Path
    start: float
    loops: int
    gain: float


@dataclass(frozen=True)
class DialogueIntervalSpec:
    start: float
    end: float
    duck_depth_db: float

    @property
    def duck_gain(self) -> float:
        return 10.0 ** (-self.duck_depth_db / 20.0)


@dataclass(frozen=True)
class SceneSpec:
    path: Path
    scene_id: str
    duration: float
    ambience: AmbienceSpec
    events: Tuple[EffectEventSpec, ...]
    music_layers: Tuple[MusicLayerSpec, ...]
    dialogue_intervals: Tuple[DialogueIntervalSpec, ...]
    camera_cuts: Tuple[float, ...]


_TOP_LEVEL_ALIASES: Dict[str, Tuple[str, ...]] = {
    "duration": ("duration", "duration_seconds", "length"),
    "ambience": ("ambience", "ambient", "ambience_bed", "ambient_bed"),
    "events": (
        "events",
        "sound_events",
        "effect_events",
        "sfx_events",
        "effects",
        "sfx",
    ),
    "music_layers": ("music_layers", "music"),
    "dialogue_intervals": (
        "dialogue_intervals",
        "dialogues",
        "dialogue",
        "dialogue_segments",
    ),
    "camera_cuts": (
        "camera_cuts",
        "camera_cut_times",
        "cuts",
        "shot_changes",
        "shot_cuts",
        "cut_points",
    ),
}


def _where(base: str, key: Any) -> str:
    return f"{base}.{key}" if base else str(key)


def _pick(mapping: Mapping[str, Any], keys: Iterable[str], where: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    expected = " or ".join(keys)
    raise SceneValidationError(f"{where}: missing required field ({expected})")


def _as_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SceneValidationError(f"{where}: expected an object")
    return value


def _as_list(value: Any, where: str) -> List[Any]:
    if not isinstance(value, list):
        raise SceneValidationError(f"{where}: expected an array")
    return value


def _as_number(
    value: Any,
    where: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SceneValidationError(f"{where}: expected a number")
    result = float(value)
    if not math.isfinite(result):
        raise SceneValidationError(f"{where}: expected a finite number")
    if minimum is not None and result < minimum:
        raise SceneValidationError(f"{where}: must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise SceneValidationError(f"{where}: must be <= {maximum}")
    if strictly_positive and result <= 0.0:
        raise SceneValidationError(f"{where}: must be > 0")
    return result


def _as_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SceneValidationError(f"{where}: expected a non-empty string")
    return value.strip()


def _relative_asset(scene_dir: Path, raw_value: Any, where: str) -> Path:
    raw = _as_string(raw_value, where)
    normalized = raw.replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", normalized) or normalized.startswith("//"):
        raise SceneValidationError(f"{where}: audio path must be relative")
    candidate = Path(normalized)
    if candidate.is_absolute():
        raise SceneValidationError(f"{where}: audio path must be relative")
    resolved = (scene_dir / candidate).resolve()
    if not resolved.is_file():
        raise SceneValidationError(f"{where}: audio file not found: {raw}")
    return resolved


def _gain(mapping: Mapping[str, Any], where: str, *, required: bool) -> float:
    has_linear = "gain" in mapping
    has_db = "gain_db" in mapping
    if has_linear and has_db:
        raise SceneValidationError(f"{where}: use only one of gain or gain_db")
    if not has_linear and not has_db:
        if required:
            raise SceneValidationError(f"{where}: missing required field (gain/gain_db)")
        return 1.0
    if has_linear:
        return _as_number(mapping["gain"], _where(where, "gain"), minimum=0.0)
    gain_db = _as_number(mapping["gain_db"], _where(where, "gain_db"))
    if gain_db > 300.0:
        raise SceneValidationError(f"{where}.gain_db: value is unreasonably large")
    return float(10.0 ** (gain_db / 20.0))


def _audio_info(path: Path, where: str) -> Tuple[int, int, int]:
    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise SceneValidationError(f"{where}: unable to read WAV metadata: {exc}") from exc
    if info.samplerate <= 0 or info.frames <= 0:
        raise SceneValidationError(f"{where}: audio has no samples")
    if info.channels not in (1, 2):
        raise SceneValidationError(
            f"{where}: only mono or stereo source audio is supported "
            f"(got {info.channels} channels)"
        )
    return int(info.samplerate), int(info.frames), int(info.channels)


def _parse_samples(
    scene_dir: Path, data: Mapping[str, Any]
) -> Dict[str, Path]:
    raw = None
    for key in (
        "samples",
        "sound_files",
        "effects_files",
        "effect_samples",
        "sfx_files",
    ):
        if key in data:
            raw = data[key]
            break
    if raw is None:
        if "effects" in data and isinstance(data["effects"], dict):
            effects_value = data["effects"]
            if all(
                isinstance(value, (str, dict))
                and not (isinstance(value, dict) and "time" in value)
                for value in effects_value.values()
            ):
                raw = effects_value
    if raw is None:
        return {}
    if isinstance(raw, dict):
        mapping_items = list(raw.items())
    elif isinstance(raw, list):
        mapping_items = []
        for index, item in enumerate(raw):
            where = f"scene.samples[{index}]"
            entry = _as_mapping(item, where)
            name = _pick(entry, ("name", "id"), where)
            file_value = _pick(entry, ("file", "path"), where)
            mapping_items.append((name, file_value))
    else:
        raise SceneValidationError("scene.samples: expected an object or array")
    samples: Dict[str, Path] = {}
    for name, value in mapping_items:
        sample_name = _as_string(name, "scene.samples key")
        where = f"scene.samples.{sample_name}"
        if isinstance(value, str):
            file_value: Any = value
        elif isinstance(value, dict):
            file_value = _pick(value, ("file", "path"), where)
        else:
            raise SceneValidationError(f"{where}: expected a path or object")
        samples[sample_name] = _relative_asset(scene_dir, file_value, f"{where}.file")
    return samples


def _parse_ambience(scene_dir: Path, raw: Any) -> AmbienceSpec:
    mapping = _as_mapping(raw, "scene.ambience")
    file_path = _relative_asset(
        scene_dir, _pick(mapping, ("file", "path"), "scene.ambience"), "scene.ambience.file"
    )
    loop_raw: Any = mapping.get("loop")
    if loop_raw is not None:
        loop = _as_mapping(loop_raw, "scene.ambience.loop")
        start_value = _pick(loop, ("start", "loop_start"), "scene.ambience.loop")
        end_value = _pick(loop, ("end", "loop_end"), "scene.ambience.loop")
    else:
        start_value = _pick(
            mapping, ("loop_start", "start"), "scene.ambience"
        )
        end_value = _pick(mapping, ("loop_end", "end"), "scene.ambience")
    loop_start = _as_number(
        start_value, "scene.ambience.loop_start", minimum=0.0
    )
    loop_end = _as_number(end_value, "scene.ambience.loop_end", minimum=0.0)
    if loop_end <= loop_start:
        raise SceneValidationError(
            "scene.ambience: loop_end must be greater than loop_start"
        )
    sample_rate, frames, _ = _audio_info(file_path, "scene.ambience.file")
    source_duration = frames / sample_rate
    if loop_end > source_duration + 1e-9:
        raise SceneValidationError(
            "scene.ambience.loop_end: exceeds source audio duration "
            f"({source_duration:.6f} s)"
        )
    return AmbienceSpec(
        file=file_path,
        loop_start=loop_start,
        loop_end=loop_end,
        gain=_gain(mapping, "scene.ambience", required=False),
    )


def _parse_events(
    scene_dir: Path, raw: Any, samples: Mapping[str, Path]
) -> Tuple[EffectEventSpec, ...]:
    items = _as_list(raw, "scene.events")
    events: List[EffectEventSpec] = []
    for index, item in enumerate(items):
        where = f"scene.events[{index}]"
        mapping = _as_mapping(item, where)
        event_time = _as_number(
            _pick(mapping, ("time", "at", "timestamp", "start"), where),
            _where(where, "time"),
            minimum=0.0,
        )
        gain = _gain(mapping, where, required=True)
        if "file" in mapping or "path" in mapping:
            file_value = _pick(mapping, ("file", "path"), where)
            event_file = _relative_asset(scene_dir, file_value, f"{where}.file")
            raw_name = mapping.get(
                "sample",
                mapping.get(
                    "sample_name",
                    mapping.get("name", Path(str(file_value)).stem),
                ),
            )
            name = _as_string(raw_name, f"{where}.sample")
        else:
            name = _as_string(
                _pick(
                    mapping,
                    ("sample", "name", "sound", "sample_name", "clip"),
                    where,
                ),
                f"{where}.sample",
            )
            if name not in samples:
                raise SceneValidationError(
                    f"{where}.sample: unknown sample name '{name}'"
                )
            event_file = samples[name]
        _audio_info(event_file, f"{where}.file")
        events.append(
            EffectEventSpec(time=event_time, file=event_file, gain=gain, name=name)
        )
    return tuple(events)


def _parse_music(scene_dir: Path, raw: Any) -> Tuple[MusicLayerSpec, ...]:
    items = _as_list(raw, "scene.music_layers")
    layers: List[MusicLayerSpec] = []
    for index, item in enumerate(items):
        where = f"scene.music_layers[{index}]"
        mapping = _as_mapping(item, where)
        file_path = _relative_asset(
            scene_dir,
            _pick(mapping, ("file", "path"), where),
            f"{where}.file",
        )
        _audio_info(file_path, f"{where}.file")
        start = _as_number(
            _pick(mapping, ("start", "start_time", "offset"), where),
            _where(where, "start"),
            minimum=0.0,
        )
        loops_value = _pick(
            mapping, ("loops", "loop_count", "repeats", "repeat"), where
        )
        if isinstance(loops_value, bool) or not isinstance(loops_value, int):
            raise SceneValidationError(f"{where}.loops: expected an integer")
        if loops_value <= 0:
            raise SceneValidationError(f"{where}.loops: must be >= 1")
        layers.append(
            MusicLayerSpec(
                file=file_path,
                start=start,
                loops=int(loops_value),
                gain=_gain(mapping, where, required=True),
            )
        )
    return tuple(layers)


def _parse_dialogues(raw: Any) -> Tuple[DialogueIntervalSpec, ...]:
    items = _as_list(raw, "scene.dialogue_intervals")
    dialogues: List[DialogueIntervalSpec] = []
    for index, item in enumerate(items):
        where = f"scene.dialogue_intervals[{index}]"
        mapping = _as_mapping(item, where)
        start = _as_number(
            _pick(mapping, ("start", "start_time"), where),
            _where(where, "start"),
            minimum=0.0,
        )
        end = _as_number(
            _pick(mapping, ("end", "end_time"), where),
            _where(where, "end"),
            minimum=0.0,
        )
        if end <= start:
            raise SceneValidationError(f"{where}: end must be greater than start")
        depth_value = _pick(
            mapping,
            (
                "duck_depth_db",
                "duck_db",
                "ducking_db",
                "duck_depth",
                "depth_db",
            ),
            where,
        )
        depth = _as_number(
            depth_value,
            f"{where}.duck_depth_db",
            minimum=0.0,
            maximum=120.0,
        )
        dialogues.append(
            DialogueIntervalSpec(start=start, end=end, duck_depth_db=depth)
        )
    return tuple(dialogues)


def _parse_cuts(raw: Any) -> Tuple[float, ...]:
    items = _as_list(raw, "scene.camera_cuts")
    cuts: List[float] = []
    for index, item in enumerate(items):
        where = f"scene.camera_cuts[{index}]"
        if isinstance(item, dict):
            value = _pick(item, ("time", "at"), where)
        else:
            value = item
        cuts.append(_as_number(value, where, minimum=0.0))
    return tuple(sorted(cuts))


def load_scene(path: Path) -> SceneSpec:
    """Load and validate one scene description."""

    scene_path = Path(path).resolve()
    try:
        with scene_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SceneValidationError(
            f"{scene_path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise SceneValidationError(f"{scene_path}: cannot read scene JSON: {exc}") from exc

    root = _as_mapping(data, "scene")
    duration = _as_number(
        _pick(root, _TOP_LEVEL_ALIASES["duration"], "scene"),
        "scene.duration",
        strictly_positive=True,
    )
    ambience = _parse_ambience(
        scene_path.parent, _pick(root, _TOP_LEVEL_ALIASES["ambience"], "scene")
    )
    samples = _parse_samples(scene_path.parent, root)
    events = _parse_events(
        scene_path.parent,
        _pick(root, _TOP_LEVEL_ALIASES["events"], "scene"),
        samples,
    )
    music = _parse_music(
        scene_path.parent, _pick(root, _TOP_LEVEL_ALIASES["music_layers"], "scene")
    )
    dialogues = _parse_dialogues(
        _pick(root, _TOP_LEVEL_ALIASES["dialogue_intervals"], "scene")
    )
    cuts = _parse_cuts(
        _pick(root, _TOP_LEVEL_ALIASES["camera_cuts"], "scene")
    )

    for index, event in enumerate(events):
        if event.time >= duration:
            raise SceneValidationError(
                f"scene.events[{index}].time: must be less than scene duration"
            )
    for index, layer in enumerate(music):
        if layer.start >= duration:
            raise SceneValidationError(
                f"scene.music_layers[{index}].start: must be less than scene duration"
            )
    for index, dialogue in enumerate(dialogues):
        if dialogue.end > duration + 1e-9:
            raise SceneValidationError(
                f"scene.dialogue_intervals[{index}].end: exceeds scene duration"
            )
    for index, cut in enumerate(cuts):
        if cut <= 0.0 or cut >= duration:
            raise SceneValidationError(
                f"scene.camera_cuts[{index}]: must be inside the scene"
            )

    scene_id_raw = root.get("scene_id", root.get("id", scene_path.parent.name))
    if isinstance(scene_id_raw, bool) or not isinstance(
        scene_id_raw, (str, int, float)
    ):
        raise SceneValidationError("scene.scene_id: expected a scalar value")
    scene_id = str(scene_id_raw).strip() or scene_path.parent.name
    return SceneSpec(
        path=scene_path,
        scene_id=scene_id,
        duration=duration,
        ambience=ambience,
        events=events,
        music_layers=music,
        dialogue_intervals=dialogues,
        camera_cuts=cuts,
    )
