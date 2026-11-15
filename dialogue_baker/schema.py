from __future__ import annotations

import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .errors import BakerError
from .timebase import Timebase, fraction_text, parse_rate
from .util import (
    canonical_json_bytes,
    finite_float,
    load_json_bytes,
    read_bytes,
    require_mapping,
    sha256_bytes,
    sha256_file,
)


SCHEMA_VERSION = 1


@dataclass
class DialogueSpec:
    id: str
    speaker: str
    start_frame: int
    end_frame: int
    text: str
    audio_path: Path
    audio_logical: str
    audio_sha256: str
    start_sample: int
    end_sample: int


@dataclass
class TrackSpec:
    positions: np.ndarray
    forwards: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.positions.shape[0])


@dataclass
class ProjectSpec:
    project_path: Path
    base_dir: Path
    fps: Fraction
    sample_rate: int
    duration_frames: int
    target_lufs: float
    true_peak_dbtp: float
    distance_reference_m: float
    pan_strength: float
    fade_ms: float
    speed_of_sound_mps: float
    limiter: Dict[str, float]
    camera: Dict[str, List[float]]
    dialogues: List[DialogueSpec]
    tracks: Dict[str, TrackSpec]
    input_digest: str
    input_manifest: Dict[str, Any]

    @property
    def timebase(self) -> Timebase:
        return Timebase(self.fps, self.sample_rate)

    @property
    def total_samples(self) -> int:
        return self.timebase.total_samples(self.duration_frames)

    @property
    def fps_text(self) -> str:
        return fraction_text(self.fps)


def _schema_version(document: Mapping[str, Any], logical_name: str) -> None:
    if "schema_version" not in document:
        return
    value = document["schema_version"]
    if isinstance(value, bool) or not isinstance(value, int) or value != SCHEMA_VERSION:
        raise BakerError(
            "E_SCHEMA_VERSION",
            "Unsupported input schema version.",
            {
                "logical_name": logical_name,
                "expected": SCHEMA_VERSION,
                "actual": value,
            },
        )


def _resolve(base: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BakerError(
            "E_INVALID_SCHEMA",
            "Expected a non-empty file path.",
            {"field": field, "value": value},
        )
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _logical_path(base: Path, path: Path) -> str:
    return os.path.relpath(str(path), str(base)).replace(os.sep, "/")


def _load_json(path: Path, logical_name: str) -> tuple[Dict[str, Any], bytes]:
    data = read_bytes(path, logical_name)
    document = load_json_bytes(data, logical_name)
    return dict(require_mapping(document, logical_name)), data


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BakerError(
            "E_INVALID_SCHEMA",
            "Expected an integer.",
            {"field": field, "value": value},
        )
    return value


def _vec3(value: Any, field: str, normalize: bool = False) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise BakerError(
            "E_INVALID_SCHEMA",
            "Expected a three-component numeric vector.",
            {"field": field, "value": value},
        )
    vector = np.array(
        [finite_float(component, f"{field}[{index}]") for index, component in enumerate(value)],
        dtype=np.float64,
    )
    if normalize:
        length = float(np.linalg.norm(vector))
        if length <= 1e-12:
            raise BakerError(
                "E_INVALID_SCHEMA",
                "Direction vector must be non-zero.",
                {"field": field, "value": value},
            )
        vector = vector / length
    return [float(component) for component in vector]


def _vector_array(vector: Sequence[float]) -> np.ndarray:
    return np.asarray(vector, dtype=np.float64)


def _target_rate(
    documents: Sequence[tuple[str, Mapping[str, Any]]],
) -> tuple[Fraction, str]:
    candidates: List[tuple[str, Fraction]] = []
    for label, document in documents:
        if "fps" in document:
            candidates.append((label, parse_rate(document["fps"], f"{label}.fps")))
    if not candidates:
        raise BakerError(
            "E_INVALID_FPS",
            "No frame rate was provided.",
            {},
        )
    rate = candidates[0][1]
    mismatched = [
        {"source": label, "fps": fraction_text(candidate)}
        for label, candidate in candidates
        if candidate != rate
    ]
    if mismatched:
        raise BakerError(
            "E_FPS_MISMATCH",
            "Frame rates in project, script, and scene must match.",
            {
                "reference_source": candidates[0][0],
                "reference_fps": fraction_text(rate),
                "mismatched": mismatched,
            },
        )
    return rate, candidates[0][0]


def _parse_tracks(
    scene: Mapping[str, Any],
    duration_frames: int,
) -> Dict[str, TrackSpec]:
    raw_tracks = scene.get("tracks")
    if not isinstance(raw_tracks, dict):
        raise BakerError(
            "E_INVALID_SCHEMA",
            "Scene tracks must be a JSON object.",
            {"field": "scene.tracks"},
        )
    tracks: Dict[str, TrackSpec] = {}
    for speaker in sorted(raw_tracks):
        raw_track = require_mapping(raw_tracks[speaker], f"scene.tracks.{speaker}")
        keyframes = raw_track.get("keyframes")
        if not isinstance(keyframes, list):
            raise BakerError(
                "E_INVALID_SCHEMA",
                "A speaker track must contain a keyframes array.",
                {"speaker": speaker},
            )
        parsed: Dict[int, tuple[List[float], List[float]]] = {}
        for index, raw_frame in enumerate(keyframes):
            frame_object = require_mapping(
                raw_frame, f"scene.tracks.{speaker}.keyframes[{index}]"
            )
            frame = _integer(
                frame_object.get("frame"),
                f"scene.tracks.{speaker}.keyframes[{index}].frame",
            )
            if frame < 0 or frame >= duration_frames:
                raise BakerError(
                    "E_FRAME_OUT_OF_RANGE",
                    "A keyframe lies outside the scene duration.",
                    {
                        "speaker": speaker,
                        "frame": frame,
                        "duration_frames": duration_frames,
                    },
                )
            if frame in parsed:
                raise BakerError(
                    "E_INVALID_SCHEMA",
                    "Duplicate speaker keyframe.",
                    {"speaker": speaker, "frame": frame},
                )
            parsed[frame] = (
                _vec3(
                    frame_object.get("position"),
                    f"scene.tracks.{speaker}.keyframes[{index}].position",
                ),
                _vec3(
                    frame_object.get("forward"),
                    f"scene.tracks.{speaker}.keyframes[{index}].forward",
                    normalize=True,
                ),
            )
        missing = [frame for frame in range(duration_frames) if frame not in parsed]
        if missing:
            raise BakerError(
                "E_TRACK_RANGE",
                "Every speaker track must provide exactly one keyframe per scene frame.",
                {
                    "speaker": speaker,
                    "duration_frames": duration_frames,
                    "first_missing_frame": missing[0],
                    "missing_frame_count": len(missing),
                },
            )
        positions = np.empty((duration_frames, 3), dtype=np.float64)
        forwards = np.empty((duration_frames, 3), dtype=np.float64)
        for frame in range(duration_frames):
            positions[frame] = _vector_array(parsed[frame][0])
            forwards[frame] = _vector_array(parsed[frame][1])
        tracks[speaker] = TrackSpec(positions=positions, forwards=forwards)
    return tracks


def _parse_camera(scene: Mapping[str, Any]) -> Dict[str, List[float]]:
    raw_camera = scene.get("camera")
    if not isinstance(raw_camera, dict):
        raise BakerError(
            "E_INVALID_SCHEMA",
            "Scene camera must be a JSON object.",
            {"field": "scene.camera"},
        )
    position = _vec3(raw_camera.get("position"), "scene.camera.position")
    if "forward" in raw_camera:
        forward = _vec3(
            raw_camera["forward"], "scene.camera.forward", normalize=True
        )
    elif "target" in raw_camera:
        target = _vector_array(
            _vec3(raw_camera["target"], "scene.camera.target")
        )
        direction = target - _vector_array(position)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise BakerError(
                "E_INVALID_SCHEMA",
                "Camera target must differ from its position.",
                {},
            )
        forward = [float(component) for component in direction / norm]
    else:
        raise BakerError(
            "E_INVALID_SCHEMA",
            "Camera requires either forward or target.",
            {"field": "scene.camera"},
        )
    up = (
        _vec3(raw_camera["up"], "scene.camera.up", normalize=True)
        if "up" in raw_camera
        else [0.0, 1.0, 0.0]
    )
    right = np.cross(_vector_array(up), _vector_array(forward))
    norm = float(np.linalg.norm(right))
    if norm <= 1e-12:
        raise BakerError(
            "E_INVALID_SCHEMA",
            "Camera up and forward vectors must not be parallel.",
            {"camera": raw_camera},
        )
    right = right / norm
    return {
        "position": position,
        "forward": forward,
        "up": up,
        "right": [float(component) for component in right],
    }


def _dialogue_audio(
    raw_dialogue: Mapping[str, Any],
    audio_root: Path,
    base_dir: Path,
    index: int,
) -> tuple[Path, str, str]:
    value = raw_dialogue.get("audio")
    if value is None:
        value = raw_dialogue.get("audio_file")
    if not isinstance(value, str) or not value:
        raise BakerError(
            "E_INVALID_SCHEMA",
            "Each dialogue requires an audio file path.",
            {"dialogue_index": index, "field": "audio"},
        )
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (audio_root / candidate).resolve()
    logical = _logical_path(base_dir, path)
    if not path.is_file():
        raise BakerError(
            "E_MISSING_INPUT",
            "Required input file is missing.",
            {"path": str(path), "logical_name": logical},
        )
    try:
        digest = sha256_file(path)
    except OSError as exc:
        raise BakerError(
            "E_INPUT_READ",
            "Required input file could not be read.",
            {"path": str(path), "logical_name": logical, "reason": str(exc)},
        ) from exc
    return path, logical, digest


def _validate_overlaps(dialogues: Sequence[DialogueSpec]) -> None:
    by_speaker: Dict[str, List[DialogueSpec]] = {}
    for dialogue in dialogues:
        by_speaker.setdefault(dialogue.speaker, []).append(dialogue)
    for speaker in sorted(by_speaker):
        ordered = sorted(
            by_speaker[speaker],
            key=lambda item: (item.start_frame, item.end_frame, item.id),
        )
        active_end = -1
        active_id = ""
        for dialogue in ordered:
            if dialogue.start_frame < active_end:
                raise BakerError(
                    "E_SAME_SPEAKER_OVERLAP",
                    "Dialogue intervals for one speaker must not overlap.",
                    {
                        "speaker": speaker,
                        "first_id": active_id,
                        "second_id": dialogue.id,
                        "first_end_frame": active_end,
                        "second_start_frame": dialogue.start_frame,
                    },
                )
            if dialogue.end_frame > active_end:
                active_end = dialogue.end_frame
                active_id = dialogue.id


def load_project(project_path: Path) -> ProjectSpec:
    project_path = Path(project_path).resolve()
    project, project_bytes = _load_json(project_path, "project.json")
    _schema_version(project, "project.json")
    base_dir = project_path.parent
    script_path = _resolve(base_dir, project.get("script"), "project.script")
    scene_path = _resolve(base_dir, project.get("scene"), "project.scene")
    audio_root = _resolve(
        base_dir, project.get("audio_root", "."), "project.audio_root"
    )
    script, script_bytes = _load_json(script_path, _logical_path(base_dir, script_path))
    scene, scene_bytes = _load_json(scene_path, _logical_path(base_dir, scene_path))
    _schema_version(script, _logical_path(base_dir, script_path))
    _schema_version(scene, _logical_path(base_dir, scene_path))

    fps, fps_source = _target_rate(
        (
            ("project", project),
            ("script", script),
            ("scene", scene),
        )
    )
    sample_rate = _integer(project.get("sample_rate", 48000), "project.sample_rate")
    if sample_rate != 48000:
        raise BakerError(
            "E_UNSUPPORTED_SAMPLE_RATE",
            "The output sample rate must be 48000 Hz.",
            {"sample_rate": sample_rate},
        )
    duration_frames = _integer(
        scene.get("duration_frames"), "scene.duration_frames"
    )
    if duration_frames <= 0:
        raise BakerError(
            "E_INVALID_FRAME_RANGE",
            "Scene duration must contain at least one frame.",
            {"duration_frames": duration_frames},
        )
    if "duration_frames" in script:
        script_duration = _integer(
            script["duration_frames"], "script.duration_frames"
        )
        if script_duration != duration_frames:
            raise BakerError(
                "E_DURATION_MISMATCH",
                "Script and scene durations must match.",
                {
                    "script_duration_frames": script_duration,
                    "scene_duration_frames": duration_frames,
                },
            )
    timebase = Timebase(fps, sample_rate)
    if not timebase.validate_roundtrip(duration_frames):
        raise BakerError(
            "E_TIMEBASE_ROUNDTRIP",
            "Frame boundaries cannot be round-tripped exactly through samples.",
            {
                "fps": fraction_text(fps),
                "sample_rate": sample_rate,
                "duration_frames": duration_frames,
            },
        )

    camera = _parse_camera(scene)
    tracks = _parse_tracks(scene, duration_frames)

    raw_dialogues = script.get("dialogues")
    if raw_dialogues is None:
        raw_dialogues = script.get("lines")
    if not isinstance(raw_dialogues, list) or not raw_dialogues:
        raise BakerError(
            "E_EMPTY_SCRIPT",
            "The script must contain at least one dialogue.",
            {"field": "script.dialogues"},
        )

    dialogues: List[DialogueSpec] = []
    seen_ids: set[str] = set()
    audio_hashes: Dict[str, str] = {}
    for index, raw_line in enumerate(raw_dialogues):
        line = require_mapping(raw_line, f"script.dialogues[{index}]")
        identifier = line.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise BakerError(
                "E_INVALID_SCHEMA",
                "Each dialogue requires a non-empty string id.",
                {"dialogue_index": index},
            )
        if identifier in seen_ids:
            raise BakerError(
                "E_DUPLICATE_DIALOGUE_ID",
                "Dialogue ids must be unique.",
                {"id": identifier},
            )
        seen_ids.add(identifier)
        speaker = line.get("speaker")
        if not isinstance(speaker, str) or not speaker:
            raise BakerError(
                "E_INVALID_SCHEMA",
                "Each dialogue requires a non-empty speaker.",
                {"dialogue_index": index, "id": identifier},
            )
        start_frame = _integer(
            line.get("start_frame"),
            f"script.dialogues[{index}].start_frame",
        )
        end_frame = _integer(
            line.get("end_frame"),
            f"script.dialogues[{index}].end_frame",
        )
        if start_frame < 0 or end_frame <= start_frame:
            raise BakerError(
                "E_INVALID_FRAME_RANGE",
                "Dialogue frame intervals must be non-empty and non-negative.",
                {
                    "id": identifier,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                },
            )
        if end_frame > duration_frames:
            raise BakerError(
                "E_FRAME_OUT_OF_RANGE",
                "Dialogue frame interval lies outside the scene duration.",
                {
                    "id": identifier,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "duration_frames": duration_frames,
                },
            )
        text = line.get("text")
        if not isinstance(text, str):
            raise BakerError(
                "E_INVALID_SCHEMA",
                "Each dialogue requires text.",
                {"dialogue_index": index, "id": identifier},
            )
        if speaker not in tracks:
            raise BakerError(
                "E_MISSING_TRACK",
                "A speaker has no keyframe track.",
                {"speaker": speaker, "dialogue_id": identifier},
            )
        track = tracks[speaker]
        if start_frame < 0 or end_frame > track.frame_count:
            raise BakerError(
                "E_FRAME_OUT_OF_RANGE",
                "Dialogue frame interval lies outside its speaker track.",
                {
                    "id": identifier,
                    "speaker": speaker,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "track_frame_count": track.frame_count,
                },
            )
        audio_path, audio_logical, audio_sha = _dialogue_audio(
            line, audio_root, base_dir, index
        )
        audio_hashes[audio_logical] = audio_sha
        dialogues.append(
            DialogueSpec(
                id=identifier,
                speaker=speaker,
                start_frame=start_frame,
                end_frame=end_frame,
                text=text,
                audio_path=audio_path,
                audio_logical=audio_logical,
                audio_sha256=audio_sha,
                start_sample=timebase.frame_to_sample(start_frame),
                end_sample=timebase.frame_to_sample(end_frame),
            )
        )

    _validate_overlaps(dialogues)

    target_lufs = finite_float(
        project.get("target_lufs", -23.0), "project.target_lufs"
    )
    true_peak_dbtp = finite_float(
        project.get("true_peak_dbtp", -1.0), "project.true_peak_dbtp"
    )
    if not (-70.0 <= target_lufs <= -5.0):
        raise BakerError(
            "E_INVALID_CONFIG",
            "Target loudness is outside the supported range.",
            {"target_lufs": target_lufs, "minimum": -70.0, "maximum": -5.0},
        )
    if not (-20.0 <= true_peak_dbtp <= 0.0):
        raise BakerError(
            "E_INVALID_CONFIG",
            "True-peak ceiling is outside the supported range.",
            {"true_peak_dbtp": true_peak_dbtp, "minimum": -20.0, "maximum": 0.0},
        )
    distance_reference_m = finite_float(
        project.get("distance_reference_m", 1.5),
        "project.distance_reference_m",
    )
    pan_strength = finite_float(
        project.get("pan_strength", 0.9), "project.pan_strength"
    )
    fade_ms = finite_float(project.get("fade_ms", 5.0), "project.fade_ms")
    speed_of_sound_mps = finite_float(
        project.get("speed_of_sound_mps", 343.0),
        "project.speed_of_sound_mps",
    )
    if distance_reference_m <= 0:
        raise BakerError(
            "E_INVALID_CONFIG",
            "distance_reference_m must be positive.",
            {"distance_reference_m": distance_reference_m},
        )
    if not (0.0 <= pan_strength <= 1.0):
        raise BakerError(
            "E_INVALID_CONFIG",
            "pan_strength must be between zero and one.",
            {"pan_strength": pan_strength},
        )
    if not (0.0 <= fade_ms <= 100.0):
        raise BakerError(
            "E_INVALID_CONFIG",
            "fade_ms must be between zero and 100.",
            {"fade_ms": fade_ms},
        )
    if speed_of_sound_mps <= 0:
        raise BakerError(
            "E_INVALID_CONFIG",
            "speed_of_sound_mps must be positive.",
            {"speed_of_sound_mps": speed_of_sound_mps},
        )
    raw_limiter = project.get("limiter", {})
    raw_limiter = dict(require_mapping(raw_limiter, "project.limiter"))
    lookahead_ms = finite_float(
        raw_limiter.get("lookahead_ms", 5.0), "project.limiter.lookahead_ms"
    )
    release_ms = finite_float(
        raw_limiter.get("release_ms", 80.0), "project.limiter.release_ms"
    )
    oversample = _integer(
        raw_limiter.get("true_peak_oversample", 4),
        "project.limiter.true_peak_oversample",
    )
    if lookahead_ms <= 0 or release_ms <= 0:
        raise BakerError(
            "E_INVALID_CONFIG",
            "Limiter lookahead and release must be positive.",
            {"lookahead_ms": lookahead_ms, "release_ms": release_ms},
        )
    if oversample not in (2, 4, 8):
        raise BakerError(
            "E_INVALID_CONFIG",
            "true_peak_oversample must be 2, 4, or 8.",
            {"true_peak_oversample": oversample},
        )

    input_manifest = {
        "audio_files": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(audio_hashes.items())
        ],
        "fps": fraction_text(fps),
        "fps_source": fps_source,
        "project_sha256": sha256_bytes(project_bytes),
        "scene_sha256": sha256_bytes(scene_bytes),
        "script_sha256": sha256_bytes(script_bytes),
        "target_lufs": target_lufs,
        "true_peak_dbtp": true_peak_dbtp,
    }
    input_digest = sha256_bytes(canonical_json_bytes(input_manifest))

    return ProjectSpec(
        project_path=project_path,
        base_dir=base_dir,
        fps=fps,
        sample_rate=sample_rate,
        duration_frames=duration_frames,
        target_lufs=target_lufs,
        true_peak_dbtp=true_peak_dbtp,
        distance_reference_m=distance_reference_m,
        pan_strength=pan_strength,
        fade_ms=fade_ms,
        speed_of_sound_mps=speed_of_sound_mps,
        limiter={
            "lookahead_ms": lookahead_ms,
            "release_ms": release_ms,
            "true_peak_oversample": float(oversample),
        },
        camera=camera,
        dialogues=dialogues,
        tracks=tracks,
        input_digest=input_digest,
        input_manifest=input_manifest,
    )

