from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .errors import BusinessError
from .models import AudioSpec, LanguageSpec, Manifest, SwitchPoint


_RATIONAL_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BusinessError(1003, f"{name} must be a JSON object")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BusinessError(1003, f"{name} must be a non-empty string")
    return value.strip()


def _require_int(value: Any, name: str, minimum: int | None = None) -> int:
    if not _is_int(value):
        raise BusinessError(1003, f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise BusinessError(1003, f"{name} must be >= {minimum}")
    return value


def parse_frame_rate(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        raise BusinessError(1004, "frame_rate must be a positive number or rational string")
    if isinstance(value, str):
        match = _RATIONAL_RE.match(value)
        if not match:
            raise BusinessError(1004, "frame_rate rational string must have the form numerator/denominator")
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        if denominator <= 0:
            raise BusinessError(1004, "frame_rate denominator must be positive")
        result = numerator / denominator
    elif isinstance(value, (int, float)):
        result = float(value)
    else:
        raise BusinessError(1004, "frame_rate must be a positive number or rational string")
    if not math.isfinite(result) or result <= 0 or result > 1000:
        raise BusinessError(1004, "frame_rate must be finite and in the range (0, 1000]")
    return result


def _resolve(base: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _audio_spec(raw: Any) -> AudioSpec:
    obj = _require_object(raw, "audio")
    sample_rate = _require_int(obj.get("sample_rate"), "audio.sample_rate", 8000)
    channels = _require_int(obj.get("channels"), "audio.channels", 1)
    bit_depth = _require_int(obj.get("bit_depth"), "audio.bit_depth", 1)
    if channels not in (1, 2):
        raise BusinessError(1005, "audio.channels must be 1 or 2")
    if sample_rate > 384000:
        raise BusinessError(1005, "audio.sample_rate must be <= 384000")
    if bit_depth != 24:
        raise BusinessError(1006, "audio.bit_depth must be 24 for the required PCM master")
    return AudioSpec(sample_rate=sample_rate, channels=channels, bit_depth=bit_depth)


def _check_language_spec(obj: dict[str, Any], name: str, audio: AudioSpec) -> None:
    for field, expected in (
        ("sample_rate", audio.sample_rate),
        ("channels", audio.channels),
        ("bit_depth", audio.bit_depth),
    ):
        if field not in obj:
            continue
        value = _require_int(obj[field], f"{name}.{field}", 1)
        if value != expected:
            code = 1005 if field == "sample_rate" else 1006
            raise BusinessError(code, f"{name}.{field}={value} is inconsistent with audio.{field}={expected}")


def load_manifest(path: str | Path) -> Manifest:
    manifest_path = Path(path).expanduser().resolve()
    try:
        text = manifest_path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise BusinessError(1001, f"manifest not found: {manifest_path}") from exc
    except OSError as exc:
        raise BusinessError(1001, f"cannot read manifest: {manifest_path}: {exc}") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BusinessError(1002, f"invalid manifest JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    root = _require_object(raw, "manifest")

    frame_rate = parse_frame_rate(root.get("frame_rate"))
    audio = _audio_spec(root.get("audio"))

    video_raw = root.get("video")
    reference_stream_index: int | None = None
    if isinstance(video_raw, str):
        video_path = _resolve(manifest_path.parent, _require_string(video_raw, "video"))
    else:
        video_obj = _require_object(video_raw, "video")
        video_path = _resolve(manifest_path.parent, _require_string(video_obj.get("path"), "video.path"))
        if "reference_stream_index" in video_obj and video_obj["reference_stream_index"] is not None:
            reference_stream_index = _require_int(
                video_obj["reference_stream_index"], "video.reference_stream_index", 0
            )
    if reference_stream_index is None and "reference_stream_index" in root:
        if root["reference_stream_index"] is not None:
            reference_stream_index = _require_int(
                root["reference_stream_index"], "reference_stream_index", 0
            )

    languages_raw = root.get("languages")
    if not isinstance(languages_raw, list) or not languages_raw:
        raise BusinessError(1003, "languages must be a non-empty array")
    languages: list[LanguageSpec] = []
    seen_ids: set[str] = set()
    for index, item_raw in enumerate(languages_raw):
        item = _require_object(item_raw, f"languages[{index}]")
        language_id = _require_string(item.get("id"), f"languages[{index}].id")
        if language_id in seen_ids:
            raise BusinessError(1007, f"duplicate language id: {language_id}")
        seen_ids.add(language_id)
        wav_path = _resolve(manifest_path.parent, _require_string(item.get("path"), f"languages[{index}].path"))
        _check_language_spec(item, f"languages[{index}]", audio)
        languages.append(LanguageSpec(language_id=language_id, path=wav_path))

    switches_raw = root.get("switch_points", root.get("switches"))
    if not isinstance(switches_raw, list) or not switches_raw:
        raise BusinessError(1008, "switch_points must be a non-empty array")
    switches: list[SwitchPoint] = []
    seen_frames: set[int] = set()
    for index, switch_raw in enumerate(switches_raw):
        item = _require_object(switch_raw, f"switch_points[{index}]")
        frame = _require_int(item.get("frame"), f"switch_points[{index}].frame", 0)
        language_id = _require_string(
            item.get("language", item.get("language_id")), f"switch_points[{index}].language"
        )
        if language_id not in seen_ids:
            raise BusinessError(
                1007, f"switch_points[{index}] refers to unknown language id: {language_id}"
            )
        if frame in seen_frames:
            raise BusinessError(1008, f"duplicate switch frame: {frame}")
        seen_frames.add(frame)
        switches.append(
            SwitchPoint(original_index=index, frame=frame, language_id=language_id)
        )
    switches.sort(key=lambda item: item.frame)
    if switches[0].frame != 0:
        switches.insert(
            0,
            SwitchPoint(
                original_index=-1,
                frame=0,
                language_id=languages[0].language_id,
                implicit=True,
            ),
        )

    duration_frames: int | None = None
    if "duration_frames" in root and root["duration_frames"] is not None:
        duration_frames = _require_int(root["duration_frames"], "duration_frames", 1)
        if switches[-1].frame >= duration_frames:
            raise BusinessError(1008, "switch_points must be before duration_frames")

    crossfade_ms = 15.0
    if "crossfade_ms" in root:
        value = root["crossfade_ms"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BusinessError(1003, "crossfade_ms must be a number")
        crossfade_ms = float(value)
        if not math.isclose(crossfade_ms, 15.0, abs_tol=1e-9):
            raise BusinessError(1008, "crossfade_ms must be 15 for broadcast delivery")

    return Manifest(
        manifest_path=manifest_path,
        frame_rate=frame_rate,
        audio=audio,
        languages=languages,
        switches=switches,
        video_path=video_path,
        reference_stream_index=reference_stream_index,
        duration_frames=duration_frames,
        crossfade_ms=crossfade_ms,
        raw=root,
    )

