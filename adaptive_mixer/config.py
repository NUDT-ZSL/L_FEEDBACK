"""Scene configuration parsing, validation and deterministic versioning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import MixerError

REQUIRED_TOP_FIELDS = (
    "sample_rate", "frame_rate", "total_frames", "target_lufs",
    "crossfade_ms", "crossfade_type", "tracks", "scenes",
)


@dataclass
class TrackSpec:
    name: str
    path: str


@dataclass
class SceneSpec:
    scene_id: str
    name: str
    start_frame: int
    end_frame: int
    target_lufs: float
    crossfade_ms: float
    crossfade_type: str
    active_tracks: List[str]


@dataclass
class SceneConfig:
    source_path: Optional[Path]
    sample_rate: int
    frame_rate: int
    total_frames: int
    target_lufs: float
    crossfade_ms: float
    crossfade_type: str
    tracks: List[TrackSpec]
    scenes: List[SceneSpec]
    output_format: Dict[str, Any]
    raw: Dict[str, Any] = field(default_factory=dict)
    version: str = ""

    @property
    def samples_per_frame(self) -> int:
        if self.sample_rate % self.frame_rate:
            raise MixerError(
                "E_SAMPLE_RATE_MISMATCH",
                details={"sample_rate": self.sample_rate,
                         "frame_rate": self.frame_rate})
        return self.sample_rate // self.frame_rate

    @property
    def total_samples(self) -> int:
        return self.total_frames * self.samples_per_frame


def config_fingerprint(raw: Dict[str, Any]) -> str:
    """Stable SHA-256 version fingerprint of the configuration content."""
    canonical = json.dumps(raw, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def load_config(path) -> SceneConfig:
    path = Path(path)
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MixerError("E_CONFIG_PARSE", str(exc),
                         {"path": str(path)}) from exc
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise MixerError("E_CONFIG_PARSE", f"JSON 解析失败: {exc.msg}",
                         {"line": exc.lineno, "col": exc.colno}) from exc
    config = validate_config(raw)
    config.source_path = path
    return config


def validate_config(raw: Dict[str, Any]) -> SceneConfig:
    if not isinstance(raw, dict):
        raise MixerError("E_CONFIG_PARSE", "配置根节点必须是 JSON 对象")
    missing = [key for key in REQUIRED_TOP_FIELDS if key not in raw]
    if missing:
        raise MixerError("E_CONFIG_MISSING_FIELD",
                         details={"missing": missing})

    sample_rate = _as_int(raw["sample_rate"], "sample_rate")
    frame_rate = _as_int(raw["frame_rate"], "frame_rate")
    total_frames = _as_int(raw["total_frames"], "total_frames")
    if sample_rate <= 0 or frame_rate <= 0 or total_frames <= 0:
        raise MixerError("E_CONFIG_VALUE",
                         "sample_rate/frame_rate/total_frames 必须为正整数")
    if sample_rate % frame_rate != 0:
        raise MixerError("E_SAMPLE_RATE_MISMATCH",
                         "采样率必须是帧率的整数倍以保证帧对齐",
                         {"sample_rate": sample_rate,
                          "frame_rate": frame_rate})

    target_lufs = _as_float(raw["target_lufs"], "target_lufs")
    if not (-70.0 <= target_lufs <= -5.0):
        raise MixerError("E_LUFS_TARGET",
                         details={"target_lufs": target_lufs})
    crossfade_ms = _as_float(raw["crossfade_ms"], "crossfade_ms")
    crossfade_type = raw["crossfade_type"]
    if crossfade_type not in ("linear", "equal_power", "s_curve"):
        raise MixerError("E_CONFIG_VALUE",
                         f"未知交叉淡化类型 {crossfade_type!r}",
                         {"allowed": ["linear", "equal_power", "s_curve"]})

    tracks_raw = raw["tracks"]
    if not isinstance(tracks_raw, list) or not tracks_raw:
        raise MixerError("E_CONFIG_MISSING_FIELD", "tracks 必须是非空数组")
    tracks: List[TrackSpec] = []
    track_names = set()
    for item in tracks_raw:
        if not isinstance(item, dict) or "name" not in item \
                or "path" not in item:
            raise MixerError("E_CONFIG_MISSING_FIELD",
                             "每个分轨需要 name 与 path 字段")
        name, path = str(item["name"]), str(item["path"])
        if name in track_names:
            raise MixerError("E_CONFIG_VALUE", f"分轨名称重复: {name}")
        track_names.add(name)
        tracks.append(TrackSpec(name=name, path=path))

    scenes_raw = raw["scenes"]
    if not isinstance(scenes_raw, list) or not scenes_raw:
        raise MixerError("E_CONFIG_MISSING_FIELD", "scenes 必须是非空数组")
    samples_per_frame = sample_rate // frame_rate
    scenes: List[SceneSpec] = []
    expected_start = 0
    seen_ids = set()
    for index, item in enumerate(scenes_raw):
        scene = _parse_scene(item, index, track_names, target_lufs,
                             crossfade_ms, crossfade_type)
        if scene.scene_id in seen_ids:
            raise MixerError("E_CONFIG_VALUE",
                             f"场景 ID 重复: {scene.scene_id}")
        seen_ids.add(scene.scene_id)
        if scene.start_frame != expected_start:
            raise MixerError(
                "E_SCENE_OVERLAP",
                f"场景 {index} 起始帧 {scene.start_frame} 与期望 "
                f"{expected_start} 不连续",
                {"scene_index": index, "expected_start": expected_start})
        if scene.end_frame <= scene.start_frame:
            raise MixerError("E_SCENE_OVERLAP",
                             f"场景 {scene.scene_id} 区间长度必须大于 0")
        crossfade_samples = int(round(
            scene.crossfade_ms / 1000.0 * sample_rate))
        span_samples = (scene.end_frame - scene.start_frame) * samples_per_frame
        if index > 0 and not (0 < crossfade_samples <= span_samples):
            raise MixerError(
                "E_CROSSFADE_RANGE",
                f"场景 {scene.scene_id} 交叉淡化时长超出场景长度",
                {"crossfade_ms": scene.crossfade_ms})
        scenes.append(scene)
        expected_start = scene.end_frame

    if expected_start != total_frames:
        raise MixerError(
            "E_SCENE_OVERLAP",
            f"场景结束帧 {expected_start} 与 total_frames {total_frames} 不一致",
            {"expected": total_frames, "actual": expected_start})

    output_format = raw.get("output_format", {})
    if output_format:
        if output_format.get("sample_rate", sample_rate) != sample_rate:
            raise MixerError("E_SAMPLE_RATE_MISMATCH",
                             "输出采样率与分轨采样率不一致")
        if output_format.get("channels", 1) != 1:
            raise MixerError("E_CONFIG_VALUE", "仅支持单声道输出")
        if output_format.get("subtype", "PCM_24") not in ("PCM_24",):
            raise MixerError("E_CONFIG_VALUE", "输出位深必须为 PCM_24")

    config = SceneConfig(
        source_path=None,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        total_frames=total_frames,
        target_lufs=target_lufs,
        crossfade_ms=crossfade_ms,
        crossfade_type=crossfade_type,
        tracks=tracks,
        scenes=scenes,
        output_format=dict(output_format or {}),
        raw=raw,
        version=config_fingerprint(raw),
    )
    return config


def _parse_scene(item: Any, index: int, track_names: set,
                 default_lufs: float, default_crossfade_ms: float,
                 default_crossfade_type: str) -> SceneSpec:
    if not isinstance(item, dict):
        raise MixerError("E_CONFIG_PARSE", f"场景 {index} 必须是 JSON 对象")
    for field_name in ("scene_id", "start_frame", "end_frame",
                       "active_tracks"):
        if field_name not in item:
            raise MixerError("E_CONFIG_MISSING_FIELD",
                             f"场景 {index} 缺少字段 {field_name}",
                             {"scene_index": index, "field": field_name})
    scene_id = str(item["scene_id"])
    start_frame = _as_int(item["start_frame"], "start_frame")
    end_frame = _as_int(item["end_frame"], "end_frame")
    lufs = _as_float(item.get("target_lufs", default_lufs), "target_lufs")
    if not (-70.0 <= lufs <= -5.0):
        raise MixerError("E_LUFS_TARGET",
                         {"scene_id": scene_id, "target_lufs": lufs})
    crossfade_ms = _as_float(
        item.get("crossfade_ms", default_crossfade_ms), "crossfade_ms")
    if crossfade_ms < 0.0:
        raise MixerError("E_CROSSFADE_RANGE",
                         {"scene_id": scene_id,
                          "crossfade_ms": crossfade_ms})
    crossfade_type = str(item.get("crossfade_type", default_crossfade_type))
    if crossfade_type not in ("linear", "equal_power", "s_curve"):
        raise MixerError("E_CONFIG_VALUE",
                         f"场景 {scene_id} 未知交叉淡化类型")
    active = item["active_tracks"]
    if not isinstance(active, list) or not active:
        raise MixerError("E_CONFIG_MISSING_FIELD",
                         f"场景 {scene_id} active_tracks 必须为非空数组")
    unknown = [name for name in active if name not in track_names]
    if unknown:
        raise MixerError("E_CONFIG_MISSING_FIELD",
                         f"场景 {scene_id} 引用了未定义分轨: {unknown}")
    if len(set(active)) != len(active):
        raise MixerError("E_CONFIG_VALUE",
                         f"场景 {scene_id} active_tracks 存在重复项")
    return SceneSpec(
        scene_id=scene_id,
        name=str(item.get("name", scene_id)),
        start_frame=start_frame,
        end_frame=end_frame,
        target_lufs=lufs,
        crossfade_ms=crossfade_ms,
        crossfade_type=crossfade_type,
        active_tracks=list(active),
    )


def _as_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MixerError("E_CONFIG_VALUE",
                         f"字段 {field_name} 必须是整数",
                         {"field": field_name, "value": value})
    return int(value)


def _as_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MixerError("E_CONFIG_VALUE",
                         f"字段 {field_name} 必须是数值",
                         {"field": field_name, "value": value})
    return float(value)


def verify_track_files(config: SceneConfig, base_dir: Path) -> None:
    """Check every referenced WAV exists and matches the declared rate."""
    from .wavio import read_wav

    base_dir = Path(base_dir)
    for track in config.tracks:
        candidate = Path(track.path)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        if not candidate.is_file():
            raise MixerError("E_TRACK_NOT_FOUND",
                             details={"track": track.name,
                                      "path": str(candidate)})
        _data, sample_rate = read_wav(candidate)
        if sample_rate != config.sample_rate:
            raise MixerError("E_SAMPLE_RATE_MISMATCH",
                             f"分轨 {track.name} 采样率 {sample_rate} "
                             f"与配置 {config.sample_rate} 不一致",
                             {"track": track.name,
                              "file_rate": sample_rate,
                              "config_rate": config.sample_rate})
