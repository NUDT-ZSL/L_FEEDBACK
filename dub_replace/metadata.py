"""Animation metadata parsing, validation and dialect-clip matching."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ChannelLayoutError, MetadataError, OverlapError

# Declared channel layouts -> ordered channel labels.
CHANNEL_LAYOUTS: dict[str, list[str]] = {
    "mono": ["mono"],
    "stereo": ["L", "R"],
    "5.1": ["L", "R", "C", "LFE", "BL", "BR"],
}

VOICE_ROLES = {"voice", "target_voice", "other_voice"}


@dataclass
class ChannelSpec:
    index: int
    label: str
    role: str
    language: str | None = None

    @property
    def is_target_voice(self) -> bool:
        return self.role in ("voice", "target_voice")


@dataclass
class DialogueLine:
    line_id: str
    start_ms: int
    end_ms: int
    text: str

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class SubtitleEvent:
    start_ms: int
    end_ms: int
    text: str


@dataclass
class LipEvent:
    start_ms: int
    end_ms: int
    line_id: str | None = None


@dataclass
class DialectClip:
    clip_id: str | None
    line_id: str | None
    start_ms: int | None
    path: str


@dataclass
class AnimationMeta:
    animation_id: str
    base_dir: Path
    original_track: Path
    sample_rate: int
    channel_layout: str
    channels: list[ChannelSpec]
    dialogue: list[DialogueLine]
    subtitles: list[SubtitleEvent]
    lip_events: list[LipEvent]
    dialect_clips: list[DialectClip]
    stretch_limits: tuple[float, float]
    raw: dict = field(default_factory=dict)

    @property
    def target_voice_indexes(self) -> list[int]:
        return [c.index for c in self.channels if c.is_target_voice]


def _require(mapping: dict, key: str, ctx: str):
    if key not in mapping:
        raise MetadataError(f"{ctx} 缺少必填字段 '{key}'")
    return mapping[key]


def parse_metadata(meta_path: str | Path) -> AnimationMeta:
    meta_path = Path(meta_path)
    if not meta_path.is_file():
        from .errors import MissingAudioError

        raise MissingAudioError(f"元数据文件不存在: {meta_path}")
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MetadataError(f"元数据 JSON 解析失败: {meta_path} ({exc})") from exc

    base_dir = meta_path.parent
    animation_id = str(_require(raw, "animation_id", "meta"))
    audio = _require(raw, "audio", "meta")
    original_track = base_dir / str(_require(audio, "original_track", "audio"))
    sample_rate = int(_require(audio, "sample_rate", "audio"))
    if sample_rate <= 0:
        raise MetadataError("sample_rate 必须为正整数")
    channel_layout = str(_require(audio, "channel_layout", "audio"))
    if channel_layout not in CHANNEL_LAYOUTS:
        raise ChannelLayoutError(
            f"不支持的 channel_layout '{channel_layout}'，可选: {sorted(CHANNEL_LAYOUTS)}"
        )
    declared = CHANNEL_LAYOUTS[channel_layout]

    raw_channels = _require(audio, "channels", "audio")
    if not isinstance(raw_channels, list) or len(raw_channels) != len(declared):
        raise ChannelLayoutError(
            f"channels 数量 {0 if not isinstance(raw_channels, list) else len(raw_channels)} "
            f"与布局 '{channel_layout}' 要求的 {len(declared)} 声道不一致"
        )
    channels: list[ChannelSpec] = []
    seen_indexes: set[int] = set()
    for i, ch in enumerate(raw_channels):
        idx = int(ch.get("index", i))
        if idx < 0 or idx >= len(declared) or idx in seen_indexes:
            raise ChannelLayoutError(f"声道索引非法或重复: {ch}")
        seen_indexes.add(idx)
        label = str(ch.get("label", declared[idx]))
        if label != declared[idx]:
            raise ChannelLayoutError(
                f"声道 {idx} 标签 '{label}' 与布局 '{channel_layout}' 期望的 '{declared[idx]}' 不一致"
            )
        role = str(_require(ch, "role", f"channels[{i}]"))
        if role not in ("sfx", "voice", "target_voice", "other_voice"):
            raise MetadataError(f"未知声道角色 '{role}'（合法值: sfx/voice/target_voice/other_voice）")
        channels.append(ChannelSpec(idx, label, role, ch.get("language")))
    if not any(c.is_target_voice for c in channels):
        raise ChannelLayoutError("声道布局中必须存在一个目标语言配音声道 (role=voice/target_voice)")

    dialogue: list[DialogueLine] = []
    for i, line in enumerate(_require(raw, "dialogue", "meta")):
        line_id = str(_require(line, "line_id", f"dialogue[{i}]"))
        start_ms = int(_require(line, "start_ms", f"dialogue[{i}]"))
        end_ms = int(_require(line, "end_ms", f"dialogue[{i}]"))
        if end_ms <= start_ms:
            raise MetadataError(f"台词 {line_id} 的 end_ms 必须大于 start_ms")
        dialogue.append(DialogueLine(line_id, start_ms, end_ms, str(line.get("text", ""))))
    _check_overlap(dialogue)

    subtitles = [
        SubtitleEvent(int(s["start_ms"]), int(s["end_ms"]), str(s.get("text", "")))
        for s in raw.get("subtitles", [])
    ]
    lip_events = [
        LipEvent(int(e["start_ms"]), int(e["end_ms"]), e.get("line_id"))
        for e in raw.get("lip_events", [])
    ]

    dialect_clips: list[DialectClip] = []
    for i, clip in enumerate(raw.get("dialect_clips", [])):
        path = str(_require(clip, "path", f"dialect_clips[{i}]"))
        dialect_clips.append(
            DialectClip(
                clip_id=clip.get("clip_id"),
                line_id=clip.get("line_id"),
                start_ms=int(clip["start_ms"]) if clip.get("start_ms") is not None else None,
                path=path,
            )
        )

    limits = raw.get("stretch_limits", {"min": 0.5, "max": 2.0})
    stretch_min = float(limits.get("min", 0.5))
    stretch_max = float(limits.get("max", 2.0))
    if not (0.0 < stretch_min <= 1.0 <= stretch_max):
        raise MetadataError("stretch_limits 须满足 0 < min <= 1.0 <= max")

    return AnimationMeta(
        animation_id=animation_id,
        base_dir=base_dir,
        original_track=original_track,
        sample_rate=sample_rate,
        channel_layout=channel_layout,
        channels=channels,
        dialogue=dialogue,
        subtitles=subtitles,
        lip_events=lip_events,
        dialect_clips=dialect_clips,
        stretch_limits=(stretch_min, stretch_max),
        raw=raw,
    )


def _check_overlap(dialogue: list[DialogueLine]) -> None:
    ordered = sorted(dialogue, key=lambda d: d.start_ms)
    for prev, cur in zip(ordered, ordered[1:]):
        if cur.start_ms < prev.end_ms:
            raise OverlapError(
                f"台词片段时间重叠: '{prev.line_id}'({prev.start_ms}-{prev.end_ms}) 与 "
                f"'{cur.line_id}'({cur.start_ms}-{cur.end_ms})"
            )


def match_dialect_clips(
    meta: AnimationMeta,
) -> tuple[dict[str, DialectClip], list[DialectClip]]:
    """Match dialect clips to dialogue lines.

    Prefer explicit line_id; otherwise match the clip with the nearest
    start_ms. Returns (matched_by_line_id, unmatched_extra_clips).
    Missing lines simply have no entry and fall back to the original voice.
    """
    lines_by_id = {line.line_id: line for line in meta.dialogue}
    matched: dict[str, DialectClip] = {}
    used: set[int] = set()

    # First pass: explicit, *known* line_id matches. Clips whose line_id does
    # not correspond to any dialogue line stay in the free pool and may end up
    # discarded as extras.
    for idx, clip in enumerate(meta.dialect_clips):
        if clip.line_id and clip.line_id in lines_by_id and clip.line_id not in matched:
            matched[clip.line_id] = clip
            used.add(idx)

    # Second pass: temporal nearest-neighbour, but only for clips that carry
    # no usable line_id at all. Clips with an unknown line_id are treated as
    # extras rather than silently hijacking another sentence's slot.
    unmatched_lines = [line for line in meta.dialogue if line.line_id not in matched]
    free_clips = [
        (idx, clip)
        for idx, clip in enumerate(meta.dialect_clips)
        if idx not in used and not clip.line_id
    ]
    for line in unmatched_lines:
        if not free_clips:
            break
        best_pos = min(
            range(len(free_clips)),
            key=lambda pos: abs((free_clips[pos][1].start_ms or 0) - line.start_ms),
        )
        idx, clip = free_clips.pop(best_pos)
        matched[line.line_id] = clip
        used.add(idx)

    extras = [clip for idx, clip in enumerate(meta.dialect_clips) if idx not in used]
    return matched, extras
