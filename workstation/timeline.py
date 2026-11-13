"""台词表解析与采样级时间轴映射。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from .sources import SourceLibrary


def round_half_up(value: float) -> int:
    """四舍五入（半值向上取整），用于浮点起始时间到采样偏移的映射。"""
    return int(math.floor(float(value) + 0.5))


@dataclass
class Clip:
    """时间轴上的一个对齐片段。"""

    clip_id: str
    source_id: str
    text: str
    track: str
    start_time: float          # JSON 中的原始浮点起始秒数，原样保留
    start_frame: int           # 四舍五入后的实际采样偏移
    length: int                # 项目采样率下的帧数
    source_rate: int
    project_rate: int
    source_path: str
    source_fingerprint: str
    fade_in: float = 0.0
    fade_out: float = 0.0

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.length

    def to_dict(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "source_id": self.source_id,
            "text": self.text,
            "track": self.track,
            "start_time": self.start_time,
            "start_frame": self.start_frame,
            "length": self.length,
            "end_frame": self.end_frame,
            "source_rate": self.source_rate,
            "project_rate": self.project_rate,
            "source_path": self.source_path,
            "source_fingerprint": self.source_fingerprint,
            "fade_in": self.fade_in,
            "fade_out": self.fade_out,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Clip":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


@dataclass
class Timeline:
    """全部片段构成的采样级时间轴。"""

    project_rate: int
    clips: list[Clip] = field(default_factory=list)

    @property
    def length(self) -> int:
        if not self.clips:
            return 0
        return max(clip.end_frame for clip in self.clips)

    def clips_for_range(self, start_frame: int, frame_count: int) -> list[Clip]:
        """仅返回与请求区间相交的片段，避免全量重算。"""
        end_frame = start_frame + frame_count
        return [
            clip
            for clip in self.clips
            if clip.start_frame < end_frame and clip.end_frame > start_frame
        ]

    def by_track(self) -> dict[str, list[Clip]]:
        tracks: dict[str, list[Clip]] = {}
        for clip in self.clips:
            tracks.setdefault(clip.track, []).append(clip)
        for track_clips in tracks.values():
            track_clips.sort(key=lambda clip: (clip.start_frame, clip.clip_id))
        return tracks

    def to_list(self) -> list[dict]:
        return [clip.to_dict() for clip in self.clips]

    @classmethod
    def from_list(cls, project_rate: int, data: list[dict]) -> "Timeline":
        return cls(
            project_rate=int(project_rate),
            clips=[Clip.from_dict(item) for item in data],
        )


def build_timeline(
    script: dict,
    library: SourceLibrary,
    default_fade_in: float = 0.0,
    default_fade_out: float = 0.0,
) -> tuple[Timeline, list[dict]]:
    """解析台词表并把每条台词映射到采样级时间轴。

    返回时间轴与对齐报告。素材加载失败的条目跳过（错误登记在 library 中）。
    """
    project_rate = int(script.get("project_rate", library.project_rate))
    timeline = Timeline(project_rate=project_rate)
    report: list[dict] = []
    for index, entry in enumerate(script.get("lines", [])):
        source_id = str(entry["source_id"])
        source = library.get(source_id)
        if source is None:
            continue  # 错误已由 SourceLibrary 登记
        start_time = float(entry["start"])
        start_frame = round_half_up(start_time * project_rate)
        clip_id = str(entry.get("clip_id", f"clip_{index:04d}"))
        fade_in = float(entry.get("fade_in", default_fade_in))
        fade_out = float(entry.get("fade_out", default_fade_out))
        clip = Clip(
            clip_id=clip_id,
            source_id=source_id,
            text=str(entry.get("text", "")),
            track=str(entry.get("track", "dialog")),
            start_time=start_time,
            start_frame=start_frame,
            length=source.length,
            source_rate=source.original_rate,
            project_rate=project_rate,
            source_path=str(source.path),
            source_fingerprint=source.fingerprint,
            fade_in=fade_in,
            fade_out=fade_out,
        )
        timeline.clips.append(clip)
        report.append(
            {
                "clip_id": clip_id,
                "source_id": source_id,
                "text": clip.text,
                "track": clip.track,
                "start_time": start_time,
                "start_frame": start_frame,
                "rounding_delta_frames": start_frame - start_time * project_rate,
                "length": source.length,
                "end_frame": clip.end_frame,
                "source_rate": source.original_rate,
                "project_rate": project_rate,
            }
        )
    return timeline, report

