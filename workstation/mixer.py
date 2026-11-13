"""多轨混音引擎、实时预览与混音快照增量更新。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .fades import compute_envelopes
from .sources import SourceLibrary
from .timeline import Clip, Timeline


@dataclass
class TrackConfig:
    """单轨混音参数。"""

    gain: float = 1.0
    pan: float = 0.0
    mute: bool = False
    solo: bool = False

    def to_dict(self) -> dict:
        return {
            "gain": float(self.gain),
            "pan": float(self.pan),
            "mute": bool(self.mute),
            "solo": bool(self.solo),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrackConfig":
        return cls(
            gain=float(data.get("gain", 1.0)),
            pan=float(data.get("pan", 0.0)),
            mute=bool(data.get("mute", False)),
            solo=bool(data.get("solo", False)),
        )


class Mixer:
    """按时间轴与轨道参数渲染立体声 PCM。"""

    def __init__(
        self,
        timeline: Timeline,
        library: SourceLibrary,
        tracks: dict[str, TrackConfig] | None = None,
    ) -> None:
        self.timeline = timeline
        self.library = library
        self.tracks = tracks or {}

    def update_tracks(self, tracks: dict[str, TrackConfig]) -> None:
        self.tracks = dict(tracks)

    def _active_tracks(self) -> set[str]:
        solo_tracks = {
            name
            for name, cfg in self.tracks.items()
            if (cfg or TrackConfig()).solo
        }
        active: set[str] = set()
        for name, cfg in self.tracks.items():
            cfg = cfg or TrackConfig()
            if cfg.mute:
                continue  # 被标记静音时即使独奏也静音
            if solo_tracks and name not in solo_tracks:
                continue
            active.add(name)
        return active

    def render_range(self, start_frame: int, frame_count: int) -> np.ndarray:
        """渲染 [start_frame, start_frame+frame_count)，输出 (n, 2) float64。

        所有包络按绝对帧号计算且渲染无跨块状态，因此块大小不影响结果。
        """
        start_frame = int(start_frame)
        frame_count = int(frame_count)
        output = np.zeros((frame_count, 2), dtype=np.float64)
        if frame_count <= 0:
            return output
        active_tracks = self._active_tracks()
        grouped = self.timeline.by_track()
        for track_name, clips in grouped.items():
            config = self.tracks.get(track_name) or TrackConfig()
            if track_name not in active_tracks:
                continue
            in_range = [
                clip
                for clip in clips
                if clip.start_frame < start_frame + frame_count
                and clip.end_frame > start_frame
            ]
            if not in_range:
                continue
            envelopes = compute_envelopes(in_range, start_frame, frame_count)
            track_bus = np.zeros((frame_count, 2), dtype=np.float64)
            for clip in in_range:
                source = self.library.get(clip.source_id)
                if source is None:
                    continue
                lo = max(start_frame, clip.start_frame)
                hi = min(start_frame + frame_count, clip.end_frame)
                src_lo = lo - clip.start_frame
                src_hi = src_lo + (hi - lo)
                audio = source.frames[src_lo:src_hi]
                if audio.ndim == 1:
                    stereo = np.stack([audio, audio], axis=1)
                else:
                    stereo = np.asarray(audio[:, :2], dtype=np.float64)
                    if stereo.shape[1] == 1:
                        stereo = np.repeat(stereo, 2, axis=1)
                gain_env = envelopes[clip.clip_id][lo - start_frame : hi - start_frame]
                track_bus[lo - start_frame : hi - start_frame] += (
                    stereo * gain_env[:, None]
                )
            left_gain = config.gain * math_pan_left(config.pan)
            right_gain = config.gain * math_pan_right(config.pan)
            output[:, 0] += track_bus[:, 0] * left_gain
            output[:, 1] += track_bus[:, 1] * right_gain
        return np.clip(output, -1.0, 1.0)

    def stream(
        self, total_frames: int, block_size: int = 512
    ) -> Iterator[tuple[int, np.ndarray]]:
        """按可配置块大小流式产出 (起始帧, PCM 块)。"""
        block_size = int(block_size)
        for offset in range(0, int(total_frames), block_size):
            count = min(block_size, int(total_frames) - offset)
            yield offset, self.render_range(offset, count)


def math_pan_left(pan: float) -> float:
    """线性等功率声像的左声道增益（pan ∈ [-1, 1]）。"""
    return float(np.clip((1.0 - pan) * 0.5, 0.0, 1.0))


def math_pan_right(pan: float) -> float:
    """线性等功率声像的右声道增益（pan ∈ [-1, 1]）。"""
    return float(np.clip((1.0 + pan) * 0.5, 0.0, 1.0))


class PreviewEngine:
    """实时混音预览：按请求区间渲染，只处理覆盖片段。"""

    def __init__(self, mixer: Mixer) -> None:
        self.mixer = mixer
        self.requests_served = 0
        self.clip_touches = 0

    def preview(self, start_frame: int, frame_count: int) -> np.ndarray:
        touched = self.mixer.timeline.clips_for_range(start_frame, frame_count)
        self.requests_served += 1
        self.clip_touches += len(touched)
        return self.mixer.render_range(start_frame, frame_count)

    def reset_counters(self) -> None:
        self.requests_served = 0
        self.clip_touches = 0


class SnapshotManager:
    """混音参数快照与增量更新。"""

    def __init__(self, mixer: Mixer) -> None:
        self.mixer = mixer
        self.version: int | None = 0
        self.snapshot: dict[str, TrackConfig] = {}

    @staticmethod
    def parse_snapshot(data: dict) -> tuple[int, dict[str, TrackConfig]]:
        version = int(data.get("version", 1))
        tracks = {
            name: TrackConfig.from_dict(cfg)
            for name, cfg in data.get("tracks", {}).items()
        }
        return version, tracks

    def load_snapshot(
        self, data: dict, timeline: Timeline | None = None
    ) -> dict:
        """整体替换载入快照；版本变化时丢弃旧偏移并重新映射时间轴。"""
        version, tracks = self.parse_snapshot(data)
        old_version = self.version
        remapped = False
        if old_version is not None and version != old_version:
            if timeline is not None:
                remap_timeline(timeline, self.mixer.library)
                remapped = True
        self.version = version
        self.snapshot = tracks
        self.mixer.update_tracks(tracks)
        return {
            "old_version": old_version,
            "new_version": version,
            "remapped": remapped,
            "track_count": len(tracks),
        }

    def apply_parameter_change(
        self,
        track_name: str,
        changes: dict,
        render_length: int,
        block_size: int = 512,
    ) -> dict:
        """修改单轨参数后只重算受影响区间，并输出差异报告。"""
        config = self.snapshot.setdefault(track_name, TrackConfig())
        old = config.to_dict()
        before = self.mixer.render_range(0, render_length)
        for key, value in changes.items():
            if key in ("mute", "solo"):
                setattr(config, key, bool(value))
            elif key in ("gain", "pan"):
                setattr(config, key, float(value))
        self.mixer.update_tracks(self.snapshot)
        after = self.mixer.render_range(0, render_length)
        diff_mask = np.any(before != after, axis=1)
        changed_frames = np.flatnonzero(diff_mask)
        if changed_frames.size:
            affected = {
                "start_frame": int(changed_frames[0]),
                "end_frame": int(changed_frames[-1]) + 1,
            }
        else:
            affected = {"start_frame": None, "end_frame": None}
        return {
            "track": track_name,
            "old": old,
            "new": config.to_dict(),
            "affected_range": affected,
            "changed_frame_count": int(changed_frames.size),
            "block_size": int(block_size),
        }


def remap_timeline(timeline: Timeline, library: SourceLibrary) -> None:
    """快照版本变化时按原始浮点时间重新计算采样偏移（不沿用旧偏移）。"""
    for clip in timeline.clips:
        clip.start_frame = int(
            np.floor(clip.start_time * timeline.project_rate + 0.5)
        )
        source = library.get(clip.source_id)
        if source is not None:
            clip.length = source.length

