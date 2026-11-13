"""片段级淡入淡出与时间轴重叠片段的线性交叉淡化。

约定：半开区间 [start_frame, end_frame)；所有增益曲线按绝对采样帧计算，
因此结果与分块大小无关。交叉淡化区间内两片段增益之和恒为 1。
"""

from __future__ import annotations

import numpy as np

from .timeline import Clip, round_half_up


def _independent_envelope(clip: Clip, start: int, count: int) -> np.ndarray:
    """单个片段自身的淡入/淡出包络（尚未考虑与相邻片段的交叉淡化）。"""
    env = np.zeros(count, dtype=np.float64)
    local_lo = max(start, clip.start_frame)
    local_hi = min(start + count, clip.end_frame)
    if local_lo >= local_hi:
        return env
    env[local_lo - start : local_hi - start] = 1.0
    rate = clip.project_rate
    fade_in_frames = min(round_half_up(clip.fade_in * rate), clip.length)
    fade_out_frames = min(round_half_up(clip.fade_out * rate), clip.length)
    if fade_in_frames > 0:
        ramp_end = min(clip.start_frame + fade_in_frames, clip.end_frame)
        for frame in range(local_lo, min(local_hi, ramp_end)):
            offset = frame - clip.start_frame
            env[frame - start] = (offset + 1) / fade_in_frames
    if fade_out_frames > 0:
        ramp_begin = max(clip.start_frame, clip.end_frame - fade_out_frames)
        for frame in range(max(local_lo, ramp_begin), local_hi):
            offset = clip.end_frame - 1 - frame
            env[frame - start] = (offset + 1) / fade_out_frames
    return env


def compute_envelopes(
    clips: list[Clip], start: int, count: int
) -> dict[str, np.ndarray]:
    """计算同轨若干片段在 [start, start+count) 区间上的增益包络。

    先应用各片段独立淡入淡出，再对重叠片段做线性交叉淡化；
    三个及以上片段同时重叠时按比例归一化，保证每帧增益之和为 1。
    """
    count = int(count)
    end = start + count
    envelopes: dict[str, np.ndarray] = {}
    active: list[Clip] = []
    for clip in clips:
        if clip.start_frame < end and clip.end_frame > start:
            envelopes[clip.clip_id] = _independent_envelope(clip, start, count)
            active.append(clip)
    active.sort(key=lambda clip: (clip.start_frame, clip.clip_id))

    # 交叉淡化只依赖片段的绝对起止帧，与渲染窗口/块大小无关。
    for index, outgoing in enumerate(active[:-1]):
        incoming = active[index + 1]
        overlap_lo = max(outgoing.start_frame, incoming.start_frame)
        overlap_hi = min(outgoing.end_frame, incoming.end_frame)
        if overlap_lo >= overlap_hi:
            continue
        full_len = overlap_hi - overlap_lo
        positions = np.arange(full_len, dtype=np.float64)
        incoming_full = (positions + 1.0) / full_len
        outgoing_full = 1.0 - incoming_full
        view_lo = max(overlap_lo, start) - start
        view_hi = min(overlap_hi, end) - start
        src_lo = max(overlap_lo, start) - overlap_lo
        src_hi = src_lo + (view_hi - view_lo)
        envelopes[outgoing.clip_id][view_lo:view_hi] = outgoing_full[src_lo:src_hi]
        envelopes[incoming.clip_id][view_lo:view_hi] = incoming_full[src_lo:src_hi]
    return envelopes

