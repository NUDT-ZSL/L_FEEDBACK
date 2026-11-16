"""Per-animation replacement pipeline.

Only the target-language voice channel(s) are modified; SFX and other
language channels keep their original sample-accurate timeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import audioio
from .errors import ChannelLayoutError, DubReplaceError, SampleRateError
from .metadata import AnimationMeta, CHANNEL_LAYOUTS, match_dialect_clips, parse_metadata
from .timestretch import fit_duration
from .util import ms_to_samples, peak_dbfs, samples_to_ms

CROSSFADE_MS = 5.0


def _event(code: str, level: str, message: str, hint: str | None = None) -> dict:
    return {"code": code, "level": level, "message": message, "hint": hint}


def _crossfade_blend(original: np.ndarray, replacement: np.ndarray, sample_rate: int) -> np.ndarray:
    """Equal-length blend with short linear fades at both boundaries."""
    length = replacement.shape[0]
    fade = min(length // 4, ms_to_samples(CROSSFADE_MS, sample_rate))
    out = replacement.copy()
    if fade >= 2:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float64)
        out[:fade] = replacement[:fade] * ramp + original[:fade] * (1.0 - ramp)
        out[-fade:] = replacement[-fade:] * ramp[::-1] + original[-fade:] * (1.0 - ramp[::-1])
    return out


def _load_original(meta: AnimationMeta, work_dir: Path, events: list[dict]):
    samples, sr = audioio.read_wav(meta.original_track)
    ok, detail = audioio.verify_wav(meta.original_track)
    if not ok:
        from .errors import CorruptAudioError

        raise CorruptAudioError(f"FFmpeg 校验未通过，原始音轨可能损坏: {meta.original_track} ({detail})")
    if sr != meta.sample_rate:
        raise SampleRateError(
            f"原始音轨采样率 {sr}Hz 与元数据声明 {meta.sample_rate}Hz 不一致: {meta.original_track}"
        )
    expected_channels = len(CHANNEL_LAYOUTS[meta.channel_layout])
    if samples.ndim == 1:
        if expected_channels != 1:
            raise ChannelLayoutError(
                f"实际为单声道 WAV，元数据声明布局 '{meta.channel_layout}' 需要 {expected_channels} 声道"
            )
        frames = samples.reshape(-1, 1)
    else:
        actual_channels = samples.shape[1]
        if actual_channels != expected_channels:
            raise ChannelLayoutError(
                f"实际 {actual_channels} 声道与元数据布局 '{meta.channel_layout}' "
                f"要求的 {expected_channels} 声道不一致"
            )
        frames = samples
    return frames


def process_animation(
    meta_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    work_dir: str | Path,
) -> dict:
    """Run the full pipeline for one cutscene directory and return its report."""
    if isinstance(meta_path, AnimationMeta):
        meta = meta_path
    else:
        meta = parse_metadata(Path(meta_path))
    output_path = Path(output_path)
    report_path = Path(report_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    events: list[dict] = []
    frames = _load_original(meta, work_dir, events)
    total_frames = frames.shape[0]
    target_indexes = meta.target_voice_indexes

    matched, extras = match_dialect_clips(meta)

    sentences: list[dict] = []
    replaced_count = 0
    fallback_count = 0
    failed_lines = 0
    stretch_min, stretch_max = meta.stretch_limits

    for line in meta.dialogue:
        start = ms_to_samples(line.start_ms, meta.sample_rate)
        end = min(ms_to_samples(line.end_ms, meta.sample_rate), total_frames)
        slot_len = max(0, end - start)
        sentence_events: list[dict] = []
        method = "fallback"
        dialect_path: str | None = None
        ratio = 1.0
        limit_event = None

        clip = matched.get(line.line_id)
        if clip is not None:
            dialect_path = str((meta.base_dir / clip.path).resolve())
            try:
                dialect, dialect_sr, resampled = audioio.ensure_sample_rate(
                    meta.base_dir / clip.path, meta.sample_rate, work_dir=work_dir
                )
            except DubReplaceError as exc:
                # Missing dialect file is recoverable: keep the original line.
                if exc.code == "E_MISSING_AUDIO":
                    sentence_events.append(
                        _event(exc.code, "warning", f"方言片段缺失，沿用原句: {exc.message}", exc.hint)
                    )
                    clip = None
                else:
                    # Corrupt / unresamplable dialect: unrecoverable for this job item.
                    raise
            else:
                if resampled:
                    sentence_events.append(
                        _event(
                            "E_SAMPLE_RATE",
                            "warning",
                            f"方言片段采样率 {dialect_sr}Hz 已通过 FFmpeg 重采样至 {meta.sample_rate}Hz: {clip.path}",
                        )
                    )
                mono = audioio.downmix_to_mono(dialect)
                if dialect.ndim > 1:
                    sentence_events.append(
                        _event("E_CHANNEL", "info", f"方言片段为多声道，已按等权下混为单声道: {clip.path}")
                    )
                fitted, ratio, limit_event = fit_duration(mono, slot_len, stretch_min, stretch_max)
                if limit_event is not None:
                    sentence_events.append(
                        _event(
                            "E_STRETCH",
                            "warning",
                            (
                                f"伸缩比 {slot_len / max(1, mono.shape[0]):.3f} 超出限制 "
                                f"[{stretch_min}, {stretch_max}]，已改为{'尾部静音填充' if limit_event == 'pad' else '裁剪尾部'}"
                                f"（句 {line.line_id}）"
                            ),
                        )
                    )
                original_slot = frames[start:end, target_indexes[0]]
                blended = _crossfade_blend(original_slot, fitted, meta.sample_rate)
                for channel_index in target_indexes:
                    frames[start:end, channel_index] = blended
                method = "replaced"
                replaced_count += 1

        if clip is None:
            fallback_count += 1

        slot_peak = peak_dbfs(frames[start:end, target_indexes[0]]) if slot_len > 0 else peak_dbfs([])
        sentences.append(
            {
                "line_id": line.line_id,
                "text": line.text,
                "method": method,
                "dialect_clip": dialect_path,
                "planned_start_ms": line.start_ms,
                "planned_end_ms": line.end_ms,
                "actual_start_ms": samples_to_ms(start, meta.sample_rate),
                "actual_end_ms": samples_to_ms(end, meta.sample_rate),
                "duration_ms": samples_to_ms(slot_len, meta.sample_rate),
                "stretch_ratio": round(float(ratio), 4),
                "limit_event": limit_event,
                "peak_dbfs": slot_peak,
                "events": sentence_events,
            }
        )
        events.extend(sentence_events)

    discarded: list[dict] = []
    for clip in extras:
        anchor = clip.start_ms or 0
        nearest = min(meta.dialogue, key=lambda ln: abs(ln.start_ms - anchor)) if meta.dialogue else None
        discarded.append(
            {
                "clip_id": clip.clip_id,
                "path": str((meta.base_dir / clip.path).resolve()),
                "nearest_line_id": nearest.line_id if nearest else None,
                "distance_ms": abs(anchor - nearest.start_ms) if nearest else None,
                "reason": "多余方言片段，按时间轴最近距离匹配后仍无对应原句，已丢弃",
            }
        )
        events.append(
            _event(
                "E_EXTRA_CLIP",
                "warning",
                f"多余方言片段已丢弃: {clip.path}（最近原句 {nearest.line_id if nearest else 'N/A'}）",
            )
        )

    # Output: mono layout writes a 1-D array, otherwise interleaved 2-D.
    out_samples = frames[:, 0] if frames.shape[1] == 1 else frames
    audioio.write_wav_pcm16(output_path, out_samples, meta.sample_rate)
    ok, detail = audioio.verify_wav(output_path)
    if not ok:
        from .errors import CorruptAudioError

        raise CorruptAudioError(f"输出 WAV 的 FFmpeg 校验失败: {output_path} ({detail})")

    warning_count = sum(1 for ev in events if ev["level"] == "warning")
    report = {
        "animation_id": meta.animation_id,
        "status": "success",
        "input": {
            "original_track": str(meta.original_track.resolve()),
            "sample_rate": meta.sample_rate,
            "channel_layout": meta.channel_layout,
            "channels": [
                {"index": ch.index, "label": ch.label, "role": ch.role, "language": ch.language}
                for ch in meta.channels
            ],
        },
        "output": {
            "path": str(output_path.resolve()),
            "sample_rate": meta.sample_rate,
            "channels": frames.shape[1],
            "frames": int(total_frames),
            "duration_ms": samples_to_ms(total_frames, meta.sample_rate),
            "peak_dbfs": peak_dbfs(frames[:, target_indexes]),
        },
        "sentences": sentences,
        "discarded_clips": discarded,
        "events": events,
        "summary": {
            "total_lines": len(meta.dialogue),
            "replaced": replaced_count,
            "fallback": fallback_count,
            "failed_lines": failed_lines,
            "warnings": warning_count,
            "discarded": len(discarded),
        },
    }
    from .util import atomic_write_json

    atomic_write_json(report_path, report)
    return report
