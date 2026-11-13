"""确定性生成 WAV 素材、台词表与混音配置（任务自包含，无需外部资源）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf


def _tone(
    frequency: float,
    duration: float,
    samplerate: int,
    amplitude: float = 0.5,
    phase: float = 0.0,
) -> np.ndarray:
    count = int(round(duration * samplerate))
    t = np.arange(count, dtype=np.float64) / samplerate
    return (amplitude * np.sin(2.0 * np.pi * frequency * t + phase)).astype(
        np.float64
    )


def _stereo_music(duration: float, samplerate: int) -> np.ndarray:
    count = int(round(duration * samplerate))
    t = np.arange(count, dtype=np.float64) / samplerate
    left = 0.3 * np.sin(2 * np.pi * 220.0 * t)
    right = 0.3 * np.sin(2 * np.pi * 277.18 * t)
    return np.stack([left, right], axis=1)


def generate_workspace(workspace: Path, project_rate: int = 48000) -> dict:
    """在 ``workspace`` 下生成全部确定性输入，返回关键路径与说明。"""
    workspace = Path(workspace)
    audio_dir = workspace / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    # 台词：不同采样率以验证重采样
    sf.write(
        audio_dir / "line_1.wav",
        _tone(440.0, 0.5, project_rate, amplitude=0.6),
        project_rate,
        subtype="PCM_16",
    )
    paths["line_1"] = audio_dir / "line_1.wav"
    sf.write(
        audio_dir / "line_2.wav",
        _tone(554.37, 0.4, 44100, amplitude=0.55, phase=0.7),
        44100,
        subtype="PCM_16",
    )
    paths["line_2"] = audio_dir / "line_2.wav"
    sf.write(
        audio_dir / "line_3.wav",
        _tone(659.25, 0.6, 32000, amplitude=0.5),
        32000,
        subtype="PCM_16",
    )
    paths["line_3"] = audio_dir / "line_3.wav"
    # 与 line_1 重叠的台词，用于交叉淡化
    sf.write(
        audio_dir / "line_overlap.wav",
        _tone(330.0, 0.4, project_rate, amplitude=0.45),
        project_rate,
        subtype="PCM_16",
    )
    paths["line_overlap"] = audio_dir / "line_overlap.wav"
    # 背景音乐（立体声）与音效
    sf.write(
        audio_dir / "music.wav",
        _stereo_music(2.0, project_rate),
        project_rate,
        subtype="PCM_16",
    )
    paths["music"] = audio_dir / "music.wav"
    sf.write(
        audio_dir / "sfx.wav",
        _tone(880.0, 0.25, project_rate, amplitude=0.35),
        project_rate,
        subtype="PCM_16",
    )
    paths["sfx"] = audio_dir / "sfx.wav"

    # 异常素材 1：文件缺失（不创建）
    paths["missing"] = audio_dir / "missing_line.wav"
    # 异常素材 2：不受支持的采样格式（伪造文件头，soundfile 无法识别）
    bad_format = audio_dir / "bad_format.wav"
    bad_format.write_bytes(b"NOT_A_REAL_WAV_FILE" + bytes(128))
    paths["bad_format"] = bad_format
    # 异常素材 3：解码后帧数为零（合法 WAV 容器但无数据帧）
    empty = audio_dir / "empty_line.wav"
    sf.write(empty, np.zeros((0,), dtype=np.float64), project_rate, subtype="PCM_16")
    paths["empty_line"] = empty

    script = {
        "project_rate": project_rate,
        "sources": [
            {"source_id": "line_1", "path": "audio/line_1.wav"},
            {"source_id": "line_2", "path": "audio/line_2.wav"},
            {"source_id": "line_3", "path": "audio/line_3.wav"},
            {"source_id": "line_overlap", "path": "audio/line_overlap.wav"},
            {"source_id": "music", "path": "audio/music.wav"},
            {"source_id": "sfx", "path": "audio/sfx.wav"},
            {"source_id": "missing", "path": "audio/missing_line.wav"},
            {"source_id": "bad_format", "path": "audio/bad_format.wav"},
            {"source_id": "empty_line", "path": "audio/empty_line.wav"},
        ],
        "lines": [
            # 0.12345s 非采样周期整数倍，验证四舍五入
            {
                "clip_id": "c001",
                "source_id": "line_1",
                "start": 0.12345,
                "text": "你好，这里是第一条台词。",
                "track": "dialog",
                "fade_in": 0.01,
                "fade_out": 0.01,
            },
            {
                "clip_id": "c002",
                "source_id": "line_2",
                "start": 0.8,
                "text": "第二条来自 44.1kHz 的素材。",
                "track": "dialog",
            },
            {
                "clip_id": "c003",
                "source_id": "line_3",
                "start": 1.5,
                "text": "第三条来自 32kHz 的素材。",
                "track": "dialog2",
            },
            {
                "clip_id": "c004",
                "source_id": "line_overlap",
                "start": 0.5,
                "text": "与第一条重叠，触发交叉淡化。",
                "track": "dialog",
            },
            {
                "clip_id": "c005",
                "source_id": "music",
                "start": 0.0,
                "text": "背景音乐",
                "track": "music",
            },
            {
                "clip_id": "c006",
                "source_id": "sfx",
                "start": 1.2,
                "text": "提示音效",
                "track": "sfx",
            },
            {
                "clip_id": "c007",
                "source_id": "missing",
                "start": 2.0,
                "text": "缺失素材台词",
                "track": "dialog",
            },
            {
                "clip_id": "c008",
                "source_id": "bad_format",
                "start": 2.1,
                "text": "格式错误素材台词",
                "track": "dialog",
            },
            {
                "clip_id": "c009",
                "source_id": "empty_line",
                "start": 2.2,
                "text": "空素材台词",
                "track": "dialog",
            },
        ],
    }
    mix_config = {
        "tracks": {
            "dialog": {"gain": 1.0, "pan": -0.2, "mute": False, "solo": False},
            "dialog2": {"gain": 0.9, "pan": 0.3, "mute": False, "solo": False},
            "music": {"gain": 0.5, "pan": 0.0, "mute": False, "solo": False},
            "sfx": {"gain": 0.8, "pan": 0.6, "mute": False, "solo": False},
        }
    }
    script_path = workspace / "script.json"
    mix_path = workspace / "mix.json"
    script_path.write_text(
        json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    mix_path.write_text(
        json.dumps(mix_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "workspace": workspace,
        "script_path": script_path,
        "mix_path": mix_path,
        "audio_paths": paths,
        "project_rate": project_rate,
    }

