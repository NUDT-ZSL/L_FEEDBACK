from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dub_replace.synth import SAMPLE_RATE, speech_like, tone, write_corrupt_wav, write_pcm16  # noqa: E402


def stereo_channels():
    return [
        {"index": 0, "label": "L", "role": "sfx", "language": "none"},
        {"index": 1, "label": "R", "role": "target_voice", "language": "zh-CN"},
    ]


def surround_channels():
    return [
        {"index": 0, "label": "L", "role": "sfx", "language": "none"},
        {"index": 1, "label": "R", "role": "sfx", "language": "none"},
        {"index": 2, "label": "C", "role": "target_voice", "language": "zh-CN"},
        {"index": 3, "label": "LFE", "role": "sfx", "language": "none"},
        {"index": 4, "label": "BL", "role": "other_voice", "language": "en"},
        {"index": 5, "label": "BR", "role": "other_voice", "language": "ja"},
    ]


def write_meta(directory: Path, payload: dict) -> Path:
    path = directory / "meta.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def make_meta(animation_id: str, layout: str, channels: list[dict], dialogue: list[dict],
              clips: list[dict], limits=(0.5, 2.0)) -> dict:
    return {
        "animation_id": animation_id,
        "audio": {
            "original_track": "original.wav",
            "sample_rate": SAMPLE_RATE,
            "channel_layout": layout,
            "channels": channels,
        },
        "stretch_limits": {"min": limits[0], "max": limits[1]},
        "dialogue": dialogue,
        "subtitles": [
            {"start_ms": d["start_ms"], "end_ms": d["end_ms"], "text": d["text"]} for d in dialogue
        ],
        "lip_events": [
            {"start_ms": d["start_ms"], "end_ms": d["end_ms"], "line_id": d["line_id"]}
            for d in dialogue
        ],
        "dialect_clips": clips,
    }


def build_stereo_job(root: Path, animation_id: str = "anim_stereo", duration_s: float = 1.0,
                     dialect_seconds: float = 0.3, limits=(0.5, 2.0)):
    """Create a one-animation stereo job. Returns (manifest_path, job_dir, spec)."""
    anim_dir = root / animation_id
    anim_dir.mkdir(parents=True, exist_ok=True)
    n = int(duration_s * SAMPLE_RATE)
    sfx = tone(300, duration_s, amp=0.2, phase=0.5)
    voice = np.zeros(n)
    dialogue = [
        {"line_id": "L1", "start_ms": 100, "end_ms": 500, "text": "第一句"},
        {"line_id": "L2", "start_ms": 600, "end_ms": 900, "text": "第二句"},
    ]
    for i, line in enumerate(dialogue):
        s = int(line["start_ms"] / 1000 * SAMPLE_RATE)
        e = int(line["end_ms"] / 1000 * SAMPLE_RATE)
        voice[s:e] = speech_like((e - s) / SAMPLE_RATE, 200 + 40 * i, amp=0.4, seed=i + 1)
    write_pcm16(anim_dir / "original.wav", np.stack([sfx, voice], axis=1))
    write_pcm16(anim_dir / "d1.wav", speech_like(dialect_seconds, 150, amp=0.4, seed=11))
    clips = [{"clip_id": "D1", "line_id": "L1", "start_ms": 100, "path": "d1.wav"}]
    meta = make_meta(animation_id, "stereo", stereo_channels(), dialogue, clips, limits)
    meta_path = write_meta(anim_dir, meta)
    job_dir = root / "out"
    manifest = {
        "job_dir": "out",
        "concurrency": 1,
        "animations": [{"animation_id": animation_id, "meta": f"{animation_id}/meta.json"}],
    }
    manifest_path = root / "job_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    spec = {"dialogue": dialogue, "sfx": sfx, "voice": voice}
    return manifest_path, job_dir, spec


@pytest.fixture
def stereo_job(tmp_path):
    return build_stereo_job(tmp_path)
