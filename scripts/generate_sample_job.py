"""Generate a fully deterministic sample job under samples/sample_job/.

Run:  python scripts/generate_sample_job.py
The job contains three animations:
  cutscene_stereo  - stereo, replacement + fallback + extra clip + clamp event
  cutscene_surround- 5.1, dialect sample-rate mismatch (FFmpeg resample)
  cutscene_corrupt - intentionally corrupt original WAV (E_CORRUPT, isolated)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dub_replace.synth import SAMPLE_RATE, speech_like, tone, write_corrupt_wav, write_pcm16  # noqa: E402

JOB_ROOT = ROOT / "samples" / "sample_job"


def _meta(animation_id: str, layout: str, channels: list[dict], dialogue: list[dict],
          clips: list[dict], limits=(0.5, 2.0), subtitles=None, lip_events=None) -> dict:
    labels = {"mono": ["mono"], "stereo": ["L", "R"], "5.1": ["L", "R", "C", "LFE", "BL", "BR"]}[layout]
    return {
        "animation_id": animation_id,
        "audio": {
            "original_track": "original.wav",
            "sample_rate": SAMPLE_RATE,
            "channel_layout": layout,
            "channels": channels,
            "channel_labels": labels,
        },
        "stretch_limits": {"min": limits[0], "max": limits[1]},
        "dialogue": dialogue,
        "subtitles": subtitles or [
            {"start_ms": d["start_ms"], "end_ms": d["end_ms"], "text": d["text"]} for d in dialogue
        ],
        "lip_events": lip_events or [
            {"start_ms": d["start_ms"], "end_ms": d["end_ms"], "line_id": d["line_id"]} for d in dialogue
        ],
        "dialect_clips": clips,
    }


def _layout_channels(layout: str) -> list[dict]:
    if layout == "stereo":
        return [
            {"index": 0, "label": "L", "role": "sfx", "language": "none"},
            {"index": 1, "label": "R", "role": "target_voice", "language": "zh-CN"},
        ]
    if layout == "5.1":
        return [
            {"index": 0, "label": "L", "role": "sfx", "language": "none"},
            {"index": 1, "label": "R", "role": "sfx", "language": "none"},
            {"index": 2, "label": "C", "role": "target_voice", "language": "zh-CN"},
            {"index": 3, "label": "LFE", "role": "sfx", "language": "none"},
            {"index": 4, "label": "BL", "role": "other_voice", "language": "en"},
            {"index": 5, "label": "BR", "role": "other_voice", "language": "ja"},
        ]
    raise ValueError(layout)


def gen_stereo(base: Path) -> dict:
    base.mkdir(parents=True, exist_ok=True)
    duration = 2.5
    n = int(duration * SAMPLE_RATE)
    sfx = tone(220, duration, amp=0.25)
    voice = np.zeros(n)
    dialogue = [
        {"line_id": "L1", "start_ms": 100, "end_ms": 900, "text": "你好，旅行者。"},
        {"line_id": "L2", "start_ms": 1000, "end_ms": 1800, "text": "今天天气不错。"},
        {"line_id": "L3", "start_ms": 1900, "end_ms": 2400, "text": "该出发了。"},
    ]
    for i, line in enumerate(dialogue):
        s = int(line["start_ms"] / 1000 * SAMPLE_RATE)
        e = int(line["end_ms"] / 1000 * SAMPLE_RATE)
        voice[s:e] = speech_like((e - s) / SAMPLE_RATE, 180 + 40 * i, amp=0.4, seed=10 + i)
    stereo = np.stack([sfx, voice], axis=1)
    write_pcm16(base / "original.wav", stereo)

    # L1 dialect much longer -> will exceed max ratio -> pad/trim clamp event.
    write_pcm16(base / "dialect_l1.wav", speech_like(2.0, 150, amp=0.4, seed=21))
    # L2 dialect shorter but within limits -> normal WSOLA.
    write_pcm16(base / "dialect_l2.wav", speech_like(0.45, 150, amp=0.4, seed=22))
    # L3 intentionally missing -> fallback.
    # Extra unmatched clip -> discarded by nearest-distance rule.
    write_pcm16(base / "dialect_extra.wav", speech_like(0.4, 150, amp=0.3, seed=23))

    clips = [
        {"clip_id": "D1", "line_id": "L1", "start_ms": 100, "path": "dialect_l1.wav"},
        {"clip_id": "D2", "line_id": "L2", "start_ms": 1000, "path": "dialect_l2.wav"},
        {"clip_id": "D9", "line_id": "L9-X", "start_ms": 2300, "path": "dialect_extra.wav"},
    ]
    return _meta("cutscene_stereo", "stereo", _layout_channels("stereo"), dialogue, clips,
                 limits=(0.5, 1.5))


def gen_surround(base: Path) -> dict:
    base.mkdir(parents=True, exist_ok=True)
    duration = 2.0
    n = int(duration * SAMPLE_RATE)
    sfx_l = tone(110, duration, amp=0.2)
    sfx_r = tone(130, duration, amp=0.2)
    center = np.zeros(n)
    lfe = tone(55, duration, amp=0.3)
    en = np.zeros(n)
    ja = np.zeros(n)
    dialogue = [
        {"line_id": "S1", "start_ms": 200, "end_ms": 1000, "text": "小心脚下。"},
        {"line_id": "S2", "start_ms": 1100, "end_ms": 1800, "text": "跟我来。"},
    ]
    for i, line in enumerate(dialogue):
        s = int(line["start_ms"] / 1000 * SAMPLE_RATE)
        e = int(line["end_ms"] / 1000 * SAMPLE_RATE)
        center[s:e] = speech_like((e - s) / SAMPLE_RATE, 200 + 30 * i, amp=0.4, seed=30 + i)
        en[s:e] = speech_like((e - s) / SAMPLE_RATE, 260, amp=0.25, seed=40 + i)
        ja[s:e] = speech_like((e - s) / SAMPLE_RATE, 300, amp=0.25, seed=50 + i)
    track = np.stack([sfx_l, sfx_r, center, lfe, en, ja], axis=1)
    write_pcm16(base / "original.wav", track)

    write_pcm16(base / "dialect_s1.wav", speech_like(0.55, 160, amp=0.4, seed=61))
    # 24 kHz dialect clip -> E_SAMPLE_RATE warning + FFmpeg resample path.
    write_pcm16(base / "dialect_s2_24k.wav", speech_like(0.5, 160, sr=24000, amp=0.4, seed=62), sr=24000)
    clips = [
        {"clip_id": "DS1", "line_id": "S1", "start_ms": 200, "path": "dialect_s1.wav"},
        {"clip_id": "DS2", "line_id": "S2", "start_ms": 1100, "path": "dialect_s2_24k.wav"},
    ]
    return _meta("cutscene_surround", "5.1", _layout_channels("5.1"), dialogue, clips)


def gen_corrupt(base: Path) -> dict:
    base.mkdir(parents=True, exist_ok=True)
    write_corrupt_wav(base / "original.wav")
    dialogue = [{"line_id": "C1", "start_ms": 0, "end_ms": 500, "text": "损坏样本。"}]
    clips = []
    return _meta("cutscene_corrupt", "stereo", _layout_channels("stereo"), dialogue, clips)


def main() -> int:
    if JOB_ROOT.exists():
        import shutil

        shutil.rmtree(JOB_ROOT)
    metas = [
        ("cutscene_stereo", gen_stereo(JOB_ROOT / "cutscene_stereo")),
        ("cutscene_surround", gen_surround(JOB_ROOT / "cutscene_surround")),
        ("cutscene_corrupt", gen_corrupt(JOB_ROOT / "cutscene_corrupt")),
    ]
    for name, meta in metas:
        (JOB_ROOT / name / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    manifest = {
        "job_dir": "out",
        "concurrency": 2,
        "animations": [
            {"animation_id": name, "meta": f"{name}/meta.json"} for name, _ in metas
        ],
    }
    (JOB_ROOT / "job_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"样本作业已生成: {JOB_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
