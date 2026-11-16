from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import numpy as np


def write_mono_wav(path: Path, samples: np.ndarray, *, sample_rate: int = 48_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.rint(np.asarray(samples) * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.setcomptype("NONE", "not compressed")
        handle.writeframes(pcm.tobytes())


def tone(
    duration_s: float,
    *,
    frequency: float = 220.0,
    amplitude: float = 0.5,
    sample_rate: int = 48_000,
) -> np.ndarray:
    frames = int(round(duration_s * sample_rate))
    t = np.arange(frames, dtype=np.float64) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * frequency * t)


def silence(
    duration_s: float,
    *,
    sample_rate: int = 48_000,
) -> np.ndarray:
    return np.zeros(int(round(duration_s * sample_rate)), dtype=np.float64)


def dialogue(
    dialogue_id: str,
    start_s: float,
    end_s: float,
    *,
    subtitle_id: str | None = None,
    text: str | None = None,
    source_duration_s: float | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": dialogue_id,
        "start_s": start_s,
        "end_s": end_s,
        "subtitle_id": subtitle_id or f"sub_{dialogue_id}",
        "text": text or f"Line {dialogue_id}",
    }
    if source_duration_s is not None:
        result["source_duration_s"] = source_duration_s
    return result


def timeline_payload(
    dialogues: list[dict[str, Any]],
    *,
    duration_s: float = 4.0,
    target_language: str = "en-US",
    project_id: str = "test_project",
) -> dict[str, Any]:
    subtitles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in dialogues:
        subtitle_id = item["subtitle_id"]
        if subtitle_id in seen:
            continue
        seen.add(subtitle_id)
        subtitles.append(
            {
                "id": subtitle_id,
                "start_s": item["start_s"],
                "end_s": item["end_s"],
                "text": item["text"],
            }
        )
    return {
        "schema_version": "1.0",
        "project_id": project_id,
        "target_language": target_language,
        "duration_s": duration_s,
        "shots": [
            {
                "id": "shot_001",
                "start_s": 0.0,
                "end_s": duration_s,
                "events": [
                    {"id": "evt_001", "type": "camera_cut", "at_s": 0.0}
                ],
                "subtitles": subtitles,
                "dialogues": dialogues,
            }
        ],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_wav(path: Path) -> tuple[np.ndarray, int, int, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.getnframes()
        payload = handle.readframes(frames)
    pcm = np.frombuffer(payload, dtype="<i2")
    if channels > 1:
        pcm = pcm.reshape(-1, channels)
    return pcm, rate, channels, width

