#!/usr/bin/env python3
"""Generate deterministic 720p/30fps cut-scene and dialogue samples."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


SAMPLE_RATE = 48_000
VIDEO_DURATION = 10.0
VIDEO_FPS = 30


def _tone(
    *,
    sample_rate: int,
    channels: int,
    duration: float,
    frequency: float,
    seed: int,
) -> np.ndarray:
    count = int(round(duration * sample_rate))
    time = np.arange(count, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(seed)
    base = (
        0.34 * np.sin(2.0 * math.pi * frequency * time)
        + 0.10 * np.sin(2.0 * math.pi * frequency * 2.01 * time)
        + 0.015 * rng.standard_normal(count)
    )
    fade_count = max(1, int(round(0.005 * sample_rate)))
    ramp = np.sin(
        np.linspace(0.0, math.pi / 2.0, fade_count, endpoint=True)
    )
    base[:fade_count] *= ramp
    base[-fade_count:] *= ramp[::-1]
    if channels == 1:
        return base[:, np.newaxis]
    right = np.roll(base, max(1, int(round(0.0007 * sample_rate))))
    return np.column_stack([base, 0.91 * right])


def _write_with_padding(
    path: Path,
    audio: np.ndarray,
    sample_rate: int,
    *,
    padding: float = 0.12,
) -> None:
    padding_count = int(round(padding * sample_rate))
    padded = np.concatenate(
        (
            np.zeros((padding_count, audio.shape[1])),
            audio,
            np.zeros((padding_count, audio.shape[1])),
        ),
        axis=0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        np.clip(padded, -0.95, 0.95),
        sample_rate,
        subtype="PCM_16",
        format="WAV",
    )


def _make_source_dialogue(path: Path) -> None:
    count = int(round(VIDEO_DURATION * SAMPLE_RATE))
    audio = np.zeros((count, 2), dtype=np.float64)
    source_segments = [
        (0.5, 2.0, 220.0),
        (2.8, 4.2, 330.0),
        (4.5, 6.0, 440.0),
        (7.3, 8.5, 550.0),
    ]
    for index, (start, end, frequency) in enumerate(source_segments):
        tone = _tone(
            sample_rate=SAMPLE_RATE,
            channels=2,
            duration=end - start,
            frequency=frequency,
            seed=100 + index,
        )
        start_sample = int(round(start * SAMPLE_RATE))
        audio[start_sample : start_sample + tone.shape[0]] += tone
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        np.clip(audio, -0.95, 0.95),
        SAMPLE_RATE,
        subtype="PCM_16",
        format="WAV",
    )


def _make_video(path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to create the demo video track")
    command = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        (
            f"color=c=0x101820:s=1280x720:r={VIDEO_FPS}:"
            f"d={VIDEO_DURATION}"
        ),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "35",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        str(path),
    ]
    subprocess.run(command, check=True)


def _target_specs(language: str) -> list[dict[str, object]]:
    if language == "zh":
        return [
            {
                "source": (0.5, 2.0),
                "target": (0.5, 2.0),
                "duration": 1.20,
                "rate": 44_100,
                "channels": 1,
                "frequency": 196.0,
                "text": "欢迎来到本地化测试场景。",
            },
            {
                "source": (2.8, 4.2),
                "target": (2.8, 3.8),
                "duration": 1.55,
                "rate": 48_000,
                "channels": 2,
                "frequency": 262.0,
                "text": "这段对白比原时间槽更长。",
            },
            {
                "source": (4.5, 6.0),
                "target": (3.75, 5.30),
                "duration": 1.30,
                "rate": 48_000,
                "channels": 2,
                "frequency": 330.0,
                "text": "时间轴会自动顺延并做交叉淡化。",
            },
            {
                "source": (7.3, 8.5),
                "target": (8.70, 9.70),
                "duration": 0.90,
                "rate": 48_000,
                "channels": 2,
                "frequency": 392.0,
                "text": "末尾片段会在视频结束处截断。",
            },
        ]
    if language == "en":
        return [
            {
                "source": (0.5, 2.0),
                "target": (0.6, 2.1),
                "duration": 1.15,
                "rate": 48_000,
                "channels": 2,
                "frequency": 210.0,
                "text": "Welcome to the localization test scene.",
            },
            {
                "source": (2.8, 4.2),
                "target": (2.9, 3.9),
                "duration": 1.45,
                "rate": 44_100,
                "channels": 2,
                "frequency": 280.0,
                "text": "This line is longer than its original slot.",
            },
            {
                "source": (4.5, 6.0),
                "target": (3.95, 5.50),
                "duration": 1.20,
                "rate": 48_000,
                "channels": 2,
                "frequency": 350.0,
                "text": "The timeline shifts and crossfades safely.",
            },
            {
                "source": (7.3, 8.5),
                "target": (8.65, 9.65),
                "duration": 0.80,
                "rate": 48_000,
                "channels": 2,
                "frequency": 420.0,
                "text": "The final line is clipped at video end.",
            },
        ]
    raise ValueError(f"unsupported demo language: {language}")


def generate(out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "video_720p30.mp4"
    source_path = out_dir / "source_dialogue_48k_stereo.wav"
    _make_video(video_path)
    _make_source_dialogue(source_path)

    segments: list[dict[str, object]] = []
    languages = ("zh", "en")
    all_specs = {language: _target_specs(language) for language in languages}
    for index, zh_spec in enumerate(all_specs["zh"], start=1):
        source_start, source_end = zh_spec["source"]  # type: ignore[misc]
        targets: dict[str, object] = {}
        for language_index, language in enumerate(languages):
            spec = all_specs[language][index - 1]
            rate = int(spec["rate"])  # type: ignore[arg-type]
            channels = int(spec["channels"])  # type: ignore[arg-type]
            target_start, target_end = spec["target"]  # type: ignore[misc]
            tone = _tone(
                sample_rate=rate,
                channels=channels,
                duration=float(spec["duration"]),  # type: ignore[arg-type]
                frequency=float(spec["frequency"]),  # type: ignore[arg-type]
                seed=1000 + language_index * 100 + index,
            )
            relative_path = (
                Path("targets")
                / language
                / f"segment_{index:02d}.wav"
            )
            _write_with_padding(out_dir / relative_path, tone, rate)
            targets[language] = {
                "path": relative_path.as_posix(),
                "start": target_start,
                "end": target_end,
                "sample_rate": rate,
                "channels": channels,
            }
        segments.append(
            {
                "id": f"scene01_line{index:02d}",
                "start": source_start,
                "end": source_end,
                "text": {
                    language: all_specs[language][index - 1]["text"]
                    for language in languages
                },
                "targets": targets,
            }
        )

    manifest = {
        "schema_version": 1,
        "video": {
            "path": video_path.name,
            "duration": VIDEO_DURATION,
            "fps": VIDEO_FPS,
            "width": 1280,
            "height": 720,
        },
        "dialogue_track": {
            "path": source_path.name,
            "sample_rate": SAMPLE_RATE,
            "channels": 2,
            "validate_targets": True,
        },
        "segments": segments,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": str(manifest_path.resolve()),
        "video": str(video_path.resolve()),
        "source_dialogue": str(source_path.resolve()),
        "languages": list(languages),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="artifacts/demo",
        help="output directory for deterministic sample assets",
    )
    args = parser.parse_args()
    summary = generate(Path(args.out))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
