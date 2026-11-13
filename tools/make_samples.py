from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


TARGET_RATE = 48000


def _tone(
    sample_rate: int,
    duration_ms: int,
    frequency: float,
    *,
    channels: int,
    amplitude: float,
    phase: float = 0.0,
) -> np.ndarray:
    frames = int(round(duration_ms * sample_rate / 1000.0))
    time = np.arange(frames, dtype=np.float64) / sample_rate
    base = amplitude * np.sin(2.0 * math.pi * frequency * time + phase)
    envelope_frames = min(frames // 2, max(1, int(sample_rate * 0.005)))
    envelope = np.ones(frames, dtype=np.float64)
    if envelope_frames > 1:
        ramp = np.linspace(0.0, 1.0, envelope_frames, endpoint=False)
        envelope[:envelope_frames] *= ramp
        envelope[-envelope_frames:] *= ramp[::-1]
    base *= envelope
    if channels == 1:
        return base[:, None]
    return np.column_stack((base, 0.82 * base))


def _write_clip(
    audio_dir: Path,
    name: str,
    sample_rate: int,
    duration_ms: int,
    frequency: float,
    *,
    channels: int,
    subtype: str,
    amplitude: float,
) -> str:
    path = audio_dir / name
    data = _tone(
        sample_rate,
        duration_ms,
        frequency,
        channels=channels,
        amplitude=amplitude,
    )
    sf.write(
        path,
        data,
        sample_rate,
        subtype=subtype,
        format="WAV",
        endian="FILE",
    )
    return f"audio/{name}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate(root: Path) -> None:
    root = root.resolve()
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    clips = {
        "en_01.wav": _write_clip(
            audio_dir, "en_01.wav", 48000, 800, 440.0,
            channels=2, subtype="PCM_16", amplitude=0.55,
        ),
        "en_02.wav": _write_clip(
            audio_dir, "en_02.wav", 48000, 905, 523.25,
            channels=1, subtype="FLOAT", amplitude=0.42,
        ),
        "en_03.wav": _write_clip(
            audio_dir, "en_03.wav", 44100, 795, 659.25,
            channels=2, subtype="PCM_24", amplitude=0.48,
        ),
        "fr_01.wav": _write_clip(
            audio_dir, "fr_01.wav", 44100, 700, 392.0,
            channels=2, subtype="PCM_24", amplitude=0.5,
        ),
        "fr_02.wav": _write_clip(
            audio_dir, "fr_02.wav", 48000, 800, 349.23,
            channels=1, subtype="PCM_16", amplitude=0.52,
        ),
        "fr_03.wav": _write_clip(
            audio_dir, "fr_03.wav", 22050, 400, 466.16,
            channels=2, subtype="PCM_16", amplitude=0.46,
        ),
        "ja_01.wav": _write_clip(
            audio_dir, "ja_01.wav", 16000, 500, 293.66,
            channels=1, subtype="PCM_16", amplitude=0.5,
        ),
        "ja_02.wav": _write_clip(
            audio_dir, "ja_02.wav", 48000, 700, 329.63,
            channels=2, subtype="PCM_16", amplitude=0.47,
        ),
        "ja_03.wav": _write_clip(
            audio_dir, "ja_03.wav", 48000, 600, 220.0,
            channels=1, subtype="PCM_24", amplitude=0.54,
        ),
    }

    normal = {
        "version": 1,
        "timeline": {"duration_ms": 2600},
        "target": {"sample_rate": 48000, "channels": 2, "bit_depth": 16},
        "output": {"directory": "out", "name": "acceptance"},
        "languages": [
            {
                "code": "en",
                "name": "English",
                "clips": [
                    {
                        "id": "en-001",
                        "start_ms": 0,
                        "duration_ms": 800,
                        "path": clips["en_01.wav"],
                        "text": "Engine ready.\nPress & continue <now>.",
                    },
                    {
                        "id": "en-002",
                        "start_ms": 795,
                        "duration_ms": 905,
                        "path": clips["en_02.wav"],
                        "text": "Two overlapping lines --> aligned.",
                    },
                    {
                        "id": "en-003",
                        "start_ms": 1705,
                        "duration_ms": 795,
                        "path": clips["en_03.wav"],
                        "text": "Line three\r\nwith CRLF & 中文。",
                    },
                ],
            },
            {
                "code": "fr",
                "name": "Français",
                "clips": [
                    {
                        "id": "fr-001",
                        "start_ms": 200,
                        "duration_ms": 700,
                        "path": clips["fr_01.wav"],
                        "text": "Bonjour,\n\n<salut> & bienvenue.",
                    },
                    {
                        "id": "fr-002",
                        "start_ms": 1200,
                        "duration_ms": 800,
                        "path": clips["fr_02.wav"],
                        "text": "Deuxième ligne.",
                    },
                    {
                        "id": "fr-003",
                        "start_ms": 2050,
                        "duration_ms": 400,
                        "path": clips["fr_03.wav"],
                        "text": "Fin\u2028de la scène.",
                    },
                ],
            },
            {
                "code": "ja",
                "name": "日本語",
                "clips": [
                    {
                        "id": "ja-001",
                        "start_ms": 100,
                        "duration_ms": 500,
                        "path": clips["ja_01.wav"],
                        "text": "準備完了。",
                    },
                    {
                        "id": "ja-002",
                        "start_ms": 605,
                        "duration_ms": 700,
                        "path": clips["ja_02.wav"],
                        "text": "続行します & 待機。",
                    },
                    {
                        "id": "ja-003",
                        "start_ms": 1600,
                        "duration_ms": 600,
                        "path": clips["ja_03.wav"],
                        "text": "三行目の字幕。",
                    },
                ],
            },
        ],
    }
    _write_json(root / "manifest.json", normal)

    media_failure = {
        "timeline": {"duration_ms": 1500},
        "target": {"sample_rate": 48000, "channels": 2, "bit_depth": 16},
        "output": {"directory": "out-media", "name": "media"},
        "languages": [
            {
                "code": "en",
                "clips": [
                    {
                        "start_ms": 0,
                        "duration_ms": 800,
                        "path": clips["en_01.wav"],
                        "text": "This track succeeds.",
                    }
                ],
            },
            {
                "code": "de",
                "clips": [
                    {
                        "start_ms": 100,
                        "duration_ms": 800,
                        "path": "audio/missing-de.wav",
                        "text": "Diese Datei fehlt.",
                    }
                ],
            },
        ],
    }
    _write_json(root / "manifest-media-error.json", media_failure)

    timeline_failure = {
        "timeline": {"duration_ms": 1200},
        "target": {"sample_rate": 48000, "channels": 2, "bit_depth": 16},
        "output": {"directory": "out-timeline", "name": "bad"},
        "languages": [
            {
                "code": "en",
                "clips": [
                    {
                        "start_ms": 0,
                        "duration_ms": 700,
                        "path": clips["en_01.wav"],
                        "text": "First.",
                    },
                    {
                        "start_ms": 689,
                        "duration_ms": 500,
                        "path": clips["en_02.wav"],
                        "text": "Illegal overlap.",
                    },
                ],
            }
        ],
    }
    _write_json(root / "manifest-timeline-error.json", timeline_failure)

    interrupt_clips = []
    interrupt_languages = [
        ("en", "eng"), ("fr", "fre"), ("ja", "jpn"), ("de", "ger"),
        ("es", "spa"), ("it", "ita"), ("pt", "por"), ("ko", "kor"),
    ]
    for language_index, (code, _) in enumerate(interrupt_languages):
        pieces = []
        for clip_index in range(16):
            name = f"interrupt-{code}-{clip_index:02d}.wav"
            path = _write_clip(
                audio_dir,
                name,
                48000,
                900,
                180.0 + language_index * 23.0 + clip_index * 7.0,
                channels=2,
                subtype="PCM_16",
                amplitude=0.35,
            )
            pieces.append(
                {
                    "start_ms": clip_index * 1000,
                    "duration_ms": 1000,
                    "path": path,
                    "text": f"{code} line {clip_index + 1}",
                }
            )
        interrupt_languages[language_index] = (
            code,
            {
                "code": code,
                "clips": pieces,
            },
        )
    interrupt_manifest = {
        "timeline": {"duration_ms": 16000},
        "target": {"sample_rate": 48000, "channels": 2, "bit_depth": 16},
        "output": {"directory": "out-interrupt", "name": "interrupt"},
        "languages": [entry for _, entry in interrupt_languages],
    }
    _write_json(root / "manifest-interrupt.json", interrupt_manifest)

    print(
        json.dumps(
            {
                "root": str(root),
                "normal_manifest": str(root / "manifest.json"),
                "media_error_manifest": str(root / "manifest-media-error.json"),
                "timeline_error_manifest": str(root / "manifest-timeline-error.json"),
                "interrupt_manifest": str(root / "manifest-interrupt.json"),
            },
            ensure_ascii=False,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Create deterministic acceptance WAVs")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    generate(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
