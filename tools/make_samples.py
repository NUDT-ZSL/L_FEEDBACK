"""Deterministic local sample generator for dubmux verification.

Creates, entirely offline and reproducibly (fixed seeds, fixed parameters):
  - a reference video with a synthetic "dialogue" audio track,
  - per-language dub WAVs with different leading silences and speech rates,
  - per-language subtitle timeline JSON files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

# Repo root on sys.path so `dubmux` imports work when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dubmux.audio import write_wav  # noqa: E402

SAMPLE_RATE = 48000
VIDEO_DURATION = 18.0
REFERENCE_SEGMENTS = [(1.0, 3.0), (4.0, 6.5), (8.0, 11.0), (12.5, 15.0)]

# Per-language deterministic dub characteristics.
LANGUAGES = {
    "en": {"factor": 1.00, "leading_silence": 0.30, "seed": 1000},
    "ja": {"factor": 0.82, "leading_silence": 0.90, "seed": 2000},
    "de": {"factor": 1.18, "leading_silence": 1.60, "seed": 3000},
}

DUB_GAP = 0.6  # silence between dub segments (s)


def speech_signal(duration: float, seed: int, rate: int = SAMPLE_RATE) -> np.ndarray:
    """Deterministic speech-like signal: harmonics + vibrato + envelope."""
    rng = np.random.default_rng(seed)
    n = round(duration * rate)
    t = np.arange(n) / rate
    signal = np.zeros(n)
    for _ in range(4):
        freq = 120.0 + rng.uniform(0, 400)
        amp = rng.uniform(0.1, 0.4)
        vibrato = 1.0 + 0.02 * np.sin(2 * np.pi * 5.0 * t)
        signal += amp * np.sin(2 * np.pi * freq * t * vibrato)
    # Syllable-like amplitude modulation.
    signal *= 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t) ** 2
    # Attack / decay envelope at segment edges.
    edge = min(int(0.05 * rate), n // 4)
    if edge > 0:
        signal[:edge] *= np.linspace(0.0, 1.0, edge)
        signal[-edge:] *= np.linspace(1.0, 0.0, edge)
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal *= 0.5 / peak
    return signal


def build_dub_track(
    segments: list[tuple[float, float]],
    factor: float,
    leading_silence: float,
    seed: int,
    rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Dub track: leading silence, then rate-shifted segments with gaps."""
    parts = [np.zeros(round(leading_silence * rate))]
    for i, (s, e) in enumerate(segments):
        dur = (e - s) * factor
        parts.append(speech_signal(dur, seed + i, rate))
        parts.append(np.zeros(round(DUB_GAP * rate)))
    return np.concatenate(parts)


def build_reference_audio(rate: int = SAMPLE_RATE) -> np.ndarray:
    track = np.zeros(round(VIDEO_DURATION * rate))
    for i, (s, e) in enumerate(REFERENCE_SEGMENTS):
        sig = speech_signal(e - s, seed=100 + i, rate=rate)
        offset = round(s * rate)
        track[offset: offset + len(sig)] = sig
    return track


def make_subtitles(lang: str) -> list[dict]:
    entries = []
    for i, (s, e) in enumerate(REFERENCE_SEGMENTS):
        dur = e - s
        entries.append(
            {
                "id": f"{lang}-{i}",
                "start": round(s + 0.15 * dur, 6),
                "end": round(s + 0.85 * dur, 6),
                "text": f"[{lang}] dialogue line number {i}",
                "segment": i,
            }
        )
    return entries


def generate(out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    paths: dict[str, object] = {}

    ref_wav = os.path.join(out_dir, "reference_audio.wav")
    write_wav(ref_wav, build_reference_audio(), SAMPLE_RATE)
    video = os.path.join(out_dir, "video.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi",
            "-i", f"testsrc2=duration={VIDEO_DURATION}:size=320x180:rate=24",
            "-i", ref_wav,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-shortest",
            video,
        ],
        check=True,
    )
    paths["video"] = video

    dubs, subs = {}, {}
    for lang, spec in LANGUAGES.items():
        wav_path = os.path.join(out_dir, f"dub_{lang}.wav")
        track = build_dub_track(
            REFERENCE_SEGMENTS, spec["factor"], spec["leading_silence"], spec["seed"]
        )
        write_wav(wav_path, track, SAMPLE_RATE)
        dubs[lang] = wav_path
        subs_path = os.path.join(out_dir, f"subs_{lang}.json")
        with open(subs_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(make_subtitles(lang), fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        subs[lang] = subs_path
    paths["dubs"] = dubs
    paths["subs"] = subs
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic samples.")
    parser.add_argument("--out", default="samples")
    args = parser.parse_args()
    paths = generate(args.out)
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
