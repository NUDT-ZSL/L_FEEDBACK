"""Deterministic test-project generator (no external assets required)."""

from __future__ import annotations

import json
import os

import numpy as np
import soundfile as sf

from . import TARGET_SR

DEFAULT_SEED = 20261117
PROJECT_SECONDS = 30.0

# (language label, file name, sample rate).
VOICE_SPECS = [
    ("en", "voice_en.wav", TARGET_SR),
    ("ja", "voice_ja.wav", 32000),
    ("fr", "voice_fr.wav", 16000),
]


def _tone_burst(rng: np.random.Generator, seconds: float, sample_rate: int,
                base_freq: float, amplitude: float = 0.55) -> np.ndarray:
    """Speech-like signal: AM/FM modulated tone plus low-level noise."""
    length = int(round(seconds * sample_rate))
    t = np.arange(length) / sample_rate
    am = 0.55 + 0.45 * np.sin(2.0 * np.pi * 3.5 * t + rng.uniform(0, 6.28))
    vibrato = 6.0 * np.sin(2.0 * np.pi * 5.0 * t)
    carrier = np.sin(2.0 * np.pi * (base_freq * t + vibrato * t / (2.0 * np.pi)))
    noise = rng.normal(0.0, 0.02, size=length)
    fade = min(length, 200)
    env = np.ones(length)
    if fade > 1:
        env[:fade] = np.linspace(0.0, 1.0, fade)
        env[-fade:] = np.minimum(env[-fade:], np.linspace(1.0, 0.0, fade))
    return amplitude * env * am * carrier + noise


def _write_wav(path: str, signal: np.ndarray, sample_rate: int,
               subtype: str = "PCM_16") -> None:
    signal = np.clip(signal, -1.0, 1.0)
    sf.write(path, signal.astype(np.float32), sample_rate, subtype=subtype)


def generate_project(output_dir: str, seed: int = DEFAULT_SEED) -> dict:
    """Create voices, music, SFX and a timeline JSON; return project metadata."""
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    voice_paths: dict[str, str] = {}
    for language, file_name, sample_rate in VOICE_SPECS:
        # Each language voice file is a long concatenation of several bursts;
        # timeline segments reference sub-ranges of these files.
        bursts = [
            _tone_burst(rng, 2.2, sample_rate, rng.uniform(180.0, 320.0)),
            _tone_burst(rng, 3.0, sample_rate, rng.uniform(180.0, 320.0)),
            _tone_burst(rng, 1.4, sample_rate, rng.uniform(180.0, 320.0)),
            _tone_burst(rng, 4.5, sample_rate, rng.uniform(180.0, 320.0)),
            _tone_burst(rng, 2.6, sample_rate, rng.uniform(180.0, 320.0)),
            np.zeros(int(round(1.0 * sample_rate))),  # silent section
            _tone_burst(rng, 3.5, sample_rate, rng.uniform(180.0, 320.0)),
            _tone_burst(rng, 5.0, sample_rate, rng.uniform(180.0, 320.0)),
        ]
        voice = np.concatenate(bursts)
        path = os.path.join(audio_dir, file_name)
        _write_wav(path, voice, sample_rate)
        voice_paths[language] = path

    _generate_music(audio_dir, rng)
    _generate_sfx(audio_dir, rng)
    clipped_path = _generate_clipped_voice(audio_dir, rng)

    timeline = _build_timeline(voice_paths, clipped_path)
    timeline_path = os.path.join(output_dir, "timeline.json")
    with open(timeline_path, "w", encoding="utf-8") as handle:
        json.dump(timeline, handle, indent=2, ensure_ascii=False)

    return {
        "output_dir": os.path.abspath(output_dir),
        "timeline": os.path.abspath(timeline_path),
        "seed": seed,
        "segments": len(timeline["segments"]),
        "languages": [spec[0] for spec in VOICE_SPECS],
    }


def _generate_music(audio_dir: str, rng: np.random.Generator) -> None:
    length = int(PROJECT_SECONDS * TARGET_SR)
    t = np.arange(length) / TARGET_SR
    chord = sum(np.sin(2.0 * np.pi * f * t) for f in (220.0, 277.18, 329.63)) / 3.0
    tremolo = 0.85 + 0.15 * np.sin(2.0 * np.pi * 0.3 * t)
    music = 0.32 * tremolo * chord + rng.normal(0.0, 0.004, size=length)
    _write_wav(os.path.join(audio_dir, "music.wav"), music, TARGET_SR)


def _generate_sfx(audio_dir: str, rng: np.random.Generator) -> None:
    length = int(PROJECT_SECONDS * TARGET_SR)
    sfx = np.zeros(length)
    # Periodic soft "blip" sound effects, deliberately not ducked.
    for onset in np.arange(2.0, PROJECT_SECONDS, 5.0):
        start = int(onset * TARGET_SR)
        blip_len = int(0.35 * TARGET_SR)
        tt = np.arange(blip_len) / TARGET_SR
        blip = 0.25 * np.sin(2.0 * np.pi * 880.0 * tt) * np.exp(-6.0 * tt)
        sfx[start : start + blip_len] += blip[: max(0, min(blip_len, length - start))]
    sfx += rng.normal(0.0, 0.002, size=length)
    _write_wav(os.path.join(audio_dir, "sfx.wav"), sfx, TARGET_SR)


def _generate_clipped_voice(audio_dir: str, rng: np.random.Generator) -> str:
    """A deliberately hot/clipped line that must engage the global limiter."""
    length = int(2.0 * TARGET_SR)
    t = np.arange(length) / TARGET_SR
    hot = 1.35 * np.sin(2.0 * np.pi * 240.0 * t)
    hot += rng.normal(0.0, 0.03, size=length)
    clipped = np.clip(hot, -1.0, 1.0)  # hard-clipped source samples
    path = os.path.join(audio_dir, "voice_clipped.wav")
    _write_wav(path, clipped, TARGET_SR)
    return path


def _build_timeline(voice_paths: dict[str, str], clipped_path: str) -> dict:
    def rel(path: str) -> str:
        return os.path.relpath(path, os.path.dirname(os.path.dirname(path)))

    # Voice file layout per language (seconds):
    # bursts 2.2 | 3.0 | 1.4 | 4.5 | 2.6 | silence 1.0 | 3.5 | 5.0
    # offsets 0.0  2.2   5.2   6.6  11.1  13.7         14.7  18.2
    segments = [
        # index, timeline start/end, language, source range, target LUFS.
        # Ratio 1.11 (2.0 s source -> 1.8 s target), within [0.8, 1.25].
        _seg(0, 1.0, 2.8, "en", (0.0, 2.0), -23.0),
        _seg(1, 3.2, 6.0, "ja", (0.0, 2.8), -24.0),
        # Timeline overlap with segment 1 (4.5 s < 6.0 s).
        # Ratio 0.91 (2.0 s source -> 2.2 s target, slow-down).
        _seg(2, 4.5, 6.7, "fr", (0.0, 2.0), -22.0),
        # Ratio 1.2 (1.2 s source -> 1.0 s target, acceleration).
        _seg(3, 7.0, 8.0, "en", (2.4, 3.6), -23.0),
        # Zero-length timeline segment: validation passes, alignment reports error.
        _seg(4, 8.5, 8.5, "ja", (3.0, 3.0), -24.0),
        # Source range covers the 1 s silent section (13.7-14.7).
        _seg(5, 9.0, 9.8, "fr", (13.8, 14.6), -23.0),
        _seg(6, 10.5, 14.5, "en", (6.8, 10.8), -24.0),
        # Ratio 0.2 (1 s source -> 5 s target): outside [0.8, 1.25].
        _seg(7, 15.5, 20.5, "ja", (8.4, 9.4), -23.0),
        # Mild acceleration: 2.5 s source -> 2.0 s target, ratio 1.25 (boundary).
        _seg(8, 21.0, 23.0, "fr", (14.8, 17.3), -22.0),
        # Hot/clipped line normalized to a loud target -> limiter engagement.
        _seg(9, 24.0, 26.0, "clipped", (0.0, 2.0), -5.0),
    ]
    for segment in segments:
        language = segment.pop("_lang")
        segment["language"] = language
        if language == "clipped":
            segment["file"] = os.path.join("audio", os.path.basename(clipped_path))
        else:
            segment["file"] = _resolve_voice(voice_paths, language)

    return {
        "project": {
            "name": "deterministic_demo",
            "sample_rate": TARGET_SR,
            "duration_seconds": PROJECT_SECONDS,
        },
        "reference_tracks": {
            "music": "audio/music.wav",
            "sfx": ["audio/sfx.wav"],
        },
        "segments": segments,
    }


def _seg(index, start, end, language, source_range, target_lufs):
    return {
        "index": index,
        "start": start,
        "end": end,
        "_lang": language,
        "source_range": list(source_range),
        "target_lufs": target_lufs,
    }


def _resolve_voice(voice_paths: dict[str, str], language: str) -> str:
    path = voice_paths[language]
    return os.path.join("audio", os.path.basename(path))
