#!/usr/bin/env python3
"""Create deterministic dialogue, music, and SFX WAV fixtures.

The generated files intentionally contain steady noise, a short synthetic room
tail, level modulation, and dialogue pauses. They require no network access.
"""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 48_000
DURATION_S = 6.4
ACTIVE_INTERVALS = ((0.55, 1.75), (2.45, 3.65), (4.35, 5.55))


def _smooth_activity(times: np.ndarray, intervals: tuple[tuple[float, float], ...]) -> np.ndarray:
    mask = np.zeros_like(times)
    ramp = 0.008
    for start, end in intervals:
        rise = np.clip((times - start) / ramp, 0.0, 1.0)
        fall = np.clip((end - times) / ramp, 0.0, 1.0)
        window = 0.5 - 0.5 * np.cos(np.pi * rise)
        window *= 0.5 - 0.5 * np.cos(np.pi * fall)
        mask = np.maximum(mask, window)
    return mask


def _speech_like(times: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f0 = 132.0 + 13.0 * np.sin(2.0 * math.pi * 0.71 * times)
    phase = 2.0 * math.pi * np.cumsum(f0) / SAMPLE_RATE
    voiced = (
        0.55 * np.sin(phase)
        + 0.24 * np.sin(2.02 * phase + 0.3)
        + 0.13 * np.sin(3.07 * phase + 1.1)
        + 0.07 * np.sin(4.91 * phase + 0.4)
    )
    syllable = 0.52 + 0.48 * np.sin(
        2.0 * math.pi * 3.4 * times + 0.4 * np.sin(2.0 * math.pi * 0.9 * times)
    )
    breath = rng.normal(0.0, 1.0, times.size)
    # A short deterministic smoothing filter for a noise-like consonant bed.
    kernel = np.ones(24, dtype=np.float64) / 24.0
    breath = np.convolve(breath, kernel, mode="same")
    return voiced * syllable + 0.18 * breath


def _room_tail(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = signal.copy()
    delay_samples = [int(0.017 * SAMPLE_RATE), int(0.043 * SAMPLE_RATE), int(0.091 * SAMPLE_RATE)]
    gains = [0.20, 0.12, 0.07]
    for delay, gain in zip(delay_samples, gains):
        if delay < output.size:
            output[delay:] += gain * signal[:-delay]
    diffuse = rng.normal(0.0, 1.0, signal.size)
    kernel = np.ones(97, dtype=np.float64) / 97.0
    diffuse = np.convolve(diffuse, kernel, mode="same")
    output += 0.012 * diffuse
    return output


def synthesize_audio(
    output_dir: str | Path,
    *,
    seed: int = 20261117,
    duration_s: float = DURATION_S,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    samples = int(round(duration_s * sample_rate))
    times = np.arange(samples, dtype=np.float64) / sample_rate

    activity = _smooth_activity(times, ACTIVE_INTERVALS)
    level_mod = 0.72 + 0.18 * np.sin(2.0 * math.pi * 0.23 * times + 0.7)
    speech = _speech_like(times, rng) * activity * level_mod
    speech = _room_tail(speech, rng)
    speech /= max(1e-12, np.max(np.abs(speech)))
    speech *= 0.32

    noise_floor = 10.0 ** (-58.0 / 20.0)
    dialogue_left = speech + rng.normal(0.0, noise_floor, samples)
    dialogue_right = np.roll(speech, int(0.0007 * sample_rate))
    dialogue_right[: int(0.0007 * sample_rate)] = 0.0
    dialogue_right *= 0.94
    dialogue_right += rng.normal(0.0, noise_floor, samples)
    dialogue = np.stack((dialogue_left, dialogue_right), axis=1)

    chord = (
        0.45 * np.sin(2.0 * math.pi * 110.0 * times)
        + 0.28 * np.sin(2.0 * math.pi * 164.81 * times + 0.5)
        + 0.20 * np.sin(2.0 * math.pi * 220.0 * times + 1.2)
    )
    pad_lfo = 0.72 + 0.28 * np.sin(2.0 * math.pi * 0.11 * times)
    pulse = np.zeros_like(times)
    for beat in np.arange(0.2, duration_s, 0.8):
        local = times - beat
        envelope = np.where(local >= 0.0, np.exp(-local * 9.0), 0.0)
        pulse += envelope * np.sin(2.0 * math.pi * 82.0 * local)
    music_left = (0.14 * chord * pad_lfo + 0.08 * pulse)
    music_right = (0.14 * np.roll(chord, 240) * pad_lfo + 0.075 * pulse)
    music = np.stack((music_left, music_right), axis=1)

    sfx_left = np.zeros(samples, dtype=np.float64)
    sfx_right = np.zeros(samples, dtype=np.float64)
    for onset, frequency in ((0.35, 1450.0), (2.15, 920.0), (4.05, 1800.0)):
        start = int(onset * sample_rate)
        length = int(0.22 * sample_rate)
        local_t = np.arange(length) / sample_rate
        envelope = np.exp(-local_t * 20.0)
        tone = np.sin(2.0 * math.pi * frequency * local_t + 0.2)
        hit = 0.22 * envelope * tone
        end = min(samples, start + length)
        sfx_left[start:end] += hit[: end - start]
        sfx_right[start:end] += 0.82 * np.roll(hit, 36)[: end - start]

    files = {
        "dialogue": output / "dialogue_stereo.wav",
        "music": output / "music_stereo.wav",
        "sfx": output / "sfx_stereo.wav",
        "mono": output / "room_mono.wav",
    }
    write_wav24(files["dialogue"], dialogue, sample_rate)
    write_wav24(files["music"], music, sample_rate)
    write_wav24(files["sfx"], np.stack((sfx_left, sfx_right), axis=1), sample_rate)
    write_wav24(files["mono"], dialogue.mean(axis=1, keepdims=True) * 0.7, sample_rate)
    return files


def write_demo_manifest(
    output_dir: str | Path,
    files: dict[str, Path],
    *,
    segments: bool = True,
    output_layout: str = "stereo",
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    output_resolved = output.resolve()

    def track_reference(key: str) -> str:
        file_path = Path(files[key])
        try:
            return str(file_path.resolve().relative_to(output_resolved))
        except ValueError:
            return str(file_path)

    if output_layout == "stereo":
        output_block = {"channels": 2, "channel_layout": "stereo"}
    else:
        output_block = {"channels": 6, "channel_layout": "5.1"}

    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "sample_rate": 48_000,
        "output": output_block,
        "target_lufs": -18.0,
        "noise_gate": {
            "window_ms": 20.0,
            "hop_ms": 10.0,
            "threshold_db": -45.0,
            "attenuation_db": 18.0,
            "snr_margin_db": 6.0,
            "noise_estimate_window_ms": 500.0,
            "noise_percentile": 20.0,
            "attack_ms": 4.0,
            "release_ms": 35.0,
        },
        "loudness": {
            "window_ms": 400.0,
            "tolerance_lu": 1.0,
            "attack_ms": 35.0,
            "release_ms": 220.0,
            "max_gain_db": 30.0,
        },
        "ducking": {
            "depth_db": 9.0,
            "attack_ms": 5.0,
            "release_ms": 300.0,
            "vad_threshold_db": -40.0,
            "vad_snr_db": 6.0,
            "vad_window_ms": 10.0,
            "min_gap_ms": 70.0,
        },
        "tracks": [
            {
                "id": "dialog",
                "path": track_reference("dialogue"),
                "role": "dialogue",
                "input_channels": 2,
                "target_lufs": -18.0,
            },
            {
                "id": "music",
                "path": track_reference("music"),
                "role": "music",
                "input_channels": 2,
                "gain_db": -1.0,
                "ducking": {"depth_db": 9.0, "release_ms": 300.0},
            },
            {
                "id": "effects",
                "path": track_reference("sfx"),
                "role": "sfx",
                "input_channels": 2,
                "gain_db": -2.0,
                "ducking": {"depth_db": 7.0, "release_ms": 250.0},
            },
        ],
    }
    if segments:
        manifest["segments"] = [
            {"id": "part_01", "start_s": 0.0, "end_s": 2.0},
            {"id": "part_02", "start_s": 2.0, "end_s": 4.0},
            {"id": "part_03", "start_s": 4.0, "end_s": 6.4},
        ]
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def write_wav24(path: str | Path, signal: np.ndarray, sample_rate: int) -> None:
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("signal must be 1-D or 2-D")
    if not np.all(np.isfinite(array)):
        raise ValueError("signal contains non-finite values")
    clipped = np.clip(array, -1.0, 1.0)
    quantized = np.round(clipped * 8_388_607.0).astype("<i4")
    interleaved = np.ascontiguousarray(quantized).view(np.uint8).reshape(-1, 4)[:, :3]
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(array.shape[1])
        wav.setsampwidth(3)
        wav.setframerate(sample_rate)
        wav.writeframes(interleaved.tobytes())


def synthesize_demo(
    output_dir: str | Path,
    *,
    seed: int = 20261117,
    segments: bool = True,
    output_layout: str = "stereo",
) -> tuple[dict[str, Path], Path]:
    output = Path(output_dir)
    files = synthesize_audio(output, seed=seed)
    manifest = write_demo_manifest(
        output,
        files,
        segments=segments,
        output_layout=output_layout,
    )
    return files, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic mixer fixtures")
    parser.add_argument("output_dir", help="directory for WAV fixtures and manifest.json")
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--no-segments", action="store_true")
    parser.add_argument(
        "--layout",
        choices=("stereo", "5.1"),
        default="stereo",
    )
    args = parser.parse_args()
    _, manifest = synthesize_demo(
        args.output_dir,
        seed=args.seed,
        segments=not args.no_segments,
        output_layout=args.layout,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
