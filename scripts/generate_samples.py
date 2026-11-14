#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cinemix.processor import process_project, write_report
from cinemix.wavio import (
    write_pcm16_wav,
    write_pcm_wav,
)


def _dialogue(seed: int, sample_rate: int, duration: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = round(sample_rate * duration)
    time = np.arange(frames, dtype=np.float64) / sample_rate
    f0 = 118.0 + 3.0 * np.sin(2.0 * math.pi * 0.23 * time)
    phase = 2.0 * math.pi * np.cumsum(f0) / sample_rate
    voice = (
        0.58 * np.sin(phase)
        + 0.24 * np.sin(2.0 * phase + 0.31)
        + 0.12 * np.sin(3.0 * phase + 1.07)
        + 0.06 * np.sin(5.0 * phase + 0.53)
    )
    syllable_rate = 3.4
    syllables = np.maximum(
        0.0,
        np.sin(2.0 * math.pi * syllable_rate * time + 0.4),
    ) ** 1.7
    phrase = 0.22 + 0.78 * syllables
    phrase *= np.clip((time - 0.10) / 0.12, 0.0, 1.0)
    phrase *= np.clip((duration - 0.18 - time) / 0.22, 0.0, 1.0)
    breath = 0.012 * rng.standard_normal(frames)
    breath = np.convolve(breath, np.ones(32) / 32.0, mode="same")
    return np.clip((voice * phrase + breath) * 0.47, -0.95, 0.95)[:, None]


def _music(
    seed: int, sample_rate: int, duration: float, *, bright: bool
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = round(sample_rate * duration)
    time = np.arange(frames, dtype=np.float64) / sample_rate
    chord_roots = (220.0, 246.94, 196.0, 261.63)
    output = np.zeros((frames, 2), dtype=np.float64)
    for channel in range(2):
        detune = 1.0 + (channel * 2 - 1) * 0.0012
        signal = np.zeros(frames, dtype=np.float64)
        for index, root in enumerate(chord_roots):
            start = index * duration / len(chord_roots)
            local = np.clip((time - start) / 0.32, 0.0, 1.0)
            local *= np.clip((start + duration / len(chord_roots) + 0.32 - time) / 0.38, 0.0, 1.0)
            if bright:
                partials = (
                    (1.0, 1.0),
                    (2.0, 0.28),
                    (3.0, 0.18),
                    (5.0, 0.07),
                )
            else:
                partials = ((1.0, 1.0), (2.0, 0.18), (3.0, 0.08))
            for multiple, amplitude in partials:
                frequency = root * multiple * detune
                drift = 0.06 * np.sin(
                    2.0 * math.pi * (0.11 + index * 0.02) * time + channel
                )
                signal += amplitude * local * np.sin(
                    2.0 * math.pi * frequency * time + drift
                )
        signal *= 0.15 + 0.03 * np.sin(2.0 * math.pi * 0.17 * time)
        noise = rng.standard_normal(frames)
        noise = np.convolve(noise, np.ones(16) / 16.0, mode="same")
        output[:, channel] = np.clip(signal + noise * 0.0015, -0.85, 0.85)
    return output


def _effects(seed: int, sample_rate: int, duration: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = round(sample_rate * duration)
    time = np.arange(frames, dtype=np.float64) / sample_rate
    noise = rng.standard_normal(frames)
    kernel_width = max(2, int(sample_rate * 0.0025))
    kernel = np.hanning(kernel_width)
    kernel /= np.sum(kernel)
    noise = np.convolve(noise, kernel, mode="same")
    envelope = np.clip(time / 0.04, 0.0, 1.0) ** 2
    envelope *= np.exp(-np.maximum(0.0, time - 0.05) * 8.0)
    tone = np.sin(2.0 * math.pi * 390.0 * time) * np.exp(
        -np.maximum(0.0, time - 0.03) * 5.0
    )
    burst = (noise * 0.62 + tone * 0.18) * envelope
    tail_time = np.clip((time - 0.75) / max(1e-9, duration - 0.75), 0.0, 1.0)
    tail = 0.12 * np.sin(2.0 * math.pi * 760.0 * time) * np.exp(-tail_time * 2.5)
    combined = np.clip(burst + tail, -0.9, 0.9)
    return combined[:, None]


def _base_project(
    *,
    output: str,
    target_lufs: float,
    true_peak_dbtp: float,
    adaptive_threshold: float,
) -> dict:
    return {
        "version": 1,
        "project": {
            "sample_rate": 48000,
            "output": output,
            "output_channels": 2,
            "target_lufs": target_lufs,
            "true_peak_dbtp": true_peak_dbtp,
            "duration_seconds": 12.5,
        },
        "tracks": [
            {
                "id": "dialogue",
                "path": "audio/dialogue.wav",
                "start": 0.25,
                "gain_db": -1.0,
                "pan": 0.0,
                "fade_in": {"start": 0.25, "end": 0.55, "curve": "linear"},
                "fade_out": {"start": 10.35, "end": 10.80, "curve": "s_curve"},
                "resample_to": 48000,
            },
            {
                "id": "music_a",
                "path": "audio/music_a.wav",
                "start": 0.0,
                "gain_db": -4.0,
                "pan": 0.0,
            },
            {
                "id": "music_b",
                "path": "audio/music_b.wav",
                "start": 4.0,
                "gain_db": -5.0,
                "pan": 0.0,
            },
            {
                "id": "effects",
                "path": "audio/effects.wav",
                "start": 9.0,
                "gain_db": -1.0,
                "pan": 0.25,
                "resample_to": 48000,
            },
        ],
        "crossfades": [
            {
                "from": "music_a",
                "to": "music_b",
                "start": 5.0,
                "end": 7.0,
                "curve": "equal_power",
            },
            {
                "from": "music_b",
                "to": "effects",
                "start": 9.0,
                "end": 10.2,
                "curve": "s_curve",
            },
        ],
        "limiter": {
            "attack_ms": 5.0,
            "release_ms": 120.0,
            "ratio": 20.0,
            "lookahead_ms": 7.0,
            "safety_margin_db": 0.25,
        },
        "adaptive_gain": {
            "enabled": True,
            "threshold_lufs": adaptive_threshold,
            "max_cut_db": 8.0,
        },
    }


def _write_error_projects(project_dir: Path) -> None:
    errors = project_dir / "errors"
    errors.mkdir(parents=True, exist_ok=True)

    def fix_audio_paths(project: dict) -> None:
        for track in project["tracks"]:
            track["path"] = "../" + track["path"]

    missing = _base_project(
        output="out/missing.wav",
        target_lufs=-23.0,
        true_peak_dbtp=-1.0,
        adaptive_threshold=-23.0,
    )
    missing["tracks"][0]["path"] = "audio/not_present.wav"
    (errors / "missing_track.json").write_text(
        json.dumps(missing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    invalid = _base_project(
        output="out/invalid.wav",
        target_lufs=-23.0,
        true_peak_dbtp=-1.0,
        adaptive_threshold=-23.0,
    )
    fix_audio_paths(invalid)
    invalid["tracks"][0]["gain_db"] = "loud"
    (errors / "invalid_field.json").write_text(
        json.dumps(invalid, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    mismatch = _base_project(
        output="out/mismatch.wav",
        target_lufs=-23.0,
        true_peak_dbtp=-1.0,
        adaptive_threshold=-23.0,
    )
    fix_audio_paths(mismatch)
    mismatch["tracks"][0].pop("resample_to")
    (errors / "sample_rate_mismatch.json").write_text(
        json.dumps(mismatch, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    conflict = _base_project(
        output="out/conflict.wav",
        target_lufs=-23.0,
        true_peak_dbtp=-1.0,
        adaptive_threshold=-23.0,
    )
    fix_audio_paths(conflict)
    conflict["crossfades"].append(
        {
            "from": "music_a",
            "to": "effects",
            "start": 6.0,
            "end": 8.0,
            "curve": "linear",
        }
    )
    (errors / "crossfade_conflict.json").write_text(
        json.dumps(conflict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    clamped = _base_project(
        output="out/gain_clamped.wav",
        target_lufs=-23.0,
        true_peak_dbtp=-1.0,
        adaptive_threshold=-30.0,
    )
    fix_audio_paths(clamped)
    clamped["tracks"][0]["gain_db"] = 99.0
    (errors / "gain_clamped.json").write_text(
        json.dumps(clamped, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def generate(project_dir: Path, seed: int) -> dict:
    if project_dir.exists():
        shutil.rmtree(project_dir)
    audio_dir = project_dir / "audio"
    audio_dir.mkdir(parents=True)

    dialogue = _dialogue(seed + 1, 44100, 10.8)
    music_a = _music(seed + 2, 48000, 7.2, bright=False)
    music_b = _music(seed + 3, 48000, 7.2, bright=True)
    effects = _effects(seed + 4, 96000, 2.8)

    write_pcm16_wav(audio_dir / "dialogue.wav", dialogue, 44100)
    write_pcm_wav(audio_dir / "music_a.wav", music_a, 48000)
    write_pcm_wav(audio_dir / "music_b.wav", music_b, 48000)
    write_pcm_wav(audio_dir / "effects.wav", effects, 96000)

    broadcast = _base_project(
        output="out/broadcast_mix.wav",
        target_lufs=-23.0,
        true_peak_dbtp=-1.0,
        adaptive_threshold=-23.0,
    )
    hot = _base_project(
        output="out/hot_mix.wav",
        target_lufs=-14.0,
        true_peak_dbtp=-1.0,
        adaptive_threshold=-23.0,
    )
    stress = _base_project(
        output="out/stress_mix.wav",
        target_lufs=-10.2,
        true_peak_dbtp=-3.0,
        adaptive_threshold=-23.0,
    )
    (project_dir / "project.json").write_text(
        json.dumps(broadcast, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (project_dir / "project_hot.json").write_text(
        json.dumps(hot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (project_dir / "project_stress.json").write_text(
        json.dumps(stress, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_error_projects(project_dir)
    return {"broadcast": broadcast, "hot": hot, "stress": stress}


def build_reference(project_dir: Path, seed: int) -> dict:
    projects: dict[str, object] = {}
    for name, config_name in (
        ("broadcast", "project.json"),
        ("hot", "project_hot.json"),
        ("stress", "project_stress.json"),
    ):
        result = process_project(project_dir / config_name)
        report_path = project_dir / f"reference_{name}_report.json"
        write_report(report_path, result.report)
        projects[name] = {
            "config": config_name,
            "target_lufs": result.report["target"]["integrated_lufs"],
            "loudness_tolerance_lu": 0.5,
            "true_peak_limit_dbtp": result.report["target"]["true_peak_dbtp"],
            "expected_integrated_lufs": result.report["measurement"][
                "integrated_lufs"
            ],
            "expected_true_peak_dbtp": result.report["measurement"][
                "true_peak_dbtp"
            ],
            "maximum_crossfade_boundary_error_ms": result.report[
                "maximum_crossfade_boundary_error_ms"
            ],
            "output_sha256": result.report["output_sha256"],
        }
    reference = {
        "seed": seed,
        "generator": "scripts/generate_samples.py",
        "projects": projects,
        "error_projects": {
            "missing_track": "errors/missing_track.json",
            "invalid_field": "errors/invalid_field.json",
            "sample_rate_mismatch": "errors/sample_rate_mismatch.json",
            "crossfade_conflict": "errors/crossfade_conflict.json",
            "gain_clamped": "errors/gain_clamped.json",
        },
    }
    write_report(project_dir / "reference.json", reference)
    return reference


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic Cinemix test audio")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "generated" / "sample_project",
        help="Destination project directory",
    )
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument(
        "--no-process",
        action="store_true",
        help="Generate WAV/JSON files without running the processor",
    )
    args = parser.parse_args()
    project_dir = args.out.resolve()
    generate(project_dir, args.seed)
    if not args.no_process:
        build_reference(project_dir, args.seed)
    print(
        json.dumps(
            {
                "ok": True,
                "project_dir": str(project_dir),
                "processed": not args.no_process,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
