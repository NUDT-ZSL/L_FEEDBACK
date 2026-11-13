"""Deterministic synthetic event generator used by tests and acceptance runs."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .util import atomic_write_bytes
from .wav import float_to_pcm_bytes


def _write_plain_pcm_wav(path: Path, audio: np.ndarray, sample_rate: int, bits: int) -> None:
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio[:, None]
    channels = audio.shape[1]
    scale = float(1 << (bits - 1))
    safe = 1.0 - 0.5 / scale
    peak = float(np.max(np.abs(audio), initial=0.0))
    if peak > safe:
        audio = audio * (safe / peak)
    payload, _ = float_to_pcm_bytes(audio, bits)
    block_align = channels * ((bits + 7) // 8)
    avg_bytes = sample_rate * block_align
    fmt = struct.pack(
        "<HHIIHH",
        1,
        channels,
        sample_rate,
        avg_bytes,
        block_align,
        bits,
    )
    fmt_chunk = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    data_chunk = b"data" + struct.pack("<I", len(payload)) + payload
    if len(payload) & 1:
        data_chunk += b"\x00"
    body = b"WAVE" + fmt_chunk + data_chunk
    atomic_write_bytes(path, b"RIFF" + struct.pack("<I", len(body)) + body)


def _click_track(
    *,
    sample_rate: int,
    duration_seconds: float,
    click_times: list[float],
    channels: int,
    frequency: float,
    seed: int,
) -> np.ndarray:
    frames = int(np.ceil(duration_seconds * sample_rate))
    t = np.arange(frames, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(seed)
    audio = np.empty((frames, channels), dtype=np.float64)
    for channel in range(channels):
        tone = 0.018 * np.sin(
            2.0 * np.pi * (frequency + channel * 17.0) * t + channel * 0.37
        )
        noise = 0.0015 * rng.standard_normal(frames)
        audio[:, channel] = tone + noise
    for index, click_time in enumerate(click_times):
        start = int(round(click_time * sample_rate))
        if start < 0 or start >= frames:
            continue
        length = max(8, int(round(0.004 * sample_rate)))
        local_t = np.arange(length, dtype=np.float64) / sample_rate
        envelope = np.exp(-local_t * 900.0)
        burst = 0.55 * envelope * np.sin(2.0 * np.pi * 1800.0 * local_t)
        end = min(frames, start + length)
        for channel in range(channels):
            polarity = 1.0 if (index + channel) % 2 == 0 else -1.0
            audio[start:end, channel] += polarity * burst[: end - start]
    return audio


def _timeline(
    *,
    event_id: str,
    title: str,
    anchor_specs: dict[str, tuple[float, float]],
    cue_times: list[tuple[float, str]],
) -> dict[str, Any]:
    tracks = [
        {
            "id": "main",
            "name": "Main commentary",
            "role": "main_commentary",
            "path": "audio/main_44100_16.wav",
            "anchors": [
                {
                    "source_seconds": float(source),
                    "program_seconds": float(program),
                }
                for source, program in anchor_specs["main"]
            ],
        },
        {
            "id": "sub",
            "name": "Co-commentary",
            "role": "co_commentary",
            "path": "audio/sub_48000_24.wav",
            "anchors": [
                {
                    "source_seconds": float(source),
                    "program_seconds": float(program),
                }
                for source, program in anchor_specs["sub"]
            ],
        },
        {
            "id": "ambient",
            "name": "Arena ambience",
            "role": "ambience",
            "path": "audio/ambient_48000_32.wav",
            "anchors": [
                {
                    "source_seconds": float(source),
                    "program_seconds": float(program),
                }
                for source, program in anchor_specs["ambient"]
            ],
        },
    ]
    return {
        "schema_version": "eventmix-timeline/v1",
        "event": {
            "id": event_id,
            "title": title,
            "venue": "Shanghai Arena",
            "date": "2026-11-14",
        },
        "program": {
            "sample_rate": 48000,
            "duration_seconds": 7.0,
            "bit_depth": 24,
        },
        "policy": {
            "on_track_error": "silence",
            "silence_threshold_dbfs": -75.0,
            "min_silence_seconds": 0.25,
            "max_anchor_error_ms": 0.25,
        },
        "channel_layout": [
            {
                "name": "main.L",
                "track": "main",
                "speaker": "FL",
                "sources": [0],
            },
            {
                "name": "main.R",
                "track": "main",
                "speaker": "FR",
                "sources": [1],
            },
            {
                "name": "sub.L",
                "track": "sub",
                "speaker": "BL",
                "sources": [0],
            },
            {
                "name": "sub.R",
                "track": "sub",
                "speaker": "BR",
                "sources": [0],
            },
            {
                "name": "ambient.L",
                "track": "ambient",
                "speaker": "SL",
                "sources": [0],
            },
            {
                "name": "ambient.R",
                "track": "ambient",
                "speaker": "SR",
                "sources": [1],
            },
        ],
        "tracks": tracks,
        "cues": [
            {"id": f"cue-{index + 1}", "name": name, "program_seconds": time}
            for index, (time, name) in enumerate(cue_times)
        ],
    }


def generate_demo(output_dir: Path | str) -> dict[str, Path]:
    """Create deterministic WAV inputs, two timelines, and a batch manifest."""

    root = Path(output_dir).resolve()
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    program_duration = 7.0
    mapping = {
        "main": (-0.083, 1.000025),
        "sub": (-0.041, 0.999972),
        "ambient": (-0.127, 1.000008),
    }
    input_rates = {"main": 44100, "sub": 48000, "ambient": 48000}
    input_bits = {"main": 16, "sub": 24, "ambient": 32}
    input_channels = {"main": 2, "sub": 1, "ambient": 2}
    anchor_program_times = [0.8, 3.4, 6.1]
    anchor_specs: dict[str, list[tuple[float, float]]] = {}
    source_click_times: dict[str, list[float]] = {}
    for track_id, (offset, drift) in mapping.items():
        rate = input_rates[track_id]
        source_times: list[float] = []
        pairs: list[tuple[float, float]] = []
        for program_time in anchor_program_times:
            frame = int(round(((program_time - offset) / drift) * rate))
            source_time = frame / rate
            mapped_program = offset + drift * source_time
            source_times.append(source_time)
            pairs.append((source_time, mapped_program))
        anchor_specs[track_id] = pairs
        source_click_times[track_id] = source_times

    for track_id in ("main", "sub", "ambient"):
        rate = input_rates[track_id]
        offset, drift = mapping[track_id]
        required_seconds = (program_duration - offset) / drift + 0.08
        amplitude = {"main": 440.0, "sub": 620.0, "ambient": 170.0}[track_id]
        seed = {"main": 11, "sub": 22, "ambient": 33}[track_id]
        audio = _click_track(
            sample_rate=rate,
            duration_seconds=required_seconds,
            click_times=source_click_times[track_id],
            channels=input_channels[track_id],
            frequency=amplitude,
            seed=seed,
        )
        _write_plain_pcm_wav(
            audio_dir
            / f"{track_id}_{rate}_{input_bits[track_id]}.wav",
            audio,
            rate,
            input_bits[track_id],
        )

    cue_times = [(0.45, "Draft phase"), (3.40, "Objective fight"), (6.10, "End of game")]
    timeline_a = _timeline(
        event_id="demo-event-a",
        title="Esports Final - Main feed",
        anchor_specs=anchor_specs,
        cue_times=cue_times,
    )
    timeline_b = _timeline(
        event_id="demo-event-b",
        title="Esports Final - Backup feed",
        anchor_specs=anchor_specs,
        cue_times=[(time, f"{name} B") for time, name in cue_times],
    )
    timeline_a_path = root / "timeline-a.json"
    timeline_b_path = root / "timeline-b.json"
    atomic_write_bytes(
        timeline_a_path,
        (json.dumps(timeline_a, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    atomic_write_bytes(
        timeline_b_path,
        (json.dumps(timeline_b, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    manifest = {
        "schema_version": "eventmix-batch/v1",
        "jobs": [
            {
                "id": "demo-event-a",
                "timeline": "timeline-a.json",
                "wav": "final-a.wav",
                "timeline_output": "final-a.timeline.json",
            },
            {
                "id": "demo-event-b",
                "timeline": "timeline-b.json",
                "wav": "final-b.wav",
                "timeline_output": "final-b.timeline.json",
            },
        ],
    }
    manifest_path = root / "batch.json"
    atomic_write_bytes(
        manifest_path,
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    expected = {
        "program_duration_seconds": program_duration,
        "target_sample_rate": 48000,
        "anchor_specs": {
            track_id: [list(pair) for pair in pairs]
            for track_id, pairs in anchor_specs.items()
        },
        "cue_times": [time for time, _ in cue_times],
    }
    expected_path = root / "expected.json"
    atomic_write_bytes(
        expected_path,
        (json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    return {
        "root": root,
        "audio_dir": audio_dir,
        "manifest": manifest_path,
        "timeline_a": timeline_a_path,
        "timeline_b": timeline_b_path,
        "expected": expected_path,
    }

