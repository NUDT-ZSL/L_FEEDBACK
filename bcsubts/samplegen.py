from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .subtitles import Cue, format_srt_time, write_srt
from .wavio import write_wav16


@dataclass
class SampleSpec:
    name: str
    sample_rate: int
    channels: int
    delay_ms: int
    subtitle_format: str
    seed: int
    duration_seconds: float = 20.0


NOMINAL_CUES = [
    (2000.0, 3600.0, "The evening bulletin begins now."),
    (3200.0, 4500.0, "Markets closed higher across the region."),
    (5200.0, 8200.0, "Weather remains clear with light winds and a cool overnight period."),
    (13400.0, 15000.0, "This is a deterministic broadcast alignment test signal."),
    (16000.0, 17200.0, "End of the local news segment."),
]


REGRESSION_SPECS = [
    SampleSpec("rate44_mono_neg", 44100, 1, -850, "srt", 44101),
    SampleSpec("rate44_stereo_pos", 44100, 2, 950, "csv", 44102),
    SampleSpec("rate48_mono_pos", 48000, 1, 250, "csv", 48001),
    SampleSpec("rate48_stereo_neg", 48000, 2, -400, "srt", 48002),
    SampleSpec("rate96_mono_pos", 96000, 1, 1500, "srt", 96001),
    SampleSpec("rate96_stereo_neg", 96000, 2, -1500, "csv", 96002),
]


def _speech_signal(spec: SampleSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    sample_rate = spec.sample_rate
    total = int(round(spec.duration_seconds * sample_rate))
    time = np.arange(total, dtype=np.float64) / sample_rate
    envelope = np.zeros(total, dtype=np.float64)
    delay = spec.delay_ms / 1000.0
    weights: list[float] = []
    for start_ms, end_ms, text in NOMINAL_CUES:
        weight = 0.65 + 0.35 * min(1.0, len(text) / 52.0)
        weights.append(weight)
        start = (start_ms / 1000.0) + delay
        end = (end_ms / 1000.0) + delay
        i0 = max(0, int(math_floor(start * sample_rate)))
        i1 = min(total, int(math_ceil(end * sample_rate)))
        if i1 <= i0:
            continue
        local_t = time[i0:i1] - start
        local_len = (i1 - i0) / sample_rate
        attack = np.clip(local_t / 0.045, 0.0, 1.0)
        release = np.clip((local_len - local_t) / 0.090, 0.0, 1.0)
        syllabic = 0.58 + 0.42 * np.sin(2 * np.pi * (4.6 + 0.4 * weight) * local_t) ** 2
        gate = np.ones_like(local_t)
        cursor = 0.0
        while cursor < local_len:
            segment = min(0.18, local_len - cursor)
            level = 0.75 if rng.random() > 0.18 else 0.25
            gate[(local_t >= cursor) & (local_t < cursor + segment)] = level
            cursor += segment
        envelope[i0:i1] = np.maximum(
            envelope[i0:i1],
            weight * attack * release * syllabic * gate,
        )

    output = np.zeros((total, spec.channels), dtype=np.float64)
    for channel in range(spec.channels):
        noise = rng.standard_normal(total)
        # A gentle deterministic coloured-noise bed.
        for _ in range(2):
            noise = 0.78 * noise + 0.22 * np.roll(noise, 1)
        tone_frequency = 185.0 + 26.0 * channel + (spec.seed % 17)
        tone = np.sin(2 * np.pi * tone_frequency * time)
        pan = 1.0 if spec.channels == 1 else (0.72 if channel == 0 else 0.92)
        output[:, channel] = pan * envelope * (0.58 * tone + 0.62 * noise)
        output[:, channel] += 0.0012 * rng.standard_normal(total)
    peak = float(np.max(np.abs(output)))
    if peak > 0:
        output *= 0.82 / peak
    return output


def math_floor(value: float) -> int:
    return int(np.floor(value))


def math_ceil(value: float) -> int:
    return int(np.ceil(value))


def _write_csv(path: Path, cues: list[Cue]) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["start", "end", "text"])
    for cue in cues:
        writer.writerow([format_srt_time(cue.start_ms), format_srt_time(cue.end_ms), cue.text])
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")


def generate_sample(output_dir: str | Path, spec: SampleSpec) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio = _speech_signal(spec)
    audio_path = output_dir / "audio.wav"
    write_wav16(audio_path, audio, spec.sample_rate)
    cues = [Cue(index, start, end, text, index) for index, (start, end, text) in enumerate(NOMINAL_CUES)]
    subtitle_path = output_dir / f"subtitles.{spec.subtitle_format}"
    if spec.subtitle_format == "srt":
        write_srt(subtitle_path, cues)
    else:
        _write_csv(subtitle_path, cues)
    events = [
        {
            "event_id": 1000 + spec.seed,
            "type": "splice_insert",
            "offset_ms": 6000,
            "duration_ms": 4500,
            "segmentation_descriptor": {
                "segmentation_type_id": 52,
                "segment_num": 1,
                "segments_expected": 1,
                "upid_type": 8,
                "upid": f"sample-{spec.name}",
            },
        },
        {
            "event_id": 2000 + spec.seed,
            "type": "splice_null",
            "offset_ms": 17250,
            "duration_ms": 0,
        },
        {
            "event_id": 3000 + spec.seed,
            "type": "splice_insert",
            "offset_ms": 19300,
            "duration_ms": 500,
        },
    ]
    (output_dir / "scte35.json").write_text(
        json.dumps({"events": events}, indent=2),
        encoding="utf-8",
    )
    ground_truth = {
        "name": spec.name,
        "seed": spec.seed,
        "sample_rate": spec.sample_rate,
        "channels": spec.channels,
        "delay_ms": spec.delay_ms,
        "duration_seconds": spec.duration_seconds,
        "subtitle_format": spec.subtitle_format,
        "overlap_cue_pair": [0, 1],
        "gap_after_index": 3,
        "scte_event_ids": [event["event_id"] for event in events],
        "spec": asdict(spec),
    }
    (output_dir / "ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2),
        encoding="utf-8",
    )
    return {
        "name": spec.name,
        "audio": str(audio_path),
        "subtitles": str(subtitle_path),
        "scte35": str(output_dir / "scte35.json"),
        "ground_truth": str(output_dir / "ground_truth.json"),
    }


def generate_samples(root: str | Path, specs: list[SampleSpec] | None = None) -> list[dict[str, object]]:
    root = Path(root)
    selected = specs or REGRESSION_SPECS
    manifest = []
    for spec in selected:
        item = generate_sample(root / spec.name, spec)
        manifest.append(item)
    (root / "samples_manifest.json").write_text(
        json.dumps({"samples": manifest}, indent=2),
        encoding="utf-8",
    )
    return manifest
