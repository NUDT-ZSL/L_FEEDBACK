"""Deterministic sample-material and chart generator.

The generator creates six layer WAV files that deliberately mix the
supported container variants (44.1 kHz / 48 kHz, mono / stereo, 16-bit /
24-bit PCM) plus a complete chart JSON.  Every source starts with a
recognizable amount of lead-in silence followed by an onset marker tone, so
the build pipeline can be verified for sample-accurate alignment.

All randomness is driven by :class:`numpy.random.RandomState` seeded from a
fixed integer per source, and all math uses explicit dtypes.  Re-running the
generator with the same arguments therefore produces byte-identical files.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .waveio import write_wav

SAMPLE_RATE_INTERNAL = 48000


@dataclass(frozen=True)
class SourceSpec:
    """Static description of one deterministic generated layer source."""

    layer_id: str
    file_name: str
    sample_rate: int
    channels: int
    sample_width: int
    seed: int
    base_freq: float
    lead_in_seconds: float
    duration_seconds: float
    harmonic: int

    @property
    def onset_frame(self) -> int:
        return int(round(self.lead_in_seconds * self.sample_rate))


# The table intentionally covers every required format combination.
SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("drums", "drums_src.wav", 44100, 1, 16,
               seed=101, base_freq=62.0, lead_in_seconds=0.120,
               duration_seconds=1.600, harmonic=1),
    SourceSpec("bass", "bass_src.wav", 48000, 2, 24,
               seed=202, base_freq=82.41, lead_in_seconds=0.080,
               duration_seconds=1.800, harmonic=2),
    SourceSpec("melody", "melody_src.wav", 48000, 1, 24,
               seed=303, base_freq=392.0, lead_in_seconds=0.150,
               duration_seconds=1.500, harmonic=3),
    SourceSpec("vocal", "vocal_src.wav", 44100, 2, 16,
               seed=404, base_freq=261.63, lead_in_seconds=0.200,
               duration_seconds=2.000, harmonic=2),
    SourceSpec("pad", "pad_src.wav", 48000, 2, 16,
               seed=505, base_freq=174.61, lead_in_seconds=0.050,
               duration_seconds=2.200, harmonic=4),
    SourceSpec("fx", "fx_src.wav", 44100, 1, 24,
               seed=606, base_freq=880.0, lead_in_seconds=0.250,
               duration_seconds=1.200, harmonic=5),
)


def _envelope(length: int, attack: int, release: int) -> np.ndarray:
    env = np.ones(length, dtype=np.float64)
    if attack > 0:
        env[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float64)
    if release > 0:
        env[-release:] = np.linspace(1.0, 0.0, release, dtype=np.float64)
    return env


def _synthesize_mono(spec: SourceSpec) -> np.ndarray:
    rng = np.random.RandomState(spec.seed)
    total = int(round(spec.duration_seconds * spec.sample_rate))
    lead = spec.onset_frame
    body_len = total - lead
    time_axis = np.arange(body_len, dtype=np.float64) / spec.sample_rate

    # Stable, small-amplitude noise bed guarantees unique byte content per seed.
    noise = rng.uniform(-0.02, 0.02, size=body_len)
    phase = 2.0 * np.pi * spec.base_freq * time_axis
    tone = 0.42 * np.sin(phase)
    tone += 0.18 * np.sin(spec.harmonic * phase + 0.37 * spec.seed)
    tone += 0.08 * np.sin((spec.harmonic + 2) * phase)

    # Four smooth amplitude modulations inside the body make BPM-driven time
    # stretching audible without injecting high-crest transient spikes.
    beat_period = max(1, body_len // 4)
    pulse = np.zeros(body_len, dtype=np.float64)
    for beat_index in range(4):
        start = beat_index * beat_period
        end = min(body_len, start + beat_period)
        local = np.arange(end - start, dtype=np.float64) / max(1, end - start)
        pulse[start:end] += 0.5 + 0.5 * np.sin(np.pi * local) ** 2

    body = tone * 0.55 + noise
    body *= 0.6 + 0.4 * pulse
    attack = max(1, int(round(0.004 * spec.sample_rate)))
    release = max(1, int(round(0.060 * spec.sample_rate)))
    body *= _envelope(body_len, attack, release)

    mono = np.zeros(total, dtype=np.float64)
    mono[lead:] = body
    # Sources stay well below full scale; overlapping stems sum toward a dense
    # master while preserving headroom so loudness normalization can reach the
    # target without peak protection kicking in for the default chart.
    mono *= 0.45
    return mono.astype(np.float32)


def _to_channels(mono: np.ndarray, channels: int, seed: int) -> np.ndarray:
    if channels == 1:
        return mono[:, np.newaxis]
    rng = np.random.RandomState(seed + 777)
    # Deterministic, slightly asymmetric stereo image.
    pan = rng.uniform(0.55, 0.75)
    stereo = np.empty((mono.shape[0], 2), dtype=np.float32)
    stereo[:, 0] = mono * pan
    stereo[:, 1] = mono * (2.0 - pan)
    return stereo / (stereo.max() if stereo.max() > 0 else 1.0) * mono.max()


def render_source(spec: SourceSpec) -> np.ndarray:
    """Render one source as a ``(frames, channels)`` float32 array."""
    mono = _synthesize_mono(spec)
    return np.ascontiguousarray(_to_channels(mono, spec.channels, spec.seed))


def generate_sources(target_dir: str) -> list[dict]:
    """Generate all WAV layer sources and return their metadata list."""
    os.makedirs(target_dir, exist_ok=True)
    written: list[dict] = []
    for spec in SOURCE_SPECS:
        path = os.path.join(target_dir, spec.file_name)
        data = render_source(spec)
        write_wav(path, data, spec.sample_rate, spec.sample_width // 8)
        written.append({
            "layer_id": spec.layer_id,
            "file": spec.file_name,
            "sample_rate": spec.sample_rate,
            "channels": spec.channels,
            "sample_width_bytes": spec.sample_width,
            "frames": data.shape[0],
            "lead_in_seconds": spec.lead_in_seconds,
            "onset_frame_native": spec.onset_frame,
            "seed": spec.seed,
        })
    return written


def build_default_chart(source_meta: list[dict]) -> dict:
    """Construct the deterministic demo chart matching the generated sources."""
    by_id = {item["layer_id"]: item for item in source_meta}

    bpm_points = [
        {"beat": 0.0, "bpm": 100.0},
        {"beat": 8.0, "bpm": 120.0},
        {"beat": 16.0, "bpm": 90.0},
        {"beat": 24.0, "bpm": 110.0},
    ]
    time_signatures = [
        {"beat": 0.0, "numerator": 4, "denominator": 4},
        {"beat": 16.0, "numerator": 3, "denominator": 4},
    ]
    # Each section spans exactly one constant-BPM region.
    sections = [
        {"id": "intro", "start_beat": 0.0, "end_beat": 8.0, "native_bpm": 100.0},
        {"id": "verse", "start_beat": 8.0, "end_beat": 16.0, "native_bpm": 120.0},
        {"id": "bridge", "start_beat": 16.0, "end_beat": 24.0, "native_bpm": 100.0},
        {"id": "chorus", "start_beat": 24.0, "end_beat": 32.0, "native_bpm": 110.0},
    ]

    layer_roles = {
        "drums": True,
        "bass": True,
        "melody": True,
        "vocal": False,
        "pad": True,
        "fx": False,
    }
    layers = []
    for layer_id, enabled in layer_roles.items():
        meta = by_id[layer_id]
        layers.append({
            "id": layer_id,
            "file": meta["file"],
            "enabled": enabled,
            "gain_db": 0.0,
            "onset_beat": 0.0,
            "lead_in_seconds": meta["lead_in_seconds"],
            "native_bpm": 120.0,
        })

    notes = []
    # Half-beat scheduling keeps every enabled layer sounding nearly
    # continuously, which gives loudness normalization a dense, representative
    # signal instead of sparse isolated hits.
    for step in range(64):
        beat = step * 0.5
        beat_index = step // 2
        active = ["bass", "melody", "pad"]
        if step % 2 == 0:
            active.append("drums")
        if beat_index % 4 == 2:
            active.append("vocal")
        if beat_index % 8 == 7:
            active.append("fx")
        for layer_id in active:
            notes.append({
                "beat": beat,
                "layer": layer_id,
                "velocity": 0.8 if layer_id == "drums" else 0.6,
            })

    return {
        "schema_version": 1,
        "title": "deterministic-demo",
        "internal_sample_rate": SAMPLE_RATE_INTERNAL,
        "end_beat": 32.0,
        "target_lufs": -33.0,
        "true_peak_dbfs": -1.0,
        "crossfade_beats": 0.5,
        "bpm_points": bpm_points,
        "time_signatures": time_signatures,
        "sections": sections,
        "layers": layers,
        "notes": notes,
    }


def generate_project(samples_dir: str, chart_path: str) -> dict:
    """Generate WAV sources and chart; return the chart dictionary."""
    meta = generate_sources(samples_dir)
    chart = build_default_chart(meta)
    os.makedirs(os.path.dirname(os.path.abspath(chart_path)), exist_ok=True)
    with open(chart_path, "w", encoding="utf-8") as handle:
        json.dump(chart, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return chart
