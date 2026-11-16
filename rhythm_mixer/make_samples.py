"""Deterministic synthetic test fixtures (fixed seed, no external assets).

Generates three mono 44.1 kHz / 16-bit PCM WAV layers whose lengths do not
match the musical loop period, plus ``chart.json`` and ``events.json`` with
millisecond-precision beat-aligned control events.
"""

import argparse
import json
import os

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100
SEED = 20261117


def _time_array(duration_s):
    count = int(round(duration_s * SAMPLE_RATE))
    return np.arange(count, dtype=np.float64) / SAMPLE_RATE, count


def synth_drums(rng, duration_s, bpm):
    time_axis, count = _time_array(duration_s)
    audio = np.zeros(count, dtype=np.float64)
    beat_s = 60.0 / bpm
    strike = 0
    strike_at = 0.0
    while strike_at < duration_s:
        start = int(round(strike_at * SAMPLE_RATE))
        length = min(int(0.06 * SAMPLE_RATE), count - start)
        if length <= 0:
            break
        envelope = np.exp(-np.arange(length) / (0.012 * SAMPLE_RATE))
        noise = rng.standard_normal(length)
        tone = np.sin(2 * np.pi * 180.0 * np.arange(length) / SAMPLE_RATE)
        audio[start:start + length] += 0.7 * envelope * (0.6 * noise + 0.4 * tone)
        strike += 1
        strike_at = strike * beat_s * 0.5  # eighth-note strikes
    return np.clip(audio, -0.9, 0.9).astype(np.float32)


def synth_bass(rng, duration_s, bpm):
    time_axis, count = _time_array(duration_s)
    audio = np.zeros(count, dtype=np.float64)
    beat_s = 60.0 / bpm
    root_frequencies = (110.0, 110.0, 130.81, 98.0)
    for beat in range(int(np.ceil(duration_s / beat_s)) + 1):
        start = int(round(beat * beat_s * SAMPLE_RATE))
        end = min(start + int(round(beat_s * SAMPLE_RATE)), count)
        if start >= count:
            break
        local_t = np.arange(end - start) / SAMPLE_RATE
        frequency = root_frequencies[beat % len(root_frequencies)]
        saw = 2.0 * (frequency * local_t % 1.0) - 1.0
        envelope = np.exp(-3.0 * local_t)
        wobble = 0.05 * rng.standard_normal(end - start)
        audio[start:end] += 0.5 * envelope * saw + wobble
    return np.clip(audio, -0.9, 0.9).astype(np.float32)


def synth_melody(rng, duration_s, bpm):
    _, count = _time_array(duration_s)
    audio = np.zeros(count, dtype=np.float64)
    beat_s = 60.0 / bpm
    note_length = beat_s * 0.5
    frequencies = (440.0, 554.37, 659.25, 554.37, 493.88, 587.33)
    note = 0
    note_at = 0.0
    while note_at < duration_s:
        start = int(round(note_at * SAMPLE_RATE))
        end = min(start + int(round(note_length * SAMPLE_RATE)), count)
        if start >= count:
            break
        local_t = np.arange(end - start) / SAMPLE_RATE
        attack = np.minimum(local_t / 0.01, 1.0)
        release = np.minimum((note_length - local_t) / 0.03, 1.0)
        envelope = np.clip(np.minimum(attack, release), 0.0, 1.0)
        vibrato = 1.0 + 0.004 * np.sin(2 * np.pi * 5.5 * local_t)
        phase = 2 * np.pi * frequencies[note % len(frequencies)] * local_t
        audio[start:end] += 0.45 * envelope * np.sin(phase * vibrato)
        note += 1
        note_at = note * note_length
    audio += 0.02 * rng.standard_normal(count)
    return np.clip(audio, -0.9, 0.9).astype(np.float32)


def generate(output_dir, bpm=120.0, total_beats=16, loop_bars=2,
             beats_per_bar=4, crossfade_ms=10.0):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    beat_s = 60.0 / bpm
    total_duration_ms = round(total_beats * beat_s * 1000.0, 3)

    layers = (
        ("drums", "drums.wav", 6.7, synth_drums),
        ("bass", "bass.wav", 3.1, synth_bass),
        ("melody", "melody.wav", 2.3, synth_melody),
    )
    chart_layers = []
    for name, file_name, duration_s, synth in layers:
        samples = synth(rng, duration_s, bpm)
        sf.write(
            os.path.join(output_dir, file_name),
            samples,
            SAMPLE_RATE,
            subtype="PCM_16",
        )
        chart_layers.append({"name": name, "file": file_name})

    chart = {
        "bpm": bpm,
        "total_duration_ms": total_duration_ms,
        "crossfade_ms": crossfade_ms,
        "loop_bars": loop_bars,
        "beats_per_bar": beats_per_bar,
        "layers": chart_layers,
    }
    events = [
        {"time_ms": 0.0, "layer": "drums", "action": "enable", "gain": 0.9},
        {"time_ms": 0.0, "layer": "bass", "action": "enable", "gain": 0.8},
        {"time_ms": round(4 * beat_s * 1000.0, 3), "layer": "melody",
         "action": "enable", "gain": 0.6},
        {"time_ms": round(8 * beat_s * 1000.0, 3), "layer": "drums",
         "action": "disable"},
        {"time_ms": round(8 * beat_s * 1000.0, 3), "layer": "drums",
         "action": "disable"},
        {"time_ms": round(12 * beat_s * 1000.0, 3), "layer": "drums",
         "action": "enable", "gain": 1.0},
        {"time_ms": round(12 * beat_s * 1000.0, 3), "layer": "bass",
         "action": "enable", "gain": 0.8},
        {"time_ms": round(14 * beat_s * 1000.0, 3), "layer": "melody",
         "action": "disable"},
    ]
    with open(os.path.join(output_dir, "chart.json"), "w",
              encoding="utf-8", newline="\n") as handle:
        json.dump(chart, handle, indent=2)
        handle.write("\n")
    with open(os.path.join(output_dir, "events.json"), "w",
              encoding="utf-8", newline="\n") as handle:
        json.dump({"events": events}, handle, indent=2)
        handle.write("\n")
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="rhythm-make-samples",
        description="Generate deterministic synthetic layers and chart files.",
    )
    parser.add_argument("--output-dir", "-o", required=True)
    parser.add_argument("--bpm", type=float, default=120.0)
    args = parser.parse_args(argv)
    generate(args.output_dir, bpm=args.bpm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
