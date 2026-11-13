"""Deterministic sample generator.

All bytes come from a fixed-seed :class:`DeterministicRandom` stream, so the
same seed always reproduces identical WAV shards and event manifest without
touching the network or any real recording.  Shards intentionally cover:

* mixed sample rates (16000 / 44100 / 48000 Hz),
* recording drift both inside and outside the 20 ms tolerance,
* missing first packets (short captures),
* overlapping speech between players,
* high-gain intervals that saturate the 16-bit mixer,
* one player referenced by the manifest but with no shard files.
"""

import json
import os
import math
import array

from . import OUTPUT_SAMPLE_RATE
from .audio import write_wav
from .prng import DeterministicRandom
from .resampling import DRIFT_TOLERANCE_MS

DEFAULT_SEED = 20261114
SAMPLE_RATES = (16000, 44100, 48000)
GATE_THRESHOLD = 0.02
MANIFEST_NAME = "events.json"


def _synthesize_samples(rng, count, rate, speech_start_ms, speech_end_ms,
                        amplitude, frequency):
    fade_ms = 8.0
    fade_samples = max(1, int(round(rate * fade_ms / 1000.0)))
    speech_start_index = int(round(rate * speech_start_ms / 1000.0))
    speech_end_index = int(round(rate * speech_end_ms / 1000.0))
    values = []
    two_pi = 6.283185307179586
    phase = rng.uniform(0.0, two_pi)
    for index in range(count):
        noise = rng.uniform(-0.004, 0.004)
        value = noise
        if speech_start_index <= index < speech_end_index:
            envelope = 1.0
            distance_in = index - speech_start_index
            distance_out = speech_end_index - 1 - index
            fade_distance = min(distance_in, distance_out, fade_samples)
            if fade_distance < fade_samples:
                envelope = (fade_distance + 1) / float(fade_samples)
            tone = math.sin(two_pi * frequency * index / rate + phase)
            value += amplitude * envelope * tone
        if value > 1.0:
            value = 1.0
        elif value < -1.0:
            value = -1.0
        values.append(int(round(value * 32767.0)))
    return values


def generate_dataset(output_dir: str, seed: int = DEFAULT_SEED) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    rng = DeterministicRandom(seed)
    player_count = 3 + rng.next_u64() % 3
    players = []
    intervals = []

    for player_index in range(player_count):
        player_id = f"P{player_index}"
        segment_count = 6 + rng.next_u64() % 5
        cursor_ms = 500.0
        segment_records = []
        for segment_index in range(segment_count):
            speech_ms = 800.0 + rng.randint(0, 2200)
            gap_ms = 300.0 + rng.randint(0, 700)
            preroll_ms = 40.0 + rng.randint(0, 120)
            tail_ms = 40.0 + rng.randint(0, 160)
            speech_start_ms = cursor_ms
            speech_end_ms = cursor_ms + speech_ms
            record_offset_ms = speech_start_ms - preroll_ms
            declared_ms = preroll_ms + speech_ms + tail_ms

            if segment_index < len(SAMPLE_RATES):
                rate = SAMPLE_RATES[segment_index]
            else:
                rate = rng.choice(SAMPLE_RATES)

            drift_mode = rng.randint(0, 3)
            ideal_count = int(round(declared_ms * rate / 1000.0))
            dropped_packets = 0
            if drift_mode == 0:
                count = ideal_count
            elif drift_mode == 1:
                delta_ms = rng.choice((-1, 1)) * rng.randint(5, 18)
                count = int(round((declared_ms + delta_ms) * rate / 1000.0))
            else:
                delta_ms = rng.choice((-1, 1)) * rng.randint(25, 90)
                dropped_packets = rng.randint(1, 3)
                dropped_samples = dropped_packets * rate // 1000
                count = int(round((declared_ms + delta_ms) * rate / 1000.0))
                count = max(1, count - dropped_samples)

            amplitude = 0.25 + rng.uniform(0.0, 0.35)
            if segment_index == 0:
                amplitude = 0.5 + rng.uniform(0.0, 0.2)
            frequency = 120.0 + rng.uniform(0.0, 260.0)
            samples = _synthesize_samples(
                rng, count, rate, preroll_ms, preroll_ms + speech_ms,
                amplitude, frequency)

            segment_id = f"{player_id}-S{segment_index:02d}"
            file_name = f"{player_id}_{segment_index:02d}.wav"
            write_wav(os.path.join(output_dir, file_name), rate,
                      _short_array(samples))

            actual_ms = count * 1000.0 / rate
            gain = 0.6 + rng.uniform(0.0, 0.9)
            if segment_index == 0:
                gain = 2.5
            record = {
                "segment_id": segment_id,
                "player_id": player_id,
                "file": file_name,
                "sample_rate": rate,
                "record_offset_ms": round(record_offset_ms, 3),
                "declared_duration_ms": round(declared_ms, 3),
                "actual_duration_ms": round(actual_ms, 3),
                "drift_ms": round(actual_ms - declared_ms, 3),
                "dropped_first_packets_ms": dropped_packets,
                "speech_start_ms": round(speech_start_ms, 3),
                "speech_end_ms": round(speech_end_ms, 3),
                "threshold": GATE_THRESHOLD,
                "gain": round(gain, 6),
                "resample_required": abs(actual_ms - declared_ms)
                                      > DRIFT_TOLERANCE_MS,
            }
            segment_records.append(record)
            intervals.append({
                "interval_id": segment_id,
                "player_id": player_id,
                "segment_id": segment_id,
                "start_ms": round(speech_start_ms, 3),
                "end_ms": round(speech_end_ms, 3),
                "threshold": GATE_THRESHOLD,
                "gain": round(gain, 6),
            })
            cursor_ms = speech_end_ms + gap_ms
        players.append({"player_id": player_id, "segments": segment_records})

    missing_id = f"P{player_count}"
    missing_intervals = []
    for index in range(1 + rng.next_u64() % 2):
        start_ms = 500.0 + index * 2500.0
        missing_intervals.append({
            "interval_id": f"{missing_id}-S{index:02d}",
            "player_id": missing_id,
            "segment_id": f"{missing_id}-S{index:02d}",
            "start_ms": round(start_ms, 3),
            "end_ms": round(start_ms + 1200.0, 3),
            "threshold": GATE_THRESHOLD,
            "gain": round(0.8 + rng.uniform(0.0, 0.8), 6),
        })
    intervals.extend(missing_intervals)

    manifest = {
        "schema_version": 1,
        "seed": seed,
        "output_sample_rate": OUTPUT_SAMPLE_RATE,
        "drift_tolerance_ms": DRIFT_TOLERANCE_MS,
        "gate_threshold": GATE_THRESHOLD,
        "players": players,
        "intervals": intervals,
        "missing_players": [missing_id],
    }
    manifest_path = os.path.join(output_dir, MANIFEST_NAME)
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def _short_array(values):
    out = array.array("h")
    for value in values:
        if value > 32767:
            value = 32767
        elif value < -32768:
            value = -32768
        out.append(value)
    return out

