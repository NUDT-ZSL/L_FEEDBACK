"""Deterministic network impairment simulator.

Pure packet streams are transformed under a fixed seed using:

* independent packet loss (plus optional periodic burst loss),
* pairwise reordering (a packet is delayed past its successor),
* arrival jitter drawn from a normal (default) or uniform distribution,
* optional malformed records for the parser's error paths.

No entropy beyond the supplied seed is used, so reruns are byte-identical.
Alongside each ``stream_<id>.pkt`` the simulator writes:

* ``ground_truth.jsonl`` - per global frame, per stream origin of the clean
  samples and whether the packet survived the impairment transform;
* ``simulation_meta.json`` - arrival schedule and generation parameters used
  by the single-threaded virtual-clock event loop.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .constants import FRAME_DURATION_MS, FRAME_SAMPLES, SAMPLE_RATE
from .protocol import build_packet
from .signals import float_to_pcm_bytes, generate_stream_signal

META_FILENAME = "simulation_meta.json"
TRUTH_FILENAME = "ground_truth.jsonl"


@dataclass
class ImpairmentConfig:
    """User-facing impairment parameters."""

    seed: int = 20261114
    loss_rate: float = 0.10
    reorder_rate: float = 0.05
    jitter_mean_ms: float = 20.0
    jitter_sigma_ms: float = 15.0
    jitter_distribution: str = "normal"  # "normal" | "uniform"
    max_jitter_ms: float = 120.0
    burst_loss: bool = False
    inject_faults: bool = False


def _sample_jitter(rng: random.Random, config: ImpairmentConfig) -> float:
    if config.jitter_distribution == "uniform":
        low = max(0.0, config.jitter_mean_ms - config.jitter_sigma_ms)
        return rng.uniform(low, config.jitter_mean_ms + config.jitter_sigma_ms)
    value = rng.gauss(config.jitter_mean_ms, config.jitter_sigma_ms)
    return float(max(0.0, min(config.max_jitter_ms, value)))


def simulate(
    output_dir: str,
    stream_count: int,
    frame_count: int,
    config: ImpairmentConfig,
    base_frequency: float = 220.0,
) -> Dict[str, object]:
    """Generate clean signals, impair them, and persist packets + truth.

    Returns the metadata dictionary also written to ``simulation_meta.json``.
    """
    if not 1 <= stream_count <= 4:
        raise ValueError("stream_count must be between 1 and 4")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    os.makedirs(output_dir, exist_ok=True)

    clean_signals: List[np.ndarray] = []
    ground_truth: List[Dict[str, object]] = []
    schedule: Dict[str, List[Dict[str, float]]] = {}
    lost_frames: Dict[int, List[int]] = {}

    for stream_id in range(stream_count):
        rng = random.Random(config.seed + stream_id * 1009)
        signal = generate_stream_signal(
            stream_id, frame_count, base_frequency=base_frequency
        )
        clean_signals.append(signal)

        records: List[bytes] = []
        arrivals: List[Dict[str, float]] = []
        stream_lost: List[int] = []

        # First pass: decide loss mask (independent + optional burst).
        lost = [False] * frame_count
        for frame_index in range(frame_count):
            if rng.random() < config.loss_rate:
                lost[frame_index] = True
            if config.burst_loss and frame_index % 40 == 0 and frame_index:
                for burst_index in range(4):
                    if frame_index + burst_index < frame_count:
                        lost[frame_index + burst_index] = True

        # Second pass: build surviving packets and per-packet arrivals.
        for frame_index in range(frame_count):
            timestamp = frame_index * FRAME_SAMPLES
            seq = frame_index & 0xFFFF
            nominal_ms = frame_index * FRAME_DURATION_MS
            if lost[frame_index]:
                stream_lost.append(frame_index)
                continue

            delay_ms = _sample_jitter(rng, config)
            arrival_ms = nominal_ms + delay_ms
            frame = signal[
                frame_index * FRAME_SAMPLES:
                (frame_index + 1) * FRAME_SAMPLES
            ]
            pcm = float_to_pcm_bytes(frame)
            records.append(build_packet(seq, timestamp, pcm))
            arrivals.append(
                {
                    "frame_index": frame_index,
                    "seq": seq,
                    "arrival_ms": round(arrival_ms, 3),
                    "jitter_ms": round(delay_ms, 3),
                }
            )

        # Reordering: swap adjacent surviving packets at random (keeps the
        # arrival schedule consistent with the on-disk packet order).
        for index in range(len(arrivals) - 1):
            if rng.random() < config.reorder_rate:
                arrivals[index], arrivals[index + 1] = (
                    arrivals[index + 1],
                    arrivals[index],
                )
                records[index], records[index + 1] = (
                    records[index + 1],
                    records[index],
                )

        packet_bytes = b"".join(records)
        if config.inject_faults and stream_id == 0:
            packet_bytes += _fault_records()

        packet_path = os.path.join(output_dir, f"stream_{stream_id}.pkt")
        with open(packet_path, "wb") as handle:
            handle.write(packet_bytes)

        schedule[str(stream_id)] = arrivals
        lost_frames[stream_id] = stream_lost

    # Ground truth: one record per global frame with every stream's origin.
    truth_path = os.path.join(output_dir, TRUTH_FILENAME)
    with open(truth_path, "w", encoding="utf-8") as handle:
        for frame_index in range(frame_count):
            record = {
                "frame_index": frame_index,
                "timestamp": frame_index * FRAME_SAMPLES,
                "streams": {},
            }
            for stream_id in range(stream_count):
                frame = clean_signals[stream_id][
                    frame_index * FRAME_SAMPLES:
                    (frame_index + 1) * FRAME_SAMPLES
                ]
                record["streams"][str(stream_id)] = {
                    "source": "synthetic_tone",
                    "stream_id": stream_id,
                    "original_frame_index": frame_index,
                    "sample_offset": frame_index * FRAME_SAMPLES,
                    "lost": frame_index in lost_frames.get(stream_id, set()),
                    "clean_rms": float(
                        np.sqrt(np.mean(np.square(frame, dtype=np.float64)))
                    ),
                }
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    metadata = {
        "sample_rate": SAMPLE_RATE,
        "frame_samples": FRAME_SAMPLES,
        "frame_duration_ms": FRAME_DURATION_MS,
        "stream_count": stream_count,
        "frame_count": frame_count,
        "config": config.__dict__,
        "schedule": schedule,
        "lost_frames": {
            str(stream_id): frames for stream_id, frames in lost_frames.items()
        },
    }
    meta_path = os.path.join(output_dir, META_FILENAME)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return metadata


def _fault_records() -> bytes:
    """Append malformed records exercising parser error branches.

    * a record declaring a bad payload length but with enough bytes (skippable),
    * a record whose declared payload runs past end of file (truncated).
    """
    import struct

    bad_length = struct.pack("<HIH", 0xFFF0, 0, FRAME_SAMPLES)
    bad_length += b"\x00" * FRAME_SAMPLES
    truncated = struct.pack("<HIH", 0xFFF1, 0, FRAME_SAMPLES * 2)
    truncated += b"\x00" * 10
    return bad_length + truncated
