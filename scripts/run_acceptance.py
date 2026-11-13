#!/usr/bin/env python3
"""Generate fixtures, run serial/parallel builds, and print measured metrics."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eventmix.batch import BatchOptions, run_batch
from eventmix.demo import generate_demo
from eventmix.wav import parse_wav, read_frames, read_wav_metadata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(manifest: Path, output_dir: Path, jobs: int, memory_mb: int) -> dict:
    options = BatchOptions(
        output_dir=output_dir,
        checkpoint_path=output_dir / ".eventmix-checkpoint.json",
        jobs=jobs,
        max_open_wav=1 if jobs > 1 else 8,
        max_memory_mb=memory_mb,
    )
    return run_batch(manifest, options)


def onset_delta(
    audio: np.ndarray, channels: list[int], seconds: float, rate: int
) -> int:
    center = int(round(seconds * rate))
    radius = int(round(0.025 * rate))
    start = max(0, center - radius)
    stop = min(audio.shape[0], center + radius)
    envelope = np.max(np.abs(audio[start:stop][:, channels]), axis=1)
    hits = np.flatnonzero(envelope > 0.15)
    return start + int(hits[0]) - center


def main() -> int:
    root = ROOT / "artifacts" / "acceptance"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    demo = generate_demo(root / "fixtures")
    expected = json.loads(demo["expected"].read_text("utf-8"))
    serial_dir = root / "serial"
    parallel_dir = root / "parallel"
    serial = run(demo["manifest"], serial_dir, jobs=1, memory_mb=128)
    parallel = run(demo["manifest"], parallel_dir, jobs=2, memory_mb=16)

    output_paths = (
        "final-a.wav",
        "final-a.timeline.json",
        "final-b.wav",
        "final-b.timeline.json",
    )
    byte_identical = all(
        (serial_dir / name).read_bytes() == (parallel_dir / name).read_bytes()
        for name in output_paths
    )
    wav_path = serial_dir / "final-a.wav"
    info = parse_wav(wav_path)
    _, audio = read_frames(info, 0, info.frames)
    metadata = read_wav_metadata(wav_path)
    sidecar = (serial_dir / "final-a.timeline.json").read_bytes()
    timeline = metadata["timeline_json"]
    max_anchor_ms = max(
        track["alignment"]["max_abs_anchor_error_ms"]
        for track in timeline["tracks"]
    )
    onset_deltas = {}
    for track_id, channels in (
        ("main", [0, 1]),
        ("sub", [2, 3]),
        ("ambient", [4, 5]),
    ):
        onset_deltas[track_id] = [
            onset_delta(audio, channels, program_seconds, info.sample_rate)
            for _, program_seconds in expected["anchor_specs"][track_id]
        ]
    resumed = run(demo["manifest"], serial_dir, jobs=2, memory_mb=128)
    report = {
        "status": (
            "ok"
            if serial["exit_code"] == 0 and parallel["exit_code"] == 0
            else "failed"
        ),
        "serial_exit_code": serial["exit_code"],
        "parallel_exit_code": parallel["exit_code"],
        "byte_identical_jobs_1_vs_2": byte_identical,
        "resume_skipped": resumed["summary"]["skipped"],
        "wav": {
            "format_tag": info.format_tag,
            "channels": info.channels,
            "sample_rate": info.sample_rate,
            "bit_depth": info.bits_per_sample,
            "channel_mask": info.channel_mask,
            "frames": info.frames,
            "peak": float(np.max(np.abs(audio))),
        },
        "max_anchor_fit_error_ms": max_anchor_ms,
        "click_onset_delta_frames": onset_deltas,
        "embedded_timeline_equals_sidecar": metadata["timeline_bytes"] == sidecar,
        "cue_count": len(timeline["cues"]),
        "wav_sha256": sha256(wav_path),
        "sidecar_sha256": sha256(serial_dir / "final-a.timeline.json"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())

