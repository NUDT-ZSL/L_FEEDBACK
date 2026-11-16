"""Measure end-to-end audio placement accuracy against the timeline.

A dedicated fixture is built where each dialogue WAV starts with a short
1 kHz tone burst. After the full pipeline runs, the mixed audio is decoded
and the burst onset is located; its deviation from the timeline start must
stay within the 10 ms acceptance budget.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from locdub import ffmpeg_utils
from locdub.cli import main

SR = 48000


def _write_burst_wav(path: Path, duration_ms: int,
                     burst_ms: int = 80, channels: int = 1) -> None:
    frames = int(round(SR * duration_ms / 1000.0))
    burst_frames = int(round(SR * burst_ms / 1000.0))
    t = np.arange(burst_frames) / SR
    burst = 0.8 * np.sin(2 * np.pi * 1000.0 * t)
    signal = np.zeros(frames, dtype=np.float64)
    signal[:burst_frames] = burst
    if channels == 2:
        signal = np.stack((signal, signal), axis=1)
    else:
        signal = signal[:, np.newaxis]
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), signal, SR, subtype="PCM_16")


def _decode_audio(mp4_path: Path, out_wav: Path) -> np.ndarray:
    cmd = [
        "ffmpeg", "-y", "-i", str(mp4_path), "-ac", "1",
        "-ar", str(SR), "-f", "wav", str(out_wav),
        "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)
    data, _ = sf.read(str(out_wav), dtype="float64")
    return data


def _first_onset(signal: np.ndarray, search_start: int,
                 threshold: float = 0.2) -> int:
    region = signal[search_start:]
    idx = np.flatnonzero(np.abs(region) > threshold)
    if len(idx) == 0:
        raise AssertionError("tone burst not found")
    return search_start + int(idx[0])


def test_sync_within_10ms(tmp_path: Path) -> None:
    root = tmp_path
    video = root / "source.mp4"
    ffmpeg_utils.make_solid_video(video, 20000)

    placements = [
        ("seg_a", 1000, 3000),
        ("seg_b", 5000, 7500),
        ("seg_c", 12345, 15000),
        ("seg_d", 17000, 19500),
    ]
    dub_dir = root / "dubs" / "en"
    timeline_segments = []
    for segment_id, start_ms, end_ms in placements:
        spoken_ms = end_ms - start_ms - 400  # force real stretching
        _write_burst_wav(dub_dir / f"{segment_id}.wav", spoken_ms)
        timeline_segments.append({
            "id": segment_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "audio": f"{segment_id}.wav",
            "ratio_range": [0.3, 3.0],
        })
    timeline_path = root / "timeline.json"
    timeline_path.write_text(json.dumps({
        "video_duration_ms": 20000,
        "segments": timeline_segments,
    }), encoding="utf-8")

    out_dir = root / "out"
    code = main([
        "--timeline", str(timeline_path),
        "--video", str(video),
        "--lang", f"en={dub_dir}",
        "--out", str(out_dir),
        "--workers", "1",
        "--sample-rate", str(SR),
        "--channels", "2",
        "--quiet",
    ])
    assert code == 0

    mixed = _decode_audio(out_dir / "en" / "en.mp4", root / "decoded.wav")
    errors_ms = []
    for segment_id, start_ms, _end_ms in placements:
        expected_sample = int(round(SR * start_ms / 1000.0))
        onset = _first_onset(mixed, max(0, expected_sample - SR // 20))
        error_ms = abs(onset - expected_sample) * 1000.0 / SR
        errors_ms.append(error_ms)
        print(f"{segment_id}: expected {start_ms}ms, onset error {error_ms:.2f}ms")

    worst = max(errors_ms)
    print(f"worst sync error: {worst:.2f}ms")
    assert worst <= 10.0, f"sync error {worst}ms exceeds 10ms budget"


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        test_sync_within_10ms(Path(tmp))
        print("sync test passed")
