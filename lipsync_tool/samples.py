"""Deterministic construction of local validation samples.

Everything is generated from fixed seeds so repeated runs produce byte-stable
WAV content:

* an animation metadata JSON with frame rate, duration and phoneme events;
* a silent MP4 video (created with FFmpeg's ``testsrc``);
* English, Japanese and Chinese dub WAV files whose timing deliberately
  exercises resampling, stereo downmix, stretch-bound clamping and a long
  silence gap.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .wavio import write_wav

FRAME_RATE = 24
ANIMATION_DURATION = 6.0
PHONEMES = ["AA", "E", "IH", "OH", "UH", "M", "P", "S", "T", "K", "F", "R"]


@dataclass(frozen=True)
class DubSpec:
    language: str
    sample_rate: int
    channels: int
    sample_width: int
    duration: float
    base_frequency: float
    jitter_ms: float
    silence_gap: tuple[float, float] | None


# English is longer than the clamp upper bound (6.0 / 4.4 < 0.75 once inverted)
# and contains a > 500 ms silence gap; Japanese is shorter than the lower bound;
# Chinese sits inside the allowed stretch window.
DUB_SPECS: Dict[str, DubSpec] = {
    "en": DubSpec("en", 22050, 2, 2, 4.4, 180.0, 30.0, (3.1, 3.8)),
    "ja": DubSpec("ja", 48000, 1, 2, 8.4, 240.0, 30.0, None),
    "zh": DubSpec("zh", 44100, 1, 2, 6.0, 200.0, 30.0, None),
}


def build_metadata_events() -> List[Dict[str, Any]]:
    """Create a fixed pseudo-random phoneme event list on frame timestamps."""
    rng = np.random.default_rng(seed=20261117)
    total_frames = int(round(ANIMATION_DURATION * FRAME_RATE))
    events: List[Dict[str, Any]] = []
    cursor = 6
    event_index = 0
    while cursor < total_frames - 8:
        duration_frames = int(rng.integers(5, 10))
        events.append(
            {
                "id": f"phoneme_{event_index:03d}",
                "phoneme": PHONEMES[int(rng.integers(0, len(PHONEMES)))],
                "start_frame": cursor,
                "duration_frames": duration_frames,
            }
        )
        gap_frames = int(rng.integers(3, 7))
        cursor += duration_frames + gap_frames
        event_index += 1
    return events


def build_metadata() -> Dict[str, Any]:
    events = build_metadata_events()
    return {
        "animation_id": "cutscene_demo_001",
        "frame_rate": FRAME_RATE,
        "duration_seconds": ANIMATION_DURATION,
        "total_frames": int(round(ANIMATION_DURATION * FRAME_RATE)),
        "target_sample_rate": 32000,
        "lip_events": events,
    }


def _time_grid(duration: float, sample_rate: int) -> np.ndarray:
    frames = int(round(duration * sample_rate))
    return np.arange(frames, dtype=np.float64) / float(sample_rate)


def synthesize_dub(
    spec: DubSpec, metadata: Dict[str, Any]
) -> np.ndarray:
    """Synthesize a deterministic speech-like tone-burst dub signal."""
    language_seeds = {"en": 10101, "ja": 20202, "zh": 30303}
    rng = np.random.default_rng(seed=language_seeds.get(spec.language, 40404))
    time_axis = _time_grid(spec.duration, spec.sample_rate)
    signal = rng.normal(0.0, 0.004, size=time_axis.shape)

    for event_index, event in enumerate(metadata["lip_events"]):
        nominal_time = event["start_frame"] / float(metadata["frame_rate"])
        if nominal_time >= spec.duration - 0.1:
            break
        jitter_seconds = float(rng.uniform(-spec.jitter_ms, spec.jitter_ms)) / 1000.0
        burst_start = max(0.0, min(spec.duration - 0.05, nominal_time + jitter_seconds))
        burst_duration = min(
            event["duration_frames"] / float(metadata["frame_rate"]) * 0.9,
            spec.duration - burst_start,
        )
        frequency = spec.base_frequency + 40.0 * (event_index % 5)
        amplitude = 0.35 + 0.1 * (event_index % 3)
        mask = (time_axis >= burst_start) & (time_axis < burst_start + burst_duration)
        local_time = time_axis[mask] - burst_start
        envelope = np.sin(np.pi * np.clip(local_time / burst_duration, 0.0, 1.0)) ** 2
        tone = np.sin(2.0 * np.pi * frequency * local_time)
        overtone = 0.35 * np.sin(2.0 * np.pi * frequency * 2.0 * local_time)
        signal[mask] += amplitude * envelope * (tone + overtone)

    if spec.silence_gap is not None:
        gap_start, gap_end = spec.silence_gap
        mask = (time_axis >= gap_start) & (time_axis < gap_end)
        signal[mask] = rng.normal(0.0, 0.0008, size=int(mask.sum()))

    signal = np.clip(signal, -1.0, 1.0)
    if spec.channels == 2:
        # Slightly different level per channel, still fully deterministic.
        stereo = np.column_stack([signal * 0.9, signal * 1.0])
        return stereo
    return signal


def create_silent_video(output_path: Path, duration: float, frame_rate: int) -> None:
    """Render a deterministic silent MP4 with the libx264 testsrc pattern."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-nostdin",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=640x360:rate={frame_rate}:duration={duration}",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        str(output_path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not output_path.is_file():
        raise RuntimeError(f"failed to create sample video: {result.stderr[-500:]}")


def generate_sample_tree(output_dir: str | Path) -> Dict[str, Path]:
    """Generate the full deterministic sample tree and return key paths."""
    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio"
    video_dir = output_dir / "video"
    audio_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata()
    metadata_path = output_dir / "animation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    video_path = video_dir / "cutscene_silent.mp4"
    create_silent_video(video_path, ANIMATION_DURATION, FRAME_RATE)

    wav_paths: Dict[str, Path] = {}
    for language, spec in DUB_SPECS.items():
        audio = synthesize_dub(spec, metadata)
        wav_path = audio_dir / f"dub_{language}.wav"
        write_wav(wav_path, audio, spec.sample_rate, spec.sample_width)
        wav_paths[language] = wav_path

    return {
        "metadata": metadata_path,
        "video": video_path,
        **{f"wav_{language}": path for language, path in wav_paths.items()},
    }
