from __future__ import annotations

import json
import struct
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def write_pcm24(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = int(audio.shape[1])
    clipped = np.clip(audio, -1.0, 1.0)
    values = np.rint(clipped * 8388607.0).astype(np.int32)
    unsigned = (values & 0xFFFFFF).astype(np.uint32)
    packed = np.empty((audio.shape[0], channels, 3), dtype=np.uint8)
    packed[:, :, 0] = unsigned & 0xFF
    packed[:, :, 1] = (unsigned >> 8) & 0xFF
    packed[:, :, 2] = (unsigned >> 16) & 0xFF
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(3)
        handle.setframerate(sample_rate)
        handle.writeframes(packed.tobytes(order="C"))


def _multitone(
    duration_seconds: float,
    sample_rate: int,
    channels: int,
    seed: int,
    base_hz: float,
    level: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = int(round(duration_seconds * sample_rate))
    t = np.arange(count, dtype=np.float64) / sample_rate
    envelope = (
        0.65
        + 0.2 * np.sin(2.0 * np.pi * 0.37 * t + 0.2)
        + 0.15 * np.sin(2.0 * np.pi * 1.13 * t + 1.1)
    )
    signal = (
        np.sin(2.0 * np.pi * base_hz * t)
        + 0.37 * np.sin(2.0 * np.pi * (base_hz * 1.51) * t + 0.4)
        + 0.21 * np.sin(2.0 * np.pi * (base_hz * 2.73) * t + 1.7)
    )
    noise = rng.standard_normal(count)
    noise = np.convolve(noise, np.ones(17, dtype=np.float64) / 17.0, mode="same")
    mono = (0.55 * signal + 0.45 * noise) * envelope
    peak = float(np.max(np.abs(mono))) or 1.0
    mono = mono / peak * level
    if channels == 1:
        return mono.astype(np.float32)[:, None]
    right = np.roll(mono, 37)
    right = right / (float(np.max(np.abs(right))) or 1.0) * level
    return np.stack((mono, right), axis=1).astype(np.float32)


def _mix_reference(
    languages: dict[str, np.ndarray],
    switches: list[dict],
    duration_frames: int,
    frame_rate: float,
) -> np.ndarray:
    sample_rate = 48000
    total = int(round(duration_frames * sample_rate / frame_rate))
    output = np.zeros((total, 2), dtype=np.float32)
    ordered = sorted(switches, key=lambda item: item["frame"])
    for index, switch in enumerate(ordered):
        end_frame = (
            ordered[index + 1]["frame"] if index + 1 < len(ordered) else duration_frames
        )
        start = int(round(switch["frame"] * sample_rate / frame_rate))
        end = min(total, int(round(end_frame * sample_rate / frame_rate)))
        source = languages[switch["language"]]
        length = min(end - start, source.shape[0])
        if length > 0:
            output[start : start + length] += source[:length]
    return output


def _write_video(
    path: Path,
    reference: np.ndarray | None,
    duration_seconds: float,
    sample_rate: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if reference is None:
        command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=160x90:r=25:d={duration_seconds}",
            "-c:v",
            "ffv1",
            "-an",
            "-shortest",
            str(path),
        ]
    else:
        reference_wav = path.with_suffix(".reference.wav")
        write_pcm24(reference_wav, reference, sample_rate)
        command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=160x90:r=25:d={duration_seconds}",
            "-i",
            str(reference_wav),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s24le",
            "-shortest",
            str(path),
        ]
    subprocess.run(command, check=True)


@dataclass
class SampleScenario:
    root: Path
    manifest_path: Path
    output_path: Path
    report_path: Path
    video_path: Path
    language_paths: dict[str, Path]
    tracks: dict[str, np.ndarray]


def create_scenario(
    root: Path,
    *,
    reference_delay_ms: float = 30.0,
    include_reference: bool = True,
    short_silent_track_seconds: float | None = 4.0,
    duration_seconds: float = 8.0,
) -> SampleScenario:
    root.mkdir(parents=True, exist_ok=True)
    frame_rate = 25
    sample_rate = 48000
    duration_frames = int(round(duration_seconds * frame_rate))
    duration_samples = int(round(duration_seconds * sample_rate))

    zh = _multitone(duration_seconds, sample_rate, 2, 11, 233.0, 0.58)
    en = _multitone(duration_seconds, sample_rate, 2, 29, 311.0, 0.20)
    ja_seconds = (
        duration_seconds if short_silent_track_seconds is None else short_silent_track_seconds
    )
    ja = _multitone(ja_seconds, sample_rate, 2, 47, 389.0, 0.0025)

    spike_indices = [int(1.25 * sample_rate), int(4.15 * sample_rate)]
    zh[spike_indices[0], 0] = 0.985
    zh[spike_indices[1], 1] = 0.975

    tracks = {"zh-CN": zh, "en-US": en, "ja-JP": ja}
    switches = [
        {"frame": 0, "language": "zh-CN"},
        {"frame": 50, "language": "en-US"},
        {"frame": 150, "language": "ja-JP"},
    ]
    for language_id, track in tracks.items():
        write_pcm24(root / f"{language_id}.wav", track, sample_rate)

    reference = _mix_reference(tracks, switches, duration_frames, frame_rate)
    delay_samples = int(round(reference_delay_ms * sample_rate / 1000.0))
    if include_reference and delay_samples:
        if delay_samples > 0:
            reference = np.concatenate(
                (
                    np.zeros((delay_samples, reference.shape[1]), dtype=np.float32),
                    reference[: max(0, duration_samples - delay_samples)],
                ),
                axis=0,
            )
        else:
            reference = reference[-delay_samples:]
            reference = np.concatenate(
                (
                    reference,
                    np.zeros((-delay_samples, reference.shape[1]), dtype=np.float32),
                ),
                axis=0,
            )
    video_path = root / "program.mkv"
    _write_video(
        video_path,
        reference if include_reference else None,
        duration_seconds,
        sample_rate,
    )

    manifest = {
        "schema_version": 1,
        "frame_rate": frame_rate,
        "duration_frames": duration_frames,
        "crossfade_ms": 15,
        "audio": {"sample_rate": sample_rate, "channels": 2, "bit_depth": 24},
        "video": {"path": video_path.name, "reference_stream_index": 1}
        if include_reference
        else {"path": video_path.name},
        "languages": [
            {
                "id": language_id,
                "path": f"{language_id}.wav",
                "sample_rate": sample_rate,
                "channels": 2,
                "bit_depth": 24,
            }
            for language_id in tracks
        ],
        "switch_points": switches,
    }
    if not include_reference:
        manifest["video"].pop("reference_stream_index", None)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return SampleScenario(
        root=root,
        manifest_path=manifest_path,
        output_path=root / "master.wav",
        report_path=root / "report.json",
        video_path=video_path,
        language_paths={
            language_id: root / f"{language_id}.wav" for language_id in tracks
        },
        tracks=tracks,
    )


def corrupt_wav_block_alignment(path: Path) -> bytes:
    original = path.read_bytes()
    data = bytearray(original)
    if data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV")
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        size = struct.unpack("<I", data[offset + 4 : offset + 8])[0]
        if chunk_id == b"data":
            struct.pack_into("<I", data, offset + 4, size - 1)
            path.write_bytes(data)
            return original
        offset += 8 + size + (size & 1)
    raise ValueError("data chunk not found")

