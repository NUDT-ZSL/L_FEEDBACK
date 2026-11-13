from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import wave
from pathlib import Path


class LCG:
    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def next_u32(self) -> int:
        self.state = (
            1664525 * self.state + 1013904223
        ) & 0xFFFFFFFF
        return self.state

    def uniform(self) -> float:
        return self.next_u32() / 4294967295.0

    def signed(self) -> float:
        return self.uniform() * 2.0 - 1.0


def _pcm16(value: float) -> int:
    scaled = int(round(max(-1.0, min(1.0, value)) * 32767.0))
    return max(-32768, min(32767, scaled))


def _write_wav(path: Path, sample_rate: int, channels: int, samples: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = struct.pack("<" + "h" * len(samples), *samples)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def _gap_active(index: int, frames: int, gaps: list[tuple[float, float]]) -> bool:
    position = index / max(1, frames)
    return any(start <= position < end for start, end in gaps)


def _make_commentary(
    frames: int,
    channels: int,
    sample_rate: int,
    seed: int,
    base_frequency: float,
    gaps: list[tuple[float, float]],
) -> list[int]:
    rng = LCG(seed)
    samples: list[int] = []
    for index in range(frames):
        t = index / sample_rate
        silent = _gap_active(index, frames, gaps)
        if silent:
            noise = 0.00015 * rng.signed()
            for _ in range(channels):
                samples.append(_pcm16(noise))
            continue
        phrase = 0.55 + 0.45 * math.sin(2.0 * math.pi * 0.37 * t + 0.4)
        for channel in range(channels):
            vibrato = 1.0 + 0.025 * math.sin(2.0 * math.pi * 4.3 * t + channel)
            carrier = math.sin(
                2.0 * math.pi * base_frequency * (1.0 + channel * 0.013) * t
            )
            harmonic = 0.32 * math.sin(
                2.0 * math.pi * base_frequency * 2.03 * t + 0.7
            )
            noise = 0.018 * rng.signed()
            value = phrase * (0.20 * carrier + harmonic * 0.075) * vibrato + noise
            samples.append(_pcm16(value))
    return samples


def _make_engine(frames: int, sample_rate: int) -> list[int]:
    rng = LCG(0x51A7E)
    samples: list[int] = []
    for index in range(frames):
        t = index / sample_rate
        sweep = 72.0 + 24.0 * math.sin(2.0 * math.pi * 0.11 * t)
        left = 0.18 * math.sin(2.0 * math.pi * sweep * t)
        right = 0.17 * math.sin(2.0 * math.pi * (sweep * 1.006) * t + 0.5)
        left += 0.030 * rng.signed()
        right += 0.030 * rng.signed()
        samples.extend((_pcm16(left), _pcm16(right)))
    return samples


def _make_ambience(frames: int, sample_rate: int) -> list[int]:
    rng = LCG(0xA6B1E)
    samples: list[int] = []
    for index in range(frames):
        t = index / sample_rate
        drift = 0.010 * math.sin(2.0 * math.pi * 0.071 * t)
        left = 0.022 * rng.signed() + drift
        right = 0.022 * rng.signed() - drift
        samples.extend((_pcm16(left), _pcm16(right)))
    return samples


def _frame_timestamps(duration_ms: float) -> list[float]:
    rng = LCG(0xF12A3)
    timestamps = [0.0]
    cursor = 0.0
    while cursor < duration_ms:
        cursor += 11.0 + 12.0 * rng.uniform()
        if cursor >= duration_ms:
            break
        timestamps.append(round(cursor, 6))
    if len(timestamps) < 2:
        timestamps.append(min(duration_ms * 0.5, 1.0))
    return timestamps


def _track(
    name: str,
    path: str,
    channels: int,
    start_ms: float,
    end_ms: float,
    gain_db: float,
    pan: float,
) -> dict:
    return {
        "name": name,
        "path": path,
        "channels": channels,
        "start_ms": round(start_ms, 6),
        "end_ms": round(end_ms, 6),
        "gain_db": gain_db,
        "pan": pan,
    }


def generate(output_dir: Path, sample_rate: int, duration_ms: float, force: bool) -> None:
    if sample_rate not in {16000, 22050, 24000, 44100, 48000}:
        raise ValueError("sample rate must be one of 16000, 22050, 24000, 44100, 48000")
    if duration_ms < 1000.0:
        raise ValueError("duration must be at least 1000 ms")
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    inputs = output_dir / "inputs"
    output_dir.mkdir(parents=True)

    duration_s = duration_ms / 1000.0
    lead_frames = int(round(duration_s * 0.87 * sample_rate))
    analyst_frames = int(round(duration_s * 0.95 * sample_rate))
    engine_frames = int(round(duration_s * 0.97 * sample_rate))
    ambience_frames = int(round(duration_s * 1.08 * sample_rate))

    lead = _make_commentary(
        lead_frames,
        2,
        sample_rate,
        0x1EAD,
        173.0,
        [(0.22, 0.257), (0.51, 0.57), (0.73, 0.741)],
    )
    analyst = _make_commentary(
        analyst_frames,
        1,
        sample_rate,
        0xA1A17,
        229.0,
        [(0.30, 0.337), (0.63, 0.688)],
    )
    engine = _make_engine(engine_frames, sample_rate)
    ambience = _make_ambience(ambience_frames, sample_rate)

    _write_wav(inputs / "commentary_lead.wav", sample_rate, 2, lead)
    _write_wav(inputs / "commentary_analyst.wav", sample_rate, 1, analyst)
    _write_wav(inputs / "engine.wav", sample_rate, 2, engine)
    _write_wav(inputs / "ambience.wav", sample_rate, 2, ambience)

    lead_track = _track(
        "lead",
        "inputs\\commentary_lead.wav",
        2,
        0.0,
        duration_ms * 0.94,
        -3.0,
        -0.45,
    )
    lead_track["anchors"] = [
        {"replay_ms": 0.0, "source_ms": 0.0},
        {"replay_ms": duration_ms * 0.22, "source_ms": duration_ms * 0.20},
        {"replay_ms": duration_ms * 0.48, "source_ms": duration_ms * 0.43},
        {"replay_ms": duration_ms * 0.67, "source_ms": duration_ms * 0.63},
        {"replay_ms": duration_ms * 0.94, "source_ms": duration_ms * 0.83},
    ]
    lead_track["silence"] = {
        "window_ms": 20.0,
        "threshold_dbfs": -45.0,
        "min_duration_ms": max(20.0, duration_ms * 0.015),
    }

    analyst_track = _track(
        "analyst",
        "inputs\\commentary_analyst.wav",
        1,
        duration_ms * 0.02,
        duration_ms,
        -4.0,
        0.4,
    )
    analyst_track["anchors"] = [
        {"replay_ms": duration_ms * 0.02, "source_ms": 0.0},
        {"replay_ms": duration_ms * 0.25, "source_ms": duration_ms * 0.21},
        {"replay_ms": duration_ms * 0.50, "source_ms": duration_ms * 0.43},
        {"replay_ms": duration_ms * 0.75, "source_ms": duration_ms * 0.68},
        {"replay_ms": duration_ms, "source_ms": duration_ms * 0.91},
    ]
    analyst_track["silence"] = {
        "window_ms": 25.0,
        "threshold_dbfs": -44.0,
        "min_duration_ms": max(20.0, duration_ms * 0.015),
    }

    config = {
        "version": 1,
        "sample_rate": sample_rate,
        "bit_depth": 16,
        "channels": 2,
        "output_path": "out\\replay_mix.wav",
        "replay": {
            "duration_ms": duration_ms,
            "frame_timestamps_ms": _frame_timestamps(duration_ms),
        },
        "mix": {
            "target_rms_dbfs": -20.0,
            "max_gain_db": 18.0,
            "peak_limit_dbfs": -1.0,
            "soft_knee_ratio": 0.9,
            "block_frames": 2048,
        },
        "tracks": {
            "commentary": [lead_track, analyst_track],
            "engine": _track(
                "engine",
                "inputs\\engine.wav",
                2,
                0.0,
                duration_ms * 0.99,
                -23.0,
                0.0,
            ),
            "ambience": _track(
                "ambience",
                "inputs\\ambience.wav",
                2,
                0.0,
                duration_ms,
                -31.0,
                0.0,
            ),
        },
    }
    config_path = output_dir / "replay.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "config": str(config_path),
                "sample_rate": sample_rate,
                "duration_ms": duration_ms,
                "tracks": 4,
            },
            separators=(",", ":"),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic replay-mixer sample inputs.")
    parser.add_argument("--output", default="demo", help="Output directory.")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--duration-ms", type=float, default=9000.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        generate(Path(args.output), args.sample_rate, args.duration_ms, args.force)
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, separators=(",", ":")))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
