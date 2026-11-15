from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from pathlib import Path
from typing import Any, Dict

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_pcm16(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    quantized = np.rint(np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(int(data.shape[1]))
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(np.ascontiguousarray(quantized).tobytes())


def _synth_voice(
    path: Path,
    sample_rate: int,
    duration_seconds: float,
    seed: int,
    base_frequency: float,
    channels: int,
) -> None:
    frame_count = int(round(duration_seconds * sample_rate))
    time = np.arange(frame_count, dtype=np.float64) / float(sample_rate)
    rng = np.random.Generator(np.random.PCG64(seed))
    syllable_rate = 3.2 + float(seed % 7) * 0.19
    syllable_phase = 0.35 * float(seed % 5)
    gate = 0.18 + 0.82 * (
        0.5 + 0.5 * np.sin(2.0 * math.pi * syllable_rate * time + syllable_phase)
    ) ** 1.6
    vibrato = 1.0 + 0.035 * np.sin(2.0 * math.pi * 4.7 * time + 0.2 * seed)
    instantaneous_frequency = base_frequency * vibrato
    phase = 2.0 * math.pi * np.cumsum(instantaneous_frequency) / sample_rate
    signal = (
        0.72 * np.sin(phase)
        + 0.20 * np.sin(2.0 * phase + 0.17)
        + 0.08 * np.sin(3.0 * phase + 0.31)
    )
    noise = rng.normal(0.0, 1.0, frame_count)
    smoothed_noise = np.convolve(noise, np.ones(17) / 17.0, mode="same")
    signal = gate * (signal + 0.08 * smoothed_noise)
    fade_frames = min(frame_count // 2, int(0.012 * sample_rate))
    if fade_frames > 0:
        ramp = np.linspace(0.0, 1.0, fade_frames, endpoint=False)
        signal[:fade_frames] *= ramp
        signal[-fade_frames:] *= ramp[::-1]
    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal = signal / peak * 0.82
    if channels == 2:
        stereo = np.stack((signal, np.roll(signal, 1)), axis=1)
        _write_pcm16(path, stereo, sample_rate)
    else:
        _write_pcm16(path, signal, sample_rate)


def _track(
    duration_frames: int,
    position_function: Any,
    forward_function: Any,
) -> Dict[str, Any]:
    keyframes = []
    for frame in range(duration_frames):
        keyframes.append(
            {
                "frame": frame,
                "position": [float(value) for value in position_function(frame)],
                "forward": [float(value) for value in forward_function(frame)],
            }
        )
    return {"keyframes": keyframes}


def generate_demo(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    voices_dir = output_dir / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    fps = 24
    duration_frames = 120

    dialogue_specs = [
        (
            "d01_narrator",
            "narrator",
            0,
            48,
            "The signal begins where the corridor bends.",
            "narrator_48k.wav",
            48000,
            101,
            146.0,
            1,
        ),
        (
            "d02_alice",
            "alice",
            36,
            80,
            "I can hear the relay through the wall.",
            "alice_44k.wav",
            44100,
            202,
            196.0,
            1,
        ),
        (
            "d03_bob",
            "bob",
            52,
            100,
            "Then move before the timer resets.",
            "bob_48k_stereo.wav",
            48000,
            303,
            118.0,
            2,
        ),
        (
            "d04_alice",
            "alice",
            84,
            116,
            "The relay is already answering us.",
            "alice_44k_tail.wav",
            44100,
            404,
            204.0,
            1,
        ),
        (
            "d05_narrator",
            "narrator",
            104,
            120,
            "And the lights went out.",
            "narrator_48k_tail.wav",
            48000,
            505,
            138.0,
            1,
        ),
    ]
    script_dialogues = []
    for (
        identifier,
        speaker,
        start_frame,
        end_frame,
        text,
        filename,
        sample_rate,
        seed,
        frequency,
        channels,
    ) in dialogue_specs:
        duration_seconds = (end_frame - start_frame) / float(fps)
        _synth_voice(
            voices_dir / filename,
            sample_rate,
            duration_seconds,
            seed,
            frequency,
            channels,
        )
        script_dialogues.append(
            {
                "id": identifier,
                "speaker": speaker,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "text": text,
                "audio": filename,
            }
        )
    script = {
        "schema_version": 1,
        "fps": "24",
        "duration_frames": duration_frames,
        "dialogues": script_dialogues,
    }

    camera_position = np.array([0.0, 1.5, -6.0], dtype=np.float64)

    def narrator_position(frame: int) -> np.ndarray:
        return np.array([-1.2, 1.5, 5.0], dtype=np.float64)

    def narrator_forward(frame: int) -> np.ndarray:
        return np.array([0.12, 0.0, -1.0], dtype=np.float64)

    def alice_position(frame: int) -> np.ndarray:
        phase = frame / float(duration_frames - 1)
        return np.array(
            [-4.8 + 9.6 * phase, 1.45, 3.2 + 0.5 * math.sin(phase * math.pi)],
            dtype=np.float64,
        )

    def alice_forward(frame: int) -> np.ndarray:
        phase = frame / float(duration_frames - 1)
        return np.array(
            [math.sin(phase * math.pi * 1.2), -0.03, -math.cos(phase * math.pi * 1.2)],
            dtype=np.float64,
        )

    def bob_position(frame: int) -> np.ndarray:
        phase = frame / float(duration_frames - 1)
        return np.array(
            [2.2 + 0.7 * math.sin(phase * math.pi * 2.0), 1.35, 11.0 - 8.6 * phase],
            dtype=np.float64,
        )

    def bob_forward(frame: int) -> np.ndarray:
        phase = frame / float(duration_frames - 1)
        return np.array(
            [-0.25 + 0.25 * math.cos(phase * math.pi), -0.02, -1.0],
            dtype=np.float64,
        )

    scene = {
        "schema_version": 1,
        "fps": "24",
        "duration_frames": duration_frames,
        "camera": {
            "position": [float(value) for value in camera_position],
            "forward": [0.0, -0.035, 1.0],
            "up": [0.0, 1.0, 0.0],
        },
        "tracks": {
            "alice": _track(duration_frames, alice_position, alice_forward),
            "bob": _track(duration_frames, bob_position, bob_forward),
            "narrator": _track(duration_frames, narrator_position, narrator_forward),
        },
    }
    project = {
        "schema_version": 1,
        "fps": "24",
        "script": "script.json",
        "scene": "scene.json",
        "audio_root": "voices",
        "sample_rate": 48000,
        "target_lufs": -10.0,
        "true_peak_dbtp": -1.0,
        "distance_reference_m": 1.5,
        "pan_strength": 0.9,
        "fade_ms": 5.0,
        "speed_of_sound_mps": 343.0,
        "limiter": {
            "lookahead_ms": 5.0,
            "release_ms": 80.0,
            "true_peak_oversample": 4,
        },
    }
    for name, document in (
        ("script.json", script),
        ("scene.json", scene),
        ("project.json", project),
    ):
        (output_dir / name).write_text(
            json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return output_dir / "project.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the deterministic demo project.")
    parser.add_argument(
        "--out",
        default=str(ROOT / "demo_project"),
        help="Directory for generated project assets.",
    )
    args = parser.parse_args()
    project = generate_demo(Path(args.out))
    print(project)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

