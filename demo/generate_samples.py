"""Deterministically generate all local test audio assets and manifests.

Everything is derived from a fixed-seed NumPy generator plus known sine
frequencies, so repeated runs produce byte-identical files.  Sources are
created at mixed sample rates / channel counts (mono 22.05 kHz, stereo
44.1 kHz, stereo 48 kHz) in both WAV and FLAC to exercise the decoder.
"""

from __future__ import annotations

import json
import os
import subprocess
import wave

import numpy as np

SEED = 20261116


def _write_wav(path: str, audio: np.ndarray, rate: int) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = np.rint(pcm * 32767).astype("<i2")
    with wave.open(path, "wb") as wav:
        wav.setnchannels(audio.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm16.tobytes())


def _to_flac(wav_path: str, flac_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path, flac_path],
        check=True,
    )


def tone(freq: float, seconds: float, rate: int, channels: int, amp=0.5) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    sig = amp * np.sin(2 * np.pi * freq * t)
    sig *= 0.85 + 0.15 * np.sin(2 * np.pi * 0.5 * t)
    return np.tile(sig[:, None], (1, channels)).astype(np.float32)


def noise(seconds: float, rate: int, channels: int, rng, amp=0.2) -> np.ndarray:
    return rng.normal(0.0, amp, size=(int(seconds * rate), channels)).astype(np.float32)


def generate(out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    paths: dict[str, str] = {}

    assets = {
        "dialog_a": (tone(220.0, 3.0, 44100, 1, 0.5), 44100, "wav"),
        "dialog_b": (tone(277.0, 4.0, 48000, 2, 0.45), 48000, "flac"),
        "sfx_blip": (tone(880.0, 0.3, 48000, 2, 0.6), 48000, "wav"),
        "sfx_noise": (noise(2.0, 22050, 1, rng, 0.25), 22050, "flac"),
        "music_loop": (tone(110.0, 6.0, 44100, 2, 0.4), 44100, "wav"),
        "music_long": (tone(164.8, 12.0, 48000, 2, 0.35), 48000, "flac"),
        "ambient": (noise(8.0, 48000, 2, rng, 0.12), 48000, "wav"),
        "quiet_clip": (tone(330.0, 0.2, 48000, 2, 0.02), 48000, "wav"),
    }
    for name, (audio, rate, kind) in assets.items():
        wav_path = os.path.join(out_dir, f"{name}.wav")
        _write_wav(wav_path, audio, rate)
        if kind == "flac":
            flac_path = os.path.join(out_dir, f"{name}.flac")
            _to_flac(wav_path, flac_path)
            os.remove(wav_path)
            paths[name] = f"{name}.flac"
        else:
            paths[name] = f"{name}.wav"

    broken = os.path.join(out_dir, "broken.flac")
    with open(broken, "wb") as handle:
        handle.write(b"fLaC" + bytes(rng.integers(0, 255, 64, dtype=np.uint8)))
    paths["broken"] = "broken.flac"

    mp4 = os.path.join(out_dir, "unsupported.mp4")
    with open(mp4, "wb") as handle:
        handle.write(b"\x00\x00\x00\x18ftypmp42" + bytes(64))
    paths["unsupported"] = "unsupported.mp4"
    return paths


def build_manifest(paths: dict[str, str]) -> dict:
    """Mixed timeline exercising every requirement."""
    events = [
        {"id": "d1", "path": paths["dialog_a"], "start_ms": 0, "category": "dialog",
         "duration_ms": 3000, "priority": 5},
        {"id": "d2", "path": paths["dialog_b"], "start_ms": 2000, "category": "dialog",
         "duration_ms": 3000, "priority": 8},
        {"id": "d3", "path": paths["dialog_a"], "start_ms": 6000, "category": "dialog",
         "duration_ms": 2000, "priority": 4},
        {"id": "d4", "path": paths["dialog_b"], "start_ms": 7000, "category": "dialog",
         "duration_ms": 2000, "priority": 4},
        {"id": "m1", "path": paths["music_long"], "start_ms": 10000, "category": "music",
         "loop": True, "duration_ms": 20000, "priority": 1, "overlap": "mix"},
        {"id": "d5", "path": paths["dialog_b"], "start_ms": 12000, "category": "dialog",
         "duration_ms": 2500, "priority": 9},
        {"id": "s1", "path": paths["sfx_blip"], "start_ms": 32000, "category": "sfx",
         "loop": True, "offset_ms": 50, "duration_ms": 1500, "priority": 3},
        {"id": "s2", "path": paths["sfx_noise"], "start_ms": 35000, "category": "sfx",
         "duration_ms": 900, "priority": 2},
        {"id": "s3", "path": paths["quiet_clip"], "start_ms": 38000, "category": "sfx",
         "priority": 2},
        {"id": "m2", "path": paths["ambient"], "start_ms": 40000, "category": "music",
         "loop": True, "duration_ms": 30000, "priority": 0, "overlap": "mix"},
        {"id": "m3", "path": paths["music_loop"], "start_ms": 70000, "category": "music",
         "loop": True, "duration_ms": 45000, "priority": 1},
        {"id": "bad1", "path": paths["broken"], "start_ms": 5000, "category": "sfx",
         "duration_ms": 1000, "priority": 1},
        {"id": "bad2", "path": paths["unsupported"], "start_ms": 5000,
         "category": "sfx", "duration_ms": 1000},
        {"id": "bad3", "path": "does_not_exist.wav", "start_ms": 5000,
         "category": "sfx", "duration_ms": 1000},
        {"id": "bad4", "path": paths["dialog_a"], "start_ms": -5,
         "category": "dialog", "duration_ms": 1000},
        {"id": "bad5", "path": paths["sfx_blip"], "start_ms": 9000,
         "category": "sfx", "offset_ms": 9000, "duration_ms": 500},
    ]
    return {"total_duration_ms": 120_000, "events": events}


def build_long_manifest(paths: dict[str, str], total_ms: int = 600_000) -> dict:
    """Dense ~10 minute workload for the performance / determinism check."""
    rng = np.random.default_rng(SEED + 1)
    pool = ["dialog_a", "dialog_b", "sfx_blip", "sfx_noise", "music_loop", "ambient"]
    events = []
    cursor = 0
    idx = 0
    while cursor < total_ms - 2000:
        name = pool[int(rng.integers(0, len(pool)))]
        dur = int(rng.choice([800, 1500, 3000, 6000]))
        start = max(0, cursor + int(rng.choice([0, 200, -300, 500])))
        loop = name in ("music_loop", "ambient") and dur > 3000
        category = (
            "music" if name in ("music_loop", "ambient")
            else ("dialog" if "dialog" in name else "sfx")
        )
        events.append(
            {
                "id": f"e{idx:04d}",
                "path": paths[name],
                "start_ms": start,
                "duration_ms": dur,
                "category": category,
                "priority": int(rng.choice([0, 1, 3, 5, 8])),
                "loop": bool(loop),
                "overlap": "mix" if name == "ambient" else None,
            }
        )
        cursor += int(rng.choice([700, 1000, 2000]))
        idx += 1
    events.append(
        {"id": "failx", "path": paths["broken"], "start_ms": 1000,
         "category": "sfx", "duration_ms": 500}
    )
    return {"total_duration_ms": total_ms, "events": events}


def main(root: str = "demo/workspace") -> None:
    assets_dir = os.path.join(root, "assets")
    paths = generate(assets_dir)
    paths = {name: os.path.join("assets", rel) for name, rel in paths.items()}
    with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(build_manifest(paths), fh, ensure_ascii=False, indent=2)
    with open(os.path.join(root, "manifest_long.json"), "w", encoding="utf-8") as fh:
        json.dump(build_long_manifest(paths), fh, ensure_ascii=False, indent=2)
    print(f"样本与清单已生成于 {root}")


if __name__ == "__main__":
    main()
