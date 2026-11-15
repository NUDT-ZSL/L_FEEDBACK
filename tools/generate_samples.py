"""Generate deterministic multi-format test samples and a manifest.

All signals are synthesized from seeded numpy generators; no network or
recorded material is used. Re-running produces identical files.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SAMPLES = os.path.join(ROOT, "samples")
TMP = os.path.join(SAMPLES, "_tmp")


def tone(fs: float, dur: float, freq: float, amp: float, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(fs * dur)) / fs
    return amp * np.sin(2 * np.pi * freq * t + phase)


def probe_start_ms(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_entries", "format=start_time", path],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    return round(float(info["format"]["start_time"]) * 1000.0, 3)


def main() -> None:
    os.makedirs(TMP, exist_ok=True)
    manifest: list[dict] = []

    def add(path: str, offset_ms: float) -> None:
        manifest.append({"path": os.path.relpath(path, ROOT), "offset_ms": offset_ms})

    # 1. 48 kHz / 16-bit / stereo, moderate level (~-23 LUFS)
    rng = np.random.default_rng(1001)
    fs = 48000
    left = tone(fs, 4.0, 440.0, 0.12) + 0.02 * rng.standard_normal(int(fs * 4.0))
    right = tone(fs, 4.0, 554.0, 0.10) + 0.02 * rng.standard_normal(int(fs * 4.0))
    p = os.path.join(SAMPLES, "tone_48k_16bit_stereo.wav")
    sf.write(p, np.stack([left, right], 1), fs, subtype="PCM_16")
    add(p, 0)

    # 2. 44.1 kHz / 24-bit / mono, louder (~-17 LUFS)
    rng = np.random.default_rng(1002)
    fs = 44100
    t = np.arange(int(fs * 3.5)) / fs
    mono = 0.4 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t))
    mono += 0.01 * rng.standard_normal(t.size)
    p = os.path.join(SAMPLES, "voice_44k1_24bit_mono.wav")
    sf.write(p, mono, fs, subtype="PCM_24")
    add(p, 250)

    # 3. 96 kHz / 32-bit float / 5.1 surround
    rng = np.random.default_rng(1003)
    fs = 96000
    dur = 5.0
    chans = []
    for i, freq in enumerate([330, 349, 392, 0, 440, 494]):
        if freq == 0:  # LFE
            chans.append(tone(fs, dur, 60.0, 0.05))
        else:
            c = tone(fs, dur, freq, 0.06) + 0.005 * rng.standard_normal(int(fs * dur))
            chans.append(c)
    p = os.path.join(SAMPLES, "surround_96k_float_5_1.wav")
    sf.write(p, np.stack(chans, 1), fs, subtype="FLOAT")
    add(p, 500)

    # 4. 32 kHz / 16-bit / stereo, extremely quiet -> loudness_out_of_range
    rng = np.random.default_rng(1004)
    fs = 32000
    quiet = 0.001 * rng.standard_normal((int(fs * 3.0), 2))
    p = os.path.join(SAMPLES, "quiet_32k_16bit_stereo.wav")
    sf.write(p, quiet, fs, subtype="PCM_16")
    add(p, 0)

    # 5. 48 kHz / 24-bit / stereo: quiet bed + full-scale clicks -> limiter
    rng = np.random.default_rng(1005)
    fs = 48000
    n = int(fs * 4.0)
    hot = 0.001 * rng.standard_normal((n, 2))
    click_idx = np.sort(rng.choice(n, size=20, replace=False))
    hot[click_idx, :] = 0.99
    p = os.path.join(SAMPLES, "hot_clicks_48k_24bit_stereo.wav")
    sf.write(p, hot, fs, subtype="PCM_24")
    add(p, 100)

    # 6. 48 kHz / 16-bit / stereo, very loud clipped mix (~-5 LUFS)
    fs = 48000
    t = np.arange(int(fs * 3.0)) / fs
    loud = np.stack(
        [
            np.clip(0.8 * np.sin(2 * np.pi * 110 * t) + 0.5 * np.sin(2 * np.pi * 165 * t), -0.95, 0.95),
            np.clip(0.8 * np.sin(2 * np.pi * 112 * t) + 0.5 * np.sin(2 * np.pi * 167 * t), -0.95, 0.95),
        ],
        1,
    )
    p = os.path.join(SAMPLES, "loud_48k_16bit_stereo.wav")
    sf.write(p, loud, fs, subtype="PCM_16")
    add(p, 0)

    # 7. OGG with a 2.5 s container start offset; manifest uses the probed
    #    start time so the two agree (no timestamp_mismatch flag)
    fs = 44100
    t = np.arange(int(fs * 3.0)) / fs
    music = np.stack([0.2 * np.sin(2 * np.pi * 523 * t), 0.2 * np.sin(2 * np.pi * 659 * t)], 1)
    tmp_wav = os.path.join(TMP, "music_src.wav")
    sf.write(tmp_wav, music, fs, subtype="PCM_16")
    p = os.path.join(SAMPLES, "offset_music.ogg")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", tmp_wav, "-output_ts_offset", "2.5",
         "-codec:a", "libvorbis", "-q:a", "4", p],
        check=True,
    )
    start_ms = probe_start_ms(p)
    add(p, start_ms)

    # 8. OGG with 1.2 s start offset, manifest says 0 -> timestamp_mismatch
    fs = 48000
    t = np.arange(int(fs * 2.5)) / fs
    voice = 0.25 * np.sin(2 * np.pi * 300 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    tmp_wav = os.path.join(TMP, "voice_src.wav")
    sf.write(tmp_wav, voice, fs, subtype="PCM_16")
    p = os.path.join(SAMPLES, "mismatch_voice.ogg")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", tmp_wav, "-output_ts_offset", "1.2",
         "-codec:a", "libvorbis", "-q:a", "4", p],
        check=True,
    )
    add(p, 0)

    # 9. Garbage bytes with a .wav name -> unreadable
    rng = np.random.default_rng(1009)
    p = os.path.join(SAMPLES, "garbage.wav")
    with open(p, "wb") as fh:
        fh.write(rng.bytes(4096))
    add(p, 0)

    # 10. MP3 with corrupted frames in the middle -> partial_decode
    fs = 44100
    t = np.arange(int(fs * 3.0)) / fs
    src = 0.2 * np.sin(2 * np.pi * 440 * t)
    tmp_wav = os.path.join(TMP, "corrupt_src.wav")
    sf.write(tmp_wav, src, fs, subtype="PCM_16")
    good_mp3 = os.path.join(TMP, "corrupt_full.mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", tmp_wav, "-codec:a", "libmp3lame", "-q:a", "4", good_mp3],
        check=True,
    )
    with open(good_mp3, "rb") as fh:
        data = bytearray(fh.read())
    rng = np.random.default_rng(1010)
    mid = len(data) // 2
    for i in range(mid, mid + 512):
        data[i] = rng.integers(0, 256, dtype=np.uint8)
    p = os.path.join(SAMPLES, "corrupted_frames.mp3")
    with open(p, "wb") as fh:
        fh.write(bytes(data))
    add(p, 0)

    with open(os.path.join(ROOT, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"files": manifest}, fh, indent=2)
    print(f"wrote {len(manifest)} manifest entries; samples in {SAMPLES}")


if __name__ == "__main__":
    sys.exit(main())
