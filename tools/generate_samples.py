#!/usr/bin/env python3
"""Deterministic sample generator.

Produces all WAV assets (BGM layers + skill SFX) and the engine config
(skill timeline + per-frame combo states) from a fixed seed, so repeated
runs yield byte-identical files.

Usage:
    python3 tools/generate_samples.py [output_dir]   # default: assets
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import soundfile as sf

SEED = 20261117
BPM = 120.0
FIRST_BEAT_OFFSET = 0.05
BARS = 4
BEATS_PER_BAR = 4
COMBO_FPS = 30


def _write(path, data, sr):
    sf.write(path, np.clip(data, -1.0, 1.0), sr, subtype="PCM_16")


def _bar_times(sr):
    bar_s = BEATS_PER_BAR * 60.0 / BPM
    n = int(round(BARS * bar_s * sr))
    return np.arange(n) / sr, n


def gen_base_layer(sr):
    """Stereo chord pad (Am add9), integer number of bars."""
    t, _ = _bar_times(sr)
    bar_s = BEATS_PER_BAR * 60.0 / BPM
    phase = (t % bar_s) / bar_s
    swell = 0.6 + 0.4 * np.sin(2 * np.pi * phase - np.pi / 2)
    freqs = [220.0, 261.63, 329.63, 493.88]
    left = np.zeros_like(t)
    right = np.zeros_like(t)
    for i, f in enumerate(freqs):
        detune = 1.0 + 0.0015 * (i + 1)
        left += np.sin(2 * np.pi * f * t + i * 0.7)
        right += np.sin(2 * np.pi * f * detune * t + i * 0.9)
    layer = 0.25 / len(freqs) * swell
    return np.stack([left * layer, right * layer], axis=1)


def gen_percussion_layer(sr, rng):
    """Stereo drum pattern: kick on beats, snare on 2/4, hats on 8ths."""
    t, n = _bar_times(sr)
    beat_s = 60.0 / BPM
    out = np.zeros((n, 2))
    total_beats = BARS * BEATS_PER_BAR

    def place(start_s, sig_l, sig_r):
        i0 = int(round(start_s * sr))
        i1 = min(n, i0 + sig_l.size)
        out[i0:i1, 0] += sig_l[: i1 - i0]
        out[i0:i1, 1] += sig_r[: i1 - i0]

    kick_len = int(0.18 * sr)
    kt = np.arange(kick_len) / sr
    kick = np.sin(2 * np.pi * (120 - 70 * kt / 0.18) * kt) * np.exp(-kt * 25)
    snare_len = int(0.12 * sr)
    st = np.arange(snare_len) / sr
    snare = rng.standard_normal(snare_len) * np.exp(-st * 40) * 0.6
    hat_len = int(0.04 * sr)
    ht = np.arange(hat_len) / sr
    hat = rng.standard_normal(hat_len) * np.exp(-ht * 120) * 0.25

    for b in range(total_beats):
        place(b * beat_s, 0.8 * kick, 0.8 * kick)
        if b % 4 in (1, 3):
            place(b * beat_s, snare, np.roll(snare, 7))
        place(b * beat_s + beat_s / 2, hat, hat)
        place(b * beat_s, hat * 0.7, hat * 0.7)
    return 0.5 * out


def gen_ambience_layer(sr, rng):
    """Stereo filtered-noise ambience with a slow 2-bar swell."""
    t, n = _bar_times(sr)
    noise_l = rng.standard_normal(n)
    noise_r = rng.standard_normal(n)

    def lowpass(x):
        y = np.empty_like(x)
        acc = 0.0
        for i in range(n):
            acc += 0.05 * (x[i] - acc)
            y[i] = acc
        return y

    two_bar = 2 * BEATS_PER_BAR * 60.0 / BPM
    phase = (t % two_bar) / two_bar
    swell = 0.5 - 0.5 * np.cos(2 * np.pi * phase)
    left = lowpass(noise_l) * swell
    right = lowpass(noise_r) * np.roll(swell, int(0.25 * sr))
    return 0.2 * np.stack([left, right], axis=1)


def gen_sfx_hit(sr, rng):
    """Mono punchy hit, 0.35 s."""
    n = int(0.35 * sr)
    t = np.arange(n) / sr
    body = np.sin(2 * np.pi * (300 - 220 * t / 0.35) * t) * np.exp(-t * 18)
    click = rng.standard_normal(n) * np.exp(-t * 200) * 0.5
    return 0.9 * (body + click)


def gen_sfx_slash(sr, rng):
    """Mono noisy slash, 0.28 s (high-passed noise sweep)."""
    n = int(0.28 * sr)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n)
    hp = np.diff(noise, prepend=0.0)
    env = np.sin(np.pi * np.minimum(t / 0.28, 1.0)) ** 2
    return 0.8 * hp * env


def gen_sfx_magic(sr):
    """Mono shimmering arpeggio, 0.6 s."""
    n = int(0.6 * sr)
    t = np.arange(n) / sr
    out = np.zeros(n)
    for i, f in enumerate([660.0, 880.0, 1320.0, 1760.0]):
        start = int(i * 0.08 * sr)
        seg_t = np.arange(n - start) / sr
        out[start:] += np.sin(2 * np.pi * f * seg_t) * np.exp(-seg_t * 8) * 0.4
    return 0.85 * out * np.exp(-t * 2)


def gen_events(rng):
    """Skill timeline near 1/16 grid points (jitter <= 4 ms)."""
    step = 60.0 / BPM / 4  # 1/16 note

    def at(k, sfx, gain_db, pan, grid="1/16"):
        grid_point = FIRST_BEAT_OFFSET + k * step
        jitter = float(rng.uniform(-0.004, 0.004))
        return {
            "time": round(grid_point + jitter, 6),
            "sfx": sfx,
            "gain_db": gain_db,
            "pan": pan,
            "grid": grid,
        }

    return [
        at(8, "hit", -2.0, -0.4),
        at(12, "slash", 0.0, 0.3),
        at(16, "magic", -4.0, 0.0, grid="1/4"),
        at(20, "hit", 0.0, 0.5),
        at(21, "hit", -6.0, -0.5),
        at(28, "hit", 0.0, -0.2),
        at(28, "slash", -3.0, 0.2),
        at(36, "magic", -2.0, 0.0, grid="1/8"),
        at(44, "hit", 0.0, 0.6),
        at(52, "slash", -2.0, -0.6),
        at(60, "hit", 0.0, 0.0),
        at(64, "magic", -1.0, 0.1),
    ]


def gen_combo_states(n_frames):
    """Deterministic combo curve covering all three layer combinations."""
    states = []
    for f in range(n_frames):
        if f < 75:
            states.append(2)
        elif f < 135:
            states.append(7)
        elif f < 225:
            states.append(13)
        elif f < 270:
            states.append(8)
        else:
            states.append(11)
    return states


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "assets"
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    _write(os.path.join(out_dir, "base_layer.wav"), gen_base_layer(44100), 44100)
    _write(os.path.join(out_dir, "percussion_layer.wav"), gen_percussion_layer(48000, rng), 48000)
    _write(os.path.join(out_dir, "ambience_layer.wav"), gen_ambience_layer(44100, rng), 44100)
    _write(os.path.join(out_dir, "sfx_hit.wav"), gen_sfx_hit(44100, rng)[:, None], 44100)
    _write(os.path.join(out_dir, "sfx_slash.wav"), gen_sfx_slash(48000, rng)[:, None], 48000)
    _write(os.path.join(out_dir, "sfx_magic.wav"), gen_sfx_magic(44100)[:, None], 44100)

    events = gen_events(rng)
    last_grid = FIRST_BEAT_OFFSET + 64 * (60.0 / BPM / 4)
    total_s = last_grid + 2.0 + 0.6
    n_frames = int(np.ceil(total_s * COMBO_FPS))

    config = {
        "bpm": BPM,
        "first_beat_offset": FIRST_BEAT_OFFSET,
        "duck_gain_db": -6.0,
        "layers": {
            "base": "base_layer.wav",
            "percussion": "percussion_layer.wav",
            "ambience": "ambience_layer.wav",
        },
        "sfx": {
            "hit": "sfx_hit.wav",
            "slash": "sfx_slash.wav",
            "magic": "sfx_magic.wav",
        },
        "combo_fps": COMBO_FPS,
        "combo_states": gen_combo_states(n_frames),
        "events": events,
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"samples + config written to {out_dir}/")


if __name__ == "__main__":
    main()
