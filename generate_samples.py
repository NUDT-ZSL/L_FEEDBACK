"""Generate deterministic sample WAVs and descriptors for the mixer.

Usage: python generate_samples.py --dir samples

Everything is derived from fixed seeds / pure tones, so repeated runs
produce byte-identical inputs and reproducible mixes.
"""

import argparse
import json
import os

import numpy as np
import soundfile as sf

SEED = 20261116


def _tone(freq, seconds, rate, amplitude):
    t = np.arange(int(seconds * rate), dtype=np.float64) / rate
    return amplitude * np.sin(2.0 * np.pi * freq * t)


def make_dialog(path):
    """44100 Hz mono 16-bit, 2.0 s: voice-like harmonic stack."""
    rate = 44100
    signal = (_tone(220, 2.0, rate, 0.30)
              + _tone(440, 2.0, rate, 0.15)
              + _tone(660, 2.0, rate, 0.05))
    sf.write(path, signal, rate, subtype="PCM_16")
    return {"rate": rate, "channels": 1, "seconds": 2.0}


def make_sfx(path):
    """48000 Hz stereo 24-bit, 1.5 s: 0.4 s silence then a noise+tone burst."""
    rate = 48000
    rng = np.random.default_rng(SEED)
    burst_len = int(1.1 * rate)
    burst = 0.25 * rng.standard_normal(burst_len) + _tone(1000, 1.1, rate, 0.30)
    burst = np.clip(burst, -0.9, 0.9)
    stereo = np.stack([burst, np.roll(burst, 7)], axis=1)
    signal = np.vstack([np.zeros((int(0.4 * rate), 2)), stereo])
    sf.write(path, signal, rate, subtype="PCM_24")
    return {"rate": rate, "channels": 2, "seconds": 1.5, "onset_sample": int(0.4 * rate)}


def make_music(path):
    """32000 Hz stereo 16-bit, 3.0 s: quiet pad chord."""
    rate = 32000
    chord = (_tone(261.63, 3.0, rate, 0.10)
             + _tone(329.63, 3.0, rate, 0.10)
             + _tone(392.00, 3.0, rate, 0.10))
    stereo = np.stack([chord, chord * 0.8], axis=1)
    sf.write(path, stereo, rate, subtype="PCM_16")
    return {"rate": rate, "channels": 2, "seconds": 3.0}


def build_descriptor():
    return {
        "frame_rate": 30.0,
        "output": {"sample_rate": 48000, "bit_depth": 16, "channels": 2},
        "crossfade_ms": 10.0,
        "fade_ms": 5.0,
        "gain_limits_db": {"min": -20.0, "max": 10.0},
        "tracks": [
            {
                "id": "dialog",
                "path": "dialog.wav",
                "events": [
                    {"id": "line_a", "track_sample": 0, "frame": 0},
                    # 0.5 s into the source, anchored at frame 10 (0.333 s):
                    # the first segment overruns this anchor -> crossfade.
                    {"id": "line_b", "track_sample": 22050, "frame": 10},
                ],
            },
            {
                "id": "sfx",
                "path": "sfx.wav",
                "events": [
                    {"id": "explosion", "track_sample": 19200, "frame": 60},
                ],
            },
            {
                "id": "music",
                "path": "music.wav",
                "events": [
                    {"id": "theme", "track_sample": 0, "frame": 15},
                ],
            },
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="samples",
                        help="Output directory for WAVs and descriptors.")
    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    make_dialog(os.path.join(args.dir, "dialog.wav"))
    make_sfx(os.path.join(args.dir, "sfx.wav"))
    make_music(os.path.join(args.dir, "music.wav"))

    descriptor = build_descriptor()
    _dump(os.path.join(args.dir, "descriptor.json"), descriptor)

    fail = build_descriptor()
    fail["tracks"].append({
        "id": "missing",
        "path": "does_not_exist.wav",
        "events": [{"id": "ghost", "track_sample": 0, "frame": 0}],
    })
    _dump(os.path.join(args.dir, "descriptor_load_fail.json"), fail)

    invalid = build_descriptor()
    invalid["frame_rate"] = 0
    invalid["output"]["sample_rate"] = 4000
    invalid["output"]["bit_depth"] = 20
    invalid["gain_limits_db"] = {"min": 5.0, "max": -5.0}
    invalid["tracks"][0]["events"][0]["track_sample"] = -1
    _dump(os.path.join(args.dir, "descriptor_invalid.json"), invalid)

    print("samples written to %s" % os.path.abspath(args.dir))


def _dump(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
