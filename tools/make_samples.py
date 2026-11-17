"""Generate deterministic local WAV samples and an example manifest.

Usage: python tools/make_samples.py [output_dir]
Creates <output_dir>/samples/*.wav and <output_dir>/manifest.json.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiotl.samplegen import write_tone  # noqa: E402

SEGMENTS = [
    # id, file, rate, channels, seconds, freq, fade_in_ms, fade_out_ms, curve
    ("voice_44k_mono", "voice_44k_mono.wav", 44100, 1, 1.20, 220.0, 0, 250, "linear"),
    ("ambience_48k_stereo", "ambience_48k_stereo.wav", 48000, 2, 1.50, 110.0, 250, 300, "equal_power"),
    ("music_44k_stereo", "music_44k_stereo.wav", 44100, 2, 1.00, 330.0, 300, 0, "linear"),
]


def build(base_dir):
    samples_dir = os.path.join(base_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    manifest_segments = []
    for seg_id, name, rate, channels, seconds, freq, fi, fo, curve in SEGMENTS:
        write_tone(os.path.join(samples_dir, name), rate, channels, seconds, freq)
        manifest_segments.append(
            {
                "id": seg_id,
                "path": os.path.join("samples", name),
                "fade_in": {"length_ms": fi, "curve": curve},
                "fade_out": {"length_ms": fo, "curve": curve},
            }
        )
    manifest = {
        "version": 1,
        "output": {"sample_rate": 48000, "channels": 2, "window_ms": 100},
        "segments": manifest_segments,
    }
    manifest_path = os.path.join(base_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "example"
    path = build(out)
    print(f"wrote {path}")
