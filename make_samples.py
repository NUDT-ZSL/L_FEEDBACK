"""Generate deterministic WAV samples and JSON project files for testing.

Everything is synthesised from fixed sine/triangle formulas and a seeded RNG,
so repeated runs produce byte-identical files and no external audio data is
required.  Run directly::

    python make_samples.py

or call :func:`generate_samples` / :func:`generate_projects` from the tests.
"""

import json
import os

import numpy as np
import soundfile as sf

SR_48K = 48000
SR_44K1 = 44100


def _time_axis(duration, sample_rate):
    return np.arange(int(round(duration * sample_rate)), dtype=np.float64) / sample_rate


def _sine(freq, duration, sample_rate, amplitude=0.5, phase=0.0):
    t = _time_axis(duration, sample_rate)
    return amplitude * np.sin(2.0 * np.pi * freq * t + phase)


def _drums(duration, sample_rate):
    """Stereo 16-bit percussive-ish bed (two detuned harmonics + pulse)."""
    t = _time_axis(duration, sample_rate)
    left = (
        0.45 * np.sin(2 * np.pi * 110 * t)
        + 0.25 * np.sin(2 * np.pi * 220 * t + 0.7)
        + 0.10 * np.sign(np.sin(2 * np.pi * 55 * t))
    )
    right = (
        0.40 * np.sin(2 * np.pi * 165 * t + 1.3)
        + 0.20 * np.sin(2 * np.pi * 330 * t)
        + 0.10 * np.sign(np.sin(2 * np.pi * 82.5 * t + 0.5))
    )
    return np.stack([left, right], axis=1)


def _bass(duration, sample_rate):
    """Mono 24-bit bass line with a seeded deterministic overtone."""
    rng = np.random.default_rng(seed=20261116)
    t = _time_axis(duration, sample_rate)
    fundamental = _sine(55, duration, sample_rate, amplitude=0.5)
    overtone = 0.2 * np.sin(2 * np.pi * 110 * t + 0.4)
    grit = 0.05 * rng.standard_normal(t.shape)
    return fundamental + overtone + grit


def _pad(duration, sample_rate):
    """Stereo 32-bit float pad with a slow amplitude LFO."""
    t = _time_axis(duration, sample_rate)
    lfo_left = 0.85 + 0.15 * np.sin(2 * np.pi * 0.7 * t)
    lfo_right = 0.85 + 0.15 * np.sin(2 * np.pi * 0.7 * t + np.pi / 2)
    left = lfo_left * (
        0.4 * np.sin(2 * np.pi * 220 * t)
        + 0.2 * np.sin(2 * np.pi * 440 * t + 0.9)
    )
    right = lfo_right * (
        0.4 * np.sin(2 * np.pi * 330 * t + 0.6)
        + 0.2 * np.sin(2 * np.pi * 660 * t)
    )
    return np.stack([left, right], axis=1)


def generate_samples(samples_dir, duration=3.0):
    """Write the deterministic WAV fixture set into ``samples_dir``."""
    os.makedirs(samples_dir, exist_ok=True)

    drums = _drums(duration, SR_48K)
    bass = _bass(duration, SR_48K)
    pad = _pad(duration, SR_48K)

    paths = {
        "drums": os.path.join(samples_dir, "drums_stereo_48k_pcm16.wav"),
        "bass": os.path.join(samples_dir, "bass_mono_48k_pcm24.wav"),
        "pad": os.path.join(samples_dir, "pad_stereo_48k_float.wav"),
        "mismatch": os.path.join(samples_dir, "mismatch_mono_44k1_pcm16.wav"),
        "unsupported": os.path.join(samples_dir, "unsupported_48k_u8.wav"),
        "short": os.path.join(samples_dir, "short_mono_48k_pcm16.wav"),
    }

    sf.write(paths["drums"], drums, SR_48K, subtype="PCM_16")
    sf.write(paths["bass"], bass, SR_48K, subtype="PCM_24")
    sf.write(paths["pad"], pad, SR_48K, subtype="FLOAT")
    sf.write(
        paths["mismatch"],
        _sine(99, duration, SR_44K1, amplitude=0.4),
        SR_44K1,
        subtype="PCM_16",
    )
    sf.write(
        paths["unsupported"],
        _sine(120, duration, SR_48K, amplitude=0.4),
        SR_48K,
        subtype="PCM_U8",
    )
    sf.write(
        paths["short"],
        _sine(70, 0.5, SR_48K, amplitude=0.4),
        SR_48K,
        subtype="PCM_16",
    )
    return paths


def generate_projects(samples_dir, output_dir):
    """Write happy-path and overwrite-protected JSON project files."""
    projects = {
        "project_demo.json": {
            "project_name": "demo",
            "input_files": [
                "drums_stereo_48k_pcm16.wav",
                "bass_mono_48k_pcm24.wav",
                "pad_stereo_48k_float.wav",
            ],
            "loop_start": 0.25,
            "loop_end": 1.75,
            "crossfade_duration": 0.1,
            "target_duration": 4.5,
            "output_dir": os.path.relpath(output_dir, samples_dir),
            "overwrite": True,
        },
        "project_keep.json": {
            "project_name": "keep",
            "input_files": [
                "drums_stereo_48k_pcm16.wav",
                "bass_mono_48k_pcm24.wav",
            ],
            "loop_start": 0.25,
            "loop_end": 1.75,
            "crossfade_duration": 0.1,
            "target_duration": 3.0,
            "output_dir": os.path.relpath(output_dir, samples_dir),
            "overwrite": False,
        },
    }
    written = {}
    for name, payload in projects.items():
        path = os.path.join(samples_dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        written[name] = path
    return written


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(repo_root, "samples")
    output_dir = os.path.join(repo_root, "output")
    generate_samples(samples_dir)
    generate_projects(samples_dir, output_dir)
    print("generated deterministic samples in %s" % samples_dir)


if __name__ == "__main__":
    main()
