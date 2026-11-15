"""Shared helpers for building deterministic fixtures inside temp dirs."""

import json
import os
import sys
import tempfile

# Make the repository root importable when tests run from any working dir.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from make_samples import generate_samples  # noqa: E402


def make_workspace():
    """Create a temp workspace populated with the deterministic WAV set."""
    root = tempfile.mkdtemp(prefix="seamless_loop_test_")
    samples_dir = os.path.join(root, "samples")
    output_dir = os.path.join(root, "output")
    paths = generate_samples(samples_dir)
    return root, samples_dir, output_dir, paths


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def base_project(samples_dir, output_dir, **overrides):
    project = {
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
    }
    project.update(overrides)
    return project
