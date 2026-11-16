"""Shared helpers for the deterministic test suite."""

import json
import os
import subprocess
import sys
import tempfile

import soundfile as sf

from rhythm_mixer.make_samples import generate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_fixture(bpm=120.0):
    base = tempfile.mkdtemp(prefix="rm_fixture_")
    input_dir = os.path.join(base, "input")
    output_dir = os.path.join(base, "output")
    generate(input_dir, bpm=bpm)
    return input_dir, output_dir


def run_cli(input_dir, output_dir):
    env = dict(os.environ)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "rhythm_mixer",
            "--input-dir",
            input_dir,
            "--output-dir",
            output_dir,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )


def write_json(path, payload):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle)


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_wav(path, samples, samplerate=44100, subtype="PCM_16"):
    sf.write(path, samples, samplerate, subtype=subtype)


def read_output(output_dir):
    wav_path = os.path.join(output_dir, "mix_output.wav")
    report_path = os.path.join(output_dir, "beat_alignment_report.json")
    data, samplerate = sf.read(wav_path, dtype="float32")
    info = sf.info(wav_path)
    return data, info, read_json(report_path), wav_path, report_path
