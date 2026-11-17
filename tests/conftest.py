import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiotl.samplegen import write_tone  # noqa: E402


@pytest.fixture()
def sample_set(tmp_path):
    """Deterministic multi-rate, multi-channel sample set + manifest."""
    samples = tmp_path / "samples"
    samples.mkdir()
    specs = [
        ("a_44k_mono", 44100, 1, 0.80, 220.0, 0, 200, "linear"),
        ("b_48k_stereo", 48000, 2, 1.00, 110.0, 200, 250, "equal_power"),
        ("c_44k_stereo", 44100, 2, 0.60, 330.0, 250, 0, "linear"),
    ]
    segments = []
    for seg_id, rate, ch, secs, freq, fi, fo, curve in specs:
        name = f"{seg_id}.wav"
        write_tone(str(samples / name), rate, ch, secs, freq)
        segments.append(
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
        "segments": segments,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "dir": tmp_path,
        "samples": samples,
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


def run_cli(manifest_path, out_dir, workers=None, state_dir=None):
    from audiotl.cli import main

    argv = [str(manifest_path), str(out_dir)]
    if workers is not None:
        argv += ["--workers", str(workers)]
    if state_dir is not None:
        argv += ["--state-dir", str(state_dir)]
    return main(argv)


def read_report(out_dir):
    with open(os.path.join(out_dir, "report.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def mix_bytes(out_dir):
    with open(os.path.join(out_dir, "mix.wav"), "rb") as fh:
        return fh.read()
