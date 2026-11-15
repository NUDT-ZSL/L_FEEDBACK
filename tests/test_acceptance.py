"""Acceptance tests for the loudnorm-batch tool.

Runs the deterministic sample set end to end and checks loudness
accuracy against the ffmpeg ebur128 reference, true-peak ceilings,
alignment, output format, anomaly flags and report reproducibility.

Run directly (python3 tests/test_acceptance.py) or via pytest.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import soundfile as sf

from loudnorm_batch import truepeak
from loudnorm_batch.pipeline import OUT_RATE
from loudnorm_batch.scheduler import run_ordered

MANIFEST = os.path.join(ROOT, "manifest.json")
OUT_DIR = os.path.join(ROOT, "out")
REPORT = os.path.join(ROOT, "report.json")
TARGET_LUFS = -23.0
CEILING = -1.0

_report = None
_manifest = None


def _ensure_run():
    global _report, _manifest
    if _report is not None:
        return
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "generate_samples.py")],
        check=True, cwd=ROOT,
    )
    subprocess.run(
        [sys.executable, "-m", "loudnorm_batch", "--manifest", MANIFEST,
         "--out-dir", OUT_DIR, "--report", REPORT],
        check=True, cwd=ROOT,
    )
    with open(REPORT) as fh:
        _report = json.load(fh)
    with open(MANIFEST) as fh:
        _manifest = json.load(fh)["files"]


def _entry(name):
    _ensure_run()
    for f in _report["files"]:
        if f["input"].endswith(name):
            return f
    raise AssertionError(f"{name} not in report")


def _ffmpeg_reference(path):
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", path,
         "-filter_complex", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True, cwd=ROOT,
    )
    tail = proc.stderr[proc.stderr.rfind("Summary:"):]
    loud = float(re.search(r"I:\s+(-?[\d.]+) LUFS", tail).group(1))
    peaks = [float(x) for x in re.findall(r"Peak:\s+(-?[\d.]+) dBFS", tail)]
    return loud, max(peaks) if peaks else None


def test_all_files_reported_in_manifest_order():
    _ensure_run()
    reported = [f["input"] for f in _report["files"]]
    expected = [e["path"] for e in _manifest]
    assert reported == expected, f"order/content mismatch: {reported} vs {expected}"
    assert [f["index"] for f in _report["files"]] == list(range(len(expected)))


def test_loudness_within_0_1_lu_of_reference():
    _ensure_run()
    for f in _report["files"]:
        if f["status"] == "unreadable":
            continue
        ref_loud, _ = _ffmpeg_reference(f["input"])
        ours = f["measured"]["integrated_lufs"]
        assert abs(ours - ref_loud) <= 0.1, (
            f"{f['input']}: ours {ours} vs ref {ref_loud}"
        )


def test_true_peak_measurement_close_to_reference():
    _ensure_run()
    for f in _report["files"]:
        if f["status"] == "unreadable":
            continue
        _, ref_tp = _ffmpeg_reference(f["input"])
        ours = f["measured"]["true_peak_dbtp"]
        assert abs(ours - ref_tp) <= 0.3, (
            f"{f['input']}: ours {ours} vs ref {ref_tp}"
        )


def test_output_format_and_length():
    _ensure_run()
    total_ms = _report["parameters"]["total_duration_ms"]
    expected_samples = int(round(total_ms * OUT_RATE / 1000.0))
    for f in _report["files"]:
        if f["status"] != "ok":
            continue
        out = f["output"]
        assert out["sample_rate"] == 48000
        assert out["bit_depth"] == 24
        assert out["channels"] == 2
        assert out["samples"] == expected_samples
        data, rate = sf.read(os.path.join(ROOT, out["path"]), always_2d=True)
        assert rate == 48000 and data.shape[1] == 2
        assert data.shape[0] == expected_samples
        info = sf.info(os.path.join(ROOT, out["path"]))
        assert info.subtype == "PCM_24" and info.format == "WAV"


def test_output_true_peak_under_ceiling():
    _ensure_run()
    for f in _report["files"]:
        if f["status"] != "ok":
            continue
        data, rate = sf.read(os.path.join(ROOT, f["output"]["path"]), always_2d=True)
        tp = truepeak.true_peak_dbfs(data, rate)
        assert tp <= CEILING + 0.01, f"{f['input']}: output TP {tp} exceeds ceiling"
        assert f["output"]["true_peak_dbtp"] <= CEILING


def test_limiter_engaged_on_hot_clicks():
    f = _entry("hot_clicks_48k_24bit_stereo.wav")
    assert f["status"] == "ok"
    assert f["processing"]["limiter_reduction_db"] > 1.0
    assert f["output"]["true_peak_dbtp"] <= CEILING
    # gain is re-computable from the report
    recomputed = round(TARGET_LUFS - f["measured"]["integrated_lufs"], 2)
    assert abs(f["processing"]["applied_gain_db"] - recomputed) <= 0.01


def test_loudness_out_of_range():
    f = _entry("quiet_32k_16bit_stereo.wav")
    assert f["status"] == "loudness_out_of_range"
    assert f["output"] is None
    assert f["measured"]["integrated_lufs"] is not None
    assert "outside" in f["error"]["reason"]
    needed = TARGET_LUFS - f["measured"]["integrated_lufs"]
    assert needed > 24.0


def test_unreadable_file():
    f = _entry("garbage.wav")
    assert f["status"] == "unreadable"
    assert f["error"]["stderr_excerpt"]
    assert f["output"] is None


def test_partial_decode_flag():
    f = _entry("corrupted_frames.mp3")
    assert "partial_decode" in f["flags"]
    assert f["status"] == "ok"


def test_timestamp_mismatch_flag():
    f = _entry("mismatch_voice.ogg")
    assert "timestamp_mismatch" in f["flags"]
    assert abs(
        f["processing"]["file_start_time_ms"] - f["processing"]["alignment_offset_ms"]
    ) > 5.0
    # consistent file must NOT be flagged
    g = _entry("offset_music.ogg")
    assert "timestamp_mismatch" not in g["flags"]


def test_alignment_offset():
    _ensure_run()
    f = _entry("surround_96k_float_5_1.wav")
    data, _ = sf.read(os.path.join(ROOT, f["output"]["path"]), always_2d=True)
    pre = data[: int(0.490 * 48000)]
    assert np.max(np.abs(pre)) < 1e-4, "expected silence before 500ms offset"
    post = data[int(0.6 * 48000) : int(1.0 * 48000)]
    assert np.max(np.abs(post)) > 1e-3, "expected content after offset"


def test_output_loudness_matches_target_for_stereo_sources():
    for name in ["tone_48k_16bit_stereo.wav", "loud_48k_16bit_stereo.wav"]:
        f = _entry(name)
        assert abs(f["output"]["integrated_lufs"] - TARGET_LUFS) <= 0.5, (
            f"{name}: output {f['output']['integrated_lufs']} LUFS"
        )


def test_report_reproducible():
    _ensure_run()
    first = open(REPORT).read()
    subprocess.run(
        [sys.executable, "-m", "loudnorm_batch", "--manifest", MANIFEST,
         "--out-dir", OUT_DIR, "--report", REPORT],
        check=True, cwd=ROOT,
    )
    second = open(REPORT).read()
    strip = lambda s: "\n".join(
        line for line in s.splitlines() if '"generated_at"' not in line
    )
    assert strip(first) == strip(second), "reports differ beyond generated_at"


def test_report_metadata_fields():
    _ensure_run()
    assert _report["version"]
    assert _report["parameters"]["target_lufs"] == TARGET_LUFS
    for f in _report["files"]:
        if f["status"] != "unreadable":
            assert re.fullmatch(r"[0-9a-f]{16}", f["sha256_16"])
    tone = _entry("tone_48k_16bit_stereo.wav")
    md = tone["metadata"]
    assert md["sample_rate"] == 48000 and md["bit_depth"] == 16
    assert md["channels"] == 2 and abs(md["duration_s"] - 4.0) < 0.01


def test_scheduler_timeout_isolates_slow_item():
    def work(item):
        if item == "slow":
            time.sleep(2.0)
        return f"done-{item}"

    def on_timeout(item):
        return f"timeout-{item}"

    results = run_ordered(
        ["a", "slow", "b", "c"], work, on_timeout, jobs=4, timeout_s=0.5
    )
    assert results == ["done-a", "timeout-slow", "done-b", "done-c"]


def _run_all():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
