import json
import os

from audiotl.samplegen import write_tone

from conftest import mix_bytes, read_report, run_cli


def test_incremental_rerun_is_byte_identical_and_cached(sample_set, tmp_path):
    out = tmp_path / "out"
    assert run_cli(sample_set["manifest_path"], out) == 0
    first = mix_bytes(out)
    report1 = read_report(out)
    assert report1["cache"]["misses"] == 3

    assert run_cli(sample_set["manifest_path"], out) == 0
    second = mix_bytes(out)
    report2 = read_report(out)
    assert second == first
    assert report2["cache"]["hits"] == 3
    assert report2["cache"]["misses"] == 0
    assert report2["stages"]["mix"]["reused"] is True


def test_single_segment_change_rerenders_only_that_segment(sample_set, tmp_path):
    out = tmp_path / "out"
    assert run_cli(sample_set["manifest_path"], out) == 0

    # modify exactly one source segment (different frequency)
    target = sample_set["samples"] / "b_48k_stereo.wav"
    write_tone(str(target), 48000, 2, 1.00, 175.0)

    assert run_cli(sample_set["manifest_path"], out) == 0
    report = read_report(out)
    assert report["cache"]["misses"] == 1
    assert report["cache"]["hits"] == 2
    incremental = mix_bytes(out)

    # a full re-render against a fresh state dir must match byte-for-byte
    out_fresh = tmp_path / "out_fresh"
    assert run_cli(sample_set["manifest_path"], out_fresh) == 0
    report_fresh = read_report(out_fresh)
    assert report_fresh["cache"]["misses"] == 3
    assert mix_bytes(out_fresh) == incremental


def test_corrupted_state_file_falls_back_to_full_render(sample_set, tmp_path):
    out = tmp_path / "out"
    assert run_cli(sample_set["manifest_path"], out) == 0
    baseline = mix_bytes(out)

    state_file = out / "_state" / "state.json"
    state_file.write_text("{ this is not valid json !!!", encoding="utf-8")

    assert run_cli(sample_set["manifest_path"], out) == 0
    report = read_report(out)
    assert report["cache"]["misses"] == 3  # full re-render
    assert mix_bytes(out) == baseline


def test_corrupted_cache_segment_is_rerendered(sample_set, tmp_path):
    out = tmp_path / "out"
    assert run_cli(sample_set["manifest_path"], out) == 0
    baseline = mix_bytes(out)

    seg_dir = out / "_state" / "segments"
    victim = next(seg_dir.glob("*.f32"))
    victim.write_bytes(b"\x00" * 8)  # truncated garbage

    assert run_cli(sample_set["manifest_path"], out) == 0
    report = read_report(out)
    assert report["cache"]["misses"] == 1
    assert mix_bytes(out) == baseline


def test_summary_recomputed_after_window_change(sample_set, tmp_path):
    out = tmp_path / "out"
    assert run_cli(sample_set["manifest_path"], out) == 0
    summary1 = json.loads((out / "summary.json").read_text())

    manifest = dict(sample_set["manifest"])
    manifest["output"] = dict(manifest["output"], window_ms=37)
    mpath = tmp_path / "manifest_w37.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")

    assert run_cli(mpath, out) == 0
    summary2 = json.loads((out / "summary.json").read_text())
    assert summary2["window_frames"] != summary1["window_frames"]
    # no stale windows: window count matches the new window size
    expected = -(-summary2["total_frames"] // summary2["window_frames"])
    assert len(summary2["windows"]) == expected
