import json
import os

from audiotl.dsp import ms_to_frames, resample_length
from audiotl.mapping import input_frame_to_output, output_frame_to_input, segments_at_ms
from audiotl.summary import compute_summary
from audiotl.wavio import read_wav

from conftest import mix_bytes, read_report, run_cli


def test_full_pipeline_success(sample_set, tmp_path):
    out = tmp_path / "out"
    code = run_cli(sample_set["manifest_path"], out)
    assert code == 0
    for name in ("mix.wav", "mapping.json", "summary.json", "report.json"):
        assert (out / name).is_file(), name
    # stage artifacts exist and carry versions
    stages = out / "_stages"
    for name in ("analysis.json", "resample.json", "mix.meta.json", "mix.f32"):
        assert (stages / name).is_file(), name
    analysis = json.loads((stages / "analysis.json").read_text())
    assert analysis["version"] == 1
    report = read_report(out)
    assert report["status"] == "ok"
    assert report["exit_code"] == 0
    assert report["cache"]["misses"] == 3


def test_mapping_positions_and_bidirectional_queries(sample_set, tmp_path):
    out = tmp_path / "out"
    assert run_cli(sample_set["manifest_path"], out) == 0
    mapping = json.loads((out / "mapping.json").read_text())
    rate = mapping["sample_rate"]
    segs = mapping["segments"]

    # expected lengths after deterministic resampling
    expected_frames = [
        resample_length(int(0.8 * 44100), 44100, 48000),
        resample_length(int(1.0 * 48000), 48000, 48000),
        resample_length(int(0.6 * 44100), 44100, 48000),
    ]
    ov1 = ms_to_frames(200.0, rate)
    ov2 = ms_to_frames(250.0, rate)
    assert segs[0]["output_start_frame"] == 0
    assert segs[0]["valid_frames"] == expected_frames[0]
    assert segs[1]["output_start_frame"] == expected_frames[0] - ov1
    assert segs[2]["output_start_frame"] == (
        expected_frames[0] - ov1 + expected_frames[1] - ov2
    )
    assert mapping["total_frames"] == segs[2]["output_end_frame"]

    # mix.wav length matches the mapping exactly
    mix, mix_rate = read_wav(out / "mix.wav")
    assert mix_rate == rate
    assert mix.shape == (mapping["total_frames"], 2)

    # ms query: inside the first crossfade, two segments are active
    xfade_mid_ms = (expected_frames[0] - ov1 // 2) * 1000.0 / rate
    hits = segments_at_ms(mapping, xfade_mid_ms)
    assert {s["id"] for s in hits} == {"a_44k_mono", "b_48k_stereo"}
    # outside any overlap, exactly one segment
    hits = segments_at_ms(mapping, 10.0)
    assert [s["id"] for s in hits] == ["a_44k_mono"]

    # bidirectional mapping is single-frame exact at segment boundaries
    for seg in segs:
        start = seg["output_start_frame"]
        assert input_frame_to_output(seg, rate, 0) == start
        assert output_frame_to_input(seg, rate, start) == 0
        last_in = seg["input_frames"] - 1
        assert input_frame_to_output(seg, rate, last_in) == seg["output_end_frame"] - 1


def test_summary_matches_final_wav_sample_exact(sample_set, tmp_path):
    out = tmp_path / "out"
    assert run_cli(sample_set["manifest_path"], out) == 0
    summary = json.loads((out / "summary.json").read_text())

    # recompute independently from the written WAV bytes
    data, rate = read_wav(out / "mix.wav")
    q = (data * 32768.0).round().astype("<i2")  # read_wav normalizes by 32768
    window_frames = summary["window_frames"]
    recomputed = compute_summary(q, rate, 2, window_frames, summary["window_ms"])
    assert recomputed["windows"] == summary["windows"]

    # trailing partial window uses its actual length
    total = summary["total_frames"]
    remainder = total % window_frames
    if remainder:
        assert summary["windows"][-1]["frames"] == remainder
    # every window starts on a strict multiple of the window size
    for w in summary["windows"]:
        assert w["start_frame"] == w["index"] * window_frames


def test_mono_output_and_44k_output(sample_set, tmp_path):
    manifest = dict(sample_set["manifest"])
    manifest["output"] = {"sample_rate": 44100, "channels": 1, "window_ms": 50}
    mpath = tmp_path / "manifest_mono.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "out_mono"
    assert run_cli(mpath, out) == 0
    data, rate = read_wav(out / "mix.wav")
    assert rate == 44100
    assert data.shape[1] == 1
    mapping = json.loads((out / "mapping.json").read_text())
    assert mapping["channels"] == 1
    assert data.shape[0] == mapping["total_frames"]
