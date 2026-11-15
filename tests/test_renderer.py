"""Overlap precedence, looping, silence padding and report alignment."""

from __future__ import annotations

import json
import os

import numpy as np
import soundfile as sf

from audiopack.cli import main as cli_main
from audiopack.errors import E_CORRUPT_CLIP, E_MEMORY_LIMIT, E_TIMELINE_FORMAT
from audiopack.loudness import measure_loudness
from audiopack.pipeline import run_pipeline
from tests.samples import build_full_scenario, sine_wave, write_wav


def _run_cli(input_dir, output_wav, report_path, memory_mb=512):
    return cli_main(
        [
            input_dir,
            output_wav,
            "--report",
            report_path,
            "--memory-limit-mb",
            str(memory_mb),
        ]
    )


def _read(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data, sr


def test_full_scenario_exit_code_and_warnings(tmp_path):
    input_dir = build_full_scenario(str(tmp_path / "scenario"))
    out_wav = str(tmp_path / "mix.wav")
    report_path = str(tmp_path / "report.json")

    exit_code = _run_cli(input_dir, out_wav, report_path)
    assert exit_code == 1  # one corrupt clip is present

    report = json.load(open(report_path, encoding="utf-8"))
    assert report["error_codes"] == [E_CORRUPT_CLIP]
    assert report["warnings"] == ["W_SHORT_CLIP"]
    assert report["status"] == "error"


def test_output_is_48k_stereo_24bit_and_aligned(tmp_path):
    input_dir = build_full_scenario(str(tmp_path / "scenario"))
    out_wav = str(tmp_path / "out.wav")
    report_path = str(tmp_path / "report.json")
    _run_cli(input_dir, out_wav, report_path)

    info = sf.info(out_wav)
    assert info.samplerate == 48000
    assert info.channels == 2
    assert info.subtype == "PCM_24"

    data, sr = _read(out_wav)
    report = json.load(open(report_path, encoding="utf-8"))
    assert sr == report["sample_rate"] == 48000
    assert data.shape[0] == report["output_frames"]

    # Report sample indices follow the fixed ms*sr/1000 rule and stay in range.
    for event in report["events"]:
        assert 0 <= event["start_sample"] < event["end_sample"] <= data.shape[0]
        assert event["end_sample"] - event["start_sample"] == event["duration_samples"]


def test_overlap_priority_and_tie_records(tmp_path):
    input_dir = build_full_scenario(str(tmp_path / "scenario"))
    report_path = str(tmp_path / "report.json")
    _run_cli(input_dir, str(tmp_path / "mix.wav"), report_path)
    report = json.load(open(report_path, encoding="utf-8"))

    overlaps = report["overlaps"]
    assert len(overlaps) == 3

    # Event 1 (priority 5) beats event 0 (priority 1).
    pair_01 = next(
        item
        for item in overlaps
        if {item["winner_event_index"], item["loser_event_index"]} == {0, 1}
    )
    assert pair_01["winner_event_index"] == 1
    assert pair_01["rule"] == "higher_priority"

    # Equal priority: later event index 2 wins over event 1.
    pair_12 = next(
        item
        for item in overlaps
        if {item["winner_event_index"], item["loser_event_index"]} == {1, 2}
    )
    assert pair_12["winner_event_index"] == 2
    assert pair_12["rule"] == "later_index_on_tie"

    # Reported overlap bounds follow the round/floor sample rule.
    assert pair_01["overlap_start_sample"] == int(round(500 * 48000 / 1000))
    assert pair_01["overlap_end_sample"] == int(round(900 * 48000 / 1000))


def test_priority_override_actually_replaces_audio(tmp_path):
    """Two distinct constant tones: overlap region must equal the winner only."""
    root = str(tmp_path / "ov")
    os.makedirs(root, exist_ok=True)
    low = np.full((48000, 1), 0.2, dtype=np.float32)
    high = np.full((48000, 1), -0.4, dtype=np.float32)
    write_wav(os.path.join(root, "low.wav"), low, 48000, "PCM_16")
    write_wav(os.path.join(root, "high.wav"), high, 48000, "PCM_16")
    timeline = {
        "sample_rate": 48000,
        "events": [
            {
                "start_ms": 0, "end_ms": 1000, "clip": "low.wav",
                "fade_in_ms": 0, "fade_out_ms": 0, "gain_db": 0,
                "priority": 1, "loop": False,
            },
            {
                "start_ms": 400, "end_ms": 800, "clip": "high.wav",
                "fade_in_ms": 0, "fade_out_ms": 0, "gain_db": 0,
                "priority": 9, "loop": False,
            },
        ],
    }
    json.dump(timeline, open(os.path.join(root, "timeline.json"), "w"))

    exit_code, report, _ = run_pipeline(root, str(tmp_path / "m.wav"), str(tmp_path / "r.json"))
    assert exit_code == 0
    audio, _ = _read(str(tmp_path / "m.wav"))

    # The final mix is loudness-normalized, so compare relative levels:
    # both regions are scaled by the same single global gain factor.
    overlap_level = float(np.mean(audio[25000:30000, 0]))
    outside_level = float(np.mean(audio[5000:10000, 0]))
    gain_factor = overlap_level / -0.4
    assert abs(gain_factor) > 1.0  # normalization boosted the quiet mix
    # Tolerance covers PCM-24 quantization plus float32 rounding after gain.
    assert abs(outside_level - 0.2 * gain_factor) < 2e-4
    # And the overlap region is constant (no leftover low-priority content).
    assert np.ptp(audio[25000:30000, 0]) < 1e-6


def test_looping_tiles_while_short_clip_pads_silence(tmp_path):
    input_dir = build_full_scenario(str(tmp_path / "scenario"))
    out_wav = str(tmp_path / "mix.wav")
    report_path = str(tmp_path / "report.json")
    _run_cli(input_dir, out_wav, report_path)
    data, _ = _read(out_wav)
    report = json.load(open(report_path, encoding="utf-8"))

    loop_event = next(item for item in report["events"] if item["event_index"] == 4)
    assert loop_event["loop"] is True
    assert loop_event["status"] == "rendered"
    # 600 ms event vs 250 ms loop source must be fully populated.
    loop_region = data[loop_event["start_sample"]:loop_event["end_sample"]]
    assert np.any(np.abs(loop_region) > 0.0)

    short_event = next(item for item in report["events"] if item["event_index"] == 3)
    assert short_event["short_clip"] is True
    padding_start = short_event["start_sample"] + short_event["source_frames_used"]
    padded = data[padding_start:short_event["end_sample"]]
    assert padded.size > 0 and np.all(padded == 0.0)


def test_loudness_normalized_to_target(tmp_path):
    input_dir = build_full_scenario(str(tmp_path / "scenario"))
    out_wav = str(tmp_path / "mix.wav")
    _run_cli(input_dir, out_wav, str(tmp_path / "report.json"))
    data, sr = _read(out_wav)
    measured = measure_loudness(data, sr)
    assert abs(measured - (-23.0)) <= 0.5


def test_low_loudness_warns_and_caps_boost(tmp_path):
    root = str(tmp_path / "silent")
    os.makedirs(root, exist_ok=True)
    write_wav(
        os.path.join(root, "tiny.wav"),
        sine_wave(1.0, 300.0, 48000, amplitude=1e-5),
        48000,
        "PCM_16",
    )
    timeline = {
        "sample_rate": 48000,
        "events": [
            {
                "start_ms": 0, "end_ms": 1000, "clip": "tiny.wav",
                "fade_in_ms": 0, "fade_out_ms": 0, "gain_db": 0,
                "priority": 0, "loop": False,
            }
        ],
    }
    json.dump(timeline, open(os.path.join(root, "timeline.json"), "w"))
    exit_code, report, _ = run_pipeline(root, str(tmp_path / "m.wav"), str(tmp_path / "r.json"))
    assert exit_code == 0
    assert "W_LOW_LOUDNESS" in report["warnings"]
    assert report["loudness"]["gain_boost_capped"] is True
    assert report["loudness"]["applied_gain_db"] <= 20.0 + 1e-9


def test_corrupt_clip_does_not_stop_other_events(tmp_path):
    input_dir = build_full_scenario(str(tmp_path / "scenario"))
    out_wav = str(tmp_path / "mix.wav")
    report_path = str(tmp_path / "report.json")
    exit_code = _run_cli(input_dir, out_wav, report_path)
    assert exit_code == 1

    report = json.load(open(report_path, encoding="utf-8"))
    rendered = [item for item in report["events"] if item["status"] == "rendered"]
    errored = [item for item in report["events"] if item["status"] == "error"]
    assert len(rendered) == 6
    assert len(errored) == 1 and errored[0]["error_code"] == E_CORRUPT_CLIP
    assert os.path.exists(out_wav)


def test_deterministic_byte_identical_outputs(tmp_path):
    input_dir = build_full_scenario(str(tmp_path / "scenario"))
    outputs = []
    for run in range(2):
        wav = str(tmp_path / f"run{run}.wav")
        report = str(tmp_path / f"run{run}.json")
        assert _run_cli(input_dir, wav, report) == 1
        outputs.append((wav, report))

    assert open(outputs[0][0], "rb").read() == open(outputs[1][0], "rb").read()
    assert open(outputs[0][1], "rb").read() == open(outputs[1][1], "rb").read()


def test_memory_limit_removes_partial_output(tmp_path):
    input_dir = build_full_scenario(str(tmp_path / "scenario"))
    out_wav = str(tmp_path / "huge.wav")
    report_path = str(tmp_path / "huge.json")
    # A 1 MiB cap cannot hold the 3.6 s * 48k stereo float render buffers.
    exit_code = _run_cli(input_dir, out_wav, report_path, memory_mb=1)
    assert exit_code == 1
    assert not os.path.exists(out_wav)
    assert not os.path.exists(report_path)


def test_no_wav_directory_is_format_error(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    try:
        run_pipeline(str(root), str(tmp_path / "x.wav"))
    except Exception as exc:  # noqa: BLE001 - assert code below
        assert getattr(exc, "code", None) == E_TIMELINE_FORMAT
    else:  # pragma: no cover
        raise AssertionError("expected PipelineError")
