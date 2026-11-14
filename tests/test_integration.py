from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dialogue_master.loudness import integrated_loudness

from conftest import (
    base_config,
    clip,
    run_cli,
    write_config,
    write_tone,
)


def _write_mixed_fixture(root: Path, config_path: Path) -> dict:
    input_dir = root / "input"
    write_tone(
        input_dir / "first.wav",
        duration_ms=500,
        amplitude=0.01,
        phase=math.pi / 2.0,
    )
    write_tone(
        input_dir / "second.wav",
        duration_ms=400,
        amplitude=0.02,
        phase=math.pi / 2.0,
    )
    write_tone(
        input_dir / "bad_format.wav",
        duration_ms=400,
        amplitude=0.02,
        channels=1,
        sample_rate=44100,
    )
    (input_dir / "broken.wav").write_bytes(b"this is not a WAV file")
    write_tone(
        input_dir / "late.wav",
        duration_ms=1500,
        amplitude=0.02,
    )
    value = base_config(
        clips=[
            clip("first", "input/first.wav", 600, -28, 0, 0),
            clip("second", "input/second.wav", 600, -20, 0, 0),
            clip("bad_format", "input/bad_format.wav", 1300, -20, 5, 5),
            clip("broken", "input/broken.wav", 1300, -20, 5, 5),
            clip("late", "input/late.wav", 3000, -20, 5, 5),
        ]
    )
    write_config(config_path, value)
    return value


def test_mixed_run_alignment_loudness_warnings_and_exit_zero(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    _write_mixed_fixture(tmp_path, config_path)
    output_dir = tmp_path / "out"
    completed = run_cli(config_path, output_dir)

    assert completed.returncode == 0
    assert "W_FORMAT" in completed.stderr
    assert "E_AUDIO" in completed.stderr
    assert "W_LATE" in completed.stderr
    assert "W_SHIFTED" in completed.stderr

    log = json.loads((output_dir / "processing_log.json").read_text())
    first, second, bad_format, broken, late = log["clips"]
    assert first["status"] == "success"
    assert first["final_start_sample"] == 28800
    assert first["integrated_lufs"] is not None
    assert abs(first["gain_db"]) > 10.0
    assert abs(first["gain_db"]) <= 20.0

    assert second["final_start_sample"] == 52800
    assert second["requested_start_sample"] == 28800
    assert second["sample_shift"] == 24000
    assert second["status"] == "warning"
    assert any(w["code"] == "W_SHIFTED" for w in second["warnings"])

    assert bad_format["status"] == "skipped"
    assert any(w["code"] == "W_FORMAT" for w in bad_format["warnings"])
    assert broken["status"] == "skipped"
    assert any(e["code"] == "E_AUDIO" for e in broken["errors"])
    assert late["status"] == "skipped"
    assert any(w["code"] == "W_LATE" for w in late["warnings"])

    summary = log["summary"]
    assert summary["placed"] == 2
    assert summary["skipped"] == 3
    assert summary["timeline_samples"] == 192000
    assert summary["exit_code"] == 0
    assert summary["master_over_0_dbfs"] is False

    info = sf.info(str(output_dir / "master.wav"))
    assert info.frames == 192000
    assert info.samplerate == 48000
    assert info.channels == 2
    assert info.subtype == "PCM_24"
    assert info.format == "WAVEX"

    master, rate = sf.read(str(output_dir / "master.wav"), always_2d=True)
    assert rate == 48000
    assert master.shape == (192000, 2)
    nonzero_frames = np.flatnonzero(np.max(np.abs(master), axis=1) > 0.0)
    assert int(nonzero_frames[0]) == 28800
    assert int(nonzero_frames[-1]) == 71999
    assert np.max(np.abs(master)) <= 1.0

    first_region = master[28800:52800, :]
    assert abs(integrated_loudness(first_region, rate, ("FL", "FR")) - (-28.0)) < 0.2
    second_region = master[52800:72000, :]
    assert abs(integrated_loudness(second_region, rate, ("FL", "FR")) - (-20.0)) < 0.2

    if shutil.which("ffprobe"):
        import subprocess

        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_name,sample_rate,channels,channel_layout,"
                "bits_per_sample,duration_ts",
                "-of",
                "default=nw=1",
                str(output_dir / "master.wav"),
            ],
            text=True,
            capture_output=True,
        )
        assert probe.returncode == 0
        assert "codec_name=pcm_s24le" in probe.stdout
        assert "channels=2" in probe.stdout
        assert "channel_layout=stereo" in probe.stdout
        assert "duration_ts=192000" in probe.stdout


def test_repeated_run_produces_byte_identical_log_and_master(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    _write_mixed_fixture(tmp_path, config_path)
    out_one = tmp_path / "out_one"
    out_two = tmp_path / "out_two"
    first = run_cli(config_path, out_one)
    second = run_cli(config_path, out_two)
    assert first.returncode == second.returncode == 0
    assert (
        out_one / "processing_log.json"
    ).read_bytes() == (
        out_two / "processing_log.json"
    ).read_bytes()
    assert (
        out_one / "master.wav"
    ).read_bytes() == (out_two / "master.wav").read_bytes()


def test_log_is_independent_of_config_workspace_location(
    tmp_path: Path,
) -> None:
    config_a = tmp_path / "workspace_a" / "config.json"
    config_b = tmp_path / "workspace_b" / "config.json"
    _write_mixed_fixture(tmp_path / "workspace_a", config_a)
    _write_mixed_fixture(tmp_path / "workspace_b", config_b)
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    assert run_cli(config_a, out_a).returncode == 0
    assert run_cli(config_b, out_b).returncode == 0
    assert (
        out_a / "processing_log.json"
    ).read_bytes() == (
        out_b / "processing_log.json"
    ).read_bytes()
    assert (out_a / "master.wav").read_bytes() == (
        out_b / "master.wav"
    ).read_bytes()


def test_all_clips_skipped_exits_two_and_writes_log_only(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    write_tone(
        input_dir / "bad_format.wav",
        duration_ms=400,
        amplitude=0.02,
        channels=1,
        sample_rate=44100,
    )
    (input_dir / "broken.wav").write_bytes(b"not a wave")
    config_path = tmp_path / "config.json"
    write_config(
        config_path,
        base_config(
            clips=[
                clip("bad_format", "input/bad_format.wav", 0, -20, 5, 5),
                clip("broken", "input/broken.wav", 0, -20, 5, 5),
            ]
        ),
    )
    output_dir = tmp_path / "out"
    completed = run_cli(config_path, output_dir)
    assert completed.returncode == 2
    assert "E_AUDIO" in completed.stderr
    assert "W_FORMAT" in completed.stderr
    log = json.loads((output_dir / "processing_log.json").read_text())
    assert log["summary"]["placed"] == 0
    assert log["summary"]["exit_code"] == 2
    assert not (output_dir / "master.wav").exists()


def test_only_format_skips_also_exit_two(tmp_path: Path) -> None:
    write_tone(
        tmp_path / "input" / "wrong_rate.wav",
        sample_rate=44100,
        duration_ms=400,
    )
    config_path = tmp_path / "config.json"
    write_config(
        config_path,
        base_config(
            clips=[
                clip("wrong_rate", "input/wrong_rate.wav", 0, -20, 0, 0)
            ]
        ),
    )
    completed = run_cli(config_path, tmp_path / "out")
    assert completed.returncode == 2
    assert "W_FORMAT" in completed.stderr
    assert "E_CFG" not in completed.stderr
    assert "E_AUDIO" not in completed.stderr


@pytest.mark.parametrize(
    "broken_config",
    [
        "missing_field",
        "wrong_type",
        "bad_json",
    ],
)
def test_config_error_exits_one_without_creating_output(
    tmp_path: Path, broken_config: str
) -> None:
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "must_not_exist"
    if broken_config == "missing_field":
        value = base_config(
            clips=[clip("line", "input/a.wav", 0, -23, 0, 0)]
        )
        del value["clips"][0]["target_lufs"]
        write_config(config_path, value)
    elif broken_config == "wrong_type":
        value = base_config(
            clips=[
                {
                    "id": "line",
                    "input_path": "input/a.wav",
                    "start_ms": "zero",
                    "target_lufs": -23,
                    "fade_in_ms": 0,
                    "fade_out_ms": 0,
                }
            ]
        )
        write_config(config_path, value)
    else:
        config_path.write_text("{not json", encoding="utf-8")

    completed = run_cli(config_path, output_dir)
    assert completed.returncode == 1
    assert "E_CFG" in completed.stderr
    assert "E_AUDIO" not in completed.stderr
    assert not output_dir.exists()


def test_gain_is_peak_constrained_and_warning_is_logged(
    tmp_path: Path,
) -> None:
    sample_rate = 48000
    frame_count = sample_rate * 2
    data = np.zeros((frame_count, 2), dtype=np.float64)
    burst_start = sample_rate // 2
    burst_frames = int(sample_rate * 0.1)
    data[burst_start:burst_start + burst_frames, :] = 0.9
    input_path = tmp_path / "input" / "crest.wav"
    input_path.parent.mkdir(parents=True)
    sf.write(
        str(input_path),
        data,
        sample_rate,
        format="WAVEX",
        subtype="PCM_24",
    )
    config_path = tmp_path / "config.json"
    write_config(
        config_path,
        base_config(
            timeline_ms=2000,
            clips=[
                clip("crest", "input/crest.wav", 0, 20.0, 0, 0),
            ],
        ),
    )
    completed = run_cli(config_path, tmp_path / "out")
    assert completed.returncode == 0
    assert "W_PEAK" in completed.stderr
    log = json.loads((tmp_path / "out" / "processing_log.json").read_text())
    entry = log["clips"][0]
    assert entry["peak_constrained"] is True
    assert entry["gain_db"] <= 20.0
    assert entry["peak_after_gain_dbfs"] <= 1e-5
    assert any(w["code"] == "W_PEAK" for w in entry["warnings"])
    master, _ = sf.read(
        str(tmp_path / "out" / "master.wav"), always_2d=True
    )
    assert np.max(np.abs(master)) <= 1.0


def test_float32_input_matches_32_bit_baseline_and_is_safely_constrained(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input" / "float_clip.wav"
    input_path.parent.mkdir(parents=True)
    data = np.ones((48000, 2), dtype=np.float32) * 1.25
    sf.write(
        str(input_path),
        data,
        48000,
        format="WAVEX",
        subtype="FLOAT",
    )
    config_path = tmp_path / "config.json"
    write_config(
        config_path,
        base_config(
            timeline_ms=1000,
            bit_depth=32,
            clips=[clip("float_clip", "input/float_clip.wav", 0, 0, 0, 0)],
        ),
    )
    completed = run_cli(config_path, tmp_path / "out")
    assert completed.returncode == 0
    log = json.loads((tmp_path / "out" / "processing_log.json").read_text())
    entry = log["clips"][0]
    assert entry["status"] == "warning"
    assert entry["peak_constrained"] is True
    assert entry["peak_after_gain_dbfs"] <= 1e-5
    output_info = sf.info(str(tmp_path / "out" / "master.wav"))
    assert output_info.subtype == "PCM_32"
    output, _ = sf.read(
        str(tmp_path / "out" / "master.wav"), always_2d=True
    )
    assert np.max(np.abs(output)) <= 1.0
