from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from loudness_pipeline.audio import read_wav
from loudness_pipeline.errors import PipelineError
from loudness_pipeline.pipeline import process_file

from conftest import run_cli, tone


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"])
def test_reads_all_supported_wav_subtypes(tmp_path: Path, subtype: str) -> None:
    path = tmp_path / "source_{0}.wav".format(subtype)
    audio = tone(duration=0.5)
    sf.write(str(path), audio, 48000, subtype=subtype, format="WAV")
    loaded = read_wav(path)
    assert loaded.frames == 24000
    assert loaded.channels == 2
    assert loaded.sample_rate == 48000
    assert loaded.subtype == subtype
    assert loaded.samples.shape == (24000, 2)
    assert loaded.samples.dtype == np.float64


def test_error_codes_for_empty_corrupt_and_unsupported_rate(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty.wav"
    sf.write(str(empty), np.zeros((0, 2)), 48000, subtype="PCM_16")
    with pytest.raises(PipelineError) as empty_error:
        read_wav(empty)
    assert empty_error.value.code == "E_EMPTY"

    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"broken wave")
    with pytest.raises(PipelineError) as corrupt_error:
        read_wav(corrupt)
    assert corrupt_error.value.code == "E_FORMAT"

    unsupported_rate = tmp_path / "rate.wav"
    sf.write(
        str(unsupported_rate),
        tone(duration=0.5),
        22050,
        subtype="PCM_16",
    )
    with pytest.raises(PipelineError) as rate_error:
        read_wav(unsupported_rate)
    assert rate_error.value.code == "E_SAMPLERATE"


def test_pipeline_preserves_input_and_is_byte_deterministic(
    tmp_path: Path,
    surround_path: Path,
) -> None:
    input_before = _digest(surround_path)
    output = tmp_path / "normalized.wav"
    report = tmp_path / "report.json"

    first = process_file(
        surround_path,
        output,
        target_lufs=-23.0,
        limiter_enabled=True,
        report_path=report,
    )
    first_wav = output.read_bytes()
    first_report = report.read_bytes()

    second = process_file(
        surround_path,
        output,
        target_lufs=-23.0,
        limiter_enabled=True,
        report_path=report,
    )
    assert output.read_bytes() == first_wav
    assert report.read_bytes() == first_report
    assert _digest(surround_path) == input_before
    assert first.input_loudness_lufs == pytest.approx(
        second.input_loudness_lufs, abs=0.0
    )
    assert first.output_loudness_lufs == pytest.approx(-23.0, abs=0.02)

    output_info = sf.info(str(output))
    source_info = sf.info(str(surround_path))
    assert output_info.channels == source_info.channels == 6
    assert output_info.samplerate == source_info.samplerate == 48000
    assert output_info.subtype == source_info.subtype == "PCM_24"
    assert output_info.frames == source_info.frames

    rendered, _ = sf.read(str(output), dtype="float64", always_2d=True)
    assert np.max(np.abs(rendered)) <= 1.0
    parsed = json.loads(report.read_text(encoding="utf-8"))
    assert parsed["audio"]["format"] == "WAV"
    assert parsed["audio"]["channel_weights"] == [
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
    ]
    assert parsed["segments"]
    assert parsed["overall"]["before"]["integrated_lufs"] is not None
    assert parsed["overall"]["after"]["integrated_lufs"] is not None


def test_report_marks_partial_final_segment_and_two_decimal_numbers(
    tmp_path: Path,
    stereo_path: Path,
) -> None:
    output = tmp_path / "out.wav"
    result = process_file(
        stereo_path,
        output,
        target_lufs=-23.0,
        report_path=tmp_path / "report.json",
        segment_seconds=5.0,
    )
    report_text = Path(result.report_path).read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert len(report["segments"]) == 1
    segment = report["segments"][0]
    assert segment["duration_seconds"] == 4.0
    assert segment["is_partial_segment"] is True
    assert segment["is_partial_window"] is False

    output2 = tmp_path / "out2.wav"
    result2 = process_file(
        stereo_path,
        output2,
        target_lufs=-23.0,
        report_path=tmp_path / "report2.json",
        segment_seconds=3.0,
    )
    report2 = json.loads(Path(result2.report_path).read_text(encoding="utf-8"))
    assert len(report2["segments"]) == 2
    last = report2["segments"][-1]
    assert last["is_partial_segment"] is True
    assert last["duration_seconds"] == 1.0
    assert last["window_duration_seconds"] == 0.4

    output3 = tmp_path / "out3.wav"
    result3 = process_file(
        stereo_path,
        output3,
        target_lufs=-23.0,
        report_path=tmp_path / "report3.json",
        segment_seconds=3.9,
    )
    report3_text = Path(result3.report_path).read_text(encoding="utf-8")
    report3 = json.loads(report3_text)
    last3 = report3["segments"][-1]
    assert last3["is_partial_segment"] is True
    assert last3["is_partial_window"] is True
    assert last3["duration_seconds"] == 0.1
    assert last3["window_duration_seconds"] == 0.1

    assert re.search(r'"duration_seconds": 0\.10,', report3_text)


def test_limiter_records_reduction_and_never_overshoots(tmp_path: Path) -> None:
    sample_rate = 48000
    time = np.arange(sample_rate * 3) / float(sample_rate)
    base = 0.012 * np.sin(2.0 * np.pi * 440.0 * time)
    pulse = (time >= 1.5) & (time < 1.51)
    base[pulse] += 0.40 * np.sin(2.0 * np.pi * 1000.0 * time[pulse])
    audio = np.stack([base, base], axis=1)
    source = tmp_path / "limiter.wav"
    sf.write(str(source), audio, sample_rate, subtype="FLOAT")

    output = tmp_path / "limited.wav"
    result = process_file(source, output, target_lufs=-23.0)
    rendered, _ = sf.read(str(output), dtype="float64", always_2d=True)
    assert result.limiter_applied
    assert result.max_limiting_db > 0.0
    assert result.output_loudness_lufs == pytest.approx(-23.0, abs=0.02)
    assert np.max(np.abs(rendered)) <= 1.0
    assert result.report["processing"]["maximum_overshoot"] == 0.0
    assert result.report["processing"]["requested_gain_db"] > 0.0
    assert result.report["processing"]["applied_base_gain_db"] > (
        result.report["processing"]["requested_gain_db"]
    )


def test_cli_success_and_business_error_exit_codes(
    tmp_path: Path,
    stereo_path: Path,
) -> None:
    output = tmp_path / "cli_out.wav"
    success = run_cli([str(stereo_path), str(output), "--target", "-23"])
    assert success.returncode == 0, success.stderr
    assert "input_lufs:" in success.stdout
    assert "output_lufs:" in success.stdout
    assert "gain_db:" in success.stdout
    assert "elapsed_seconds:" in success.stdout
    assert output.exists()
    assert Path(str(output) + ".json").exists()

    invalid = run_cli([str(stereo_path), str(tmp_path / "bad.wav"), "--target", "-40"])
    assert invalid.returncode != 0
    assert "usage:" in invalid.stderr.lower()

    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"not audio")
    format_error = run_cli([str(corrupt), str(tmp_path / "corrupt_out.wav")])
    assert format_error.returncode == 3
    assert "E_FORMAT" in format_error.stderr

    empty = tmp_path / "empty.wav"
    sf.write(str(empty), np.zeros((0, 2)), 48000, subtype="PCM_16")
    empty_error = run_cli([str(empty), str(tmp_path / "empty_out.wav")])
    assert empty_error.returncode == 4
    assert "E_EMPTY" in empty_error.stderr

    rate = tmp_path / "rate.wav"
    sf.write(str(rate), tone(duration=0.5), 22050, subtype="PCM_16")
    rate_error = run_cli([str(rate), str(tmp_path / "rate_out.wav")])
    assert rate_error.returncode == 5
    assert "E_SAMPLERATE" in rate_error.stderr

    input_before = _digest(stereo_path)
    destination_directory = tmp_path / "is_a_directory"
    destination_directory.mkdir()
    write_error = run_cli([str(stereo_path), str(destination_directory)])
    assert write_error.returncode == 6
    assert "E_WRITE" in write_error.stderr
    assert _digest(stereo_path) == input_before
    assert list(tmp_path.glob(".is_a_directory.*.tmp")) == []


def test_no_limiter_uses_peak_safe_static_gain(tmp_path: Path) -> None:
    sample_rate = 48000
    time = np.arange(sample_rate * 3) / float(sample_rate)
    base = 0.012 * np.sin(2.0 * np.pi * 440.0 * time)
    pulse = (time >= 1.5) & (time < 1.51)
    base[pulse] += 0.40 * np.sin(2.0 * np.pi * 1000.0 * time[pulse])
    audio = np.stack([base, base], axis=1)
    source = tmp_path / "no_limiter.wav"
    sf.write(str(source), audio, sample_rate, subtype="FLOAT")
    output = tmp_path / "safe.wav"
    result = process_file(
        source,
        output,
        target_lufs=-23.0,
        limiter_enabled=False,
    )
    rendered, _ = sf.read(str(output), dtype="float64", always_2d=True)
    assert not result.limiter_applied
    assert np.max(np.abs(rendered)) <= 1.0
    assert result.applied_base_gain_db < result.requested_gain_db


def test_output_path_cannot_replace_input(stereo_path: Path) -> None:
    input_before = _digest(stereo_path)
    with pytest.raises(PipelineError) as error:
        process_file(stereo_path, stereo_path, target_lufs=-23.0)
    assert error.value.code == "E_USAGE"
    assert _digest(stereo_path) == input_before

