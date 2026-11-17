from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import (
    error_payload,
    invoke_cli,
    make_static_config,
    make_stereo_wav,
)
from spatialmix.hrtf import (
    AZIMUTH_COUNT,
    ELEVATION_COUNT,
    generate_hrtf,
    overlap_save_convolve,
)


def _run_ok(
    input_wav: Path,
    config_path: Path,
    output_wav: Path,
    *,
    output_format: str = "float32",
    block_size: int = 256,
    extra: tuple[str, ...] = (),
):
    result = invoke_cli(
        [
            str(input_wav),
            str(config_path),
            str(output_wav),
            "--output-format",
            output_format,
            "--block-size",
            str(block_size),
            "--json",
            *extra,
        ]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result, json.loads(result.stdout)


def _read(path: Path) -> np.ndarray:
    data, _ = sf.read(str(path), dtype="float64", always_2d=True)
    return data


def test_hrtf_dataset_is_deterministic_and_nearest() -> None:
    first = generate_hrtf(48000)
    second = generate_hrtf(48000)
    assert first.shape == (
        ELEVATION_COUNT,
        AZIMUTH_COUNT,
        2,
        first.ir_length,
    )
    assert first.version == "local-hrtf-v1"
    assert np.array_equal(first.impulse_responses, second.impulse_responses)
    assert first.fingerprint() == second.fingerprint()
    index, ir = first.select(90.0, 0.0)
    assert index[1] == 18
    assert ir.shape == (2, first.ir_length)


def test_hrtf_info_command_needs_no_positional_arguments() -> None:
    result = invoke_cli(["--hrtf-info", "--json"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["azimuth_count"] == 24
    assert payload["elevation_count"] == 12
    assert len(payload["fingerprint_sha256"]) == 64


@pytest.mark.parametrize("block_size", [128, 256, 512, 1000])
def test_overlap_save_matches_one_shot_convolution(block_size: int) -> None:
    rng = np.random.default_rng(20261117)
    signal = rng.normal(size=1000)
    impulse = rng.normal(size=73)
    expected = np.convolve(signal, impulse, mode="full")
    actual = overlap_save_convolve(signal, impulse, block_size)
    assert actual.shape == expected.shape
    assert np.max(np.abs(actual - expected)) <= 1.0e-11


def test_three_source_spatial_mix_and_block_consistency(demo_files) -> None:
    input_wav, config_path = demo_files
    outputs = {}
    for block_size in (128, 256, 512):
        output = input_wav.parent / f"out_{block_size}.wav"
        _, payload = _run_ok(input_wav, config_path, output, block_size=block_size)
        assert payload["source_count"] == 3
        assert payload["channels"] == 2
        outputs[block_size] = _read(output)

    reference = outputs[256]
    assert reference.shape == (24239, 2)
    assert np.max(np.abs(reference[:, 0] - reference[:, 1])) > 0.02
    assert np.max(np.abs(reference)) < 1.0
    for block_size in (128, 512):
        difference = np.max(np.abs(outputs[block_size] - reference))
        relative_db = 20.0 * math.log10(
            max(difference, 1.0e-30) / max(np.max(np.abs(reference)), 1.0e-30)
        )
        assert relative_db <= -60.0


def test_filter_and_hrtf_order_equivalence_across_block_sizes(demo_files) -> None:
    input_wav, config_path = demo_files
    before_128 = input_wav.parent / "before_128.wav"
    after_512 = input_wav.parent / "after_512.wav"
    _run_ok(
        input_wav,
        config_path,
        before_128,
        block_size=128,
        extra=("--filter-order", "before"),
    )
    _run_ok(
        input_wav,
        config_path,
        after_512,
        block_size=512,
        extra=("--filter-order", "after"),
    )
    difference = np.max(np.abs(_read(before_128) - _read(after_512)))
    reference_peak = np.max(np.abs(_read(before_128)))
    relative_db = 20.0 * math.log10(
        max(difference, 1.0e-30) / max(reference_peak, 1.0e-30)
    )
    assert relative_db <= -60.0


def test_resume_is_sample_exact_and_rejects_block_size_change(demo_files) -> None:
    input_wav, config_path = demo_files
    reference = input_wav.parent / "reference.wav"
    resumed = input_wav.parent / "resumed.wav"
    _run_ok(input_wav, config_path, reference, block_size=128)

    stopped = invoke_cli(
        [
            str(input_wav),
            str(config_path),
            str(resumed),
            "--output-format",
            "float32",
            "--block-size",
            "128",
            "--stop-after-blocks",
            "50",
            "--json",
        ]
    )
    interrupted = error_payload(stopped)
    assert stopped.returncode == 130
    assert interrupted["error_code"] == "E_INTERRUPTED"
    state_path = Path(str(resumed) + ".state.json")
    assert state_path.exists()
    assert not resumed.exists()

    _, payload = _run_ok(input_wav, config_path, resumed, block_size=128)
    assert payload["resumed"] is True
    assert np.array_equal(_read(reference), _read(resumed))
    assert not state_path.exists()

    stopped_again = invoke_cli(
        [
            str(input_wav),
            str(config_path),
            str(resumed),
            "--output-format",
            "float32",
            "--block-size",
            "128",
            "--stop-after-blocks",
            "5",
            "--json",
        ]
    )
    assert error_payload(stopped_again)["error_code"] == "E_INTERRUPTED"
    mismatch = invoke_cli(
        [
            str(input_wav),
            str(config_path),
            str(resumed),
            "--output-format",
            "float32",
            "--block-size",
            "256",
            "--json",
        ]
    )
    mismatch_payload = error_payload(mismatch)
    assert mismatch_payload["error_code"] == "E_STATE_MISMATCH"
    assert "block_size" in mismatch_payload["details"]["mismatches"]


def test_mix_stage_interruption_resumes_sample_exact(demo_files) -> None:
    input_wav, config_path = demo_files
    reference = input_wav.parent / "mix_reference.wav"
    resumed = input_wav.parent / "mix_resumed.wav"
    _run_ok(input_wav, config_path, reference, block_size=128)

    stopped = invoke_cli(
        [
            str(input_wav),
            str(config_path),
            str(resumed),
            "--output-format",
            "float32",
            "--block-size",
            "128",
            "--stop-after-mix-blocks",
            "10",
            "--json",
        ]
    )
    assert error_payload(stopped)["error_code"] == "E_INTERRUPTED"
    state_path = Path(str(resumed) + ".state.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["stage"] == "mix"
    assert state["mix_offset"] == 1280
    assert not resumed.exists()

    _, payload = _run_ok(input_wav, config_path, resumed, block_size=128)
    assert payload["resumed"] is True
    assert np.array_equal(_read(reference), _read(resumed))


@pytest.mark.parametrize(
    ("input_subtype", "output_format", "expected_subtype"),
    [
        ("PCM_U8", "pcm8", "PCM_U8"),
        ("PCM_16", "pcm16", "PCM_16"),
        ("PCM_24", "pcm24", "PCM_24"),
        ("PCM_32", "pcm32", "PCM_32"),
        ("FLOAT", "float32", "FLOAT"),
        ("DOUBLE", "float64", "DOUBLE"),
    ],
)
def test_supported_wav_bit_depths(
    tmp_path: Path,
    input_subtype: str,
    output_format: str,
    expected_subtype: str,
) -> None:
    input_wav = make_stereo_wav(
        tmp_path / f"in_{input_subtype}.wav", subtype=input_subtype
    )
    config_path = make_static_config(tmp_path / f"cfg_{input_subtype}.json")
    output_wav = tmp_path / f"out_{input_subtype}.wav"
    _run_ok(
        input_wav,
        config_path,
        output_wav,
        output_format=output_format,
        block_size=128,
    )
    info = sf.info(str(output_wav))
    assert info.format == "WAV"
    assert info.subtype == expected_subtype
    assert info.channels == 2
    assert info.frames > 0


def test_incompatible_bit_depth_preserves_existing_output(tmp_path: Path) -> None:
    input_wav = make_stereo_wav(tmp_path / "input.wav", subtype="PCM_16")
    config_path = make_static_config(tmp_path / "config.json")
    output_wav = tmp_path / "output.wav"
    output_wav.write_bytes(b"existing-output-sentinel")
    result = invoke_cli(
        [
            str(input_wav),
            str(config_path),
            str(output_wav),
            "--output-format",
            "pcm8",
            "--json",
        ]
    )
    payload = error_payload(result)
    assert result.returncode == 41
    assert payload["error_code"] == "E_BIT_DEPTH"
    assert output_wav.read_bytes() == b"existing-output-sentinel"
    assert not list(tmp_path.glob(f".{output_wav.name}.tmp-*"))


def test_business_error_codes_for_bad_inputs(tmp_path: Path, demo_files) -> None:
    input_wav, config_path = demo_files

    flac_path = tmp_path / "not_wav.flac"
    data = np.zeros((48000, 2), dtype=np.float64)
    sf.write(str(flac_path), data, 48000, format="FLAC")
    result = invoke_cli(
        [
            str(flac_path),
            str(config_path),
            str(tmp_path / "out1.wav"),
            "--output-format",
            "float32",
            "--json",
        ]
    )
    assert error_payload(result)["error_code"] == "E_INPUT_FORMAT"

    missing_config = tmp_path / "missing.json"
    missing_config.write_text(
        json.dumps({"sources": [{"id": "missing-position", "channel": 0}]}),
        encoding="utf-8",
    )
    result = invoke_cli(
        [
            str(input_wav),
            str(missing_config),
            str(tmp_path / "out2.wav"),
            "--output-format",
            "float32",
            "--json",
        ]
    )
    payload = error_payload(result)
    assert payload["error_code"] == "E_CONFIG_MISSING_FIELD"

    version_config = tmp_path / "version.json"
    version_config.write_text(
        json.dumps(
            {
                "hrtf_version": "local-hrtf-v9",
                "sources": [{"position": [1, 0, 2]}],
            }
        ),
        encoding="utf-8",
    )
    result = invoke_cli(
        [
            str(input_wav),
            str(version_config),
            str(tmp_path / "out3.wav"),
            "--output-format",
            "float32",
            "--json",
        ]
    )
    payload = error_payload(result)
    assert result.returncode == 30
    assert payload["error_code"] == "E_HRTF_VERSION"


def test_memory_limit_returns_business_error(tmp_path: Path) -> None:
    input_wav = make_stereo_wav(tmp_path / "input.wav", subtype="PCM_16")
    config_path = make_static_config(tmp_path / "config.json")
    output_wav = tmp_path / "output.wav"
    output_wav.write_bytes(b"sentinel")
    result = invoke_cli(
        [
            str(input_wav),
            str(config_path),
            str(output_wav),
            "--output-format",
            "float32",
            "--block-size",
            "50000",
            "--json",
        ]
    )
    payload = error_payload(result)
    assert result.returncode == 50
    assert payload["error_code"] == "E_MEMORY_LIMIT"
    assert (
        payload["details"]["estimated_peak_bytes"] > payload["details"]["limit_bytes"]
    )
    assert output_wav.read_bytes() == b"sentinel"
    assert not Path(str(output_wav) + ".state.json").exists()


def test_resume_rejects_changed_input_hash(tmp_path: Path) -> None:
    input_wav = make_stereo_wav(tmp_path / "input.wav", duration=0.5)
    config_path = make_static_config(tmp_path / "config.json")
    output_wav = tmp_path / "output.wav"
    stopped = invoke_cli(
        [
            str(input_wav),
            str(config_path),
            str(output_wav),
            "--output-format",
            "float32",
            "--block-size",
            "128",
            "--stop-after-blocks",
            "5",
            "--json",
        ]
    )
    assert error_payload(stopped)["error_code"] == "E_INTERRUPTED"

    data, sample_rate = sf.read(str(input_wav), dtype="float64", always_2d=True)
    data[0, 0] += 0.001
    sf.write(str(input_wav), data, sample_rate, format="WAV", subtype="PCM_16")
    result = invoke_cli(
        [
            str(input_wav),
            str(config_path),
            str(output_wav),
            "--output-format",
            "float32",
            "--block-size",
            "128",
            "--json",
        ]
    )
    payload = error_payload(result)
    assert result.returncode == 61
    assert payload["error_code"] == "E_STATE_MISMATCH"
    assert "input_sha256" in payload["details"]["mismatches"]


def test_resume_rejects_changed_config_hash(tmp_path: Path) -> None:
    input_wav = make_stereo_wav(tmp_path / "input.wav", duration=0.5)
    config_path = make_static_config(tmp_path / "config.json")
    output_wav = tmp_path / "output.wav"
    stopped = invoke_cli(
        [
            str(input_wav),
            str(config_path),
            str(output_wav),
            "--output-format",
            "float32",
            "--block-size",
            "128",
            "--stop-after-blocks",
            "5",
            "--json",
        ]
    )
    assert error_payload(stopped)["error_code"] == "E_INTERRUPTED"

    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["note"] = "semantic content unchanged, bytes changed"
    config_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    result = invoke_cli(
        [
            str(input_wav),
            str(config_path),
            str(output_wav),
            "--output-format",
            "float32",
            "--block-size",
            "128",
            "--json",
        ]
    )
    payload = error_payload(result)
    assert result.returncode == 61
    assert payload["error_code"] == "E_STATE_MISMATCH"
    assert "source_config_sha256" in payload["details"]["mismatches"]


def test_doppler_phase_drift_is_below_one_sample(tmp_path: Path) -> None:
    sample_rate = 48000
    frames = 96000
    input_wav = make_stereo_wav(
        tmp_path / "input.wav", sample_rate=sample_rate, frames=frames
    )
    config_path = make_static_config(
        tmp_path / "config.json",
        positions=[
            [0.0, 0.0, 0.0, 5.0],
            [4.0, 0.0, 0.0, 1.0],
        ],
        channel=0,
    )
    output_wav = tmp_path / "output.wav"
    block_size = 512
    block_count = math.ceil(frames / block_size)
    result = invoke_cli(
        [
            str(input_wav),
            str(config_path),
            str(output_wav),
            "--output-format",
            "float32",
            "--block-size",
            str(block_size),
            "--listener-pos",
            "0",
            "0",
            "0",
            "--stop-after-blocks",
            str(block_count),
            "--json",
        ]
    )
    assert error_payload(result)["error_code"] == "E_INTERRUPTED"
    state = json.loads(
        Path(str(output_wav) + ".state.json").read_text(encoding="utf-8")
    )
    expected_rate = 343.0 / 342.0
    ideal_phase = frames * expected_rate
    drift = abs(float(state["phase"]) - ideal_phase)
    assert drift <= 1.0
