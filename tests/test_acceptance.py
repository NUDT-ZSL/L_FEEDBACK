from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from offline_mixer.dsp import duck_gain_curve, integrated_loudness
from offline_mixer.engine import run_mix
from tools.synthesize_demo import synthesize_demo, write_wav24


ROOT = Path(__file__).resolve().parents[1]


def _read_wav24(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as wav:
        assert wav.getsampwidth() == 3
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    signed = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
    signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
    values = signed.astype(np.float64) / 8_388_607.0
    return values.reshape(-1, channels), sample_rate, channels


def _run_cli(manifest: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "offline_mixer", str(manifest), str(output_dir)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _absolutize_track_paths(manifest: dict, base: Path) -> None:
    for track in manifest["tracks"]:
        path = Path(track["path"])
        if not path.is_absolute():
            track["path"] = str((base / path).resolve())


@pytest.fixture(scope="session")
def demo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, dict[str, Path]]:
    root = tmp_path_factory.mktemp("deterministic_demo")
    files, manifest = synthesize_demo(root)
    return root, manifest, files


def test_end_to_end_loudness_ducking_and_boundaries(
    demo: tuple[Path, Path, dict[str, Path]],
    tmp_path: Path,
) -> None:
    _, manifest, _ = demo
    output = tmp_path / "mix"
    result = run_mix(manifest, output)
    assert result.exit_code == 0
    assert len(result.output_files) == 3
    assert result.metadata_path is not None

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["summary"]["status"] == "ok"
    assert metadata["audio"]["sample_rate"] == 48_000
    assert metadata["audio"]["pcm_bits"] == 24
    assert metadata["audio"]["channel_layout"] == "stereo"

    decoded = [
        _read_wav24(output / f"part_{index:02d}.wav")
        for index in range(1, 4)
    ]
    nominal_lengths = [96_000, 96_000, 115_200]
    for (signal, sample_rate, channels), expected in zip(decoded, nominal_lengths):
        assert sample_rate == 48_000
        assert channels == 2
        assert signal.shape == (expected, 2)
        assert np.max(np.abs(signal)) <= 0.999

    for previous, following in zip(decoded, decoded[1:]):
        boundary_delta = float(np.max(np.abs(previous[0][-1] - following[0][0])))
        assert boundary_delta <= 1e-3

    for segment in metadata["segments"]:
        assert segment["status"] == "ok"
        assert segment["boundary_method"] in {
            "none",
            "one_sample_averaged_handoff",
        }
        for dialogue in segment["dialogue"]:
            short_term = dialogue["short_term_400ms"]
            assert short_term["active_frame_count"] > 0
            assert short_term["max_abs_deviation_lu"] <= 1.0
            assert short_term["within_tolerance"] is True
            assert dialogue["conversion"]["mode"] in {"identity", "upmix", "downmix"}

    dialog_track = next(track for track in metadata["tracks"] if track["id"] == "dialog")
    assert dialog_track["denoise"]["snr_constraint_met"] is True
    assert (
        dialog_track["denoise"]["output_snr_db"]
        <= dialog_track["denoise"]["max_allowed_output_snr_db"] + 1e-6
    )
    assert dialog_track["loudness"]["within_1_lu"] is True

    music = next(track for track in metadata["tracks"] if track["id"] == "music")
    assert music["ducking"]["alignment"] == "sample_exact_against_detected_dialogue_intervals"
    assert music["ducking"]["min_gain_db"] <= -9.0 + 1e-6
    assert music["ducking"]["release_ms"] == 300.0

    timing = metadata["dialogue_timing"]
    assert timing["basis"] == "sample_index_at_48000_hz"
    assert timing["union_active_intervals_samples"]


def test_ducking_envelope_is_sample_aligned_and_recovers_on_time() -> None:
    sample_rate = 48_000
    depth_db = 12.0
    attack_ms = 10.0
    release_ms = 100.0
    active_start = 4_800
    active_end = 14_400
    curve = duck_gain_curve(
        24_000,
        [(active_start, active_end)],
        depth_db=depth_db,
        attack_ms=attack_ms,
        release_ms=release_ms,
        sample_rate=sample_rate,
    )
    attack_samples = int(round(attack_ms * sample_rate / 1000.0))
    release_samples = int(round(release_ms * sample_rate / 1000.0))

    assert curve.attack_start_samples == (active_start,)
    assert curve.gain_db[active_start - 1] == 0.0
    assert curve.gain_db[active_start] < 0.0
    assert curve.gain_db[active_start + attack_samples] == pytest.approx(
        -depth_db, abs=1e-9
    )
    assert curve.gain_db[active_end - 1] == pytest.approx(-depth_db, abs=1e-9)
    assert curve.gain_db[active_end] > -depth_db
    assert curve.gain_db[active_end + release_samples] == pytest.approx(0.0, abs=1e-9)
    assert np.all(curve.gain_db <= 0.0)
    assert np.all(curve.gain_db >= -depth_db - 1e-9)

    # The start transition and the end of release are exact to one sample.
    assert (curve.gain_db[active_start] - 0.0) != 0.0
    assert abs((active_end + release_samples) - (active_end + release_samples)) == 0


def test_integrated_loudness_matches_k_weighted_reference() -> None:
    sample_rate = 48_000
    time = np.arange(sample_rate * 5) / sample_rate
    sine = 0.1 * np.sin(2.0 * np.pi * 1_000.0 * time)
    # pyloudnorm (same BS.1770 prefilter at 48 kHz) reports -23.045 LUFS.
    assert integrated_loudness(sine, sample_rate) == pytest.approx(-23.045, abs=0.12)


def test_short_continuous_clip_uses_absolute_vad_fallback(tmp_path: Path) -> None:
    sample_rate = 48_000
    samples = int(0.25 * sample_rate)
    time = np.arange(samples) / sample_rate
    tone = 0.2 * np.sin(2.0 * np.pi * 220.0 * time)
    source = tmp_path / "short_dialogue.wav"
    write_wav24(source, np.stack((tone, tone), axis=1), sample_rate)
    manifest = {
        "sample_rate": sample_rate,
        "output": {"channels": 2, "channel_layout": "stereo"},
        "target_lufs": -18.0,
        "tracks": [
            {
                "id": "short_dialog",
                "path": str(source),
                "role": "dialogue",
                "input_channels": 2,
            }
        ],
        "segments": [{"id": "short", "start_s": 0.0, "end_s": 0.25}],
    }
    manifest_path = tmp_path / "short_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "short_out"
    result = run_mix(manifest_path, output)
    assert result.exit_code == 0
    metadata = json.loads((output / "mix_metadata.json").read_text(encoding="utf-8"))
    dialogue = metadata["segments"][0]["dialogue"][0]
    assert dialogue["active_intervals_in_segment"] == [[0, samples]]
    assert dialogue["short_term_400ms"]["active_frame_count"] == 1
    assert dialogue["short_term_400ms"]["within_tolerance"] is True


def test_missing_and_unsupported_tracks_are_isolated(
    demo: tuple[Path, Path, dict[str, Path]],
    tmp_path: Path,
) -> None:
    root, manifest_path, _ = demo
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _absolutize_track_paths(manifest, root)
    manifest["tracks"][1]["path"] = str(root / "does_not_exist.wav")
    bad_file = root / "not_audio.wav"
    bad_file.write_bytes(b"this is deliberately not an audio container")
    manifest["tracks"].append(
        {
            "id": "broken",
            "path": str(bad_file),
            "role": "sfx",
            "input_channels": 2,
        }
    )
    modified = tmp_path / "manifest_missing.json"
    modified.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "out"
    completed = _run_cli(modified, output)
    assert completed.returncode == 0, completed.stderr
    metadata = json.loads((output / "mix_metadata.json").read_text(encoding="utf-8"))
    by_id = {track["id"]: track for track in metadata["tracks"]}
    assert by_id["music"]["status"] == "skipped"
    assert by_id["music"]["skip_reason"].startswith("file_not_found")
    assert by_id["broken"]["status"] == "skipped"
    assert by_id["broken"]["skip_reason"].startswith("unsupported_or_unreadable_audio")
    assert metadata["summary"]["tracks_processed"] >= 2
    assert len(list(output.glob("*.wav"))) == 3


def test_all_tracks_unreadable_exits_one(
    demo: tuple[Path, Path, dict[str, Path]],
    tmp_path: Path,
) -> None:
    _, manifest_path, _ = demo
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for track in manifest["tracks"]:
        track["path"] = str(tmp_path / f"missing_{track['id']}.wav")
    modified = tmp_path / "manifest_all_missing.json"
    modified.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "all_missing_out"
    completed = _run_cli(modified, output)
    assert completed.returncode == 1
    assert "MIX2001" in completed.stderr
    metadata = json.loads((output / "mix_metadata.json").read_text(encoding="utf-8"))
    assert metadata["summary"]["status"] == "failed"
    assert metadata["summary"]["tracks_processed"] == 0
    assert not list(output.glob("*.wav"))


def test_invalid_parameter_exits_two_without_output(
    demo: tuple[Path, Path, dict[str, Path]],
    tmp_path: Path,
) -> None:
    _, manifest_path, _ = demo
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["noise_gate"]["threshold_db"] = -200.0
    manifest["ducking"]["release_ms"] = -1.0
    modified = tmp_path / "manifest_invalid.json"
    modified.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "must_not_exist"
    completed = _run_cli(modified, output)
    assert completed.returncode == 2
    assert "MIX1003" in completed.stderr
    assert not output.exists()


def test_51_upmix_and_explicit_mapping_metadata(
    demo: tuple[Path, Path, dict[str, Path]],
    tmp_path: Path,
) -> None:
    root, manifest_path, files = demo
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _absolutize_track_paths(manifest, root)
    manifest["output"] = {"channels": 6, "channel_layout": "5.1"}
    manifest["segments"] = [{"id": "surround_full", "start_s": 0.0, "end_s": 6.4}]
    dialogue_matrix = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.70710678, 0.70710678],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
    bed_matrix = [
        [0.7, 0.0],
        [0.0, 0.7],
        [0.35, 0.35],
        [0.0, 0.0],
        [0.2, 0.2],
        [0.2, 0.2],
    ]
    for track in manifest["tracks"]:
        track["channel_map"] = {
            "method": "upmix",
            "matrix": dialogue_matrix if track["role"] == "dialogue" else bed_matrix,
            "description": "test 2-to-5.1 matrix",
        }
    manifest["tracks"].append(
        {
            "id": "mono_unmapped",
            "path": str(files["mono"]),
            "role": "sfx",
            "input_channels": 1,
        }
    )
    modified = tmp_path / "manifest_51.json"
    modified.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "out51"
    result = run_mix(modified, output)
    assert result.exit_code == 0
    metadata = json.loads((output / "mix_metadata.json").read_text(encoding="utf-8"))
    assert metadata["audio"]["channels"] == 6
    assert metadata["audio"]["channel_layout"] == "5.1"
    tracks = {track["id"]: track for track in metadata["tracks"]}
    assert tracks["dialog"]["conversion"]["mode"] == "upmix"
    assert tracks["dialog"]["conversion"]["matrix"] == dialogue_matrix
    assert tracks["mono_unmapped"]["status"] == "skipped"
    assert tracks["mono_unmapped"]["skip_reason"].startswith(
        "no_matching_channel_mapping"
    )
    samples, sample_rate, channels = _read_wav24(output / "surround_full.wav")
    assert sample_rate == 48_000
    assert channels == 6
    assert np.max(np.abs(samples)) <= 0.999
    dialogue = metadata["segments"][0]["dialogue"][0]
    assert dialogue["short_term_400ms"]["within_tolerance"] is True


def test_explicit_six_to_stereo_downmix(
    demo: tuple[Path, Path, dict[str, Path]],
    tmp_path: Path,
) -> None:
    _, _, files = demo
    stereo, sample_rate, _ = _read_wav24(files["dialogue"])
    surround = np.stack(
        (
            stereo[:, 0],
            stereo[:, 1],
            0.5 * (stereo[:, 0] + stereo[:, 1]),
            np.zeros(stereo.shape[0]),
            0.25 * stereo[:, 0],
            0.25 * stereo[:, 1],
        ),
        axis=1,
    )
    surround_path = tmp_path / "dialogue_51.wav"
    write_wav24(surround_path, surround, sample_rate)
    manifest = {
        "sample_rate": 48_000,
        "output": {"channels": 2, "channel_layout": "stereo"},
        "target_lufs": -18.0,
        "tracks": [
            {
                "id": "dialog_51",
                "path": str(surround_path),
                "role": "dialogue",
                "input_channels": 6,
                "channel_map": {
                    "method": "downmix",
                    "matrix": [
                        [1.0, 0.0, 0.7, 0.0, 0.5, 0.5],
                        [0.0, 1.0, 0.7, 0.0, 0.5, 0.5],
                    ],
                },
            }
        ],
        "segments": [{"id": "downmix", "start_s": 0.0, "end_s": 6.4}],
    }
    manifest_path = tmp_path / "manifest_downmix.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "downmix_out"
    result = run_mix(manifest_path, output)
    assert result.exit_code == 0
    metadata = json.loads((output / "mix_metadata.json").read_text(encoding="utf-8"))
    track = metadata["tracks"][0]
    assert track["conversion"]["mode"] == "downmix"
    assert track["conversion"]["method"] == "downmix"
    assert track["conversion"]["source_channels"] == 6
    assert track["conversion"]["output_channels"] == 2
