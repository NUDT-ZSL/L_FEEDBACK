"""Acceptance and unit tests for the cutscene mixer."""

import json
import os
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cutscene_mixer import mixer
from cutscene_mixer.audio import load_track_audio
from cutscene_mixer.cli import main as cli_main
from cutscene_mixer.errors import SCHEMA_INVALID, SchemaError
from cutscene_mixer.loudness import integrated_loudness
from cutscene_mixer.schema import validate_descriptor
from cutscene_mixer.timeline import frame_to_sample, ms_to_samples

import generate_samples


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("samples")
    generate_samples.make_dialog(str(directory / "dialog.wav"))
    generate_samples.make_sfx(str(directory / "sfx.wav"))
    generate_samples.make_music(str(directory / "music.wav"))
    return directory


def _write_descriptor(directory, descriptor):
    path = directory / "descriptor.json"
    path.write_text(json.dumps(descriptor), encoding="utf-8")
    return str(path)


def _base_descriptor():
    return {
        "frame_rate": 30.0,
        "output": {"sample_rate": 48000, "bit_depth": 16, "channels": 2},
        "tracks": [
            {"id": "dialog", "path": "dialog.wav",
             "events": [{"id": "line_a", "track_sample": 0, "frame": 0}]},
        ],
    }


# --- time conversion -------------------------------------------------------

def test_frame_to_sample_rounding_half_up():
    assert frame_to_sample(30, 30.0, 48000) == 48000
    assert frame_to_sample(1, 30.0, 48000) == 1600
    assert frame_to_sample(1, 29.97, 48000) == 1602  # 1601.6 -> half-up
    assert frame_to_sample(0, 24.0, 44100) == 0


def test_ms_to_samples():
    assert ms_to_samples(10.0, 48000) == 480
    assert ms_to_samples(5.0, 48000) == 240


# --- schema validation -----------------------------------------------------

def test_schema_accepts_valid_descriptor():
    normalized = validate_descriptor(_base_descriptor())
    assert normalized["crossfade_ms"] == 10.0
    assert normalized["fade_ms"] == 5.0
    assert normalized["gain_limits_db"] == {"min": -20.0, "max": 10.0}


@pytest.mark.parametrize("mutate,paths", [
    (lambda d: d.update(frame_rate=0), ["frame_rate"]),
    (lambda d: d.update(frame_rate=-24), ["frame_rate"]),
    (lambda d: d["output"].update(sample_rate=7999), ["output.sample_rate"]),
    (lambda d: d["output"].update(sample_rate=192001), ["output.sample_rate"]),
    (lambda d: d["output"].update(sample_rate=48000.5), ["output.sample_rate"]),
    (lambda d: d["output"].update(bit_depth=20), ["output.bit_depth"]),
    (lambda d: d["output"].update(channels=3), ["output.channels"]),
    (lambda d: d.update(gain_limits_db={"min": 5, "max": -5}), ["gain_limits_db"]),
    (lambda d: d["tracks"][0].update(path=""), ["tracks[0].path"]),
    (lambda d: d["tracks"][0]["events"][0].update(track_sample=-1),
     ["tracks[0].events[0].track_sample"]),
    (lambda d: d["tracks"][0]["events"][0].update(frame=-2),
     ["tracks[0].events[0].frame"]),
])
def test_schema_rejects_invalid_fields(mutate, paths):
    descriptor = _base_descriptor()
    mutate(descriptor)
    with pytest.raises(SchemaError) as excinfo:
        validate_descriptor(descriptor)
    assert excinfo.value.paths == paths


# --- audio loading ---------------------------------------------------------

def test_load_resamples_and_duplicates_mono(sample_dir):
    audio = load_track_audio(str(sample_dir / "dialog.wav"), 48000, 2)
    assert audio.shape == (96000, 2)  # 2.0 s @ 48 kHz, mono -> stereo
    np.testing.assert_allclose(audio[:, 0], audio[:, 1])


def test_load_keeps_stereo(sample_dir):
    audio = load_track_audio(str(sample_dir / "sfx.wav"), 48000, 2)
    assert audio.shape == (72000, 2)


def test_load_failure_raises(tmp_path):
    with pytest.raises(Exception):
        load_track_audio(str(tmp_path / "missing.wav"), 48000, 2)


# --- fades and crossfade ---------------------------------------------------

def test_fades_shrink_to_half_segment():
    data = np.ones((100, 2))
    mixer.apply_fades(data, 240)  # fade longer than length//2
    assert data[0, 0] == 0.0
    assert data[-1, 0] == pytest.approx(0.02)
    # fade collapsed to 50 samples: middle of segment is the peak
    assert np.argmax(np.abs(data[:, 0])) in (49, 50)


def test_fades_do_not_cross_boundaries():
    data = np.ones((10, 2))
    mixer.apply_fades(data, 5)  # exactly length//2
    assert data[4, 0] > 0.0 and data[5, 0] > 0.0


def test_crossfade_shrinks_to_actual_overlap():
    audio = np.ones((1000, 2))
    events = [
        {"id": "a", "track_sample": 0, "frame": 0},
        # second anchor 100 samples before the first segment ends
        {"id": "b", "track_sample": 500, "frame": 30},  # abs 48000? no: 30fps
    ]
    # force abs positions manually instead
    segments = [
        mixer.Segment(events[0], 0, 0, 1000),
        mixer.Segment(events[1], 900, 500, 1000),
    ]
    out = mixer.assemble_track(audio, segments, crossfade_samples=480,
                               fade_samples=0, channels=2)
    # overlap is 100 < crossfade 480: total length = 900 + 500
    assert out.shape == (1400, 2)
    # crossfaded region stays bounded (no clipping, no silence)
    assert np.all(out[900:1000] >= 0.0)
    assert np.all(out[900:1000] <= 1.0)


def test_gap_between_segments_is_silent():
    audio = np.ones((500, 2))
    segments = [
        mixer.Segment({"id": "a"}, 0, 0, 500),
        mixer.Segment({"id": "b"}, 1000, 0, 500),
    ]
    out = mixer.assemble_track(audio, segments, crossfade_samples=480,
                               fade_samples=0, channels=2)
    assert out.shape == (1500, 2)
    np.testing.assert_array_equal(out[500:1000], np.zeros((500, 2)))


# --- loudness ---------------------------------------------------------------

def test_integrated_loudness_sine_close_to_expected():
    rate = 48000
    t = np.arange(rate, dtype=np.float64) / rate
    sine = 0.5 * np.sin(2 * np.pi * 997 * t)
    stereo = np.stack([sine, sine], axis=1)
    loudness = integrated_loudness(stereo)
    # 0.5 amplitude 997 Hz stereo sine: -0.691 + 10*log10(2*0.125) + K ~= -6.0
    assert loudness == pytest.approx(-6.0, abs=0.3)


def test_gain_clamp():
    limits = {"min": -20.0, "max": 10.0}
    assert mixer.clamp_db(15.0, limits) == 10.0
    assert mixer.clamp_db(-25.0, limits) == -20.0


# --- end to end -------------------------------------------------------------

def test_end_to_end_success(sample_dir, tmp_path):
    descriptor = generate_samples.build_descriptor()
    desc_path = _write_descriptor(sample_dir, descriptor)
    out_wav = tmp_path / "nested" / "mix.wav"   # directory auto-created
    out_report = tmp_path / "nested" / "report.json"

    exit_code = cli_main(["--input", desc_path,
                          "--output", str(out_wav),
                          "--report", str(out_report)])
    assert exit_code == 0

    info = sf.info(str(out_wav))
    assert info.samplerate == 48000
    assert info.channels == 2
    assert info.subtype == "PCM_16"

    report = json.loads(out_report.read_text())
    # music anchored at frame 15 (abs 24000) runs 3.0 s -> longest track
    assert report["mix"]["duration_samples"] == 24000 + 144000
    assert info.frames == report["mix"]["duration_samples"]
    assert report["mix"]["loudness_after_lufs"] == pytest.approx(-16.0, abs=0.1)

    # event positions in the report match the shared conversion basis
    for track in report["tracks"]:
        for event in track["events"]:
            assert event["absolute_sample"] == frame_to_sample(
                event["frame"], report["frame_rate"], report["sample_rate"])

    # per-track loudness after gain stays within +/-2 LU of the mean
    adjusted = [t["loudness_lufs"] + t["gain_db"] for t in report["tracks"]]
    mean = sum(adjusted) / len(adjusted)
    for value in adjusted:
        assert abs(value - mean) <= 2.0 + 1e-6


def test_event_onset_matches_report(sample_dir, tmp_path):
    """The sfx burst onset lands exactly at the reported absolute sample."""
    descriptor = {
        "frame_rate": 30.0,
        "output": {"sample_rate": 48000, "bit_depth": 24, "channels": 2},
        "tracks": [
            {"id": "sfx", "path": "sfx.wav",
             "events": [{"id": "explosion", "track_sample": 19200, "frame": 60}]},
        ],
    }
    desc_path = _write_descriptor(sample_dir, descriptor)
    out_wav = tmp_path / "mix.wav"
    out_report = tmp_path / "report.json"
    assert cli_main(["--input", desc_path, "--output", str(out_wav),
                     "--report", str(out_report)]) == 0

    report = json.loads(out_report.read_text())
    onset = report["tracks"][0]["events"][0]["absolute_sample"]
    assert onset == 96000
    assert sf.info(str(out_wav)).subtype == "PCM_24"

    data, _ = sf.read(str(out_wav), always_2d=True)
    fade = ms_to_samples(report["fade_ms"], report["sample_rate"])
    threshold = np.abs(data).max() * 0.05
    # silence before the anchor (only the fade ramp may start earlier? no:
    # the segment starts exactly at the anchor)
    assert np.abs(data[:onset - 1]).max() == 0.0
    # energy appears within the fade-in window after the anchor
    assert np.abs(data[onset:onset + fade + 8]).max() > 0.0
    first_hot = int(np.argmax(np.abs(data[:, 0]) > threshold))
    assert onset <= first_hot <= onset + fade + 8


def test_load_failure_exit_1_still_writes_outputs(sample_dir, tmp_path, capsys):
    descriptor = _base_descriptor()
    descriptor["tracks"].append(
        {"id": "ghost", "path": "nope.wav",
         "events": [{"id": "e", "track_sample": 0, "frame": 0}]})
    desc_path = _write_descriptor(sample_dir, descriptor)
    out_wav = tmp_path / "mix.wav"
    out_report = tmp_path / "report.json"

    exit_code = cli_main(["--input", desc_path, "--output", str(out_wav),
                          "--report", str(out_report)])
    assert exit_code == 1
    assert out_wav.exists() and out_report.exists()

    report = json.loads(out_report.read_text())
    statuses = {t["id"]: t["status"] for t in report["tracks"]}
    assert statuses == {"dialog": "ok", "ghost": "AUDIO_LOAD_FAIL"}


def test_schema_invalid_exit_2(sample_dir, tmp_path, capsys):
    descriptor = _base_descriptor()
    descriptor["frame_rate"] = 0
    descriptor["output"]["bit_depth"] = 20
    desc_path = _write_descriptor(sample_dir, descriptor)

    exit_code = cli_main(["--input", desc_path,
                          "--output", str(tmp_path / "mix.wav"),
                          "--report", str(tmp_path / "report.json")])
    assert exit_code == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"] == SCHEMA_INVALID
    assert "frame_rate" in payload["fields"]
    assert "output.bit_depth" in payload["fields"]
    assert not (tmp_path / "mix.wav").exists()


def test_target_lufs_override(sample_dir, tmp_path):
    desc_path = _write_descriptor(sample_dir, _base_descriptor())
    out_report = tmp_path / "report.json"
    exit_code = cli_main(["--input", desc_path,
                          "--output", str(tmp_path / "mix.wav"),
                          "--report", str(out_report),
                          "--target-lufs", "-14"])
    assert exit_code == 0
    report = json.loads(out_report.read_text())
    assert report["target_lufs"] == -14.0
    assert report["mix"]["loudness_after_lufs"] == pytest.approx(-14.0, abs=0.1)
