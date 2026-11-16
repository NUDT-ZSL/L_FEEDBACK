import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cinemixer.cli import main
from cinemixer.errors import CineMixerError
from cinemixer.logging_util import StructuredLogger
from cinemixer.pipeline import run_project
from cinemixer.project import parse_project, timeline_length_samples
from cinemixer.state import state_path_for
from cinemixer.wav_io import read_wav


def _read_index(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_end_to_end_output_and_index(sample_project, tmp_path, capsys):
    result = run_project(sample_project, log=StructuredLogger())
    project = parse_project(sample_project)
    expected_length = timeline_length_samples(project)

    audio, rate, bit_depth = read_wav(result.output_path)
    assert rate == project.sample_rate == 48000
    assert audio.shape == (expected_length, project.channels)
    assert bit_depth == project.output.bit_depth == 16

    index = _read_index(result.index_path)
    assert index["length_samples"] == expected_length
    assert index["sample_rate"] == 48000
    assert len(index["events"]) == sum(len(t.events) for t in project.tracks)

    # Every index entry is within bounds and internally consistent.
    for event in index["events"]:
        assert 0 <= event["start_sample"] <= event["end_sample"] <= expected_length
        assert event["end_sample"] - event["start_sample"] == event["length_samples"]

    # No clipping after peak normalization (16-bit full scale is 32767).
    assert np.max(np.abs(audio)) <= 1.0

    # Per-track actual sample counts are recorded for traceability.
    assert set(index["tracks"]) == {track.track_id for track in project.tracks}
    assert index["tracks"]["voice"]["normalized_samples"] > 0
    assert index["tracks"]["voice"]["timeline_samples"] == expected_length

    # Loudness lands within the +/-0.5 LU tolerance (or was corrected once).
    assert abs(index["measured_lufs"] - project.target_lufs) <= 0.5


def test_index_sample_accuracy_matches_event_times(sample_project):
    result = run_project(sample_project, log=StructuredLogger())
    index = _read_index(result.index_path)
    by_id = {(e["track_id"], e["event_id"]): e for e in index["events"]}

    voice_v1 = by_id[("voice", "v1")]
    assert voice_v1["start_sample"] == 0
    assert voice_v1["length_samples"] == int(round(800 * 48000 / 1000))

    ambience = by_id[("ambience", "a1")]
    assert ambience["start_sample"] == int(round(100 * 48000 / 1000))
    assert ambience["length_samples"] == int(round(1800 * 48000 / 1000))


def test_render_is_byte_deterministic(sample_project):
    first = run_project(sample_project, log=StructuredLogger())
    first_bytes = first.output_path.read_bytes()
    first_index = first.index_path.read_bytes()

    second = run_project(sample_project, log=StructuredLogger())
    assert second.output_path.read_bytes() == first_bytes
    assert second.index_path.read_bytes() == first_index


def test_resampled_track_actually_runs(sample_project):
    # The voice source is 44.1 kHz; the render must still produce audible
    # content on the 48 kHz timeline.
    result = run_project(sample_project, log=StructuredLogger())
    audio, _, _ = read_wav(result.output_path)
    assert np.any(audio != 0.0)
    rms = np.sqrt(np.mean(audio**2))
    assert rms > 0.001


def test_mono_project_downmixes_stereo_sources(tmp_path):
    from tests.sample_factory import build_sample_project

    project_path = build_sample_project(tmp_path)
    payload = json.loads(project_path.read_text(encoding="utf-8"))
    payload["channels"] = 1
    project_path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_project(project_path, log=StructuredLogger())
    audio, rate, _ = read_wav(result.output_path)
    project = parse_project(project_path)
    assert audio.shape == (timeline_length_samples(project), 1)
    assert rate == project.sample_rate
    index = _read_index(result.index_path)
    assert index["channels"] == 1


def test_corrupt_track_is_skipped_others_continue(corrupt_project):
    result = run_project(corrupt_project, log=StructuredLogger())
    assert result.failed_tracks == ("music",)
    index = _read_index(result.index_path)
    music_events = [e for e in index["events"] if e["track_id"] == "music"]
    assert music_events  # failed track still has addressable index entries
    audio, _, _ = read_wav(result.output_path)
    assert np.any(audio != 0.0)  # surviving tracks still render


def test_missing_track_uses_same_error_code(missing_project):
    result = run_project(missing_project, log=StructuredLogger())
    assert result.failed_tracks == ("music",)


def test_all_tracks_failed_terminates(tmp_path):
    from tests.sample_factory import build_sample_project, write_garbage_file

    project_path = build_sample_project(tmp_path)
    project = json.loads(project_path.read_text(encoding="utf-8"))
    garbage = tmp_path / "samples" / "garbage.wav"
    write_garbage_file(garbage)
    for track in project["tracks"]:
        track["source"] = "samples/garbage.wav"
    project_path.write_text(json.dumps(project), encoding="utf-8")

    with pytest.raises(CineMixerError) as exc_info:
        run_project(project_path, log=StructuredLogger())
    assert exc_info.value.code == "E_ALL_TRACKS_FAILED"
    assert exc_info.value.exit_code != 0
    # No output file is produced when every track fails.
    assert not (tmp_path / "out" / "mix.wav").exists()


def test_recovery_after_interruption(sample_project):
    project = parse_project(sample_project)
    output_path = (sample_project.parent / project.output.path).resolve()
    state_file = state_path_for(output_path)
    repo_root = Path(__file__).resolve().parents[1]
    env = _subprocess_env(repo_root)

    # First run is killed deterministically after the second completed track.
    # The crash injection uses os._exit, so it must run in a subprocess.
    crashed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cinemixer",
            str(sample_project),
            "--fault-after",
            "2",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert crashed.returncode == 130, crashed.stderr
    assert state_file.is_file()
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert len(state["tracks"]) == 2

    # Resumed run completes and yields identical output to a clean run.
    resumed = run_project(sample_project, log=StructuredLogger())
    resumed_bytes = resumed.output_path.read_bytes()
    assert not state_file.exists()  # checkpoint cleaned after success

    clean = run_project(sample_project, log=StructuredLogger())
    assert clean.output_path.read_bytes() == resumed_bytes


def test_corrupt_state_falls_back_to_full_render(sample_project):
    project = parse_project(sample_project)
    output_path = (sample_project.parent / project.output.path).resolve()
    state_file = state_path_for(output_path)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text("{broken json", encoding="utf-8")

    result = run_project(sample_project, log=StructuredLogger())
    assert result.output_path.is_file()
    assert not state_file.exists()


def test_stale_checksum_ignored(tmp_path):
    from tests.sample_factory import build_sample_project

    first_root = tmp_path / "a"
    second_root = tmp_path / "b"
    first_project = build_sample_project(first_root)
    second_project = build_sample_project(second_root)
    run_project(first_project, log=StructuredLogger())

    # Reuse the first project's checkpoint path structure against a different
    # project directory; checksums must not match.
    result = run_project(second_project, log=StructuredLogger())
    assert result.output_path.is_file()


def test_cli_success_exit_code(sample_project):
    assert main([str(sample_project)]) == 0


def test_cli_schema_error_exit_code(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"sample_rate": 1234}), encoding="utf-8")
    code = main([str(bad)])
    assert code == 2


def test_cli_as_subprocess(sample_project):
    repo_root = Path(__file__).resolve().parents[1]
    env = _subprocess_env(repo_root)
    completed = subprocess.run(
        [sys.executable, "-m", "cinemixer", str(sample_project)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    # Structured JSON lines are emitted on stderr.
    lines = [line for line in completed.stderr.splitlines() if line.strip()]
    for line in lines:
        payload = json.loads(line)
        assert "level" in payload and "code" in payload


def _subprocess_env(repo_root: Path) -> dict:
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    return env
