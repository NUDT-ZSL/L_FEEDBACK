from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import soundfile as sf

from game_audio_mixer.cli import main
from game_audio_mixer.dsp import (
    SAMPLE_RATE,
    true_peak,
)
from game_audio_mixer.engine import _build_mix, resolve_track_states
from game_audio_mixer.model import EventSpec, RenderedEvent, TrackConfig


def _sine(rate: int, seconds: float, frequency: float, amplitude: float) -> np.ndarray:
    time = np.arange(int(round(rate * seconds)), dtype=np.float64) / rate
    return (amplitude * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)


def _write_json(path: Path, document: Dict[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def _run(
    tmp_path: Path,
    timeline: Dict[str, Any],
    *extra_args: str,
    output_name: str = "out",
) -> tuple[int, Path, Path, Dict[str, Any]]:
    timeline_path = tmp_path / "timeline.json"
    _write_json(timeline_path, timeline)
    output_dir = tmp_path / output_name
    arguments: List[str] = [
        "--input-dir",
        str(tmp_path),
        "--timeline",
        str(timeline_path),
        "--output-dir",
        str(output_dir),
    ]
    arguments.extend(extra_args)
    exit_code = main(arguments)
    report_path = output_dir / "loudness_report.json"
    wav_path = output_dir / "mix.wav"
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    return exit_code, wav_path, report_path, report


def _base_timeline(assets: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "title": "deterministic-test",
        "duration_ms": 5000,
        "tracks": [
            {"id": "sfx", "name": "SFX", "mute": False, "solo": False},
            {"id": "amb", "name": "Ambience", "mute": True, "solo": False},
            {"id": "vo", "name": "Voice", "mute": False, "solo": False},
        ],
        "assets": assets,
        "events": events,
    }


def test_end_to_end_resampling_alignment_cache_and_error_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    sf.write(
        audio_dir / "step44.wav",
        _sine(44100, 0.5, 440.0, 0.2),
        44100,
        subtype="PCM_24",
    )
    sf.write(
        audio_dir / "amb22.wav",
        _sine(22050, 0.3, 220.0, 0.05),
        22050,
        subtype="PCM_16",
    )
    sf.write(
        audio_dir / "voice48.wav",
        _sine(48000, 0.3, 880.0, 0.1),
        48000,
        subtype="PCM_24",
    )
    sf.write(
        audio_dir / "unsupported37.wav",
        _sine(37000, 0.2, 440.0, 0.1),
        37000,
        subtype="PCM_16",
    )
    (audio_dir / "corrupt.wav").write_bytes(b"this is not a wav file")

    timeline = _base_timeline(
        [
            {"id": "step", "path": "audio/step44.wav"},
            {"id": "ambience", "path": "audio/amb22.wav"},
            {"id": "voice", "path": "audio/voice48.wav"},
            {"id": "unsupported", "path": "audio/unsupported37.wav"},
            {"id": "corrupt", "path": "audio/corrupt.wav"},
        ],
        [
            {"id": "step1", "type": "footstep", "time_ms": 100, "track": "sfx", "asset": "step"},
            {"id": "step2", "type": "footstep", "time_ms": 900, "track": "sfx", "asset": "step"},
            {"id": "amb1", "type": "ambient", "time_ms": 0, "track": "amb", "asset": "ambience"},
            {"id": "voice1", "type": "dialogue", "time_ms": 1800, "track": "vo", "asset": "voice", "gain_db": -3},
            {"id": "negative", "type": "footstep", "time_ms": -5, "track": "sfx", "asset": "step"},
            {"id": "outside", "type": "footstep", "time_ms": 6000, "track": "sfx", "asset": "step"},
            {"id": "missing", "type": "footstep", "time_ms": 300, "track": "sfx", "asset": "not_there"},
            {"id": "badrate", "type": "ambient", "time_ms": 400, "track": "sfx", "asset": "unsupported"},
            {"id": "baddecode", "type": "ambient", "time_ms": 500, "track": "sfx", "asset": "corrupt"},
        ],
    )

    read_counts: Dict[str, int] = {}
    original_read = sf.read

    def counted_read(path: str, *args: Any, **kwargs: Any) -> Any:
        read_counts[path] = read_counts.get(path, 0) + 1
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(sf, "read", counted_read)
    exit_code, wav_path, report_path, report = _run(tmp_path, timeline)

    assert exit_code == 0
    assert wav_path.exists() and report_path.exists()
    info = sf.info(str(wav_path))
    assert info.samplerate == 48000
    assert info.channels == 2
    assert info.subtype == "PCM_24"
    # The last valid event ends at 1800 + 300 ms.
    assert info.frames == 100800
    assert report["output"]["frames"] == 100800
    assert report["output"]["duration_s"] == pytest.approx(2.1)

    assets = {Path(item["path"]).name: item for item in report["assets"]}
    assert assets["step44.wav"]["original_sample_rate"] == 44100
    assert assets["amb22.wav"]["original_sample_rate"] == 22050
    assert assets["voice48.wav"]["original_sample_rate"] == 48000
    assert "resample_poly" in assets["step44.wav"]["resampling"]
    assert "none" in assets["voice48.wav"]["resampling"]
    assert assets["step44.wav"]["load_count"] == 1
    assert assets["step44.wav"]["referenced_by"] == ["step"]
    assert read_counts[str((audio_dir / "step44.wav").resolve())] == 1
    assert report["asset_cache"]["cache_hits"] == 1
    assert report["asset_cache"]["unique_files_referenced"] == 5
    assert report["asset_cache"]["successful_loads"] == 3
    assert report["asset_cache"]["failed_loads"] == 2

    event_ids = {event["event_id"]: event for event in report["events"]}
    assert event_ids["step1"]["status"] == "mixed"
    assert event_ids["step2"]["status"] == "mixed"
    assert event_ids["amb1"]["status"] == "skipped"
    assert event_ids["voice1"]["status"] == "mixed"
    assert event_ids["outside"]["status"] == "skipped"
    assert event_ids["missing"]["status"] == "skipped"
    assert event_ids["badrate"]["status"] == "skipped"
    assert event_ids["baddecode"]["status"] == "skipped"
    assert {"negative", "outside", "missing", "badrate", "baddecode"}.issubset(
        {error.get("event_id") for error in report["errors"]}
    )
    assert any("unsupported sample rate 37000" in error["reason"] for error in report["errors"])
    assert any("cannot inspect/decode" in error["reason"] for error in report["errors"])

    mixed, _ = sf.read(str(wav_path), dtype="float64", always_2d=True)
    assert float(np.max(np.abs(mixed[:4800]))) < 1e-8
    assert float(np.max(np.abs(mixed[4800:4810]))) > 1e-4
    assert report["target"]["achieved"] is True
    assert abs(report["loudness"]["integrated_lufs"] - (-23.0)) <= 0.5
    assert report["loudness"]["short_term"]
    assert true_peak(mixed)[0] <= 1.0


def test_initial_mute_and_cli_solo_override(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    sf.write(audio_dir / "a.wav", _sine(48000, 0.5, 330.0, 0.25), 48000, subtype="PCM_24")
    sf.write(audio_dir / "b.wav", _sine(48000, 0.5, 550.0, 0.25), 48000, subtype="PCM_24")
    sf.write(audio_dir / "c.wav", _sine(48000, 0.5, 770.0, 0.25), 48000, subtype="PCM_24")
    timeline = _base_timeline(
        [
            {"id": "a", "path": "audio/a.wav"},
            {"id": "b", "path": "audio/b.wav"},
            {"id": "c", "path": "audio/c.wav"},
        ],
        [
            {"id": "ea", "type": "footstep", "time_ms": 0, "track": "amb", "asset": "a"},
            {"id": "eb", "type": "footstep", "time_ms": 500, "track": "sfx", "asset": "b"},
            {"id": "ec", "type": "footstep", "time_ms": 1000, "track": "vo", "asset": "c"},
        ],
    )

    exit_code, wav_initial, _, report_initial = _run(
        tmp_path, timeline, output_name="initial"
    )
    assert exit_code == 0
    states = {track["id"]: track for track in report_initial["tracks"]}
    assert states["amb"]["mute"] is True
    assert states["amb"]["audible"] is False
    assert states["amb"]["mute_source"] == "timeline"

    audio_initial, _ = sf.read(str(wav_initial), dtype="float64")
    assert np.sqrt(np.mean(audio_initial[:24000] ** 2)) < 1e-6
    assert np.sqrt(np.mean(audio_initial[24000:48000] ** 2)) > 1e-3

    exit_code, wav_solo, _, report_solo = _run(
        tmp_path, timeline, "--solo", "sfx", output_name="solo"
    )
    assert exit_code == 0
    states = {track["id"]: track for track in report_solo["tracks"]}
    assert states["sfx"]["audible"] is True
    assert states["amb"]["audible"] is False
    assert states["vo"]["audible"] is False
    assert states["sfx"]["solo_source"] == "cli"
    audio_solo, _ = sf.read(str(wav_solo), dtype="float64")
    assert np.sqrt(np.mean(audio_solo[:24000] ** 2)) < 1e-6
    assert np.sqrt(np.mean(audio_solo[24000:48000] ** 2)) > 1e-3
    assert np.sqrt(np.mean(audio_solo[48000:72000] ** 2)) < 1e-6


def test_solo_has_priority_over_mute() -> None:
    tracks = [
        TrackConfig(id="muted", name="Muted", initial_mute=True),
        TrackConfig(
            id="muted_solo", name="Muted Solo", initial_mute=True, initial_solo=True
        ),
        TrackConfig(id="normal", name="Normal"),
    ]
    states, warnings = resolve_track_states(tracks, None, None)
    lookup = {state.id: state for state in states}
    assert warnings == []
    assert lookup["muted"].audible is False
    assert lookup["muted_solo"].audible is True
    assert lookup["normal"].audible is False


def test_overlapping_events_are_added_on_the_sample_grid() -> None:
    first_event = EventSpec(
        id="first",
        event_type="step",
        time_ms=0.0,
        track_id="sfx",
        asset_id="one",
    )
    second_event = EventSpec(
        id="second",
        event_type="step",
        time_ms=500.0,
        track_id="sfx",
        asset_id="one",
    )
    first = RenderedEvent(
        event=first_event,
        start_frame=0,
        end_frame=48000,
        audio=np.ones((48000, 2), dtype=np.float64),
        start_s=0.0,
        end_s=1.0,
        audible=True,
    )
    second = RenderedEvent(
        event=second_event,
        start_frame=24000,
        end_frame=72000,
        audio=np.ones((48000, 2), dtype=np.float64) * 0.5,
        start_s=0.5,
        end_s=1.5,
        audible=True,
    )
    mix = _build_mix([first, second], 72000)
    assert np.allclose(mix[:24000], 1.0)
    assert np.allclose(mix[24000:48000], 1.5)
    assert np.allclose(mix[48000:72000], 0.5)


def test_external_asset_manifest_and_clip_selection(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    sf.write(
        audio_dir / "external.wav",
        _sine(48000, 2.0, 440.0, 0.2),
        48000,
        subtype="PCM_24",
    )
    (tmp_path / "assets.json").write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "id": "external_tone",
                        "file": "audio/external.wav",
                        "offset_ms": 100,
                        "duration_ms": 500,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    timeline = {
        "title": "external-manifest",
        "duration_ms": 2000,
        "asset_manifest": "assets.json",
        "tracks": [{"id": "sfx", "mute": False, "solo": False}],
        "events": [
            {
                "id": "external1",
                "type": "tone",
                "time_ms": 250,
                "track": "sfx",
                "clip": "external_tone",
            }
        ],
    }
    exit_code, wav_path, _, report = _run(tmp_path, timeline)
    assert exit_code == 0
    assert report["assets"][0]["path"].endswith("external.wav")
    assert report["events"][0]["status"] == "mixed"
    assert report["events"][0]["end_s"] == pytest.approx(0.75)
    assert sf.info(str(wav_path)).frames == 36000


def test_limiter_trigger_records_track_event_and_time(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    bed_seconds = 4.0
    time = np.arange(int(SAMPLE_RATE * bed_seconds)) / SAMPLE_RATE
    bed = (0.035 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)
    sf.write(audio_dir / "bed.wav", bed, SAMPLE_RATE, subtype="PCM_24")
    click = np.zeros(9600, dtype=np.float32)
    click[1000:1020] = 0.98
    sf.write(audio_dir / "hot.wav", click, SAMPLE_RATE, subtype="PCM_24")
    events: List[Dict[str, Any]] = [
        {"id": "bed0", "type": "ambient", "time_ms": 0, "track": "amb", "asset": "bed"}
    ]
    for index, start_ms in enumerate((300, 900, 1500, 2100, 2700, 3300)):
        events.append(
            {
                "id": f"hit{index}",
                "type": "impact",
                "time_ms": start_ms + 40,
                "track": "sfx",
                "asset": "hot",
                "gain_db": -2,
            }
        )
    timeline = _base_timeline(
        [
            {"id": "bed", "path": "audio/bed.wav"},
            {"id": "hot", "path": "audio/hot.wav"},
        ],
        events,
    )
    timeline["tracks"][1]["mute"] = False

    exit_code, wav_path, _, report = _run(
        tmp_path, timeline, "--target-lufs", "-14"
    )
    assert exit_code == 0
    assert report["processing"]["limiter"]["enabled"] is True
    assert report["processing"]["limiter"]["trigger_count"] >= 1
    assert len(report["limiter_triggers"]) == report["processing"]["limiter"][
        "trigger_count"
    ]
    for trigger in report["limiter_triggers"]:
        assert trigger["track"]
        assert trigger["event_id"]
        assert trigger["time_s"] >= 0.0
        assert trigger["contributors"]
    assert report["loudness"]["true_peak_dbtp"] <= 0.0
    assert abs(report["loudness"]["integrated_lufs"] - (-14.0)) <= 0.5
    audio, _ = sf.read(str(wav_path), dtype="float64", always_2d=True)
    assert true_peak(audio)[0] <= 1.0


def test_fatal_timeline_returns_nonzero_and_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad_timeline = tmp_path / "broken.json"
    bad_timeline.write_text("{not valid json", encoding="utf-8")
    exit_code = main(
        [
            "--input-dir",
            str(tmp_path),
            "--timeline",
            str(bad_timeline),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code != 0
    assert "fatal:" in captured.err
    assert "cannot parse timeline JSON" in captured.err
    assert not (tmp_path / "out" / "mix.wav").exists()


def test_silence_when_all_events_are_invalid(tmp_path: Path) -> None:
    timeline = _base_timeline(
        [{"id": "step", "path": "audio/missing.wav"}],
        [
            {"id": "negative", "type": "step", "time_ms": -1, "track": "sfx", "asset": "step"},
            {"id": "missing", "type": "step", "time_ms": 100, "track": "sfx", "asset": "step"},
        ],
    )
    exit_code, wav_path, _, report = _run(tmp_path, timeline)
    assert exit_code == 0
    assert report["status"] == "silent"
    assert report["output"]["frames"] == 0
    assert report["loudness"]["integrated_lufs"] is None
    assert report["target"]["achieved"] is False
    assert sf.info(str(wav_path)).frames == 0
