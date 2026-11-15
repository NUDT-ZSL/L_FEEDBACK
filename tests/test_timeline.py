"""Timeline structural and range validation error codes."""

from __future__ import annotations

import json
import os

import pytest

from audiopack.errors import E_MISSING_CLIP, E_RANGE, E_TIMELINE_FORMAT, PipelineError
from audiopack.timeline import load_timeline
from tests.samples import sine_wave, write_wav


def _make_dir(tmp_path, events, sample_rate=48000, wavs=("a.wav",)):
    root = str(tmp_path)
    for name in wavs:
        write_wav(os.path.join(root, name), sine_wave(0.1, 440, 48000), 48000, "PCM_16")
    with open(os.path.join(root, "timeline.json"), "w", encoding="utf-8") as handle:
        json.dump({"sample_rate": sample_rate, "events": events}, handle)
    return root


def _event(**overrides):
    base = {
        "start_ms": 0,
        "end_ms": 100,
        "clip": "a.wav",
        "fade_in_ms": 0,
        "fade_out_ms": 0,
        "gain_db": 0,
        "priority": 0,
        "loop": False,
    }
    base.update(overrides)
    return base


def test_missing_timeline_file(tmp_path):
    with pytest.raises(PipelineError) as exc:
        load_timeline(str(tmp_path))
    assert exc.value.code == E_TIMELINE_FORMAT


def test_invalid_json(tmp_path):
    (tmp_path / "timeline.json").write_text("{not json")
    with pytest.raises(PipelineError) as exc:
        load_timeline(str(tmp_path))
    assert exc.value.code == E_TIMELINE_FORMAT


def test_wrong_field_type(tmp_path):
    root = _make_dir(tmp_path, [_event(start_ms="soon")])
    with pytest.raises(PipelineError) as exc:
        load_timeline(root)
    assert exc.value.code == E_TIMELINE_FORMAT
    assert exc.value.details["field"] == "start_ms"


def test_boolean_rejected_as_priority(tmp_path):
    root = _make_dir(tmp_path, [_event(priority=True)])
    with pytest.raises(PipelineError) as exc:
        load_timeline(root)
    assert exc.value.code == E_TIMELINE_FORMAT


def test_end_before_start_is_range_error(tmp_path):
    root = _make_dir(tmp_path, [_event(start_ms=200, end_ms=100)])
    with pytest.raises(PipelineError) as exc:
        load_timeline(root)
    assert exc.value.code == E_RANGE


def test_fades_exceed_duration(tmp_path):
    root = _make_dir(tmp_path, [_event(fade_in_ms=60, fade_out_ms=60)])
    with pytest.raises(PipelineError) as exc:
        load_timeline(root)
    assert exc.value.code == E_RANGE


def test_gain_out_of_range(tmp_path):
    root = _make_dir(tmp_path, [_event(gain_db=999)])
    with pytest.raises(PipelineError) as exc:
        load_timeline(root)
    assert exc.value.code == E_RANGE


def test_missing_clip_file(tmp_path):
    root = _make_dir(tmp_path, [_event(clip="ghost.wav")])
    with pytest.raises(PipelineError) as exc:
        load_timeline(root)
    assert exc.value.code == E_MISSING_CLIP
    assert exc.value.details["event_index"] == 0


def test_defaults_are_filled(tmp_path):
    root = _make_dir(
        tmp_path,
        [
            {
                "start_ms": 0,
                "end_ms": 100,
                "clip": "a.wav",
                "fade_in_ms": 0,
                "fade_out_ms": 0,
                "gain_db": 0,
            }
        ],
    )
    events, sr = load_timeline(root)
    assert sr == 48000
    assert events[0].priority == 0
    assert events[0].loop is False
