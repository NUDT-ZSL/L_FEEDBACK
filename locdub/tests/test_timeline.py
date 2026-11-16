import json

import pytest

from locdub.timeline import TimelineFatalError, load_timeline, validate_segments


def _write(tmp_path, payload):
    path = tmp_path / "timeline.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_flattens_nested_groups(tmp_path):
    path = _write(tmp_path, {
        "segments": [
            {"id": "g1", "segments": [
                {"id": "a", "start_ms": 0, "end_ms": 100, "audio": "a.wav"},
                {"id": "g2", "segments": [
                    {"id": "b", "start_ms": 200, "end_ms": 300,
                     "audio": "b.wav"},
                ]},
            ]},
        ]
    })
    segments, _ = load_timeline(path)
    assert [s.segment_id for s in segments] == ["a", "b"]
    assert segments[0].group == "g1"
    assert segments[1].group == "g1/g2"


def test_default_and_segment_ratio_ranges(tmp_path):
    path = _write(tmp_path, {
        "default_ratio_range": [0.6, 1.7],
        "segments": [
            {"id": "a", "start_ms": 0, "end_ms": 100, "audio": "a.wav"},
            {"id": "b", "start_ms": 200, "end_ms": 300, "audio": "b.wav",
             "ratio_range": [0.9, 1.1]},
        ],
    })
    segments, _ = load_timeline(path)
    assert segments[0].min_ratio == 0.6
    assert segments[0].max_ratio == 1.7
    assert segments[1].min_ratio == 0.9


def test_bad_json_is_fatal(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(TimelineFatalError):
        load_timeline(path)


def test_missing_audio_field_is_fatal(tmp_path):
    path = _write(tmp_path, {"segments": [
        {"id": "a", "start_ms": 0, "end_ms": 100},
    ]})
    with pytest.raises(TimelineFatalError):
        load_timeline(path)


def test_non_integer_time_is_fatal(tmp_path):
    path = _write(tmp_path, {"segments": [
        {"id": "a", "start_ms": 0.5, "end_ms": 100, "audio": "a.wav"},
    ]})
    with pytest.raises(TimelineFatalError):
        load_timeline(path)


def test_validation_detects_reverse_overlap_and_out_of_range(tmp_path):
    path = _write(tmp_path, {
        "video_duration_ms": 1000,
        "segments": [
            {"id": "ok", "start_ms": 0, "end_ms": 100, "audio": "ok.wav"},
            {"id": "reverse", "start_ms": 200, "end_ms": 100,
             "audio": "r.wav"},
            {"id": "overlap", "start_ms": 50, "end_ms": 120,
             "audio": "o.wav"},
            {"id": "outside", "start_ms": 900, "end_ms": 1100,
             "audio": "x.wav"},
        ],
    })
    segments, meta = load_timeline(path)
    report = validate_segments(segments, meta["video_duration_ms"])
    valid_ids = {s.segment_id for s in report.valid}
    invalid_ids = {seg_id for seg_id, _ in report.invalid}
    assert valid_ids == {"ok"}
    assert invalid_ids == {"reverse", "overlap", "outside"}
