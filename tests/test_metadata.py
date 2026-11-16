import json

import pytest

from dub_replace.errors import ChannelLayoutError, MetadataError, OverlapError
from dub_replace.metadata import match_dialect_clips, parse_metadata

from tests.conftest import make_meta, stereo_channels, write_meta


def _write(tmp_path, payload):
    return write_meta(tmp_path, payload)


def test_parse_and_match(tmp_path):
    dialogue = [
        {"line_id": "L1", "start_ms": 0, "end_ms": 100, "text": "a"},
        {"line_id": "L2", "start_ms": 200, "end_ms": 300, "text": "b"},
    ]
    clips = [
        {"clip_id": "D2", "line_id": "L2", "start_ms": 200, "path": "d2.wav"},
        {"clip_id": "D1", "line_id": None, "start_ms": 30, "path": "d1.wav"},
        {"clip_id": "DX", "line_id": "LX", "start_ms": 9000, "path": "dx.wav"},
    ]
    meta = make_meta("m", "stereo", stereo_channels(), dialogue, clips)
    parsed = parse_metadata(_write(tmp_path, meta))
    matched, extras = match_dialect_clips(parsed)
    assert set(matched) == {"L1", "L2"}
    assert matched["L2"].clip_id == "D2"
    assert matched["L1"].clip_id == "D1"  # nearest temporal match
    assert [c.clip_id for c in extras] == ["DX"]


def test_overlap_detection(tmp_path):
    dialogue = [
        {"line_id": "L1", "start_ms": 0, "end_ms": 200, "text": "a"},
        {"line_id": "L2", "start_ms": 150, "end_ms": 300, "text": "b"},
    ]
    meta = make_meta("m", "stereo", stereo_channels(), dialogue, [])
    with pytest.raises(OverlapError) as exc:
        parse_metadata(_write(tmp_path, meta))
    assert exc.value.code == "E_OVERLAP"


def test_channel_layout_mismatch(tmp_path):
    dialogue = [{"line_id": "L1", "start_ms": 0, "end_ms": 100, "text": "a"}]
    meta = make_meta("m", "stereo", stereo_channels(), dialogue, [])
    meta["audio"]["channel_layout"] = "5.1"
    with pytest.raises(ChannelLayoutError) as exc:
        parse_metadata(_write(tmp_path, meta))
    assert exc.value.code == "E_CHANNEL"


def test_missing_required_field(tmp_path):
    meta = make_meta("m", "stereo", stereo_channels(), [], [])
    del meta["audio"]["sample_rate"]
    with pytest.raises(MetadataError) as exc:
        parse_metadata(_write(tmp_path, meta))
    assert exc.value.code == "E_METADATA"


def test_bad_json(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(MetadataError):
        parse_metadata(path)


def test_bad_stretch_limits(tmp_path):
    meta = make_meta("m", "stereo", stereo_channels(), [], [])
    meta["stretch_limits"] = {"min": 1.2, "max": 2.0}
    with pytest.raises(MetadataError):
        parse_metadata(_write(tmp_path, meta))
