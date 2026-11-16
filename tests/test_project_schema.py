import json

import pytest

from cinemixer.errors import CineMixerError
from cinemixer.project import build_project, parse_project, timeline_length_samples


def _base():
    return {
        "project_id": "p",
        "sample_rate": 48000,
        "channels": 2,
        "duration_ms": 1000,
        "target_lufs": -23.0,
        "output": {"path": "out/mix.wav", "format": "wav", "bit_depth": 16},
        "tracks": [
            {
                "track_id": "t1",
                "source": "a.wav",
                "gain_db": 0.0,
                "muted": False,
                "events": [
                    {"event_id": "e1", "start_ms": 0, "duration_ms": 100}
                ],
            }
        ],
    }


def test_valid_project_parses():
    project = build_project(_base())
    assert project.sample_rate == 48000
    assert timeline_length_samples(project) == 48000


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p.pop("sample_rate"),
        lambda p: p.update(sample_rate=999),
        lambda p: p.update(channels=5),
        lambda p: p.update(duration_ms=0),
        lambda p: p.update(target_lufs=10.0),
        lambda p: p.update(tracks=[]),
        lambda p: p["tracks"][0].update(gain_db=999.0),
        lambda p: p["tracks"][0]["events"][0].update(duration_ms=-1),
        lambda p: p.update(compressor={"lookahead_ms": 8.0}),
        lambda p: p["tracks"][0].update(track_id="t1")
        or p["tracks"].append(dict(p["tracks"][0])),
    ],
)
def test_invalid_projects_raise_schema_code(mutator):
    payload = _base()
    mutator(payload)
    with pytest.raises(CineMixerError) as exc_info:
        build_project(payload)
    assert exc_info.value.code == "E_PROJECT_SCHEMA"
    assert exc_info.value.exit_code != 0


def test_missing_project_file(tmp_path):
    with pytest.raises(CineMixerError) as exc_info:
        parse_project(tmp_path / "nope.json")
    assert exc_info.value.code == "E_PROJECT_NOT_FOUND"


def test_malformed_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(CineMixerError) as exc_info:
        parse_project(path)
    assert exc_info.value.code == "E_PROJECT_SCHEMA"


def test_duplicate_event_id_rejected():
    payload = _base()
    payload["tracks"][0]["events"].append(
        {"event_id": "e1", "start_ms": 50, "duration_ms": 100}
    )
    with pytest.raises(CineMixerError) as exc_info:
        build_project(payload)
    assert exc_info.value.code == "E_PROJECT_SCHEMA"


def test_parse_roundtrip_from_disk(tmp_path):
    path = tmp_path / "project.json"
    path.write_text(json.dumps(_base()), encoding="utf-8")
    project = parse_project(path)
    assert project.tracks[0].events[0].event_id == "e1"
