import numpy as np

from cinemixer.project import build_project
from cinemixer.timeline import index_entry_for, ms_to_samples, plan_placement, place_event


def _project():
    return build_project(
        {
            "project_id": "p",
            "sample_rate": 1000,
            "channels": 1,
            "duration_ms": 1000,
            "target_lufs": -23.0,
            "output": {"path": "o.wav"},
            "tracks": [
                {
                    "track_id": "t",
                    "source": "a.wav",
                    "events": [
                        {"event_id": "e", "start_ms": 100, "duration_ms": 200}
                    ],
                }
            ],
        }
    )


class _SilentLog:
    def __init__(self):
        self.warnings = []

    def warning(self, code, message, **fields):
        self.warnings.append((code, fields))

    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def test_ms_to_samples_exact():
    assert ms_to_samples(0, 48000) == 0
    assert ms_to_samples(1000, 48000) == 48000
    assert ms_to_samples(250, 48000) == 12000


def test_placement_happy_path():
    project = _project()
    event = project.tracks[0].events[0]
    placement = plan_placement(event, project, source_length=1000, timeline_length=1000)
    assert (placement.timeline_start, placement.timeline_end) == (100, 300)
    assert placement.copy_length == 200
    assert not placement.source_padded


def test_negative_start_is_clipped_with_silence():
    project = _project()
    event = project.tracks[0].events[0]
    object.__setattr__(event, "start_ms", -50.0)
    placement = plan_placement(event, project, 1000, 1000)
    assert placement.truncated_left
    assert placement.timeline_start == 0
    assert placement.source_start == 50  # source read shifts to stay aligned


def test_event_past_timeline_and_short_source():
    project = _project()
    event = project.tracks[0].events[0]
    object.__setattr__(event, "start_ms", 950.0)
    placement = plan_placement(event, project, source_length=20, timeline_length=1000)
    assert placement.truncated_right
    assert placement.timeline_end == 1000
    assert placement.copy_length == 20
    assert placement.source_padded


def test_place_event_warns_and_pads_silence():
    project = _project()
    event = project.tracks[0].events[0]
    object.__setattr__(event, "start_ms", 950.0)
    buffer = np.zeros((1000, 1))
    source = np.ones((30, 1))
    log = _SilentLog()
    entry = place_event(buffer, source, "t", event, project, 1000, log)
    codes = {code for code, _ in log.warnings}
    assert "W_EVENT_PAST_TIMELINE" in codes
    assert "W_SOURCE_PAD_SILENCE" in codes
    assert entry.start_sample == 950
    assert entry.end_sample == 1000
    assert entry.length_samples == 50
    assert buffer[950:980].sum() == 30
    assert buffer[980:].sum() == 0


def test_overlapping_events_accumulate():
    project = _project()
    buffer = np.zeros((1000, 1))
    source = np.ones((1000, 1))
    log = _SilentLog()
    first = project.tracks[0].events[0]
    second = type(first)(event_id="e2", start_ms=250.0, duration_ms=100.0)
    place_event(buffer, source, "t", first, project, 1000, log)
    place_event(buffer, source, "t", second, project, 1000, log)
    assert buffer[0:100].sum() == 0
    assert np.all(buffer[100:250] == 1.0)
    assert np.all(buffer[250:300] == 2.0)  # overlap region accumulates


def test_index_entry_matches_placement():
    project = _project()
    event = project.tracks[0].events[0]
    entry = index_entry_for("t", event, project, 1000, 1000)
    assert entry.as_dict() == {
        "track_id": "t",
        "event_id": "e",
        "start_sample": 100,
        "end_sample": 300,
        "length_samples": 200,
    }


def test_event_completely_before_timeline_yields_zero_length():
    project = _project()
    event = project.tracks[0].events[0]
    object.__setattr__(event, "start_ms", -200.0)
    object.__setattr__(event, "duration_ms", 100.0)  # ends at -100 ms
    placement = plan_placement(event, project, source_length=1000, timeline_length=1000)
    assert placement.truncated_left
    assert placement.timeline_start == 0
    assert placement.timeline_end == 0
    assert placement.event_length == 0
    entry = index_entry_for("t", event, project, 1000, 1000)
    assert entry.start_sample == entry.end_sample == 0
    assert entry.length_samples == 0
