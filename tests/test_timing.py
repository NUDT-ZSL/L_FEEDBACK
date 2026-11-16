from __future__ import annotations

import numpy as np
import pytest

from rhythmforge.errors import BuildError, ExitCode
from rhythmforge.timing import BpmPoint, TimeMap


def _map(points, rate=48000, end_beat=None):
    return TimeMap([BpmPoint(b, bpm) for b, bpm in points], rate, end_beat)


def test_beat_to_frame_known_values():
    # 120 BPM => 0.5 s/beat => 24000 frames/beat at 48 kHz.
    mapping = _map([(0.0, 120.0), (4.0, 60.0)])
    assert int(mapping.beat_to_frame(0.0)) == 0
    assert int(mapping.beat_to_frame(1.0)) == 24000
    assert int(mapping.beat_to_frame(4.0)) == 96000
    # After the switch 60 BPM => 1 s/beat => 48000 frames/beat.
    assert int(mapping.beat_to_frame(5.0)) == 96000 + 48000


def test_mapping_is_monotonic():
    mapping = _map([(0.0, 100.0), (8.0, 120.0), (16.0, 90.0), (24.0, 110.0)],
                   end_beat=32.0)
    grid = np.linspace(0.0, 32.0, 20000)
    frames = mapping.beat_to_frame(grid)
    assert np.all(np.diff(frames) > 0)


def test_switch_point_error_within_one_frame():
    mapping = _map([(0.0, 100.0), (8.0, 120.0), (16.0, 90.0), (24.0, 110.0)],
                   end_beat=32.0)
    report = mapping.validate_switch_accuracy()
    assert report["worst_rounding_error_frames"] <= 1.0
    assert report["checked_switches"] == 3


def test_continuity_at_switch_points():
    mapping = _map([(0.0, 100.0), (8.0, 120.0), (16.0, 90.0)])
    for anchor in (8.0, 16.0):
        left = mapping.beat_to_frame_exact(anchor - 1e-9)
        right = mapping.beat_to_frame_exact(anchor + 1e-9)
        # 1e-9 of a beat maps to a tiny fraction of a sample frame.
        assert abs(float(right - left)) < 1e-3


def test_rejects_zero_and_negative_bpm():
    for bad_bpm in (0.0, -5.0):
        with pytest.raises(BuildError) as exc:
            _map([(0.0, 120.0), (4.0, bad_bpm)])
        assert exc.value.code == ExitCode.MAPPING_CONFLICT


def test_rejects_non_monotonic_anchors():
    with pytest.raises(BuildError) as exc:
        _map([(0.0, 120.0), (4.0, 100.0), (4.0, 90.0)])
    assert exc.value.code == ExitCode.MAPPING_CONFLICT


def test_section_frames_exact():
    mapping = _map([(0.0, 100.0), (8.0, 120.0)], end_beat=16.0)
    # 8 beats at 100 BPM = 4.8 s -> 230400 frames at 48 kHz.
    assert mapping.section_frames(0.0, 8.0) == 230400


def test_invalid_end_beat_rejected():
    with pytest.raises(BuildError) as exc:
        _map([(0.0, 120.0)], end_beat=0.0)
    assert exc.value.code == ExitCode.MAPPING_CONFLICT
