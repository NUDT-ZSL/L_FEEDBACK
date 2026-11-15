"""Fixed millisecond/sample conversion rule."""

import math

import pytest

from audiopack.timeutils import ms_to_samples


@pytest.mark.parametrize(
    "ms,sr,expected",
    [
        (0, 48000, 0),
        (1000, 48000, 48000),
        (1, 48000, 48),
        (0.5, 48000, 24),
        (100, 44100, 4410),
        # 0.375 ms * 48 = 18 exactly; banker's rounding cases stay explicit.
        (0.375, 48000, 18),
    ],
)
def test_ms_to_samples(ms, sr, expected):
    assert ms_to_samples(ms, sr) == expected


def test_rule_matches_floor_round_formula():
    for ms, sr in [(1.234, 48000), (7.7, 44100), (33.333, 32000)]:
        assert ms_to_samples(ms, sr) == math.floor(round(ms * sr / 1000))
