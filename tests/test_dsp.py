from __future__ import annotations

import numpy as np
import pytest

from dialogue_master.dsp import (
    apply_cosine_squared_fades,
    cosine_squared_fade_in,
    cosine_squared_fade_out,
    find_placement,
    milliseconds_to_samples,
)


def test_sample_rounding_uses_half_up() -> None:
    assert milliseconds_to_samples(0.48958, 48000) == 23
    assert milliseconds_to_samples(0.5104166667, 48000) == 25
    assert milliseconds_to_samples(0.0104166666666667, 48000) == 1


def test_cosine_squared_fades_are_sample_exact() -> None:
    samples = np.ones((100, 2), dtype=np.float64)
    result = apply_cosine_squared_fades(samples, 10, 12)
    assert result.shape == samples.shape
    assert result[0, 0] == 0.0
    assert result[-1, 1] == 0.0
    assert np.allclose(result[10:88], 1.0)
    assert np.allclose(result[:10, 0], cosine_squared_fade_in(10))
    assert np.allclose(result[-12:, 0], cosine_squared_fade_out(12))
    assert np.max(np.abs(result)) <= 1.0


def test_single_sample_fade_is_muted() -> None:
    assert cosine_squared_fade_in(1).tolist() == [0.0]
    assert cosine_squared_fade_out(1).tolist() == [0.0]


def test_degenerate_fade_on_one_sample_clip() -> None:
    result = apply_cosine_squared_fades(
        np.ones((1, 1), dtype=np.float64), 1, 1
    )
    assert result[0, 0] == 0.0


def test_greedy_placement_finds_earliest_fitting_gap() -> None:
    placed = [(100, 150), (160, 170)]
    assert find_placement(100, 5, 200, placed) == 150
    assert find_placement(100, 15, 200, placed) == 170
    assert find_placement(150, 5, 200, placed) == 150
    assert find_placement(180, 21, 200, placed) is None
    assert find_placement(0, 50, 200, []) == 0


def test_greedy_placement_rejects_clip_longer_than_timeline() -> None:
    assert find_placement(0, 201, 200, []) is None
