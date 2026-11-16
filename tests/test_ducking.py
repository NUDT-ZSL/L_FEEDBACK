from __future__ import annotations

import numpy as np
import pytest

from cutscene_dub.config import RenderConfig
from cutscene_dub.ducking import apply_ducking
from cutscene_dub.errors import BusinessError


def test_nearby_dialogues_are_merged_without_gain_jump() -> None:
    sample_rate = 48_000
    music = np.ones(sample_rate * 4, dtype=np.float64)
    config = RenderConfig(
        duck_depth_db=18.0,
        duck_fade_in_ms=100.0,
        duck_fade_out_ms=100.0,
        max_gain_jump_db=0.5,
    )
    result = apply_ducking(
        music,
        dialogue_intervals=[
            (sample_rate // 2, sample_rate, "a"),
            (sample_rate + 600, sample_rate * 2, "b"),
        ],
        config=config,
    )

    assert len(result.intervals) == 1
    assert result.intervals[0].dialogue_ids == ("a", "b")
    floor_gain = 10.0 ** (-18.0 / 20.0)
    # The short 12.5 ms gap remains held at full attenuation.
    assert result.gain_envelope[sample_rate + 300] == pytest.approx(floor_gain)
    assert result.gain_envelope[1000] == pytest.approx(1.0)
    assert result.max_gain_jump_db < 0.5
    assert result.measured_dialogue_attenuation_db == pytest.approx(-18.0)


def test_separated_dialogues_restore_unity_gain_between_them() -> None:
    sample_rate = 48_000
    music = np.ones(sample_rate * 4, dtype=np.float64)
    config = RenderConfig(
        duck_depth_db=12.0,
        duck_fade_in_ms=50.0,
        duck_fade_out_ms=50.0,
    )
    result = apply_ducking(
        music,
        dialogue_intervals=[
            (sample_rate // 2, sample_rate, "a"),
            (sample_rate * 2, sample_rate * 3, "b"),
        ],
        config=config,
    )

    assert len(result.intervals) == 2
    assert result.gain_envelope[int(1.5 * sample_rate)] == pytest.approx(1.0)
    assert result.gain_envelope[sample_rate // 4] == pytest.approx(1.0)


def test_impossible_gain_ramp_is_rejected() -> None:
    config = RenderConfig(
        duck_depth_db=40.0,
        duck_fade_in_ms=0.01,
        duck_fade_out_ms=0.01,
        max_gain_jump_db=0.5,
    )
    with pytest.raises(BusinessError) as captured:
        apply_ducking(
            np.ones(48_000, dtype=np.float64),
            dialogue_intervals=[(10_000, 20_000, "dlg_001")],
            config=config,
        )
    assert captured.value.code == "E_DUCK_RAMP_UNSAFE"
    assert captured.value.entry_id == "dlg_001"

