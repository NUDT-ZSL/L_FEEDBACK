from __future__ import annotations

import numpy as np
import pytest

from game_audio_mixer.dsp import (
    SAMPLE_RATE,
    integrated_loudness,
    short_term_loudness,
    true_peak,
)


def test_bs1770_reference_sine_loudness() -> None:
    seconds = 6.0
    time = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    mono = 0.1 * np.sin(2.0 * np.pi * 1000.0 * time)
    # A -20 dBFS-peak 1 kHz mono sine is the EBU R128 calibration tone.
    assert integrated_loudness(mono[:, None]) == pytest.approx(-23.0, abs=0.1)
    # A duplicated stereo signal has 3 LU more energy.
    assert integrated_loudness(np.column_stack([mono, mono])) == pytest.approx(
        -20.0, abs=0.1
    )

    peak, _ = true_peak(mono)
    assert peak == pytest.approx(0.1, abs=1e-4)
    segments = short_term_loudness(mono)
    assert segments
    assert all(
        {"start_s", "end_s", "lufs"}.issubset(segment) for segment in segments
    )
    assert all(segment["lufs"] == pytest.approx(-23.0, abs=0.2) for segment in segments)
