from __future__ import annotations

import math

import numpy as np
import pytest

from loudness_pipeline.channels import channel_weights
from loudness_pipeline.kweighting import (
    filter_kweighting,
    kweighting_coefficients,
)
from loudness_pipeline.loudness import measure_loudness


def _direct_biquad(
    samples: np.ndarray,
    b: tuple,
    a: tuple,
) -> np.ndarray:
    result = np.empty_like(samples)
    state_1 = 0.0
    state_2 = 0.0
    for index, value in enumerate(samples):
        output = b[0] * value + state_1
        state_1 = b[1] * value - a[1] * output + state_2
        state_2 = b[2] * value - a[2] * output
        result[index] = output
    return result


def test_48k_coefficients_match_bs1770_table() -> None:
    coefficients = kweighting_coefficients(48000)
    assert coefficients.shelf.b == pytest.approx(
        (1.53512485958697, -2.69169618940638, 1.19839281085285),
        abs=1.0e-14,
    )
    assert coefficients.shelf.a == pytest.approx(
        (1.0, -1.69065929318241, 0.73248077421585),
        abs=1.0e-14,
    )
    assert coefficients.rlb.b == (1.0, -2.0, 1.0)
    assert coefficients.rlb.a == pytest.approx(
        (1.0, -1.99004745483398, 0.99007225036621),
        abs=1.0e-14,
    )


@pytest.mark.parametrize(
    "sample_rate",
    [32000, 44100, 48000, 88200, 96000, 192000],
)
def test_fft_filter_matches_cascaded_direct_form(sample_rate: int) -> None:
    rng = np.random.default_rng(7)
    samples = rng.normal(0.0, 0.1, (2048, 2))
    actual = filter_kweighting(samples, sample_rate)
    coefficients = kweighting_coefficients(sample_rate)
    expected = np.empty_like(samples)
    for channel in range(samples.shape[1]):
        stage_1 = _direct_biquad(
            samples[:, channel], coefficients.shelf.b, coefficients.shelf.a
        )
        expected[:, channel] = _direct_biquad(
            stage_1, coefficients.rlb.b, coefficients.rlb.a
        )
    assert np.max(np.abs(actual - expected)) < 1.0e-9


def test_one_khz_calibration_tone_measures_minus_23_lufs() -> None:
    sample_rate = 48000
    time = np.arange(sample_rate * 5) / float(sample_rate)
    audio = (0.1 * np.sin(2.0 * np.pi * 1000.0 * time))[:, None]
    measurement = measure_loudness(audio, sample_rate)
    assert measurement.integrated_lufs == pytest.approx(-23.0036, abs=0.01)
    assert measurement.gated_block_count > 0


def test_relative_gate_ignores_quiet_tail() -> None:
    sample_rate = 48000
    time = np.arange(sample_rate * 10) / float(sample_rate)
    audio = np.empty((time.shape[0], 2), dtype=np.float64)
    audio[: sample_rate * 5, :] = 0.5 * np.sin(
        2.0 * np.pi * 1000.0 * time[: sample_rate * 5]
    )[:, None]
    audio[sample_rate * 5 :, :] = 0.00005 * np.sin(
        2.0 * np.pi * 1000.0 * time[sample_rate * 5 :]
    )[:, None]

    whole = measure_loudness(audio, sample_rate)
    loud_only = measure_loudness(audio[: sample_rate * 5], sample_rate)
    assert whole.integrated_lufs == pytest.approx(
        loud_only.integrated_lufs, abs=0.2
    )
    assert any(not block.passes_absolute_gate for block in whole.blocks)
    assert 0 < whole.gated_block_count < len(whole.blocks)


def test_5_1_lfe_has_zero_weight() -> None:
    sample_rate = 48000
    time = np.arange(sample_rate * 4) / float(sample_rate)
    base = 0.05 * np.sin(2.0 * np.pi * 500.0 * time)
    audio = np.stack([base] * 6, axis=1)
    without_lfe_contribution = audio.copy()
    without_lfe_contribution[:, 3] *= 10.0

    first = measure_loudness(audio, sample_rate)
    second = measure_loudness(without_lfe_contribution, sample_rate)
    assert channel_weights(6) == [1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    assert first.integrated_lufs == pytest.approx(
        second.integrated_lufs, abs=1.0e-10
    )


def test_short_input_uses_actual_partial_window() -> None:
    sample_rate = 48000
    audio = 0.1 * np.sin(
        2.0 * np.pi * 1000.0 * np.arange(4800) / sample_rate
    )[:, None]
    measurement = measure_loudness(audio, sample_rate)
    assert len(measurement.blocks) == 1
    assert measurement.blocks[0].is_partial
    assert measurement.blocks[0].duration_seconds == pytest.approx(0.1)

