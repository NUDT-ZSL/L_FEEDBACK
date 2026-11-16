import numpy as np

from cinemixer.compressor import (
    SoftKneeCompressor,
    compressor_latency_ms,
    db_to_linear,
    linear_to_db,
)
from cinemixer.project import CompressorSpec


def test_db_conversions_roundtrip():
    values = np.array([0.1, 0.5, 1.0, 2.0])
    np.testing.assert_allclose(db_to_linear(linear_to_db(values)), values, rtol=1e-9)


def test_signal_below_threshold_passes_unchanged():
    spec = CompressorSpec(threshold_db=-18.0, ratio=4.0, knee_db=6.0)
    compressor = SoftKneeCompressor(spec, 48000)
    audio = np.full((4800, 1), 0.05)  # about -26 dBFS, below knee
    out = compressor.process(audio)
    np.testing.assert_allclose(out, audio, atol=1e-6)


def test_loud_signal_is_attenuated_and_capped():
    spec = CompressorSpec(
        threshold_db=-18.0,
        ratio=4.0,
        knee_db=0.0,
        attack_ms=1.0,
        release_ms=10.0,
    )
    compressor = SoftKneeCompressor(spec, 48000)
    audio = np.full((48000, 1), 0.9)
    out = compressor.process(audio)
    steady = out[-1000:]
    # 0.9 ~= -0.9 dBFS; reduction = (1 - 1/4) * (-0.9 - (-18)) ~= 12.8 dB
    expected = 0.9 * db_to_linear(-12.8)
    np.testing.assert_allclose(steady[:, 0], expected, rtol=0.05)
    assert np.max(np.abs(out)) <= 0.9 + 1e-9


def test_attack_then_release_envelope():
    spec = CompressorSpec(
        threshold_db=-20.0,
        ratio=10.0,
        knee_db=0.0,
        attack_ms=2.0,
        release_ms=80.0,
    )
    compressor = SoftKneeCompressor(spec, 48000)
    quiet = np.full((2400, 1), 0.01)
    loud = np.full((2400, 1), 0.8)
    tail = np.full((9600, 1), 0.01)
    out = np.concatenate([
        compressor.process(quiet),
        compressor.process(loud),
        compressor.process(tail),
    ], axis=0)
    # Attenuation ramps in during the loud segment.
    loud_region = out[2400:4800, 0]
    assert loud_region[0] > loud_region[-1]
    # Release recovers toward unity after the loud segment ends.
    tail_region = out[4800:, 0]
    assert tail_region[-1] > tail_region[0]


def test_lookahead_latency_within_five_ms():
    spec = CompressorSpec(lookahead_ms=4.0)
    assert compressor_latency_ms(spec, 48000) == 4.0
    compressor = SoftKneeCompressor(spec, 48000)
    audio = np.random.default_rng(1).standard_normal((4800, 1)) * 0.5
    out = compressor.process(audio)
    assert out.shape == audio.shape
    # Look-ahead zero-pads the beginning by exactly the delayed sample count.
    delay = int(round(48000 * 4.0 / 1000.0))
    np.testing.assert_array_equal(out[:delay, 0], 0.0)


def test_state_persists_across_calls():
    spec = CompressorSpec(threshold_db=-20.0, ratio=8.0, knee_db=0.0)
    compressor = SoftKneeCompressor(spec, 48000)
    first = compressor.process(np.full((1000, 1), 0.7))
    envelope_after_first = compressor.envelope_db
    assert envelope_after_first < 0.0
    compressor.process(np.full((1000, 1), 0.7))
    assert compressor.envelope_db <= envelope_after_first
