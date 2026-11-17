import numpy as np

from audiotl.dsp import (
    apply_fades,
    convert_channels,
    fade_gains,
    ms_to_frames,
    resample_length,
    resample_linear,
)


def test_resample_length_rounding():
    assert resample_length(44100, 44100, 48000) == 48000
    assert resample_length(48000, 48000, 44100) == 44100
    assert resample_length(1, 44100, 48000) == 1
    assert resample_length(0, 44100, 48000) == 0


def test_resample_linear_is_deterministic_and_exact_for_constant():
    signal = np.full((1000, 2), 0.5, dtype=np.float32)
    out1 = resample_linear(signal, 44100, 48000)
    out2 = resample_linear(signal, 44100, 48000)
    assert out1.shape == (resample_length(1000, 44100, 48000), 2)
    np.testing.assert_array_equal(out1, out2)
    np.testing.assert_allclose(out1, 0.5, atol=1e-6)


def test_resample_linear_interpolates_known_values():
    data = np.array([[0.0], [1.0]], dtype=np.float32)
    out = resample_linear(data, 2, 4)  # 2 -> 4 frames, positions 0, 0.5, 1.0, 1.5
    np.testing.assert_allclose(out[:, 0], [0.0, 0.5, 1.0, 1.0], atol=1e-6)


def test_channel_conversion():
    mono = np.array([[0.5], [-0.25]], dtype=np.float32)
    stereo = convert_channels(mono, 2)
    assert stereo.shape == (2, 2)
    np.testing.assert_array_equal(stereo[:, 0], mono[:, 0])
    back = convert_channels(stereo, 1)
    np.testing.assert_allclose(back[:, 0], mono[:, 0], atol=1e-7)


def test_fade_curve_boundaries():
    for curve in ("linear", "equal_power"):
        g = fade_gains(curve, 8)
        assert g[0] == 0.0
        assert g[-1] == 1.0
        assert np.all(np.diff(g) > 0)
    np.testing.assert_allclose(fade_gains("linear", 4), [0.0, 1 / 3, 2 / 3, 1.0])
    assert fade_gains("linear", 1)[0] == 1.0
    assert fade_gains("linear", 0).size == 0


def test_apply_fades_zeroes_endpoints():
    data = np.ones((16, 1), dtype=np.float32)
    out = apply_fades(data, 4, "linear", 4, "linear")
    assert out[0, 0] == 0.0
    assert out[-1, 0] == 0.0
    assert out[3, 0] == 1.0
    assert out[-4, 0] == 1.0


def test_ms_to_frames():
    assert ms_to_frames(1000.0, 48000) == 48000
    assert ms_to_frames(0.0, 48000) == 0
    assert ms_to_frames(100.0, 44100) == 4410
