import numpy as np

from cinemixer.resample import convert_channels, has_soxr, resample_audio


def test_mono_to_stereo_duplicates_channel():
    mono = np.arange(8, dtype=np.float64).reshape(8, 1)
    stereo = convert_channels(mono, 2)
    assert stereo.shape == (8, 2)
    np.testing.assert_array_equal(stereo[:, 0], stereo[:, 1])


def test_stereo_to_mono_preserves_energy():
    stereo = np.ones((100, 2), dtype=np.float64)
    mono = convert_channels(stereo, 1)
    # (1 + 1) / sqrt(2) = sqrt(2): identical L/R keeps total energy.
    np.testing.assert_allclose(mono[:, 0], np.sqrt(2.0))


def test_resample_preserves_tone_frequency_and_length():
    source_rate = 32000
    target_rate = 48000
    timeline = np.arange(source_rate) / source_rate
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * timeline)
    resampled = resample_audio(tone[:, np.newaxis], source_rate, target_rate)
    assert resampled.shape == (target_rate, 1)

    # Estimate the dominant frequency of the resampled tone via zero crossings.
    zero_crossings = np.sum(np.diff(np.sign(resampled[:, 0])) != 0)
    estimated_freq = zero_crossings / 2.0
    assert abs(estimated_freq - 440.0) < 5.0


def test_resample_downsample_keeps_amplitude_bounded():
    timeline = np.arange(48000) / 48000
    tone = 0.6 * np.sin(2 * np.pi * 1000.0 * timeline)
    out = resample_audio(tone[:, np.newaxis], 48000, 24000)
    assert out.shape == (24000, 1)
    assert np.max(np.abs(out)) < 0.75


def test_resample_identity_returns_copy():
    audio = np.zeros((16, 2), dtype=np.float64)
    out = resample_audio(audio, 48000, 48000)
    assert out is not audio
    np.testing.assert_array_equal(out, audio)


def test_soxr_backend_available():
    assert has_soxr() is True
