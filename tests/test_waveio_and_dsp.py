from __future__ import annotations

import os

import numpy as np
import pytest

from rhythmforge import dsp
from rhythmforge.errors import BuildError, ExitCode
from rhythmforge.waveio import read_wav, write_wav


@pytest.mark.parametrize("width", [2, 3, 4])
def test_wav_roundtrip_widths(tmp_path, width):
    rate = 48000
    data = (0.5 * np.sin(np.linspace(0, 20, rate // 4))).astype(np.float32)
    stereo = np.stack([data, -data], axis=1)
    path = str(tmp_path / f"w{width}.wav")
    write_wav(path, stereo, rate, width)
    decoded, info = read_wav(path)
    assert info.sample_rate == rate
    assert info.channels == 2
    assert info.sample_width == width
    # Quantization error stays within one LSB of the chosen bit depth.
    lsb = 1.0 / (2 ** (8 * width - 1) - 1)
    assert np.max(np.abs(decoded - stereo)) <= lsb + 1e-9


def test_24bit_values_are_signed():
    raw = dsp  # import touchstone; real assertion lives in roundtrip
    assert raw is not None
    negative = np.array([[-0.75], [0.75]], dtype=np.float32)
    from rhythmforge.waveio import _encode_pcm, _decode_pcm

    payload = _encode_pcm(negative, 3)
    decoded = _decode_pcm(payload, 3)
    assert decoded[0] < 0 < decoded[1]


def test_resample_preserves_onset_relationship():
    rate_src = 44100
    rate_dst = 48000
    data = np.zeros((rate_src, 1), dtype=np.float32)
    onset_native = int(0.1 * rate_src)
    data[onset_native] = 1.0
    out = dsp.resample_to_rate(data, rate_src, rate_dst)
    expected = int(round(onset_native * rate_dst / rate_src))
    peak = int(np.argmax(np.abs(out)))
    assert abs(peak - expected) <= 1


def test_resample_length_semantics():
    data = np.random.RandomState(1).randn(4410, 2).astype(np.float32)
    out = dsp.resample_to_rate(data, 44100, 48000)
    assert out.shape[0] == 4800
    assert out.shape[1] == 2


def test_mono_to_stereo_duplication():
    mono = np.arange(10, dtype=np.float32).reshape(10, 1)
    stereo = dsp.normalize_channels(mono)
    assert stereo.shape == (10, 2)
    assert np.array_equal(stereo[:, 0], stereo[:, 1])


def test_time_stretch_factor_one_is_identity():
    data = np.random.RandomState(2).randn(4800, 2).astype(np.float32)
    out = dsp.time_stretch(data, 1.0)
    assert np.array_equal(out, data)


def test_time_stretch_returns_exact_frames():
    data = np.random.RandomState(3).randn(9600, 2).astype(np.float32)
    target = 12345
    out = dsp.time_stretch(data, 1.5, exact_frames=target)
    assert out.shape[0] == target
    out2 = dsp.time_stretch(data, 0.75, exact_frames=target)
    assert out2.shape[0] == target


def test_time_stretch_invalid_factor():
    data = np.zeros((100, 2), dtype=np.float32)
    with pytest.raises(BuildError) as exc:
        dsp.time_stretch(data, 0.0)
    assert exc.value.code == ExitCode.MAPPING_CONFLICT


def test_unsupported_width_rejected(tmp_path):
    path = str(tmp_path / "u.wav")
    write_wav(path, np.zeros((10, 2), dtype=np.float32), 48000, 2)
    # Hand-craft an 8-bit wav container to trigger the read-side rejection.
    import wave

    bad = str(tmp_path / "eight.wav")
    with wave.open(bad, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(8000)
        handle.writeframes(bytes(16))
    with pytest.raises(BuildError) as exc:
        read_wav(bad)
    assert exc.value.code == ExitCode.UNSUPPORTED_FORMAT
