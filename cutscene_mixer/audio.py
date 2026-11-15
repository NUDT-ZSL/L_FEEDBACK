"""WAV loading, linear-interpolation resampling and channel conversion."""

import numpy as np
import soundfile as sf


def load_track_audio(path, target_sample_rate, target_channels):
    """Load a WAV file and convert it to the target rate/channel layout.

    - Linear interpolation resampling when the source rate differs.
    - Mono is duplicated to stereo; stereo is kept; more than two channels
      are reduced to their first two channels.
    - Samples are returned as float64 in [-1, 1]; bit depth is unified at
      render time when the mix is written.

    Returns an array of shape (samples, channels). Raises on any decode
    failure so the caller can record AUDIO_LOAD_FAIL and skip the track.
    """
    data, source_rate = sf.read(path, dtype="float64", always_2d=True)
    if data.shape[0] == 0:
        raise ValueError("empty audio file: %s" % path)

    if source_rate != target_sample_rate:
        data = _resample_linear(data, source_rate, target_sample_rate)

    source_channels = data.shape[1]
    if source_channels == 1 and target_channels == 2:
        data = np.repeat(data, 2, axis=1)
    elif source_channels >= 2 and target_channels == 2:
        data = data[:, :2]
    elif source_channels >= 2 and target_channels == 1:
        data = data[:, :2].mean(axis=1, keepdims=True)
    return data


def _resample_linear(data, source_rate, target_rate):
    """Resample (samples, channels) float audio with linear interpolation."""
    source_length = data.shape[0]
    target_length = int(round(source_length * target_rate / source_rate))
    if target_length < 1:
        target_length = 1
    source_positions = np.arange(source_length, dtype=np.float64)
    target_positions = np.linspace(
        0.0, source_length - 1, num=target_length, dtype=np.float64
    )
    out = np.empty((target_length, data.shape[1]), dtype=np.float64)
    for channel in range(data.shape[1]):
        out[:, channel] = np.interp(
            target_positions, source_positions, data[:, channel]
        )
    return out
