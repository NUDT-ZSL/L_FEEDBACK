"""Built-in deterministic test-signal synthesis.

The mixer never needs an external audio asset for its automated acceptance
tests.  These functions use only numpy and produce the same samples on every
run for the same arguments.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

import numpy as np


def _as_components(
    frequencies: Union[float, Sequence[float]],
    amplitudes: Union[float, Sequence[float], None],
) -> Tuple[Tuple[float, float], ...]:
    if isinstance(frequencies, (int, float)):
        freq_values = (float(frequencies),)
    else:
        freq_values = tuple(float(value) for value in frequencies)
    if not freq_values:
        raise ValueError("at least one frequency is required")

    if amplitudes is None:
        amp_values = (0.5,) * len(freq_values)
    elif isinstance(amplitudes, (int, float)):
        amp_values = (float(amplitudes),) * len(freq_values)
    else:
        amp_values = tuple(float(value) for value in amplitudes)
        if len(amp_values) != len(freq_values):
            raise ValueError("frequencies and amplitudes must have equal lengths")

    components = tuple(zip(freq_values, amp_values))
    for frequency, amplitude in components:
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("frequency must be finite and positive")
        if not math.isfinite(amplitude) or amplitude < 0.0:
            raise ValueError("amplitude must be finite and non-negative")
    return components


def synthesize_signal(
    sample_rate: int,
    duration_seconds: float,
    frequencies: Union[float, Sequence[float]],
    amplitudes: Union[float, Sequence[float], None] = 0.5,
    channels: int = 1,
    phases: Union[float, Sequence[float], None] = None,
    start_seconds: float = 0.0,
) -> np.ndarray:
    """Return a deterministic sine (or sum of sines) as float64 samples.

    Args:
        sample_rate: Output rate in Hz.
        duration_seconds: Exact requested duration; frame count is rounded.
        frequencies: One frequency or a sequence of frequencies.
        amplitudes: One amplitude, one per frequency, or None for 0.5 each.
        channels: 1 for mono, 2 for stereo.  Stereo uses a deterministic
            channel phase spread for the second channel when multiple
            frequencies are present.
        phases: Optional phase in radians, scalar or per-component.
        start_seconds: Timeline position of the first returned frame.  This
            lets deterministic long fixtures be generated in bounded chunks.

    Returns:
        An array with shape ``(frames, channels)`` and dtype float64.
    """

    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise ValueError("sample_rate must be an integer")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be finite and positive")
    if channels not in (1, 2):
        raise ValueError("channels must be 1 or 2")
    if not math.isfinite(start_seconds) or start_seconds < 0.0:
        raise ValueError("start_seconds must be finite and non-negative")

    components = _as_components(frequencies, amplitudes)
    if phases is None:
        phase_values = (0.0,) * len(components)
    elif isinstance(phases, (int, float)):
        phase_values = (float(phases),) * len(components)
    else:
        phase_values = tuple(float(value) for value in phases)
        if len(phase_values) != len(components):
            raise ValueError("phases and frequencies must have equal lengths")
    if any(not math.isfinite(value) for value in phase_values):
        raise ValueError("phases must be finite")

    frames = int(math.floor(duration_seconds * sample_rate + 0.5))
    if frames < 1:
        raise ValueError("duration is shorter than one sample")
    time = (
        np.arange(frames, dtype=np.float64) + start_seconds * sample_rate
    ) / float(sample_rate)

    mono = np.zeros(frames, dtype=np.float64)
    for (frequency, amplitude), phase in zip(components, phase_values):
        if frequency >= sample_rate / 2.0:
            raise ValueError("frequency must be below the Nyquist frequency")
        mono += amplitude * np.sin(2.0 * math.pi * frequency * time + phase)

    if channels == 1:
        return mono[:, np.newaxis]

    stereo = np.empty((frames, 2), dtype=np.float64)
    stereo[:, 0] = mono
    second = np.zeros(frames, dtype=np.float64)
    for index, ((frequency, amplitude), phase) in enumerate(zip(components, phase_values)):
        channel_phase = phase + (math.pi / 2.0 if index % 2 else 0.0)
        second += amplitude * np.sin(2.0 * math.pi * frequency * time + channel_phase)
    stereo[:, 1] = second
    return stereo


def synthesize_impulse_train(
    sample_rate: int,
    duration_seconds: float,
    period_seconds: float,
    amplitude: float = 0.8,
    channels: int = 1,
) -> np.ndarray:
    """Synthesize a deterministic sparse impulse train for seam tests."""

    if not math.isfinite(period_seconds) or period_seconds <= 0.0:
        raise ValueError("period_seconds must be finite and positive")
    if not math.isfinite(amplitude) or amplitude < 0.0:
        raise ValueError("amplitude must be finite and non-negative")
    frames = int(math.floor(duration_seconds * sample_rate + 0.5))
    if frames < 1:
        raise ValueError("duration is shorter than one sample")
    period_frames = max(1, int(math.floor(period_seconds * sample_rate + 0.5)))
    signal = np.zeros((frames, channels), dtype=np.float64)
    signal[::period_frames, :] = amplitude
    return signal
