"""ITU-R BS.1770-4 loudness and true-peak measurement."""

from __future__ import annotations

import math

import numpy as np

from .dynamics import TruePeakEnvelope


def safe_db(value):
    if value is None or value <= 0.0 or not math.isfinite(float(value)):
        return None
    return float(20.0 * math.log10(float(value)))


def _shelf_and_highpass(sample_rate):
    f0 = 1681.974450955533
    gain_db = 3.999843853973347
    q = 0.7071752369554196
    k = math.tan(math.pi * f0 / sample_rate)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q + k * k
    shelf_b = np.array(
        [
            (vh + vb * k / q + k * k) / a0,
            2.0 * (k * k - vh) / a0,
            (vh - vb * k / q + k * k) / a0,
        ],
        dtype=np.float64,
    )
    shelf_a = np.array(
        [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
        dtype=np.float64,
    )

    highpass_f0 = 38.13547087602444
    highpass_q = 0.5003270373238773
    w0 = 2.0 * math.pi * highpass_f0 / sample_rate
    cosine = math.cos(w0)
    alpha = math.sin(w0) / (2.0 * highpass_q)
    hp_a0 = 1.0 + alpha
    hp_b = np.array(
        [(1.0 + cosine) / 2.0, -(1.0 + cosine), (1.0 + cosine) / 2.0],
        dtype=np.float64,
    ) / hp_a0
    hp_a = np.array([1.0, -2.0 * cosine / hp_a0, (1.0 - alpha) / hp_a0])
    return shelf_b, shelf_a, hp_b, hp_a


class _Biquad:
    def __init__(self, coefficients, channels):
        b, a = coefficients
        self.b = b
        self.a = a
        self.state = np.zeros((channels, 4), dtype=np.float64)

    def process(self, block):
        output = np.empty_like(block, dtype=np.float64)
        b0, b1, b2 = self.b
        _, a1, a2 = self.a
        for channel in range(block.shape[1]):
            z1 = self.state[channel, 0]
            z2 = self.state[channel, 1]
            source = block[:, channel]
            target = output[:, channel]
            for index in range(source.shape[0]):
                value = source[index]
                result = b0 * value + z1
                z1 = b1 * value - a1 * result + z2
                z2 = b2 * value - a2 * result
                target[index] = result
            self.state[channel, 0] = z1
            self.state[channel, 1] = z2
        return output


class LoudnessMeter:
    def __init__(self, sample_rate, channel_count):
        self.sample_rate = int(sample_rate)
        self.channel_count = int(channel_count)
        self.segment_frames = max(1, int(round(self.sample_rate * 0.1)))
        coefficients = _shelf_and_highpass(self.sample_rate)
        self.shelf = _Biquad(coefficients[:2], self.channel_count)
        self.highpass = _Biquad(coefficients[2:], self.channel_count)
        self._segment_count = 0
        self._segment_sums = np.zeros(self.channel_count, dtype=np.float64)
        self._segments = []
        self.total_frames = 0
        self.weights = np.ones(self.channel_count, dtype=np.float64)
        if self.channel_count == 4:
            self.weights[:] = [1.0, 1.0, 1.41, 1.41]
        elif self.channel_count == 5:
            self.weights[:] = [1.0, 1.0, 1.0, 1.41, 1.41]
        elif self.channel_count == 6:
            self.weights[:] = [1.0, 1.0, 1.0, 0.0, 1.41, 1.41]
        elif self.channel_count == 7:
            self.weights[:] = [1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.41]
        elif self.channel_count >= 8:
            self.weights[:] = [1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.41, 1.41]

    def feed(self, block):
        data = np.asarray(block, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.channel_count:
            raise ValueError("loudness block has an unexpected channel count")
        if data.shape[0] == 0:
            return
        filtered = self.highpass.process(self.shelf.process(data))
        self.total_frames += filtered.shape[0]
        offset = 0
        while offset < filtered.shape[0]:
            take = min(self.segment_frames - self._segment_count, filtered.shape[0] - offset)
            part = filtered[offset:offset + take]
            self._segment_sums += np.sum(part * part, axis=0)
            self._segment_count += take
            offset += take
            if self._segment_count == self.segment_frames:
                self._segments.append(self._segment_sums.copy())
                self._segment_sums.fill(0.0)
                self._segment_count = 0

    def _energies(self, window_segments):
        if len(self._segments) < window_segments:
            return np.empty(0, dtype=np.float64), 0
        segments = np.asarray(self._segments, dtype=np.float64)
        cumulative = np.vstack(
            (np.zeros((1, self.channel_count)), np.cumsum(segments, axis=0))
        )
        windows = cumulative[window_segments:] - cumulative[:-window_segments]
        # Per-channel mean square: squared sums are divided by the number of
        # frames in one channel, then channel weights are applied.
        denominator = float(window_segments * self.segment_frames)
        weighted = np.sum(windows * self.weights[None, :], axis=1) / denominator
        return weighted, window_segments

    def _integrated(self):
        energies, _ = self._energies(4)
        if energies.size == 0:
            return None
        absolute = energies > 10.0 ** ((-70.0 + 0.691) / 10.0)
        if not np.any(absolute):
            return None
        ungated_mean = float(np.mean(energies[absolute]))
        relative = absolute & (energies > ungated_mean * 0.1)
        if not np.any(relative):
            return None
        mean_energy = float(np.mean(energies[relative]))
        if mean_energy <= 0.0:
            return None
        return float(-0.691 + 10.0 * math.log10(mean_energy))

    def _short_term(self):
        if not self._segments:
            return None, None
        window = min(30, len(self._segments))
        energies, segments = self._energies(window)
        if energies.size == 0:
            return None, None
        if float(np.max(energies)) <= 0.0:
            return None, segments * self.segment_frames / float(self.sample_rate)
        loudness = -0.691 + 10.0 * np.log10(np.maximum(energies, 1e-30))
        window_seconds = segments * self.segment_frames / float(self.sample_rate)
        return float(np.max(loudness)), window_seconds

    def result(self):
        short_term, window_seconds = self._short_term()
        return {
            "integrated_lufs": self._integrated(),
            "short_term_max_lufs": short_term,
            "short_term_window_seconds": window_seconds,
            "measured_frames": self.total_frames,
            "channel_weights": [float(value) for value in self.weights],
        }


class SignalAnalyzer:
    """Streaming loudness, sample-peak and true-peak analysis."""

    def __init__(self, sample_rate, channels):
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.loudness = LoudnessMeter(sample_rate, channels)
        self.true_peak_detector = TruePeakEnvelope(sample_rate, channels)
        self.sample_peak = 0.0
        self.true_peak = 0.0
        self.frames = 0

    def feed(self, block):
        data = np.asarray(block, dtype=np.float64)
        if data.shape[0] == 0:
            return
        self.frames += data.shape[0]
        if data.size:
            self.sample_peak = max(self.sample_peak, float(np.max(np.abs(data))))
        self.loudness.feed(data)
        peaks = self.true_peak_detector.feed(data)
        if peaks.size:
            self.true_peak = max(self.true_peak, float(np.max(peaks)))

    def result(self):
        peaks = self.true_peak_detector.finalize()
        if peaks.size:
            self.true_peak = max(self.true_peak, float(np.max(peaks)))
        loudness = self.loudness.result()
        loudness.update(
            {
                "sample_peak_dbfs": safe_db(self.sample_peak),
                "true_peak_dbtp": safe_db(self.true_peak),
            }
        )
        return loudness
