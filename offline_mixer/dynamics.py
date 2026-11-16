"""Streaming compressor, true-peak detector and look-ahead limiter."""

from __future__ import annotations

import math

import numpy as np


def _db_to_linear(value):
    return float(10.0 ** (value / 20.0))


class TruePeakEnvelope:
    """Four-times oversampled peak envelope (one value per base-rate frame)."""

    def __init__(self, sample_rate, channels, taps=64, beta=10.0):
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.taps = int(taps)
        self.half = self.taps // 2
        self.offsets = np.arange(-self.half + 1, self.half + 1, dtype=np.int64)
        self.phases = np.arange(4, dtype=np.float64) / 4.0
        delta = self.phases[:, None] - self.offsets[None, :].astype(np.float64)
        # Interpolation only; no decimation is involved. A unity cutoff makes
        # phase zero reproduce the source sample and leaves intersample
        # reconstruction to the other three phases.
        cutoff = 1.0
        ratio = delta / self.half
        window = np.zeros_like(delta)
        inside = np.abs(ratio) < 1.0
        i0_beta = float(np.i0(beta))
        window[inside] = (
            np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - ratio[inside] ** 2))) / i0_beta
        )
        weights = cutoff * np.sinc(cutoff * delta) * window
        weights /= weights.sum(axis=1, keepdims=True)
        self.weights = weights
        self._buffer = np.empty((0, self.channels), dtype=np.float64)
        self._buffer_start = 0
        self._total_in = 0
        self._emitted = 0
        self._finished = False

    @property
    def emitted_frames(self):
        return self._emitted

    def _append(self, block):
        if block.shape[0] == 0:
            return
        if self._buffer.shape[0] == 0:
            self._buffer = np.array(block, dtype=np.float64, copy=True)
        else:
            self._buffer = np.concatenate((self._buffer, block), axis=0)

    def _measure(self, start, stop, valid_limit):
        if stop <= start:
            return np.empty(0, dtype=np.float64)
        bases = np.arange(start, stop, dtype=np.int64)
        lattice = bases[:, None] + self.offsets[None, :]
        valid = (lattice >= 0) & (lattice < valid_limit)
        local = np.clip(lattice, 0, max(0, valid_limit - 1))
        local = np.clip(local - self._buffer_start, 0, max(0, self._buffer.shape[0] - 1))
        source = self._buffer[local] * valid[:, :, None]
        phases = np.einsum("pt,btc->bpc", self.weights, source, optimize=True)
        return np.max(np.abs(phases), axis=(1, 2))

    def _trim(self):
        keep_from = self._emitted - self.half + 1
        drop = keep_from - self._buffer_start
        if drop > 8192:
            self._buffer = self._buffer[drop:].copy()
            self._buffer_start += drop

    def feed(self, block):
        if self._finished:
            raise RuntimeError("true-peak detector already finished")
        data = np.asarray(block, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.channels:
            raise ValueError("true-peak block has an unexpected channel count")
        self._append(data)
        self._total_in += data.shape[0]
        stop = self._total_in - self.half
        if stop <= self._emitted:
            return np.empty(0, dtype=np.float64)
        start = self._emitted
        result = self._measure(start, stop, self._total_in)
        self._emitted = stop
        self._trim()
        return result

    def finalize(self):
        if self._finished:
            return np.empty(0, dtype=np.float64)
        total = self._total_in
        if total <= self._emitted:
            self._finished = True
            return np.empty(0, dtype=np.float64)
        padding = np.zeros((self.half + 2, self.channels), dtype=np.float64)
        self._append(padding)
        start = self._emitted
        result = self._measure(start, total, total)
        self._emitted = total
        self._finished = True
        return result


class Compressor:
    """Linked-channel feed-forward compressor with an optional soft knee."""

    def __init__(self, sample_rate, config):
        self.sample_rate = int(sample_rate)
        self.threshold = float(config.threshold_db)
        self.ratio = float(config.ratio)
        self.knee = float(config.knee_db)
        self.attack_coeff = math.exp(
            -1.0 / max(1.0, self.sample_rate * config.attack_ms / 1000.0)
        )
        self.release_coeff = math.exp(
            -1.0 / max(1.0, self.sample_rate * config.release_ms / 1000.0)
        )
        self.gain_reduction = 0.0

    def _desired_reduction(self, level_db):
        over = level_db - self.threshold
        if self.ratio <= 1.0 or over <= -self.knee * 0.5:
            return 0.0
        slope = 1.0 - 1.0 / self.ratio
        if self.knee > 0.0 and over < self.knee * 0.5:
            return slope * (over + self.knee * 0.5) ** 2 / (2.0 * self.knee)
        return slope * over

    def process(self, block):
        data = np.asarray(block, dtype=np.float64)
        if data.shape[0] == 0:
            return data.copy(), np.empty(0, dtype=np.float64)
        peak = np.max(np.abs(data), axis=1)
        level = np.full(peak.shape, -240.0, dtype=np.float64)
        np.log10(np.maximum(peak, 1e-12), out=level)
        level *= 20.0
        gains = np.empty(data.shape[0], dtype=np.float64)
        state = self.gain_reduction
        for index, value in enumerate(level):
            desired = self._desired_reduction(float(value))
            if desired > state:
                state = desired + (state - desired) * self.attack_coeff
            else:
                state = desired + (state - desired) * self.release_coeff
            gains[index] = state
        self.gain_reduction = state
        output = data * (10.0 ** (-gains / 20.0))[:, None]
        return output, gains


class LookaheadLimiter:
    """True-peak look-ahead limiter with smooth attack and release.

    A true-peak envelope is evaluated at four phases per base-rate frame. The
    gain requirement is held across a look-ahead window before and after the
    frame, then smoothed recursively, so attenuation precedes the transient
    and release only starts after it has passed.
    """

    def __init__(self, sample_rate, channels, config, safety_db=0.15):
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.ceiling_db = float(config.ceiling_dbfs)
        self.ceiling = _db_to_linear(self.ceiling_db - float(safety_db))
        self.lookahead_ms = float(config.lookahead_ms)
        self.lookahead = max(1, int(round(self.sample_rate * self.lookahead_ms / 1000.0)))
        self.attack_coeff = math.exp(
            -1.0 / max(1.0, self.sample_rate * max(0.1, self.lookahead_ms * 0.25) / 1000.0)
        )
        self.release_coeff = math.exp(
            -1.0 / max(1.0, self.sample_rate * config.release_ms / 1000.0)
        )
        self.detector = TruePeakEnvelope(sample_rate, channels)
        self._audio = np.empty((0, self.channels), dtype=np.float64)
        self._audio_start = 0
        self._input_count = 0
        self._next_out = 0
        self._needs = np.empty(0, dtype=np.float64)
        self._needs_start = 0
        self.gain_reduction = 0.0
        self._finished = False

    @property
    def latency_frames(self):
        return self.lookahead * 2 + 2

    def _append_audio(self, block):
        if block.shape[0] == 0:
            return
        if self._audio.shape[0] == 0:
            self._audio = np.array(block, dtype=np.float64, copy=True)
        else:
            self._audio = np.concatenate((self._audio, block), axis=0)

    def _append_needs(self, peaks):
        if peaks.size == 0:
            return
        with np.errstate(divide="ignore", invalid="ignore"):
            need = np.where(
                peaks > self.ceiling,
                20.0 * np.log10(np.maximum(peaks, 1e-20) / self.ceiling),
                0.0,
            )
        self._needs = np.concatenate((self._needs, need))

    def _trim(self):
        audio_drop = self._next_out - self._audio_start
        if audio_drop > 8192:
            self._audio = self._audio[audio_drop:].copy()
            self._audio_start += audio_drop
        keep_need = max(0, self._next_out - 1)
        need_drop = keep_need - self._needs_start
        if need_drop > 8192:
            self._needs = self._needs[need_drop:].copy()
            self._needs_start += need_drop

    def _produce(self, end_index):
        if end_index < self._next_out:
            return 0, np.empty((0, self.channels)), np.empty(0)
        count = end_index - self._next_out + 1
        window_size = self.lookahead * 2 + 3
        global_start = self._next_out - 1
        global_stop = end_index + self.lookahead * 2 + 2
        padded = np.zeros(global_stop - global_start, dtype=np.float64)
        overlap_start = max(global_start, self._needs_start)
        overlap_stop = min(global_stop, self._needs_start + self._needs.size)
        if overlap_stop > overlap_start:
            padded[overlap_start - global_start:overlap_stop - global_start] = self._needs[
                overlap_start - self._needs_start:overlap_stop - self._needs_start
            ]
        windows = np.lib.stride_tricks.sliding_window_view(padded, window_size)
        needs = np.max(windows[:count], axis=1)

        reduction = np.empty(count, dtype=np.float64)
        state = self.gain_reduction
        if self._next_out == 0:
            # There is no audio before the file start in which to ramp down.
            # Seed the envelope with the first look-ahead requirement so an
            # opening transient cannot escape the ceiling.
            state = max(state, float(needs[0]))
        for index, need in enumerate(needs):
            target = float(need)
            if target > state:
                state = target + (state - target) * self.attack_coeff
            else:
                state = target + (state - target) * self.release_coeff
            reduction[index] = state
        self.gain_reduction = state

        local_start = self._next_out - self._audio_start
        local_stop = local_start + count
        source = self._audio[local_start:local_stop]
        if source.shape[0] != count:
            raise RuntimeError("limiter audio buffer underrun")
        output = source * (10.0 ** (-reduction / 20.0))[:, None]
        start = self._next_out
        self._next_out += count
        self._trim()
        return start, output, reduction

    def feed(self, block):
        if self._finished:
            raise RuntimeError("limiter already finished")
        data = np.asarray(block, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.channels:
            raise ValueError("limiter block has an unexpected channel count")
        self._append_audio(data)
        self._input_count += data.shape[0]
        peaks = self.detector.feed(data)
        self._append_needs(peaks)
        available = self.detector.emitted_frames
        safe_end = available - self.lookahead * 2 - 3
        safe_end = min(safe_end, self._input_count - 1)
        start, output, reduction = self._produce(safe_end)
        return start, output, reduction

    def flush(self):
        if self._finished:
            return 0, np.empty((0, self.channels)), np.empty(0)
        peaks = self.detector.finalize()
        self._append_needs(peaks)
        start, output, reduction = self._produce(self._input_count - 1)
        if self._next_out != self._input_count:
            raise RuntimeError(
                "limiter produced {} of {} frames".format(
                    self._next_out, self._input_count
                )
            )
        self._finished = True
        return start, output, reduction
