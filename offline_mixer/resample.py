"""Streaming band-limited sinc sample-rate conversion."""

from __future__ import annotations

import math

import numpy as np


class SincResampler:
    """Windowed-sinc interpolator with bounded streaming state.

    Kernel centres are evaluated at exact rational output positions. Only the
    overlap required by the current block is retained in memory.
    """

    def __init__(self, input_rate, output_rate, channels, output_frames,
                 taps=64, beta=12.0, max_block=4096):
        if input_rate <= 0 or output_rate <= 0:
            raise ValueError("sample rates must be positive")
        if taps < 16 or taps % 2:
            raise ValueError("taps must be an even integer >= 16")
        self.input_rate = int(input_rate)
        self.output_rate = int(output_rate)
        self.channels = int(channels)
        self.output_frames = max(0, int(output_frames))
        self.taps = int(taps)
        self.half = self.taps // 2
        self.beta = float(beta)
        self.max_block = int(max_block)
        self.passthrough = self.input_rate == self.output_rate
        self._offsets = np.arange(-self.half + 1, self.half + 1, dtype=np.int64)
        self._i0_beta = float(np.i0(self.beta))
        self._buffer = np.empty((0, self.channels), dtype=np.float64)
        self._buffer_start = 0
        self._total_in = 0
        self._next_out = 0
        self._finished = False

    @property
    def buffered_frames(self):
        return int(self._buffer.shape[0])

    def _append(self, block):
        if block.shape[0] == 0:
            return
        if self._buffer.shape[0] == 0:
            self._buffer = np.array(block, dtype=np.float64, copy=True)
        else:
            self._buffer = np.concatenate((self._buffer, block), axis=0)

    def feed(self, block):
        if self._finished:
            raise RuntimeError("resampler already finished")
        data = np.asarray(block, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.channels:
            raise ValueError("resampler block has an unexpected channel count")
        if self.passthrough:
            self._total_in += data.shape[0]
            return np.array(data, copy=True)
        self._append(data)
        self._total_in += data.shape[0]
        return self._emit(False)

    def _trim(self):
        if self._next_out >= self.output_frames:
            self._buffer_start = self._total_in
            self._buffer = np.empty((0, self.channels), dtype=np.float64)
            return
        time = self._next_out * self.input_rate / self.output_rate
        keep_from = int(math.floor(time)) - self.half + 1
        drop = keep_from - self._buffer_start
        if drop > 8192:
            self._buffer = self._buffer[drop:].copy()
            self._buffer_start += drop

    def _emit(self, final):
        outputs = []
        while self._next_out < self.output_frames:
            count = min(self.max_block, self.output_frames - self._next_out)
            indices = np.arange(self._next_out, self._next_out + count, dtype=np.float64)
            times = indices * self.input_rate / self.output_rate
            bases = np.floor(times).astype(np.int64)
            if not final:
                ready_end = bases + self.half
                ready = int(np.searchsorted(ready_end, self._total_in, side="left"))
                if ready <= 0:
                    break
                count = ready
                indices = indices[:count]
                times = times[:count]
                bases = bases[:count]
            else:
                valid_end = bases + self.half
                if valid_end[0] > self._total_in + self.half:
                    break

            lattice = bases[:, None] + self._offsets[None, :]
            valid = (lattice >= 0) & (lattice < self._total_in)
            local = np.clip(lattice, 0, max(0, self._total_in - 1))
            local = np.clip(local - self._buffer_start, 0, max(0, self._buffer.shape[0] - 1))
            source = self._buffer[local] * valid[:, :, None]

            delta = times[:, None] - lattice
            cutoff = min(1.0, self.output_rate / self.input_rate) * 0.97
            scaled = delta * cutoff
            ratio = delta / self.half
            window = np.zeros_like(delta)
            inside = np.abs(ratio) < 1.0
            window[inside] = (
                np.i0(self.beta * np.sqrt(np.maximum(0.0, 1.0 - ratio[inside] ** 2)))
                / self._i0_beta
            )
            kernel = cutoff * np.sinc(scaled) * window
            sums = kernel.sum(axis=1)
            kernel /= np.maximum(sums[:, None], 1e-20)
            output = np.einsum("bk,bkc->bc", kernel, source, optimize=True)
            outputs.append(output)
            self._next_out += count
            self._trim()

        if not outputs:
            return np.empty((0, self.channels), dtype=np.float64)
        return np.concatenate(outputs, axis=0)

    def finish(self):
        if self._finished:
            return np.empty((0, self.channels), dtype=np.float64)
        if self.passthrough:
            self._finished = True
            return np.empty((0, self.channels), dtype=np.float64)
        padding = np.zeros((self.half + 2, self.channels), dtype=np.float64)
        self._append(padding)
        result = self._emit(True)
        if self._next_out != self.output_frames:
            raise RuntimeError(
                "resampler produced {} of {} frames".format(
                    self._next_out, self.output_frames
                )
            )
        self._finished = True
        return result
