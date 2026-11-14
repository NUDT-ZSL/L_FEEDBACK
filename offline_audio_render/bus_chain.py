"""Bus effect chain: gain, soft-knee peak limiter and feedback delay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curves import db_to_gain
from .scene import BusSpec

LIMITER_THRESHOLD_DB = -1.0
LIMITER_KNEE_DB = 3.0


class SoftKneeLimiter:
    """Peak limiter at -1 dBFS with a 3 dB soft knee and per-channel state."""

    def __init__(self, channels: int, sample_rate: int) -> None:
        self._threshold = float(db_to_gain(LIMITER_THRESHOLD_DB))
        self._knee_start = float(db_to_gain(LIMITER_THRESHOLD_DB - LIMITER_KNEE_DB))
        self._envelope = np.zeros(channels, dtype=np.float64)
        # 1 ms attack, 50 ms release as smoothing coefficients.
        self._attack = float(1.0 - np.exp(-1.0 / (0.001 * sample_rate)))
        self._release = float(1.0 - np.exp(-1.0 / (0.050 * sample_rate)))

    def process(self, block: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(block), axis=0)
        target = np.where(peak > self._envelope, self._attack, self._release)
        self._envelope = (
            (1.0 - target) * self._envelope + target * peak
        )
        gain = np.ones(block.shape[1], dtype=np.float64)
        active = self._envelope > self._knee_start
        if np.any(active):
            env = self._envelope[active]
            # Soft knee: smoothly interpolate unity gain toward hard threshold.
            blend = np.clip(
                (env - self._knee_start)
                / (self._threshold - self._knee_start + 1e-12),
                0.0,
                1.0,
            )
            knee_gain = 1.0 - blend * (1.0 - self._threshold / np.maximum(env, 1e-12))
            gain[active] = knee_gain
        reduced = block * gain[None, :]
        # Ceiling guarantee even if an inter-sample peak exceeds the envelope.
        return np.clip(reduced, -self._threshold, self._threshold)


class FeedbackDelay:
    """Simple per-channel feedback delay with a persistent ring buffer."""

    def __init__(
        self, channels: int, sample_rate: int, seconds: float, feedback: float
    ) -> None:
        self._feedback = float(feedback)
        self._delay_frames = max(1, int(round(seconds * sample_rate)))
        self._buffer = np.zeros(
            (self._delay_frames, channels), dtype=np.float64
        )
        self._write = 0

    def process(self, block: np.ndarray) -> np.ndarray:
        if self._feedback <= 0.0:
            return block
        out = np.empty_like(block)
        size = self._delay_frames
        feedback = self._feedback
        write = self._write
        remaining = block.shape[0]
        cursor = 0
        while cursor < remaining:
            contiguous = min(size - write, remaining - cursor)
            read = (write - size) % size
            # At most two ring segments are needed per contiguous write run.
            if read + contiguous <= size:
                delayed = self._buffer[read : read + contiguous].copy()
            else:
                first = size - read
                delayed = np.vstack(
                    (
                        self._buffer[read:],
                        self._buffer[: contiguous - first],
                    )
                )
            segment = block[cursor : cursor + contiguous]
            out[cursor : cursor + contiguous] = segment + delayed
            self._buffer[write : write + contiguous] = segment + feedback * delayed
            write = (write + contiguous) % size
            cursor += contiguous
        self._write = write
        return out


@dataclass
class BusProcessor:
    spec: BusSpec
    limiter: SoftKneeLimiter | None
    delay: FeedbackDelay | None
    static_gain: float

    @classmethod
    def build(cls, spec: BusSpec, channels: int, sample_rate: int) -> "BusProcessor":
        limiter = SoftKneeLimiter(channels, sample_rate) if spec.limiter else None
        delay = (
            FeedbackDelay(
                channels,
                sample_rate,
                spec.delay_seconds,
                spec.delay_feedback,
            )
            if spec.delay_seconds > 0.0
            else None
        )
        return cls(
            spec=spec,
            limiter=limiter,
            delay=delay,
            static_gain=float(db_to_gain(spec.gain_db)),
        )

    def process(self, block: np.ndarray) -> np.ndarray:
        # Chain order: gain -> limiter -> feedback delay.
        block = block * self.static_gain
        if self.limiter is not None:
            block = self.limiter.process(block)
        if self.delay is not None:
            block = self.delay.process(block)
        return block
