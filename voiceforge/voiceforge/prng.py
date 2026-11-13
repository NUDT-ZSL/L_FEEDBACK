"""A small self-contained deterministic pseudo random generator.

Relying on a fixed xorshift64* implementation keeps the generated samples
byte-for-byte reproducible across Python versions and operating systems,
independent of CPython's internal ``random`` module evolution.
"""

import math

MASK64 = (1 << 64) - 1


class DeterministicRandom:
    """Deterministic 64-bit xorshift* stream with convenience helpers."""

    def __init__(self, seed: int):
        state = int(seed) & MASK64
        if state == 0:
            state = 0x9E3779B97F4A7C15
        self._state = state

    def next_u64(self) -> int:
        state = self._state
        state ^= (state << 13) & MASK64
        state ^= state >> 7
        state ^= (state << 17) & MASK64
        state &= MASK64
        self._state = state
        return (state * 0x2545F4914F6CDD1D) & MASK64

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return low + (self.next_u64() / float(1 << 64)) * (high - low)

    def randint(self, low: int, high: int) -> int:
        """Inclusive integer in [low, high]."""
        span = high - low + 1
        return low + (self.next_u64() % span)

    def choice(self, options):
        return options[self.next_u64() % len(options)]


def phase_sin(rng: DeterministicRandom, index: int, frequency: float,
              sample_rate: int) -> float:
    """Deterministic sine value with a reproducible per-stream phase."""
    phase = rng.uniform(0.0, 2.0 * math.pi)
    return math.sin(2.0 * math.pi * frequency * index / sample_rate + phase)

