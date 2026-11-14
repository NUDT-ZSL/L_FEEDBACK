"""Deterministic per-event random variant engine.

Every event owns a dedicated :class:`numpy.random.Generator` derived from the
scene-level global seed and the event's integer ``seed_offset`` via
:class:`numpy.random.SeedSequence`. The draw order is fixed, so identical
scenes always produce byte-identical results regardless of Python hash
seeding, wall clock time, dict ordering or the platform RNG.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fixed draw order inside each event's RNG stream.
DRAW_PITCH = 0
DRAW_START = 1
DRAW_GAIN = 2

PITCH_LIMIT = 0.05  # +/- 5 percent
START_LIMIT_S = 0.020  # +/- 20 milliseconds
GAIN_LIMIT_DB = 1.5  # +/- 1.5 dB


@dataclass(frozen=True)
class EventVariants:
    """Resolved random variants for one event."""

    seed_value: int
    pitch_factor: float  # playback rate multiplier, e.g. 0.97..1.03
    start_offset_s: float  # signed source start shift, seconds
    gain_jitter_db: float


def event_seed(global_seed: int, seed_offset: int) -> int:
    """Derive the stable 64-bit seed used by one event."""
    sequence = np.random.SeedSequence([int(global_seed), int(seed_offset)])
    state = sequence.generate_state(4)
    return _fold_state(state)


def _fold_state(state: np.ndarray) -> int:
    value = 0
    for chunk in state:
        value = (value * 0x9E3779B1) ^ int(chunk)
        value &= 0xFFFFFFFFFFFFFFFF
    return value


def derive_variants(global_seed: int, seed_offset: int) -> EventVariants:
    """Draw pitch, start-offset and gain-jitter variants for an event.

    Uniform draws are mapped symmetrically to their documented ranges.
    """
    seed_value = event_seed(global_seed, seed_offset)
    rng = np.random.default_rng(np.random.SeedSequence(seed_value))

    pitch_factor = float(1.0 + (rng.random() * 2.0 - 1.0) * PITCH_LIMIT)
    start_offset_s = float((rng.random() * 2.0 - 1.0) * START_LIMIT_S)
    gain_jitter_db = float((rng.random() * 2.0 - 1.0) * GAIN_LIMIT_DB)
    return EventVariants(seed_value, pitch_factor, start_offset_s, gain_jitter_db)
