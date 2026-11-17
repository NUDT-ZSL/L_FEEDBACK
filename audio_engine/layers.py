"""Combo-state driven layered mixing (base / percussion / ambience)."""

from __future__ import annotations

import numpy as np

LAYER_NAMES = ("base", "percussion", "ambience")


def state_to_layer_gains(combo: int) -> tuple[float, float, float]:
    """Map a combo counter to (base, percussion, ambience) gains."""
    if combo <= 4:
        return (1.0, 0.0, 0.0)
    if combo <= 9:
        return (1.0, 1.0, 0.0)
    return (1.0, 1.0, 1.0)


def _boxcar_smooth(target: np.ndarray, width: int) -> np.ndarray:
    """Causal linear-ramp smoothing: a step at sample n becomes a ramp
    spanning [n, n + width), i.e. transitions complete within ``width``."""
    if width <= 1:
        return target
    padded = np.concatenate([np.full(width, target[0]), target])
    csum = np.concatenate([[0.0], np.cumsum(padded)])
    n = target.size
    return (csum[width + 1 : width + 1 + n] - csum[1 : n + 1]) / width


def layer_gain_envelopes(
    combo_states: list[int],
    fps: float,
    n_samples: int,
    sample_rate: int,
    fade_s: float = 0.050,
) -> dict[str, np.ndarray]:
    """Per-sample gain envelopes for each layer.

    ``combo_states`` is a per-frame combo counter sampled at ``fps``.
    Layer add/remove transitions complete within ``fade_s`` (linear ramps
    produced by boxcar smoothing of the step targets).
    """
    if not combo_states:
        combo_states = [0]
    frames = np.asarray(combo_states, dtype=np.int64)
    frame_idx = np.minimum(
        (np.arange(n_samples, dtype=np.int64) * int(round(fps))) // sample_rate,
        frames.size - 1,
    )
    gains = np.array([state_to_layer_gains(int(c)) for c in frames])  # (F, 3)
    width = max(1, int(round(fade_s * sample_rate)))
    envelopes = {}
    for i, name in enumerate(LAYER_NAMES):
        target = gains[frame_idx, i]
        envelopes[name] = _boxcar_smooth(target, width)
    return envelopes
