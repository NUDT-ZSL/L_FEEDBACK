"""Packet loss concealment (PLC).

Two complementary algorithms are used:

* **Pitch periodic repetition** when a reliable pitch period is found in the
  recent good history: a period-sized waveform is repeated and crossfaded.
* **Linear prediction** fallback: an order-2 LPC extrapolation driven by the
  last good samples.

At most :data:`MAX_PLC_FRAMES` consecutive frames are actively synthesized and
smoothly crossfaded with the preceding audio. Longer loss runs collapse to
silence and are flagged with a recovery point when real audio returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .constants import FRAME_SAMPLES, MAX_PLC_FRAMES

MIN_PITCH_LAG = 60  # 1.25 ms at 48 kHz (800 Hz ceiling)
MAX_PITCH_LAG = 600  # 12.5 ms (80 Hz floor)
PITCH_CONFIDENCE = 0.45
HISTORY_SAMPLES = 4 * FRAME_SAMPLES
LP_ORDER = 2


@dataclass
class PlcResult:
    """Output of one concealment / hand-off decision."""

    frame: np.ndarray
    method: str  # "pitch" | "linear_prediction" | "silence"
    concealed: bool
    recovery: bool


def estimate_pitch(history: np.ndarray) -> Tuple[Optional[int], float]:
    """Estimate pitch lag via normalized autocorrelation.

    Returns ``(lag, confidence)``; lag is ``None`` when confidence is too low.
    """
    if history.size < MAX_PITCH_LAG + FRAME_SAMPLES:
        return None, 0.0

    end = history.size
    reference = history[end - FRAME_SAMPLES:end].astype(np.float64)
    energy = float(np.dot(reference, reference))
    if energy < 1e-10:
        return None, 0.0

    best_lag: Optional[int] = None
    best_confidence = 0.0
    for lag in range(MIN_PITCH_LAG, MAX_PITCH_LAG + 1):
        candidate = history[
            end - FRAME_SAMPLES - lag:end - lag
        ].astype(np.float64)
        denom = np.sqrt(energy * float(np.dot(candidate, candidate)))
        if denom < 1e-10:
            continue
        confidence = float(np.dot(reference, candidate)) / denom
        if confidence > best_confidence:
            best_confidence = confidence
            best_lag = lag

    if best_lag is not None and best_confidence >= PITCH_CONFIDENCE:
        return best_lag, best_confidence
    return None, best_confidence


def _linear_prediction(history: np.ndarray, count: int) -> np.ndarray:
    """Order-2 LPC extrapolation estimated on recent history."""
    base = history[-(FRAME_SAMPLES + LP_ORDER):].astype(np.float64)
    if base.size < LP_ORDER + 2:
        return np.zeros(count, dtype=np.float32)

    # Solve the Yule-Walker style normal equations on local autocorrelation.
    x = base[:-LP_ORDER]
    design = np.column_stack(
        [base[LP_ORDER - k - 1:len(base) - k - 1] for k in range(LP_ORDER)]
    )
    gram = design.T @ design + 1e-8 * np.eye(LP_ORDER)
    coeffs = np.linalg.solve(gram, design.T @ x)

    tail = base[-LP_ORDER:].copy()
    output = np.zeros(count, dtype=np.float64)
    for idx in range(count):
        value = float(coeffs @ tail[::-1])
        output[idx] = value
        tail = np.roll(tail, -1)
        tail[-1] = value
    return output.astype(np.float32)


class PlcState:
    """Per-stream PLC state machine."""

    def __init__(self, stream_id: int) -> None:
        self.stream_id = stream_id
        self.history = np.zeros(HISTORY_SAMPLES, dtype=np.float32)
        self.consecutive_lost = 0
        self.long_run_active = False
        self._last_lag: Optional[int] = None

    def reset(self) -> None:
        self.history.fill(0.0)
        self.consecutive_lost = 0
        self.long_run_active = False
        self._last_lag = None

    def accept_good_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Feed a real frame back into history.

        When the previous loss run exceeded the PLC budget the returned frame
        is faded in to avoid a click at the recovery point.
        """
        frame = frame.astype(np.float32, copy=True)
        recovery = False
        if self.long_run_active:
            fade = np.linspace(0.0, 1.0, FRAME_SAMPLES, dtype=np.float32)
            frame *= fade
            recovery = True
            self.long_run_active = False
        self.consecutive_lost = 0
        self.history = np.roll(self.history, -FRAME_SAMPLES)
        self.history[-FRAME_SAMPLES:] = frame
        return frame if recovery else None

    def conceal(self) -> PlcResult:
        """Synthesize one replacement frame for a lost playout frame."""
        self.consecutive_lost += 1

        if self.consecutive_lost > MAX_PLC_FRAMES:
            self.long_run_active = True
            return PlcResult(
                frame=np.zeros(FRAME_SAMPLES, dtype=np.float32),
                method="silence",
                concealed=False,
                recovery=False,
            )

        lag, _ = estimate_pitch(self.history)
        method = "pitch"
        if lag is None:
            method = "linear_prediction"
            continuation = _linear_prediction(self.history, FRAME_SAMPLES)
        else:
            self._last_lag = lag
            period = self.history[-lag:].astype(np.float32)
            # True continuation: after period[lag - 1] comes period[0].
            continuation_offsets = (
                np.arange(FRAME_SAMPLES) + lag
            ) % lag
            continuation = period[continuation_offsets].astype(np.float32)

        # Progressive attenuation across consecutive concealed frames.
        decay = 0.85 ** (self.consecutive_lost - 1)
        continuation *= decay

        # Crossfade with the last good samples to guarantee smooth joining.
        crossfade = min(240, FRAME_SAMPLES // 3)
        previous_tail = self.history[-crossfade:].astype(np.float32)
        weight_in = np.linspace(0.0, 1.0, crossfade, dtype=np.float32)
        blended = continuation.copy()
        blended[:crossfade] = (
            previous_tail * (1.0 - weight_in)
            + continuation[:crossfade] * weight_in
        )
        synthesized = blended

        self.history = np.roll(self.history, -FRAME_SAMPLES)
        self.history[-FRAME_SAMPLES:] = synthesized
        return PlcResult(
            frame=synthesized.astype(np.float32),
            method=method,
            concealed=True,
            recovery=False,
        )
