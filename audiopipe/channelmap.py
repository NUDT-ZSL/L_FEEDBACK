"""Channel mapping / downmix to the stereo delivery layout.

5.1 -> stereo follows ITU-R BS.775 with -3 dB (1/sqrt(2)) coefficients:
    L' = L + 0.7071*C + 0.7071*Ls
    R' = R + 0.7071*C + 0.7071*Rs
    LFE is discarded.
Mono is duplicated to both channels at -3 dB so the stereo integrated
loudness of the result matches the mono measurement.
"""

from __future__ import annotations

import math

import numpy as np

from .errors import Code, PipelineError

SQRT1_2 = 1.0 / math.sqrt(2.0)


def to_stereo(x: np.ndarray) -> np.ndarray:
    ch = x.shape[1]
    if ch == 2:
        return x
    if ch == 1:
        return np.repeat(x * SQRT1_2, 2, axis=1)
    if ch == 6:
        out = np.empty((x.shape[0], 2), dtype=np.float64)
        out[:, 0] = x[:, 0] + SQRT1_2 * x[:, 2] + SQRT1_2 * x[:, 4]
        out[:, 1] = x[:, 1] + SQRT1_2 * x[:, 2] + SQRT1_2 * x[:, 5]
        return out
    raise PipelineError(Code.E_UNSUPPORTED_LAYOUT,
                        f"channel count {ch} not supported (1/2/6 only)")
