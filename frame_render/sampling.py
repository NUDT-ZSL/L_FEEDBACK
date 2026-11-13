from __future__ import annotations

import numpy as np

from .spec import SamplingConfig


def sample_offsets(
    config: SamplingConfig, width: int, height: int, frame_seed: int
) -> np.ndarray:
    """Return (height, width, axis_y, axis_x, 2) in-pixel offsets.

    Every frame uses a local, explicitly seeded PCG64 generator. No global
    numpy random state is read or modified.
    """
    count = config.samples_per_axis
    if config.jitter:
        generator = np.random.Generator(np.random.PCG64(np.uint64(frame_seed)))
        offsets = generator.random((height, width, count, count, 2), dtype=np.float64)
    else:
        offsets = np.full(
            (height, width, count, count, 2), 0.5, dtype=np.float64
        )
    return offsets

