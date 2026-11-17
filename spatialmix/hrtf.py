"""Deterministic, locally generated HRTF data set.

The data set is deliberately synthetic.  It has the exact geometry required by
the task (24 azimuths, 12 elevations and one left/right impulse response per
direction) and is generated from pure functions, so runs are reproducible and
no external audio asset is required.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

HRTF_DATASET_VERSION = "local-hrtf-v1"
AZIMUTH_COUNT = 24
ELEVATION_COUNT = 12
MAX_ITD_SECONDS = 0.00065


def _direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Return a unit vector as [right, up, forward]."""

    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return np.array(
        [
            np.cos(el) * np.sin(az),
            np.sin(el),
            np.cos(el) * np.cos(az),
        ],
        dtype=np.float64,
    )


def _synth_ir(
    azimuth_deg: float,
    elevation_deg: float,
    ear: str,
    sample_rate: int,
    length: int,
) -> np.ndarray:
    direction = _direction(azimuth_deg, elevation_deg)
    lateral, vertical, forward = direction
    n = np.arange(length, dtype=np.float64)
    max_itd = MAX_ITD_SECONDS * float(sample_rate)

    if ear == "left":
        delay = max(0.0, lateral) * max_itd
        shadow = 1.0 - 0.50 * max(0.0, lateral)
    else:
        delay = max(0.0, -lateral) * max_itd
        shadow = 1.0 - 0.50 * max(0.0, -lateral)

    window = np.exp(-n / max(8.0, length * 0.72))
    direct = np.sinc((n - delay) / 1.18) * window
    broadband = 0.16 * np.sinc((n - delay - 1.6) / 4.0) * window

    notch_hz = 3300.0 + 4300.0 * (vertical + 1.0) / 2.0
    notch = (
        -0.24
        * np.exp(-0.5 * ((n - delay - 2.2) / 1.5) ** 2)
        * np.cos(2.0 * np.pi * notch_hz / sample_rate * (n - delay))
    )
    reflection_delay = 2.0 + 5.0 * (1.0 - max(0.0, forward))
    reflection = (
        -0.10
        * np.exp(-0.5 * ((n - delay - reflection_delay) / 1.1) ** 2)
        * (1.15 + 0.25 * max(0.0, -forward))
    )

    ir = (direct + broadband + notch + reflection) * shadow
    ir *= 0.88 + 0.12 * max(0.0, forward)
    ir[0] += 0.12 * (1.0 - abs(lateral))

    denominator = float(np.sum(np.abs(ir)))
    if denominator <= np.finfo(np.float64).tiny:
        ir = np.zeros(length, dtype=np.float64)
        ir[0] = 1.0
    else:
        ir /= denominator
    return ir.astype(np.float32)


@dataclass(frozen=True)
class HRTFDataset:
    sample_rate: int
    ir_length: int
    azimuths_deg: np.ndarray
    elevations_deg: np.ndarray
    directions: np.ndarray
    impulse_responses: np.ndarray
    version: str = HRTF_DATASET_VERSION

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return self.impulse_responses.shape

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.version.encode("ascii"))
        digest.update(str(self.sample_rate).encode("ascii"))
        digest.update(self.azimuths_deg.astype(np.float64).tobytes())
        digest.update(self.elevations_deg.astype(np.float64).tobytes())
        digest.update(self.directions.astype(np.float64).tobytes())
        digest.update(self.impulse_responses.astype(np.float32).tobytes())
        return digest.hexdigest()

    def metadata(self) -> Dict[str, object]:
        return {
            "version": self.version,
            "sample_rate": self.sample_rate,
            "ir_length": self.ir_length,
            "azimuth_count": int(self.azimuths_deg.size),
            "elevation_count": int(self.elevations_deg.size),
            "azimuths_deg": self.azimuths_deg.tolist(),
            "elevations_deg": self.elevations_deg.tolist(),
            "fingerprint_sha256": self.fingerprint(),
        }

    def select(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        azimuth_offset_deg: float = 0.0,
        elevation_offset_deg: float = 0.0,
        enabled: bool = True,
    ) -> Tuple[Tuple[int, int], np.ndarray]:
        """Select the nearest grid direction using a vector-space dot product."""

        if not enabled:
            delta = np.zeros(self.ir_length, dtype=np.float32)
            delta[0] = 1.0
            return (-1, -1), np.stack((delta, delta), axis=0)

        query = _direction(
            azimuth_deg + azimuth_offset_deg,
            np.clip(elevation_deg + elevation_offset_deg, -90.0, 90.0),
        )
        dots = self.directions.reshape(-1, 3) @ query
        flat_index = int(np.argmax(dots))
        elevation_index, azimuth_index = np.unravel_index(
            flat_index, (ELEVATION_COUNT, AZIMUTH_COUNT)
        )
        return (
            int(elevation_index),
            int(azimuth_index),
        ), self.impulse_responses[elevation_index, azimuth_index]


def generate_hrtf(sample_rate: int) -> HRTFDataset:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    ir_length = max(48, int(round(0.0030 * sample_rate)))
    azimuths = np.linspace(
        -180.0, 180.0, AZIMUTH_COUNT, endpoint=False, dtype=np.float64
    )
    elevations = np.linspace(-60.0, 90.0, ELEVATION_COUNT, dtype=np.float64)

    directions = np.empty((ELEVATION_COUNT, AZIMUTH_COUNT, 3), dtype=np.float64)
    irs = np.empty((ELEVATION_COUNT, AZIMUTH_COUNT, 2, ir_length), dtype=np.float32)
    for elevation_index, elevation in enumerate(elevations):
        for azimuth_index, azimuth in enumerate(azimuths):
            directions[elevation_index, azimuth_index] = _direction(azimuth, elevation)
            irs[elevation_index, azimuth_index, 0] = _synth_ir(
                float(azimuth), float(elevation), "left", sample_rate, ir_length
            )
            irs[elevation_index, azimuth_index, 1] = _synth_ir(
                float(azimuth), float(elevation), "right", sample_rate, ir_length
            )

    return HRTFDataset(
        sample_rate=sample_rate,
        ir_length=ir_length,
        azimuths_deg=azimuths,
        elevations_deg=elevations,
        directions=directions,
        impulse_responses=irs,
    )


def overlap_save_convolve(
    signal: np.ndarray, impulse: np.ndarray, block_size: int
) -> np.ndarray:
    """Reference streaming convolution used by tests.

    This implementation writes the same B-sample output segments and uses a
    (L-1)-sample overlap context.  Its numerical result matches
    ``numpy.convolve(signal, impulse, mode="full")`` to floating point tolerance.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    x = np.asarray(signal, dtype=np.float64)
    h = np.asarray(impulse, dtype=np.float64)
    if x.ndim != 1 or h.ndim != 1:
        raise ValueError("signal and impulse must be one-dimensional")
    n = x.size
    length = h.size
    if n == 0:
        return np.zeros(length - 1, dtype=np.float64)
    if length == 0:
        return np.zeros(n, dtype=np.float64)

    out = np.zeros(n + length - 1, dtype=np.float64)
    history = np.zeros(length - 1, dtype=np.float64)
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        block = x[start:stop]
        context = np.concatenate((history, block))
        full = np.convolve(context, h, mode="full")
        segment = full[length - 1 : length - 1 + block.size]
        out[start:stop] = segment
        history = context[-(length - 1) :].copy()

    context = np.concatenate((history, np.zeros(length - 1, dtype=np.float64)))
    full = np.convolve(context, h, mode="full")
    tail_start = length - 1
    out[n : n + length - 1] = full[tail_start : tail_start + length - 1]
    return out
