from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import soundfile as sf


SUBTYPE_BITS: Dict[str, int] = {
    "PCM_U8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
}

class AudioError(RuntimeError):
    pass


class FormatMismatch(ValueError):
    pass


@dataclass(frozen=True)
class WavFormat:
    sample_rate: int
    channels: int
    bit_depth: int
    frames: int
    subtype: str


def inspect_wav(path: Path) -> WavFormat:
    try:
        info = sf.info(str(path))
    except (OSError, RuntimeError, sf.LibsndfileError) as exc:
        raise AudioError("cannot decode WAV header: {}".format(exc))

    if info.format not in ("WAV", "WAVEX"):
        raise FormatMismatch(
            "container={} (expected WAV/WAVEX)".format(info.format)
        )
    subtype = str(info.subtype)
    if subtype not in SUBTYPE_BITS:
        raise FormatMismatch(
            "subtype={} (expected a PCM WAV subtype)".format(subtype)
        )
    return WavFormat(
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        bit_depth=SUBTYPE_BITS[subtype],
        frames=int(info.frames),
        subtype=subtype,
    )


def read_matching_wav(
    path: Path,
    sample_rate: int,
    bit_depth: int,
    channels: int,
) -> Tuple[WavFormat, np.ndarray]:
    actual = inspect_wav(path)
    mismatches = []
    if actual.sample_rate != sample_rate:
        mismatches.append(
            "sample_rate={} (expected {})".format(
                actual.sample_rate, sample_rate
            )
        )
    if actual.bit_depth != bit_depth:
        mismatches.append(
            "bit_depth={} (expected {})".format(
                actual.bit_depth, bit_depth
            )
        )
    if actual.channels != channels:
        mismatches.append(
            "channels={} (expected {})".format(actual.channels, channels)
        )
    if actual.frames <= 0:
        mismatches.append("frames=0 (expected at least one sample)")
    if mismatches:
        raise FormatMismatch("; ".join(mismatches))

    try:
        samples, read_rate = sf.read(
            str(path),
            dtype="float64",
            always_2d=True,
            frames=-1,
            fill_value=0.0,
        )
    except (OSError, RuntimeError, sf.LibsndfileError) as exc:
        raise AudioError("cannot decode samples: {}".format(exc))
    if read_rate != actual.sample_rate:
        raise AudioError(
            "decoder reported a different sample rate while reading"
        )
    if samples.ndim != 2 or samples.shape[1] != actual.channels:
        raise AudioError("decoded channel shape does not match WAV header")
    if samples.shape[0] != actual.frames:
        raise AudioError("decoded frame count does not match WAV header")
    if not np.all(np.isfinite(samples)):
        raise AudioError("decoded audio contains NaN or infinite samples")
    return actual, np.asarray(samples, dtype=np.float64)
