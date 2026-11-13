"""PCM WAV helpers and deterministic 16-bit saturation helpers."""

import array
import math
import sys
import wave

INT16_MIN = -32768
INT16_MAX = 32767
FLOAT_SCALE = 32768.0


class WavData:
    """In-memory mono PCM WAV payload."""

    def __init__(self, sample_rate: int, samples: array.array):
        if samples.typecode != "h":
            raise ValueError("WavData only holds signed 16-bit samples")
        self.sample_rate = sample_rate
        self.samples = samples

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def duration_ms(self) -> float:
        return len(self.samples) * 1000.0 / self.sample_rate


def read_wav(path: str) -> WavData:
    with wave.open(path, "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        raw = wav.readframes(frame_count)
    if channels != 1:
        raise ValueError(f"{path}: only mono WAV is supported, got {channels} channels")
    if sample_width != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV is supported, got width {sample_width}")
    samples = array.array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    return WavData(sample_rate, samples)


def write_wav(path: str, sample_rate: int, samples: array.array) -> None:
    if samples.typecode != "h":
        raise ValueError("write_wav expects a signed-short array")
    payload = samples
    if sys.byteorder != "little":
        payload = array.array("h", samples)
        payload.byteswap()
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(payload.tobytes())


def int16_to_float(samples) -> list:
    """Convert signed 16-bit PCM samples to normalized floats in [-1, 1]."""
    return [sample / FLOAT_SCALE for sample in samples]


def saturate_float(value: float) -> int:
    """Round-half-to-even then clamp to the signed 16-bit integer range."""
    rounded = int(math.floor(value + 0.5)) if value >= 0 else int(math.ceil(value - 0.5))
    if rounded > INT16_MAX:
        return INT16_MAX
    if rounded < INT16_MIN:
        return INT16_MIN
    return rounded


def floats_to_int16(values) -> array.array:
    """Convert normalized float samples to saturated signed 16-bit PCM."""
    out = array.array("h")
    append = out.append
    for value in values:
        append(saturate_float(value * FLOAT_SCALE))
    return out


def is_clipped(value: float) -> bool:
    """A normalized float sample clips when saturation would clamp it."""
    return value >= INT16_MAX / FLOAT_SCALE or value <= INT16_MIN / FLOAT_SCALE

