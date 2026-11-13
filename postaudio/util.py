"""Small deterministic helpers shared by the audio pipeline."""

from __future__ import annotations

from fractions import Fraction
import math
from typing import Any


def fraction_from_number(value: float | int) -> Fraction:
    """Return an exact rational for a JSON number without binary surprises."""
    if isinstance(value, int):
        return Fraction(value, 1)
    return Fraction(str(value))


def round_half_even(value: Fraction | int | float) -> int:
    """Round exactly using IEEE round-half-to-even (banker's rounding)."""
    if isinstance(value, Fraction):
        numerator, denominator = value.numerator, value.denominator
    elif isinstance(value, int):
        return value
    else:
        exact = fraction_from_number(value)
        numerator, denominator = exact.numerator, exact.denominator
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if numerator < 0 else 1
    q, r = divmod(abs(numerator), denominator)
    twice = 2 * r
    if twice > denominator or (twice == denominator and q % 2 == 1):
        q += 1
    return sign * q


def round_half_even_array(values: Any) -> Any:
    """Vectorized round-half-to-even for a NumPy float64 array."""
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    lower = np.floor(values)
    fraction = values - lower
    result = lower.astype(np.int64)
    ties = fraction == 0.5
    odd = (result & 1) == 1
    round_up = (fraction > 0.5) | (ties & odd)
    result = np.where(round_up, result + 1, result)
    return result


def seconds_to_samples(seconds: float | int, sample_rate: int) -> int:
    return round_half_even(fraction_from_number(seconds) * sample_rate)


def samples_to_seconds(samples: int, sample_rate: int) -> float:
    return float(Fraction(samples, sample_rate))


def db_to_linear(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def linear_to_db(value: float) -> float:
    if value <= 0.0:
        return float("-inf")
    return float(20.0 * math.log10(value))


def peak_dbfs(values: Any) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("-inf")
    peak = float(np.max(np.abs(arr)))
    return linear_to_db(peak)


def rms(values: Any) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr, dtype=np.float64)))


def rms_dbfs(values: Any) -> float:
    return linear_to_db(rms(values))


def finite_number(value: float | int) -> float | None:
    value = float(value)
    if not math.isfinite(value):
        return None
    if abs(value) < 0.000000000001:
        return 0.0
    return value
