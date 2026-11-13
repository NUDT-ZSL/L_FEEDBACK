"""Map shard sample streams onto the unified output time axis.

The game server time base is monotonic milliseconds since match start.
Each shard records its own ``record_offset_ms`` (first captured sample on
that time base) and sample rate.  When the captured PCM duration drifts from
the manifest declaration by more than ``DRIFT_TOLERANCE_MS`` milliseconds the
shard is linearly re-sampled to the declared length instead of being silently
truncated or padded.
"""

import math

from . import OUTPUT_SAMPLE_RATE

DRIFT_TOLERANCE_MS = 20.0


def duration_deviation_ms(declared_ms: float, actual_ms: float) -> float:
    return abs(actual_ms - declared_ms)


def needs_resampling(declared_ms: float, actual_ms: float,
                     tolerance_ms: float = DRIFT_TOLERANCE_MS) -> bool:
    return duration_deviation_ms(declared_ms, actual_ms) > tolerance_ms


def ms_to_samples(ms: float, sample_rate: int = OUTPUT_SAMPLE_RATE) -> int:
    return int(math.floor(ms * sample_rate / 1000.0))


def samples_to_ms(sample_index: int, sample_rate: int = OUTPUT_SAMPLE_RATE) -> float:
    return sample_index * 1000.0 / sample_rate


def render_segment(source_floats, source_rate: int, record_offset_ms: float,
                   declared_ms: float, output_rate: int = OUTPUT_SAMPLE_RATE):
    """Render a shard onto the unified output grid via linear interpolation.

    Returns ``(start_output_sample, rendered_floats)``.  The effective shard
    duration is the declared duration when drift exceeds the tolerance, and
    the physically captured duration otherwise.  The source stream is always
    linearly interpolated onto the output grid so mixed sample rates align.
    """
    source_count = len(source_floats)
    actual_ms = source_count * 1000.0 / source_rate
    if needs_resampling(declared_ms, actual_ms):
        effective_ms = declared_ms
    else:
        effective_ms = actual_ms

    start_sample = ms_to_samples(record_offset_ms, output_rate)
    span = ms_to_samples(effective_ms, output_rate)
    if span <= 0 or source_count == 0:
        return start_sample, []

    rendered = [0.0] * span
    last_source = source_count - 1
    for index in range(span):
        output_time_ms = (start_sample + index) * 1000.0 / output_rate
        relative_ms = output_time_ms - record_offset_ms
        position = relative_ms / effective_ms * source_count
        if position <= 0.0:
            rendered[index] = source_floats[0]
        elif position >= last_source:
            rendered[index] = source_floats[last_source]
        else:
            lower = int(math.floor(position))
            fraction = position - lower
            low_value = source_floats[lower]
            high_value = source_floats[lower + 1]
            rendered[index] = low_value + (high_value - low_value) * fraction
    return start_sample, rendered

