"""Frame/sample time conversion shared by parsing, alignment and reporting."""

import math


def frame_to_sample(frame, frame_rate, sample_rate):
    """Convert a video frame number to an absolute sample position.

    Single conversion basis used everywhere: round-half-up of
    ``frame * sample_rate / frame_rate``.
    """
    value = frame * sample_rate / frame_rate
    return int(math.floor(value + 0.5))


def ms_to_samples(milliseconds, sample_rate):
    """Convert a duration in milliseconds to a sample count (round-half-up)."""
    return int(math.floor(milliseconds * sample_rate / 1000.0 + 0.5))
