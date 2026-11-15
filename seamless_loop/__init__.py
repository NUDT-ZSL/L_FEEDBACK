"""Offline seamless multi-track audio loop processor.

The package reads a JSON project description, slices each input track to a
sample-accurate loop region, applies an equal-power sine/cosine crossfade at
the loop seam, aligns every track to the exact same loop length, repeats the
loop to a target duration, mixes the tracks to stereo and writes a WAV file
plus a JSON processing report.
"""

from .errors import (
    EXIT_CONFIG,
    EXIT_FILE,
    EXIT_RANGE,
    EXIT_OVERWRITE,
    LoopProcessorError,
    ConfigError,
    AudioFileError,
    ParameterRangeError,
    OverwriteError,
)
from .crossfade import equal_power_windows, build_loop

__all__ = [
    "EXIT_CONFIG",
    "EXIT_FILE",
    "EXIT_RANGE",
    "EXIT_OVERWRITE",
    "LoopProcessorError",
    "ConfigError",
    "AudioFileError",
    "ParameterRangeError",
    "OverwriteError",
    "equal_power_windows",
    "build_loop",
]

__version__ = "1.0.0"
