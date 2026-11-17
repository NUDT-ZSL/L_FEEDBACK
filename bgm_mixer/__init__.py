"""Deterministic, loop-aware WAV BGM mixer."""

__version__ = "1.0.0"

from .errors import ERROR_EXIT_CODES, E_INVALID_INPUT, E_IO, E_MIX, E_RESAMPLE, MixerError

__all__ = [
    "ERROR_EXIT_CODES",
    "E_INVALID_INPUT",
    "E_IO",
    "E_MIX",
    "E_RESAMPLE",
    "MixerError",
    "__version__",
]
