"""Offline dynamic layer mixer for rhythm games.

Reads mono 44.1 kHz / 16-bit PCM WAV layer tracks, a BPM chart JSON and a
layer-control event JSON, snaps events to the beat grid, mixes the enabled
layers with seamless looped cross-fades and writes a deterministic mono WAV
plus a beat-alignment JSON report.
"""

from .errors import MixerError

__all__ = ["MixerError", "__version__"]
__version__ = "1.0.0"
