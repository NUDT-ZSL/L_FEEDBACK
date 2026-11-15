"""Offline batch audio loudness normalization tool.

Measures ITU-R BS.1770-4 integrated loudness, normalizes to an
EBU R128 style target, applies true-peak limiting, aligns files on a
common timeline and writes uniform 48 kHz / 24-bit / stereo WAV output.
"""

__version__ = "1.0.0"
