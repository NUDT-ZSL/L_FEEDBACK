"""Offline multilingual dialogue mixer.

Pipeline: timeline validation -> resample -> granular time-scale alignment ->
ITU-R BS.1770 loudness normalization -> side-chain music ducking -> sample
accurate mix -> soft peak limiting -> WAV + JSON report.
"""

__version__ = "1.0.0"

TARGET_SR = 48000
MIN_RATIO = 0.8
MAX_RATIO = 1.25
PEAK_CEILING_DBFS = -1.0
LUFS_TOLERANCE = 0.5

# Error codes emitted in JSON diagnostics (non-zero process exit).
ERR_FILE_NOT_FOUND = "E1001"
ERR_FILE_UNREADABLE = "E1002"
ERR_BAD_TIMING = "E2001"
ERR_DURATION_EXCEEDED = "E2002"
ERR_BAD_SCHEMA = "E2003"
ERR_ALIGNMENT = "E3001"
ERR_INTERNAL = "E9000"
