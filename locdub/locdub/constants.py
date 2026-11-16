"""Project-wide constants and vocabulary.

All status strings are defined once here so that reports produced for
different languages always use the exact same vocabulary.
"""

from __future__ import annotations

# --- process exit codes (non-zero integers on failure) ---------------------
EXIT_OK = 0
EXIT_PARTIAL = 2          # one or more languages finished with skipped segments
EXIT_LANG_FAILED = 3      # one or more languages failed completely
EXIT_FATAL = 4            # unrecoverable error (bad JSON, missing video, ...)

# --- language-level overall status (shared vocabulary) ---------------------
STATUS_SUCCESS = "success"
STATUS_PARTIAL = "partial_success"
STATUS_FAILED = "failed"

# --- per-segment processing status -----------------------------------------
SEGMENT_OK = "ok"
SEGMENT_SKIPPED = "skipped"

# --- log / report keys -----------------------------------------------------
KIND_ERROR = "error"
KIND_WARNING = "warning"

# --- audio defaults --------------------------------------------------------
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_BLOCK_SECONDS = 60.0

# WSOLA analysis window length / synthesis hop in milliseconds
WSOLA_FRAME_MS = 40.0
WSOLA_HOP_MS = 20.0
# Maximum +/- shift (ms) searched when aligning the next analysis frame
WSOLA_MAX_SHIFT_MS = 15.0

# Default allowed time-stretch range when a segment omits an explicit one.
DEFAULT_MIN_RATIO = 0.5
DEFAULT_MAX_RATIO = 2.0
