"""Central OpenGL import.

PyOpenGL chooses its function-loading backend at import time. Linux validations
run against Mesa's surfaceless EGL display, so select that backend before any
OpenGL import.
"""

from __future__ import annotations

import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from OpenGL import GL  # noqa: E402

