"""rhythmforge - deterministic offline audio track builder for rhythm games.

The package turns a chart JSON plus a set of local WAV layer sources into:

* a normalized master mix,
* one aligned stem per declared layer,
* sample-accurate synchronization markers at section / BPM boundaries,
* a build manifest supporting bit-stable incremental rebuilds.

Everything is deterministic and offline: no network access is required at
build time and all sample material can be produced by :mod:`rhythmforge.generator`.
"""

from .errors import BuildError, ExitCode

__all__ = ["BuildError", "ExitCode"]
__version__ = "1.0.0"
