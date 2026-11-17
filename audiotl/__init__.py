"""audiotl - deterministic audio timeline builder.

Builds a mixed WAV from PCM source segments and a JSON timeline manifest,
with content-hash based incremental rendering and a four-stage pipeline
(analyze -> resample -> mix -> writeout).
"""

__version__ = "1.0.0"
