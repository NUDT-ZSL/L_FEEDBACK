"""Project-wide constants.

Changing synthesis, packing, or resampling behavior must bump
``BAKER_VERSION`` so incremental exports cannot reuse stale audio.
"""

BAKER_VERSION = "1.0.1"
INDEX_SCHEMA_VERSION = 1
PACK_FORMAT_VERSION = 1
PACK_MAGIC = b"DLGPACK1"
PACK_HEADER_SIZE = 64
PACK_HEADER_STRUCT = "<8sIIHHIII"
SAMPLE_FORMAT = "pcm_s16le"
SAMPLE_RATE = 48000
CHANNELS = 1
SECONDS_DECIMAL_PLACES = 6

