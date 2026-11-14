"""Deterministic audio frame synchronization, PLC and mixing service.

The package implements a local, single-threaded RTP-style audio pipeline:

* custom binary packet parsing with validation and business error codes,
* per-stream adaptive jitter buffers with underflow / overflow events,
* packet loss concealment (pitch periodic repetition / linear prediction),
* 2-4 stream mixing to a 48 kHz mono 16-bit PCM WAV,
* frame-aligned JSON Lines diagnostics,
* a deterministic network impairment simulator and a virtual-clock event loop.

Everything is generated locally; no network socket or audio device is used.
"""

from .constants import (
    SAMPLE_RATE,
    FRAME_SAMPLES,
    FRAME_DURATION_MS,
    FRAME_BYTES,
    HEADER_SIZE,
    SEQ_MOD,
    MAX_STREAMS,
    MIN_STREAMS,
    MAX_BUFFER_BYTES,
    MIN_BUFFER_DEPTH,
    MAX_BUFFER_DEPTH,
)

__all__ = [
    "SAMPLE_RATE",
    "FRAME_SAMPLES",
    "FRAME_DURATION_MS",
    "FRAME_BYTES",
    "HEADER_SIZE",
    "SEQ_MOD",
    "MAX_STREAMS",
    "MIN_STREAMS",
    "MAX_BUFFER_BYTES",
    "MIN_BUFFER_DEPTH",
    "MAX_BUFFER_DEPTH",
]

__version__ = "1.0.0"
