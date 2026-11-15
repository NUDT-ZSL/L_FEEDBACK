"""Deterministic offline audio-event processing for game audio."""

from .dsp import (
    LoopCleanupResult,
    apply_loop_cleanup,
    crossfade_sample_count,
    pan_mono,
    panning_gains,
    resample_windowed_sinc,
)
from .errors import BusinessError, ErrorCode
from .events import EventSpec, LoopSpec, load_events
from .processor import process_events
from .wavio import (
    AtomicPcmWavWriter,
    WavData,
    WavInfo,
    WavReader,
    inspect_wav,
    read_wav,
    write_pcm_wav,
    write_wav_float32,
)

__all__ = [
    "BusinessError",
    "ErrorCode",
    "EventSpec",
    "LoopSpec",
    "LoopCleanupResult",
    "AtomicPcmWavWriter",
    "WavReader",
    "WavData",
    "WavInfo",
    "apply_loop_cleanup",
    "crossfade_sample_count",
    "inspect_wav",
    "load_events",
    "pan_mono",
    "panning_gains",
    "process_events",
    "read_wav",
    "resample_windowed_sinc",
    "write_pcm_wav",
    "write_wav_float32",
]

__version__ = "1.0.0"
