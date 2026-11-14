"""Adaptive game-audio offline mixing engine.

Pure-Python/NumPy implementation.  No external audio tools are required;
WAV I/O uses SoundFile for reading and a self-contained RIFF writer for
48 kHz / 24-bit PCM output (including the mandatory ``fact`` chunk).
"""

# Keep BLAS/FFT work single-threaded: the workload is many small FFTs where
# thread pool oversubscription dominates; concurrency is handled explicitly.
import os as _os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

from .errors import MixerError, ERROR_CODES
from .config import SceneConfig, load_config, validate_config
from .engine import MixEngine, MixResult
from .batch import process_batch

__all__ = [
    "MixerError",
    "ERROR_CODES",
    "SceneConfig",
    "load_config",
    "validate_config",
    "MixEngine",
    "MixResult",
    "process_batch",
]

__version__ = "1.0.0"
