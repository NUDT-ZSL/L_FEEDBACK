"""Deterministic offline dialogue/ambience/music shot mixer."""

from .config import ProjectConfig, ShotSpec, TrackSpec, load_project_config
from .engine import mix_files
from .errors import ErrorCode, MixerError
from .samples import generate_sample

__all__ = [
    "ErrorCode",
    "MixerError",
    "ProjectConfig",
    "ShotSpec",
    "TrackSpec",
    "generate_sample",
    "load_project_config",
    "mix_files",
]

__version__ = "1.0.0"
