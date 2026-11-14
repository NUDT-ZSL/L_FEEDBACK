"""Cinemix offline multitrack audio processor."""

from .errors import CinemixError, Diagnostics, Issue
from .processor import ProcessingResult, process_project

__all__ = [
    "CinemixError",
    "Diagnostics",
    "Issue",
    "ProcessingResult",
    "process_project",
]

__version__ = "1.0.0"

