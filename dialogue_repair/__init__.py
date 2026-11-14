"""Deterministic, offline dialogue repair."""

from .errors import RepairFailure
from .processor import ProcessResult, process_files

__all__ = ["ProcessResult", "RepairFailure", "process_files"]

