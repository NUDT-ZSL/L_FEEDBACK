from __future__ import annotations


class PipelineError(Exception):
    """Base class for errors that can be reported by the pipeline."""


class ConfigurationError(PipelineError):
    """Raised when configuration or lens parameters are invalid."""


class InputDataError(PipelineError):
    """Raised when an input image or depth file is malformed."""


class FocusTrackError(PipelineError):
    """Metadata error tied to a concrete output frame."""

    def __init__(self, frame_index: int, reason: str):
        self.frame_index = int(frame_index)
        self.reason = str(reason)
        super().__init__(
            f"focus track invalid at frame {self.frame_index}: {self.reason}"
        )
