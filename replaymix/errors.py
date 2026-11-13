"""Business error codes used by the replay mixer."""

from __future__ import annotations

from dataclasses import dataclass


E_TRACK_TIMELINE_MISMATCH = "E_TRACK_TIMELINE_MISMATCH"
E_SAMPLE_RATE_UNSUPPORTED = "E_SAMPLE_RATE_UNSUPPORTED"
E_TRACK_EXCEEDS_SESSION = "E_TRACK_EXCEEDS_SESSION"
E_TRACK_FILE_MISSING = "E_TRACK_FILE_MISSING"
E_TRACK_WAV_INVALID = "E_TRACK_WAV_INVALID"
E_TRACK_PROCESSING_FAILED = "E_TRACK_PROCESSING_FAILED"
E_EVENT_TABLE_INVALID = "E_EVENT_TABLE_INVALID"
E_OUTPUT_WRITE_FAILED = "E_OUTPUT_WRITE_FAILED"


@dataclass(frozen=True)
class TrackError:
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


def error(code: str, message: str) -> TrackError:
    return TrackError(code=code, message=message)


class EventTableError(ValueError):
    """Raised when the event table cannot be interpreted at all."""

    code = E_EVENT_TABLE_INVALID


def add_error(errors: list[TrackError], code: str, message: str) -> None:
    item = TrackError(code=code, message=message)
    if item not in errors:
        errors.append(item)
