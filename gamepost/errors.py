from __future__ import annotations


BUSINESS_CODES: dict[str, dict[str, object]] = {
    "OK": {
        "number": 0,
        "exit_code": 0,
        "description": "Export completed or validation passed.",
    },
    "EVENT_JSON_INVALID": {
        "number": 1001,
        "exit_code": 2,
        "description": "Event table is not valid JSON or has an unsupported top-level shape.",
    },
    "EVENT_ITEM_INVALID": {
        "number": 1002,
        "exit_code": 2,
        "description": "An event item is missing a required field or contains a malformed value.",
    },
    "EVENT_TIMESTAMP_INVALID": {
        "number": 1003,
        "exit_code": 2,
        "description": "An event timestamp is negative or cannot form a sample interval.",
    },
    "EVENT_ORDER": {
        "number": 1004,
        "exit_code": 2,
        "description": "Events are not sorted by ascending timestamp.",
    },
    "EVENT_OVERLAP": {
        "number": 1005,
        "exit_code": 2,
        "description": "Two event windows overlap in source sample time.",
    },
    "EVENT_OUT_OF_RANGE": {
        "number": 1006,
        "exit_code": 2,
        "description": "An event window extends beyond the narration track.",
    },
    "EVENT_UNKNOWN_CATEGORY": {
        "number": 1007,
        "exit_code": 2,
        "description": "An event category is not in the configured allow-list.",
    },
    "EVENT_RATE_INVALID": {
        "number": 1008,
        "exit_code": 2,
        "description": "An event playback rate is missing, non-positive, or outside the supported range.",
    },
    "EVENT_DURATION_INVALID": {
        "number": 1009,
        "exit_code": 2,
        "description": "An event duration is non-positive or too short to occupy one sample.",
    },
    "INPUT_WAV_INVALID": {
        "number": 2001,
        "exit_code": 3,
        "description": "An input WAV is missing, malformed, or is not 48 kHz/16-bit PCM mono/stereo.",
    },
    "INPUT_FINGERPRINT_MISMATCH": {
        "number": 2002,
        "exit_code": 3,
        "description": "A persistent export state belongs to different inputs or parameters.",
    },
    "STATE_INVALID": {
        "number": 2003,
        "exit_code": 3,
        "description": "Persistent state is unreadable or inconsistent with the partial output.",
    },
    "MEMORY_QUOTA_INVALID": {
        "number": 2004,
        "exit_code": 3,
        "description": "A job cannot be admitted with the configured memory accounting rules.",
    },
    "CANCELLED": {
        "number": 3001,
        "exit_code": 130,
        "description": "Cancellation was observed at a chunk boundary; completed state was preserved.",
    },
    "INTERNAL_ERROR": {
        "number": 9001,
        "exit_code": 1,
        "description": "An unexpected internal error occurred.",
    },
}


class PipelineError(Exception):
    code = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        if code is not None:
            if code not in BUSINESS_CODES:
                raise ValueError(f"unknown business code: {code}")
            self.code = code

    @property
    def exit_code(self) -> int:
        return int(BUSINESS_CODES[self.code]["exit_code"])

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "code_number": BUSINESS_CODES[self.code]["number"],
            "exit_code": self.exit_code,
            "message": self.message,
            "details": self.details,
        }


class EventValidationError(PipelineError):
    code = "EVENT_ITEM_INVALID"

    def __init__(self, errors: list[dict]):
        self.errors = errors
        primary = errors[0] if errors else {
            "code": "EVENT_ITEM_INVALID",
            "message": "Event table validation failed.",
            "details": {},
        }
        super().__init__(
            str(primary["message"]),
            details={"primary_code": primary["code"], "errors": errors},
        )
        self.code = str(primary["code"])


class CancelledError(PipelineError):
    code = "CANCELLED"

    def __init__(self, message: str = "Cancelled at a safe chunk boundary."):
        super().__init__(message)

