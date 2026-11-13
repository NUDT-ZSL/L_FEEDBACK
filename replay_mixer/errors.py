from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_MISSING_INPUT = 2
EXIT_INVALID_CONFIG = 3
EXIT_UNSUPPORTED_SAMPLE_RATE = 4
EXIT_CHANNEL_COUNT = 5
EXIT_CHECKPOINT_INVALID = 6
EXIT_AUDIO_IO = 7
EXIT_UNSUPPORTED_BIT_DEPTH = 8
EXIT_OUTPUT = 9
EXIT_INTERRUPTED = 75


@dataclass
class AppError(Exception):
    code: str
    exit_code: int
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code,
            "exit_code": self.exit_code,
            "message": self.message,
            "context": self.context,
        }

    def to_json_line(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, separators=(",", ":"))


def invalid_config(message: str, **context: Any) -> AppError:
    return AppError("INVALID_CONFIG", EXIT_INVALID_CONFIG, message, context)


def missing_input(path: str, role: str) -> AppError:
    return AppError(
        "MISSING_INPUT",
        EXIT_MISSING_INPUT,
        "required input file does not exist",
        {"path": path, "role": role},
    )


def unsupported_sample_rate(message: str, **context: Any) -> AppError:
    return AppError("UNSUPPORTED_SAMPLE_RATE", EXIT_UNSUPPORTED_SAMPLE_RATE, message, context)


def channel_count_error(message: str, **context: Any) -> AppError:
    return AppError("CHANNEL_COUNT_OUT_OF_RANGE", EXIT_CHANNEL_COUNT, message, context)


def checkpoint_invalid(message: str, **context: Any) -> AppError:
    return AppError("CHECKPOINT_INVALID", EXIT_CHECKPOINT_INVALID, message, context)


def audio_io_error(message: str, **context: Any) -> AppError:
    return AppError("AUDIO_IO_ERROR", EXIT_AUDIO_IO, message, context)


def unsupported_bit_depth(message: str, **context: Any) -> AppError:
    return AppError("UNSUPPORTED_BIT_DEPTH", EXIT_UNSUPPORTED_BIT_DEPTH, message, context)


def output_error(message: str, **context: Any) -> AppError:
    return AppError("OUTPUT_ERROR", EXIT_OUTPUT, message, context)


def interrupted_export(message: str, **context: Any) -> AppError:
    return AppError("EXPORT_INTERRUPTED", EXIT_INTERRUPTED, message, context)


class CheckpointValidationError(Exception):
    """Internal checkpoint validation failure that can trigger a clean restart."""

    def __init__(self, reason: str, detail: str, **context: Any) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        self.context = context
