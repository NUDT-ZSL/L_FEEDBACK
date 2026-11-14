"""Stable business error codes and exit status mapping."""

from __future__ import annotations

from dataclasses import dataclass


ERROR_EXIT_CODES = {
    "E_USAGE": 2,
    "E_TELEMETRY_IO": 10,
    "E_TELEMETRY_JSON": 11,
    "E_TELEMETRY_SCHEMA": 12,
    "E_TELEMETRY_MISSING_FRAME": 13,
    "E_TELEMETRY_TIMESTAMP": 14,
    "E_TELEMETRY_FIELD": 15,
    "E_TELEMETRY_RANGE": 16,
    "E_PARAMS_IO": 20,
    "E_PARAMS_JSON": 21,
    "E_PARAMS_SCHEMA": 22,
    "E_PARAMS_RANGE": 23,
    "E_ALIGNMENT": 30,
    "E_OUTPUT_IO": 31,
    "E_WAV_IO": 40,
    "E_WAV_HEADER": 41,
    "E_REPORT_IO": 42,
    "E_SYNTH": 50,
    "E_INTERNAL": 99,
}


@dataclass
class SynthError(Exception):
    """An expected failure with a stable machine-readable code."""

    code: str
    message: str
    exit_code: int | None = None

    def __post_init__(self) -> None:
        if self.exit_code is None:
            self.exit_code = ERROR_EXIT_CODES.get(self.code, ERROR_EXIT_CODES["E_INTERNAL"])
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

