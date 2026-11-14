"""Stable business errors and process exit codes."""

from __future__ import annotations

from dataclasses import dataclass


EXIT_INTERNAL = 1
EXIT_USAGE = 2
EXIT_INPUT = 3
EXIT_VALIDATION = 4
EXIT_CAPACITY = 5
EXIT_TOLERANCE = 6
EXIT_OUTPUT = 7
EXIT_GL = 8


@dataclass
class BusinessError(Exception):
    code: str
    message: str
    exit_code: int = EXIT_VALIDATION

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


def scene_io(message: str) -> BusinessError:
    return BusinessError("E_SCENE_IO", message, EXIT_INPUT)


def scene_json(message: str) -> BusinessError:
    return BusinessError("E_SCENE_JSON", message, EXIT_INPUT)


def scene_hash(message: str) -> BusinessError:
    return BusinessError("E_SCENE_HASH", message, EXIT_INPUT)


def schema(message: str) -> BusinessError:
    return BusinessError("E_SCHEMA", message, EXIT_VALIDATION)


def unsupported_version(message: str) -> BusinessError:
    return BusinessError("E_VERSION", message, EXIT_VALIDATION)


def numeric(message: str) -> BusinessError:
    return BusinessError("E_NUMERIC", message, EXIT_VALIDATION)


def timeline(message: str) -> BusinessError:
    return BusinessError("E_TIMELINE", message, EXIT_VALIDATION)


def atlas_capacity(message: str) -> BusinessError:
    return BusinessError("E_ATLAS_CAPACITY", message, EXIT_CAPACITY)


def tolerance(message: str) -> BusinessError:
    return BusinessError("E_TOLERANCE_EXCEEDED", message, EXIT_TOLERANCE)


def output_io(message: str) -> BusinessError:
    return BusinessError("E_OUTPUT_IO", message, EXIT_OUTPUT)


def texture_corrupt(message: str) -> BusinessError:
    return BusinessError("E_TEXTURE_CORRUPT", message, EXIT_INPUT)


def gl_unavailable(message: str) -> BusinessError:
    return BusinessError("E_GL_UNAVAILABLE", message, EXIT_GL)


def gl_runtime(message: str) -> BusinessError:
    return BusinessError("E_GL_RUNTIME", message, EXIT_GL)

