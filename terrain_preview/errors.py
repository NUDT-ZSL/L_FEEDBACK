"""Stable error categories used by the command line entry point."""

from __future__ import annotations


class PreviewError(Exception):
    """An expected failure with a stable exit code and user-facing message."""

    def __init__(self, code: str, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code


def config_error(message: str) -> PreviewError:
    return PreviewError("E_CONFIG", message, 3)


def config_parse_error(message: str) -> PreviewError:
    return PreviewError("E_CONFIG_PARSE", message, 2)


def terrain_error(message: str) -> PreviewError:
    return PreviewError("E_TERRAIN", message, 4)


def render_error(message: str) -> PreviewError:
    return PreviewError("E_RENDER", message, 5)


def cache_error(message: str) -> PreviewError:
    return PreviewError("E_CACHE", message, 6)


def resource_error(message: str) -> PreviewError:
    return PreviewError("E_RESOURCE", message, 7)


def output_error(message: str) -> PreviewError:
    return PreviewError("E_OUTPUT", message, 8)


EXIT_CODES = {
    "success": 0,
    "unexpected": 1,
    "config_parse": 2,
    "config_validation": 3,
    "terrain": 4,
    "render": 5,
    "cache": 6,
    "resource": 7,
    "output": 8,
}

