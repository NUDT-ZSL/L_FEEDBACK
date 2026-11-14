"""Cross-platform path handling used by the CLI."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from .errors import BusinessError, invalid_output_dir, invalid_scene


PathLike = Union[str, os.PathLike]

_WINDOWS_ABSOLUTE = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def normalize_path(value: PathLike) -> Path:
    """Normalize POSIX and Windows-style user paths.

    On Windows, pathlib performs the native normalization. On POSIX, backslashes
    are treated as separators and a Windows drive path is mapped to WSL's
    /mnt/<drive> mount when that mount exists. Other drive paths remain explicit
    paths rather than being silently changed into something unrelated.
    """

    try:
        text = os.fspath(value)
    except TypeError as exc:
        raise invalid_scene("path is not a string", value=repr(value)) from exc
    if not isinstance(text, str) or not text.strip():
        raise invalid_scene("path must be a non-empty string", value=repr(value))

    text = os.path.expandvars(os.path.expanduser(text.strip()))
    if os.name == "nt":
        return Path(text)

    match = _WINDOWS_ABSOLUTE.match(text)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2).replace("\\", "/")
        wsl_mount = Path("/mnt") / drive
        if wsl_mount.exists():
            return wsl_mount / rest
        return Path(text)

    if "\\" in text:
        text = text.replace("\\", "/")
    return Path(text)


def _probe_directory(directory: Path) -> None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise invalid_output_dir(
            "output directory cannot be created",
            path=str(directory),
            reason=exc.strerror or str(exc),
        ) from exc

    if not directory.is_dir():
        raise invalid_output_dir("output path is not a directory", path=str(directory))

    probe_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".write-probe-", dir=str(directory), delete=False
        ) as probe:
            probe_path = Path(probe.name)
            probe.write(b"probe")
        probe_path.unlink()
    except OSError as exc:
        if probe_path is not None:
            try:
                probe_path.unlink()
            except OSError:
                pass
        raise invalid_output_dir(
            "output directory is not writable",
            path=str(directory),
            reason=exc.strerror or str(exc),
        ) from exc


def ensure_output_dir(path: PathLike,
                      files_to_overwrite: Optional[List[str]] = None) -> Path:
    """Create and validate an output directory without relying on POSIX only APIs."""

    try:
        directory = normalize_path(path)
    except BusinessError as exc:
        raise invalid_output_dir(
            "output directory path is invalid",
            path=repr(path),
            reason=exc.message,
        ) from exc
    _probe_directory(directory)

    for filename in files_to_overwrite or []:
        target = directory / filename
        if target.exists() and not os.access(str(target), os.W_OK):
            raise invalid_output_dir(
                "existing output file is not writable",
                path=str(target),
            )
    return directory


def ensure_snapshot_dir(path: PathLike) -> Path:
    try:
        directory = normalize_path(path)
    except BusinessError as exc:
        raise invalid_output_dir(
            "snapshot directory path is invalid",
            path=repr(path),
            reason=exc.message,
        ) from exc
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise invalid_output_dir(
            "snapshot directory cannot be created",
            path=str(directory),
            reason=exc.strerror or str(exc),
        ) from exc
    if not directory.is_dir():
        raise invalid_output_dir("snapshot path is not a directory", path=str(directory))
    _probe_directory(directory)
    return directory

