from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Mapping

from .errors import output_error


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def publish_directory(
    output_dir: Path,
    files: Mapping[str, bytes],
    *,
    replace: bool = True,
) -> None:
    """Materialize all files in a staging directory, then publish as one directory."""
    output_dir = output_dir.resolve()
    parent = output_dir.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise output_error(
            "E_OUTPUT_IO",
            f"cannot create output parent directory: {type(exc).__name__}",
        )

    if output_dir.exists() and not output_dir.is_dir():
        raise output_error("E_OUTPUT_EXISTS", "output path exists and is not a directory")
    if output_dir.exists() and not replace:
        raise output_error("E_OUTPUT_EXISTS", "output directory already exists")

    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=parent))
    backup = None
    try:
        for name in sorted(files):
            if Path(name).name != name:
                raise output_error("E_OUTPUT_NAME", "output file names must be flat")
            _write_bytes(stage / name, files[name])

        if output_dir.exists():
            backup = output_dir.with_name(
                f".{output_dir.name}.backup-{uuid.uuid4().hex}"
            )
            os.replace(output_dir, backup)
        try:
            os.replace(stage, output_dir)
        except Exception:
            if backup is not None and backup.exists() and not output_dir.exists():
                os.replace(backup, output_dir)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
            backup = None
    except Exception as exc:
        if getattr(exc, "code", None) is not None:
            raise
        raise output_error(
            "E_OUTPUT_IO",
            f"failed to publish output directory: {type(exc).__name__}",
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        if backup is not None and backup.exists():
            if not output_dir.exists():
                os.replace(backup, output_dir)
            else:
                shutil.rmtree(backup, ignore_errors=True)


def write_atomic_file(path: Path, payload: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
