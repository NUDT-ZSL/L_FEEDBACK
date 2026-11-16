"""Alignment report: versioned snapshot with atomic whole-file replacement."""

from __future__ import annotations

import hashlib
import json
import os

from . import REPORT_FORMAT_VERSION


def file_digest(path: str) -> dict:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {
        "path": os.path.abspath(path),
        "sha256": h.hexdigest(),
        "size": os.path.getsize(path),
    }


def read_existing_version(path: str) -> int | None:
    """Return the format version of an existing report, or None."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data, _ = json.JSONDecoder().raw_decode(fh.read().lstrip())
        version = data.get("format_version")
        return int(version) if version is not None else None
    except (OSError, json.JSONDecodeError, ValueError, AttributeError):
        return None


def write_report(path: str, report: dict) -> None:
    """Atomically replace the report with a complete new snapshot.

    Never appends or reuses offsets from a previous file: the payload is
    serialized in full, written to a temp file in the same directory, and
    moved over the target with os.replace. A version change therefore
    always yields a complete rewrite.
    """
    report = dict(report)
    report["format_version"] = REPORT_FORMAT_VERSION
    previous = read_existing_version(path)
    report["previous_format_version"] = previous
    payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
