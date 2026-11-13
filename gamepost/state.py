from __future__ import annotations

import json
import os
from pathlib import Path

from .fingerprint import write_json_atomic


STATE_SCHEMA_VERSION = "gamepost-state-v1"


def state_path_for(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    return output_path.with_name(output_path.name + ".state.json")


def partial_path_for(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    return output_path.with_name(output_path.name + ".part.wav")


def load_state(path: str | Path) -> dict | None:
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def write_state(path: str | Path, state: dict) -> None:
    write_json_atomic(path, state)


def remove_if_exists(path: str | Path) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def fsync_parent(path: str | Path) -> None:
    directory = Path(path).parent
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
