"""Small JSON helpers with atomic writes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Union


PathLike = Union[str, os.PathLike]


def atomic_write_json(path: PathLike, payload: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    )
    fd, temp_name = tempfile.mkstemp(
        prefix=output_path.name + ".", suffix=".tmp", dir=str(output_path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, str(output_path))
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise

