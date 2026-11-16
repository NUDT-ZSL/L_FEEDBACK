"""Small filesystem / numeric helpers."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    """Write *data* via a same-directory temp file then atomic replace."""
    target = Path(path)
    ensure_dir(target.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path: str | os.PathLike, payload) -> None:
    import json

    data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    atomic_write_bytes(path, data)


def ms_to_samples(ms: float, sample_rate: int) -> int:
    return int(round(ms * sample_rate / 1000.0))


def samples_to_ms(samples: int, sample_rate: int) -> int:
    return int(round(samples * 1000.0 / sample_rate))


SILENCE_DBFS = -120.0


def peak_dbfs(samples) -> float:
    """Peak level of an array (any shape), dBFS relative to full-scale 1.0.

    Digital silence is reported as the documented sentinel -120.00 dBFS.
    """
    import numpy as np

    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return SILENCE_DBFS
    peak = float(np.max(np.abs(arr)))
    if peak <= 0.0:
        return SILENCE_DBFS
    return round(20.0 * math.log10(peak), 2)
