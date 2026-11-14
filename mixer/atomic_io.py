"""Crash-safe file output that is safe for concurrent invocations."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def _temporary_path(target: Path, suffix: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=suffix,
        dir=str(target.parent),
    )
    os.close(descriptor)
    return Path(name)


def write_json_atomic(path: str | Path, payload: Any) -> None:
    target = Path(path).resolve()
    temporary = _temporary_path(target, ".json.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def write_wav_atomic(
    path: str | Path,
    audio: np.ndarray,
    *,
    sample_rate: int,
    subtype: str,
) -> None:
    target = Path(path).resolve()
    temporary = _temporary_path(target, ".wav.tmp")
    try:
        signal = np.asarray(audio, dtype=np.float64)
        if signal.ndim == 1:
            output = signal[:, np.newaxis]
        elif signal.ndim == 2 and signal.shape[1] > 0:
            output = signal
        else:
            raise ValueError("WAV writer expects a non-empty 1-D or 2-D signal")
        if not np.all(np.isfinite(output)):
            raise ValueError("WAV writer received non-finite samples")
        sf.write(
            str(temporary),
            output,
            int(sample_rate),
            format="WAV",
            subtype=subtype,
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
