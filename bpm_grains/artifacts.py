"""Atomic artifact and WAV writes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Callable


def _json_bytes(payload: object) -> bytes:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def write_artifacts_atomically(
    output_path: Path,
    *,
    json_artifacts: dict[str, object],
    wav_writer: Callable[[Path], None],
) -> None:
    """Stage every artifact, then atomically replace final paths.

    The WAV is replaced last, so a failure cannot expose a partially written
    output WAV. Temporary files are always removed.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    try:
        for filename, payload in json_artifacts.items():
            final_path = output_path.parent / filename
            descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{filename}.", suffix=".tmp", dir=output_path.parent
            )
            temp_path = Path(temp_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(_json_bytes(payload))
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                temp_path.unlink(missing_ok=True)
                raise
            staged.append((temp_path, final_path))

        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
        )
        os.close(descriptor)
        wav_temp = Path(temp_name)
        try:
            wav_writer(wav_temp)
            staged.append((wav_temp, output_path))
        except BaseException:
            wav_temp.unlink(missing_ok=True)
            raise

        for temp_path, final_path in staged:
            os.replace(temp_path, final_path)
        staged.clear()
    finally:
        for temp_path, _ in staged:
            temp_path.unlink(missing_ok=True)

