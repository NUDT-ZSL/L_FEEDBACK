"""Checkpoint state and deterministic array streams."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .errors import MaterialError
from .util import atomic_write_bytes, atomic_write_json, sha256_bytes

CHECKPOINT_SCHEMA = "material-bake-checkpoints/1"


class CheckpointManager:
    """Persist completed row chunks and resume only matching runs."""

    def __init__(
        self,
        out_dir: Path,
        fingerprint: str,
        width: int,
        height: int,
        chunk_rows: int,
        *,
        force: bool = False,
    ) -> None:
        self.out_dir = out_dir
        self.work_dir = out_dir / ".work"
        self.chunks_dir = self.work_dir / "chunks"
        self.state_path = self.work_dir / "state.json"
        self.fingerprint = fingerprint
        self.width = int(width)
        self.height = int(height)
        self.chunk_rows = int(chunk_rows)
        self.total_chunks = (self.height + self.chunk_rows - 1) // self.chunk_rows
        self.completed: dict[int, dict[str, Any]] = {}
        self.resumed_chunks = 0

        if force and self.work_dir.exists():
            shutil.rmtree(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self._clean_stale_temporaries()
        self._load_or_create()

    def _clean_stale_temporaries(self) -> None:
        for path in list(self.work_dir.glob(".*.tmp-*")):
            if path.is_file():
                path.unlink()
        for path in list(self.chunks_dir.glob(".*.tmp-*")):
            if path.is_file():
                path.unlink()

    def _chunk_rows_span(self, index: int) -> tuple[int, int]:
        start = index * self.chunk_rows
        end = min(start + self.chunk_rows, self.height)
        return start, end

    def _chunk_name(self, index: int) -> str:
        return f"chunks/bake_{index:05d}.npy"

    def _chunk_path(self, index: int) -> Path:
        return self.work_dir / self._chunk_name(index)

    def _load_or_create(self) -> None:
        if not self.state_path.exists():
            self._write_state()
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MaterialError(
                "E010",
                f"checkpoint manifest is unreadable; rerun with --force: {exc}",
            ) from exc
        if state.get("schema") != CHECKPOINT_SCHEMA:
            raise MaterialError(
                "E010", "checkpoint schema mismatch; rerun with --force"
            )
        if state.get("fingerprint") != self.fingerprint:
            raise MaterialError(
                "E010",
                "output directory belongs to a different input/config; rerun with --force",
            )
        if (
            state.get("chunk_rows") != self.chunk_rows
            or state.get("total_chunks") != self.total_chunks
            or state.get("width") != self.width
            or state.get("height") != self.height
        ):
            raise MaterialError(
                "E010",
                "checkpoint geometry does not match the current run; rerun with --force",
            )
        for entry in state.get("completed", []):
            try:
                index = int(entry["index"])
                rows = [int(entry["rows"][0]), int(entry["rows"][1])]
                expected = list(self._chunk_rows_span(index))
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            if rows != expected or not 0 <= index < self.total_chunks:
                continue
            path = self._chunk_path(index)
            if not path.exists():
                continue
            try:
                digest = sha256_bytes(path.read_bytes())
            except OSError:
                continue
            if digest != entry.get("sha256"):
                continue
            self.completed[index] = {
                "index": index,
                "rows": rows,
                "file": self._chunk_name(index),
                "sha256": digest,
            }
        self.resumed_chunks = len(self.completed)

    def _state_payload(self) -> dict[str, Any]:
        return {
            "schema": CHECKPOINT_SCHEMA,
            "fingerprint": self.fingerprint,
            "width": self.width,
            "height": self.height,
            "chunk_rows": self.chunk_rows,
            "total_chunks": self.total_chunks,
            "completed": [self.completed[key] for key in sorted(self.completed)],
        }

    def _write_state(self) -> None:
        atomic_write_json(self.state_path, self._state_payload())

    def is_complete(self, index: int) -> bool:
        return index in self.completed

    def rows_for(self, index: int) -> tuple[int, int]:
        return self._chunk_rows_span(index)

    def save(self, index: int, data: np.ndarray) -> None:
        start, end = self._chunk_rows_span(index)
        if data.ndim != 3 or data.shape[0] != end - start:
            raise MaterialError(
                "E009",
                f"checkpoint chunk {index} has invalid shape {data.shape}",
            )
        buffer = io.BytesIO()
        np.save(buffer, np.ascontiguousarray(data), allow_pickle=False)
        payload = buffer.getvalue()
        atomic_write_bytes(self._chunk_path(index), payload)
        self.completed[index] = {
            "index": index,
            "rows": [start, end],
            "file": self._chunk_name(index),
            "sha256": sha256_bytes(payload),
        }
        self._write_state()

    def load(self, index: int) -> np.ndarray:
        entry = self.completed.get(index)
        if entry is None:
            raise MaterialError("E010", f"checkpoint chunk {index} is not complete")
        path = self._chunk_path(index)
        try:
            data = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise MaterialError(
                "E010", f"cannot read checkpoint chunk {index}: {exc}"
            ) from exc
        start, end = self._chunk_rows_span(index)
        if data.shape[0] != end - start or data.shape[1] != self.width:
            raise MaterialError(
                "E010", f"checkpoint chunk {index} has an unexpected shape"
            )
        return data

    def iter_chunks(self) -> Iterator[tuple[int, tuple[int, int], np.ndarray]]:
        for index in range(self.total_chunks):
            yield index, self._chunk_rows_span(index), self.load(index)


class NpyStreamWriter:
    """Write a deterministic C-order float32 .npy header then raw rows."""

    def __init__(
        self,
        path: Path,
        shape: tuple[int, ...],
        dtype: str = "<f4",
    ) -> None:
        self.path = path
        self.shape = tuple(int(v) for v in shape)
        self.dtype = np.dtype(dtype)
        self._temporary = path.with_name(f".{path.name}.tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._temporary.open("wb")
        header = {
            "descr": self.dtype.str,
            "fortran_order": False,
            "shape": self.shape,
        }
        np.lib.format.write_array_header_1_0(self._handle, header)
        self._rows = 0

    def append(self, rows: np.ndarray) -> None:
        values = np.ascontiguousarray(rows, dtype=self.dtype)
        if values.ndim < 1 or values.shape[0] == 0:
            return
        if tuple(values.shape[1:]) != tuple(self.shape[1:]):
            raise MaterialError("E009", "npy stream row shape mismatch")
        self._handle.write(values.tobytes(order="C"))
        self._rows += int(values.shape[0])

    def finalize(self) -> str:
        if self._rows != self.shape[0]:
            self._handle.close()
            raise MaterialError(
                "E009",
                f"npy stream wrote {self._rows} of {self.shape[0]} rows",
            )
        self._handle.flush()
        self._handle.close()
        self._temporary.replace(self.path)
        from .util import sha256_file

        return sha256_file(self.path)
