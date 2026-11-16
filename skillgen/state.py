from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


def _json_line(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


class StateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._records: dict[tuple[str, int], dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue
                    skill_id = record.get("skill_id")
                    variant_index = record.get("variant_index")
                    if (
                        isinstance(skill_id, str)
                        and isinstance(variant_index, int)
                        and not isinstance(variant_index, bool)
                    ):
                        self._records[(skill_id, variant_index)] = record
        except OSError:
            return

    def get(self, skill_id: str, variant_index: int) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get((skill_id, variant_index))
            return dict(record) if record is not None else None

    def append(self, record: dict[str, Any]) -> None:
        data = _json_line(record)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self.path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o644,
            )
            try:
                os.write(descriptor, data)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._records[
                (str(record["skill_id"]), int(record["variant_index"]))
            ] = dict(record)

    def clear(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + ".tmp")
            temporary.write_bytes(b"")
            os.replace(temporary, self.path)
            self._records.clear()

    def rewrite(self, records: list[dict[str, Any]]) -> None:
        temporary = self.path.with_name(self.path.name + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with temporary.open("wb") as handle:
                for record in records:
                    handle.write(_json_line(record))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            self._records = {
                (str(record["skill_id"]), int(record["variant_index"])): dict(record)
                for record in records
            }
