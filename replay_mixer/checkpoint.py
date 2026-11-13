from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import CheckpointValidationError, invalid_config
from .util import atomic_write_bytes, canonical_json, is_int, is_number, sha256_bytes


CHECKPOINT_MAGIC = "REPLAY_MIX_CHECKPOINT"
CHECKPOINT_VERSION = 1
PHASE_MEASURE = "measure"
PHASE_FINAL = "final"


@dataclass(frozen=True)
class BlockLayout:
    total_frames: int
    block_frames: int
    channels: int
    frame_counts: tuple[int, ...]
    measure_offsets: tuple[int, ...]
    final_offsets: tuple[int, ...]

    @classmethod
    def build(cls, total_frames: int, block_frames: int, channels: int) -> "BlockLayout":
        if total_frames <= 0 or block_frames <= 0 or channels not in (1, 2):
            raise invalid_config("invalid block layout parameters")
        counts: list[int] = []
        remaining = total_frames
        while remaining > 0:
            counts.append(min(block_frames, remaining))
            remaining -= counts[-1]
        measure_offsets = [0]
        final_offsets = [0]
        for count in counts:
            measure_offsets.append(
                measure_offsets[-1] + count * channels * 4
            )
            final_offsets.append(
                final_offsets[-1] + count * channels * 2
            )
        return cls(
            total_frames=total_frames,
            block_frames=block_frames,
            channels=channels,
            frame_counts=tuple(counts),
            measure_offsets=tuple(measure_offsets),
            final_offsets=tuple(final_offsets),
        )

    @property
    def total_blocks(self) -> int:
        return len(self.frame_counts)

    @property
    def measure_bytes(self) -> int:
        return self.measure_offsets[-1]

    @property
    def final_bytes(self) -> int:
        return self.final_offsets[-1]


@dataclass(frozen=True)
class CheckpointState:
    phase: str
    completed_blocks: int
    data_bytes: int
    data_sha256: str
    config_digest: str
    block_frames: int
    total_blocks: int
    total_frames: int
    channels: int
    normalization: dict[str, Any] | None = None


def checkpoint_payload(state: CheckpointState) -> dict[str, Any]:
    return {
        "magic": CHECKPOINT_MAGIC,
        "version": CHECKPOINT_VERSION,
        "phase": state.phase,
        "completed_blocks": state.completed_blocks,
        "data_bytes": state.data_bytes,
        "data_sha256": state.data_sha256,
        "config_digest": state.config_digest,
        "block_frames": state.block_frames,
        "total_blocks": state.total_blocks,
        "total_frames": state.total_frames,
        "channels": state.channels,
        "normalization": state.normalization,
    }


def write_checkpoint(path: Path, state: CheckpointState) -> None:
    payload = checkpoint_payload(state)
    integrity = sha256_bytes(canonical_json(payload))
    payload["integrity_sha256"] = integrity
    data = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    atomic_write_bytes(path, data)


def _corrupt(detail: str, **context: Any) -> CheckpointValidationError:
    return CheckpointValidationError("checkpoint_corrupt", detail, **context)


def _validate_normalization(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _corrupt("checkpoint normalization must be an object")
    required = {
        "target_rms_dbfs",
        "measured_rms_dbfs",
        "measured_peak_dbfs",
        "applied_gain_db",
        "peak_limit_triggered",
        "soft_limit_triggered",
        "peak_limit_dbfs",
        "max_gain_triggered",
    }
    if set(value) != required:
        raise _corrupt(
            "checkpoint normalization fields are invalid",
            missing=sorted(required - set(value)),
            unknown=sorted(set(value) - required),
        )
    for key in (
        "target_rms_dbfs",
        "measured_rms_dbfs",
        "measured_peak_dbfs",
        "applied_gain_db",
        "peak_limit_dbfs",
    ):
        if not is_number(value[key]):
            raise _corrupt("checkpoint normalization number is invalid", field=key)
    for key in (
        "peak_limit_triggered",
        "soft_limit_triggered",
        "max_gain_triggered",
    ):
        if not isinstance(value[key], bool):
            raise _corrupt("checkpoint normalization flag is invalid", field=key)
    return dict(value)


def read_checkpoint(
    path: Path,
    config_digest: str,
    layout: BlockLayout,
    measure_raw: Path,
    final_raw: Path,
) -> CheckpointState | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _corrupt("checkpoint JSON is unreadable", error=str(exc)) from None
    if not isinstance(value, dict):
        raise _corrupt("checkpoint root must be an object")

    integrity = value.pop("integrity_sha256", None)
    if not isinstance(integrity, str):
        raise _corrupt("checkpoint integrity digest is missing")
    expected_integrity = sha256_bytes(canonical_json(value))
    if integrity != expected_integrity:
        raise _corrupt("checkpoint metadata integrity check failed")

    required = {
        "magic",
        "version",
        "phase",
        "completed_blocks",
        "data_bytes",
        "data_sha256",
        "config_digest",
        "block_frames",
        "total_blocks",
        "total_frames",
        "channels",
        "normalization",
    }
    if set(value) != required:
        raise _corrupt(
            "checkpoint fields are invalid",
            missing=sorted(required - set(value)),
            unknown=sorted(set(value) - required),
        )
    if value["magic"] != CHECKPOINT_MAGIC or value["version"] != CHECKPOINT_VERSION:
        raise _corrupt("checkpoint magic/version is unsupported")
    if value["config_digest"] != config_digest:
        raise CheckpointValidationError(
            "config_digest_mismatch",
            "checkpoint configuration digest does not match current inputs/configuration",
            checkpoint_digest=value["config_digest"],
            expected_digest=config_digest,
        )
    if value["block_frames"] != layout.block_frames:
        raise _corrupt("checkpoint block_frames does not match configuration")
    if value["total_blocks"] != layout.total_blocks:
        raise _corrupt("checkpoint total_blocks does not match configuration")
    if value["total_frames"] != layout.total_frames:
        raise _corrupt("checkpoint total_frames does not match configuration")
    if value["channels"] != layout.channels:
        raise _corrupt("checkpoint channels do not match configuration")
    if value["phase"] not in (PHASE_MEASURE, PHASE_FINAL):
        raise _corrupt("checkpoint phase is invalid")
    if not is_int(value["completed_blocks"]):
        raise _corrupt("checkpoint completed_blocks is not an integer")
    completed = value["completed_blocks"]
    if not (0 <= completed <= layout.total_blocks):
        raise _corrupt("checkpoint completed_blocks is out of range")
    if not isinstance(value["data_sha256"], str) or len(value["data_sha256"]) != 64:
        raise _corrupt("checkpoint data_sha256 is invalid")
    if not is_int(value["data_bytes"]):
        raise _corrupt("checkpoint data_bytes is not an integer")

    phase = value["phase"]
    expected_bytes = (
        layout.measure_offsets[completed]
        if phase == PHASE_MEASURE
        else layout.final_offsets[completed]
    )
    if value["data_bytes"] != expected_bytes:
        raise _corrupt(
            "checkpoint data_bytes does not match completed_blocks",
            data_bytes=value["data_bytes"],
            expected=expected_bytes,
        )
    normalization = value["normalization"]
    if phase == PHASE_FINAL:
        normalization = _validate_normalization(normalization)
    elif normalization is not None:
        raise _corrupt("measure checkpoint must not contain normalization data")

    raw_path = measure_raw if phase == PHASE_MEASURE else final_raw
    if expected_bytes > 0:
        if not raw_path.is_file():
            raise _corrupt("checkpoint data file is missing", path=str(raw_path))
        actual_size = raw_path.stat().st_size
        if actual_size < expected_bytes:
            raise _corrupt(
                "checkpoint data file is shorter than checkpoint byte count",
                expected_bytes=expected_bytes,
                actual_bytes=actual_size,
            )
        digest = hashlib.sha256()
        remaining = expected_bytes
        with raw_path.open("rb") as handle:
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise _corrupt("checkpoint data file ended during validation")
                digest.update(chunk)
                remaining -= len(chunk)
        if digest.hexdigest() != value["data_sha256"]:
            raise _corrupt("checkpoint data digest does not match checkpoint")
        if actual_size > expected_bytes:
            with raw_path.open("r+b") as handle:
                handle.truncate(expected_bytes)
                handle.flush()
                os.fsync(handle.fileno())
    elif raw_path.exists():
        with raw_path.open("r+b") as handle:
            handle.truncate(0)
            handle.flush()
            os.fsync(handle.fileno())

    return CheckpointState(
        phase=phase,
        completed_blocks=completed,
        data_bytes=value["data_bytes"],
        data_sha256=value["data_sha256"],
        config_digest=value["config_digest"],
        block_frames=value["block_frames"],
        total_blocks=value["total_blocks"],
        total_frames=value["total_frames"],
        channels=value["channels"],
        normalization=normalization,
    )


class RawPcmStore:
    def __init__(
        self,
        path: Path,
        layout: BlockLayout,
        phase: str,
        completed_blocks: int,
        data_sha256: str,
    ) -> None:
        self.path = path
        self.layout = layout
        self.phase = phase
        self.completed_blocks = completed_blocks
        self._digest = hashlib.sha256()
        path.parent.mkdir(parents=True, exist_ok=True)
        offsets = layout.measure_offsets if phase == PHASE_MEASURE else layout.final_offsets
        expected = offsets[completed_blocks]
        if completed_blocks == 0 and data_sha256 != sha256_bytes(b""):
            raise invalid_config("invalid initial checkpoint digest for empty raw stream")
        if path.exists():
            self._file = path.open("r+b")
            actual_size = path.stat().st_size
            if actual_size < expected:
                self._file.close()
                raise invalid_config(
                    "raw data stream is shorter than checkpoint byte count",
                    expected_bytes=expected,
                    actual_bytes=actual_size,
                )
            self._file.truncate(expected)
            self._file.seek(0)
            remaining = expected
            while remaining:
                chunk = self._file.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise invalid_config("raw data stream ended during resume")
                self._digest.update(chunk)
                remaining -= len(chunk)
            if self._digest.hexdigest() != data_sha256:
                self._file.close()
                raise invalid_config("raw data stream digest changed after validation")
            self._file.seek(expected)
            self._file.truncate(expected)
        else:
            if expected != 0:
                raise invalid_config("raw data stream is missing during resume")
            self._file = path.open("wb")
        self.data_bytes = expected

    @property
    def digest_hex(self) -> str:
        return self._digest.hexdigest()

    def append(self, data: bytes) -> None:
        self._file.write(data)
        self._file.flush()
        os.fsync(self._file.fileno())
        self._digest.update(data)
        self.data_bytes += len(data)
        self.completed_blocks += 1

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "RawPcmStore":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
