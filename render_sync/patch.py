"""Deterministic field-level delta patches and strict replay."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

from .errors import ErrorCode, make_error
from .model import StateSnapshot


PathPart = Union[str, int]
Path = Tuple[PathPart, ...]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def state_digest(state: StateSnapshot) -> str:
    return hashlib.sha256(canonical_json(state.to_dict()).encode("utf-8")).hexdigest()


def digest_to_dict(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PatchOperation:
    op: str
    path: Path
    value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"op": self.op, "path": list(self.path)}
        if self.op == "set":
            data["value"] = copy.deepcopy(self.value)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PatchOperation":
        op = str(data["op"])
        if op not in ("set", "remove"):
            raise ValueError(f"unsupported patch operation: {op}")
        raw_path = data["path"]
        if not isinstance(raw_path, list) or not raw_path:
            raise ValueError("patch operation path must be a non-empty list")
        path: List[PathPart] = []
        for part in raw_path:
            if isinstance(part, bool) or not isinstance(part, (str, int)):
                raise ValueError("patch path parts must be strings or integers")
            path.append(part)
        return cls(op=op, path=tuple(path), value=copy.deepcopy(data.get("value")))


@dataclass(frozen=True)
class DeltaPatch:
    base_version: int
    target_version: int
    base_digest: str
    target_digest: str
    operations: Tuple[PatchOperation, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "base_version": self.base_version,
            "target_version": self.target_version,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "operations": [operation.to_dict() for operation in self.operations],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeltaPatch":
        if data.get("schema_version") != 1:
            raise ValueError("unsupported patch schema_version")
        operations = data["operations"]
        if not isinstance(operations, list):
            raise ValueError("patch operations must be a list")
        return cls(
            base_version=int(data["base_version"]),
            target_version=int(data["target_version"]),
            base_digest=str(data["base_digest"]),
            target_digest=str(data["target_digest"]),
            operations=tuple(PatchOperation.from_dict(item) for item in operations),
        )


def _path_related(path: Path, dirty: Sequence[Path]) -> bool:
    for prefix in dirty:
        shorter = min(len(path), len(prefix))
        if path[:shorter] == prefix[:shorter]:
            return True
    return False


def _diff_value(
    old: Any,
    new: Any,
    path: Path,
    dirty: Sequence[Path],
    operations: List[PatchOperation],
) -> None:
    if old == new or not _path_related(path, dirty):
        return

    if isinstance(old, Mapping) and isinstance(new, Mapping):
        old_keys = set(old)
        new_keys = set(new)
        for key in sorted(old_keys - new_keys):
            operations.append(PatchOperation("remove", path + (str(key),)))
        for key in sorted(new_keys - old_keys):
            operations.append(
                PatchOperation("set", path + (str(key),), copy.deepcopy(new[key]))
            )
        for key in sorted(old_keys & new_keys):
            _diff_value(old[key], new[key], path + (str(key),), dirty, operations)
        return

    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            operations.append(PatchOperation("set", path, copy.deepcopy(new)))
            return
        for index, (old_item, new_item) in enumerate(zip(old, new)):
            _diff_value(old_item, new_item, path + (index,), dirty, operations)
        return

    operations.append(PatchOperation("set", path, copy.deepcopy(new)))


def build_delta_patch(
    base: StateSnapshot,
    current: StateSnapshot,
    dirty: Iterable[Path],
) -> DeltaPatch:
    """Build a patch containing only changed fields under dirty prefixes."""

    dirty_paths = tuple(tuple(prefix) for prefix in dirty)
    if current.state_version < base.state_version:
        raise make_error(
            ErrorCode.INTERNAL,
            "target state version is older than patch base",
            details={
                "base_version": base.state_version,
                "target_version": current.state_version,
            },
            state_version=current.state_version,
            last_valid_version=current.state_version,
        )

    operations: List[PatchOperation] = []
    if base != current:
        _diff_value(base.to_dict(), current.to_dict(), (), dirty_paths, operations)

    patch = DeltaPatch(
        base_version=base.state_version,
        target_version=current.state_version,
        base_digest=state_digest(base),
        target_digest=state_digest(current),
        operations=tuple(operations),
    )
    if base != current and not operations:
        raise make_error(
            ErrorCode.INTERNAL,
            "state changed without any patch operation",
            details={
                "base_version": base.state_version,
                "target_version": current.state_version,
            },
            state_version=current.state_version,
            last_valid_version=current.state_version,
        )
    return patch


def _set_at_path(root: Any, path: Path, value: Any) -> None:
    current = root
    for index, part in enumerate(path[:-1]):
        try:
            if isinstance(current, Mapping):
                current = current[part]
            elif isinstance(current, list) and isinstance(part, int):
                current = current[part]
            else:
                raise TypeError("path traverses a scalar")
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"patch set path missing at segment {index}: {path}"
            ) from exc

    final = path[-1]
    if isinstance(current, dict) and isinstance(final, str):
        current[final] = copy.deepcopy(value)
        return
    if isinstance(current, list) and isinstance(final, int):
        if not 0 <= final < len(current):
            raise ValueError(f"patch list index out of range: {path}")
        current[final] = copy.deepcopy(value)
        return
    raise ValueError(f"patch set target is not mutable: {path}")


def _remove_at_path(root: Any, path: Path) -> None:
    current = root
    for index, part in enumerate(path[:-1]):
        try:
            if isinstance(current, Mapping):
                current = current[part]
            elif isinstance(current, list) and isinstance(part, int):
                current = current[part]
            else:
                raise TypeError("path traverses a scalar")
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"patch remove path missing at segment {index}: {path}"
            ) from exc

    final = path[-1]
    if isinstance(current, dict) and isinstance(final, str):
        if final not in current:
            raise ValueError(f"patch remove key not found: {path}")
        del current[final]
        return
    if isinstance(current, list) and isinstance(final, int):
        if not 0 <= final < len(current):
            raise ValueError(f"patch list index out of range: {path}")
        del current[final]
        return
    raise ValueError(f"patch remove target is not mutable: {path}")


def replay_delta_patch(base: StateSnapshot, patch: DeltaPatch) -> StateSnapshot:
    """Replay a patch against a snapshot and verify all hashes strictly."""

    actual_base_digest = state_digest(base)
    if base.state_version != patch.base_version:
        raise make_error(
            ErrorCode.PATCH_BASE_MISMATCH,
            "patch base version does not match snapshot",
            details={
                "expected": patch.base_version,
                "actual": base.state_version,
            },
            state_version=base.state_version,
            last_valid_version=base.state_version,
        )
    if actual_base_digest != patch.base_digest:
        raise make_error(
            ErrorCode.PATCH_BASE_MISMATCH,
            "patch base digest does not match snapshot",
            details={
                "expected": patch.base_digest,
                "actual": actual_base_digest,
            },
            state_version=base.state_version,
            last_valid_version=base.state_version,
        )

    data = copy.deepcopy(base.to_dict())
    try:
        for operation in patch.operations:
            if operation.op == "set":
                _set_at_path(data, operation.path, operation.value)
            else:
                _remove_at_path(data, operation.path)
        replayed = StateSnapshot.from_dict(data)
    except Exception as exc:
        raise make_error(
            ErrorCode.PATCH_REPLAY_FAILED,
            "patch operation replay failed",
            details={
                "cause": f"{type(exc).__name__}: {exc}",
                "base_version": patch.base_version,
                "target_version": patch.target_version,
            },
            recoverable=True,
            state_version=base.state_version,
            last_valid_version=base.state_version,
        ) from exc

    if replayed.state_version != patch.target_version:
        raise make_error(
            ErrorCode.PATCH_REPLAY_FAILED,
            "replayed state version does not match patch target",
            details={
                "expected": patch.target_version,
                "actual": replayed.state_version,
            },
            recoverable=True,
            state_version=replayed.state_version,
            last_valid_version=base.state_version,
        )
    actual_target_digest = state_digest(replayed)
    if actual_target_digest != patch.target_digest:
        raise make_error(
            ErrorCode.PATCH_REPLAY_FAILED,
            "replayed state digest does not match patch target",
            details={
                "expected": patch.target_digest,
                "actual": actual_target_digest,
            },
            recoverable=True,
            state_version=replayed.state_version,
            last_valid_version=base.state_version,
        )
    return replayed
