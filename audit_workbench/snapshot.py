"""Atomic, versioned audit snapshots and hash-aware baseline differences."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import RULES_VERSION, __version__
from .errors import WorkbenchError
from .hashing import canonical_json, sha256_text


SNAPSHOT_FORMAT = "audit-snapshot-v1"
SNAPSHOT_ID_RE = __import__("re").compile(r"^[0-9a-f]{64}$")


def ensure_state_dir(state_dir: Path) -> None:
    state_dir = Path(state_dir)
    try:
        (state_dir / "snapshots").mkdir(parents=True, exist_ok=True)
        (state_dir / "tmp").mkdir(parents=True, exist_ok=True)
        probe = state_dir / "tmp" / f".write-probe-{os.getpid()}"
        with probe.open("w", encoding="utf-8") as handle:
            handle.write("ok\n")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        raise WorkbenchError(
            "E_IO",
            f"state directory is not writable: {exc}",
            {"file": str(state_dir), "line": 1},
        ) from exc


def snapshot_payload(
    *,
    roots: list[dict[str, Any]],
    configs: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    parse_failures: list[dict[str, Any]],
    combinations: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "audit-snapshot-payload-v1",
        "rules_version": RULES_VERSION,
        "tool_version": __version__,
        "roots": sorted(roots, key=lambda item: (item["name"], item["root_hash"])),
        "configs": sorted(configs, key=lambda item: (item["name"], item["config_hash"])),
        "calls": sorted(
            calls,
            key=lambda item: (
                item["root"],
                item["config"],
                item["location"]["file"],
                item["location"]["line"],
                item["location"]["column"],
                item["id"],
            ),
        ),
        "parse_failures": sorted(
            parse_failures,
            key=lambda item: (
                item["root"],
                item["config"],
                item["file"],
                item["line"],
                item["reason"],
            ),
        ),
        "combinations": sorted(
            combinations,
            key=lambda item: (item["root"], item["config"]),
        ),
        "errors": sorted(
            errors,
            key=lambda item: (
                item["code"],
                item["message"],
                canonical_json(item["context"]),
            ),
        ),
        "summary": summary,
    }


def write_snapshot(state_dir: Path, payload: dict[str, Any]) -> str:
    ensure_state_dir(state_dir)
    snapshot_id = sha256_text(canonical_json(payload))
    wrapper = {
        "format": SNAPSHOT_FORMAT,
        "snapshot_id": snapshot_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "payload_sha256": snapshot_id,
        "payload": payload,
    }
    snapshots_dir = Path(state_dir) / "snapshots"
    final_path = snapshots_dir / f"{snapshot_id}.json"
    data = json.dumps(
        wrapper,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=".tmp-",
        suffix=".json",
        dir=str(snapshots_dir),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, final_path)
        try:
            dir_fd = os.open(snapshots_dir, os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise WorkbenchError(
            "E_IO",
            f"cannot write snapshot: {exc}",
            {"file": str(final_path), "line": 1},
        ) from exc
    return snapshot_id


def _corrupt(snapshot: str, path: Path | None, reason: str) -> WorkbenchError:
    return WorkbenchError(
        "E_SNAPSHOT_CORRUPT",
        f"snapshot rejected: {reason}",
        {
            "snapshot": snapshot,
            "file": str(path) if path is not None else snapshot,
            "line": 1,
            "reason": reason,
        },
    )


def load_snapshot(state_dir: Path, baseline: str) -> tuple[dict[str, Any], Path]:
    """Load only a fully committed, checksum-valid snapshot."""

    baseline = str(baseline)
    candidate = Path(baseline)
    explicit_path = candidate.is_file() or os.sep in baseline or (
        os.altsep is not None and os.altsep in baseline
    )
    if explicit_path:
        path = candidate
        requested_id: str | None = None
    else:
        path = Path(state_dir) / "snapshots" / f"{baseline}.json"
        requested_id = baseline

    lower_name = path.name.lower()
    if ".tmp" in lower_name or path.suffix.lower() != ".json":
        raise _corrupt(baseline, path, "temporary or non-committed snapshot file")
    if requested_id is not None and not SNAPSHOT_ID_RE.fullmatch(requested_id):
        raise _corrupt(baseline, path, "snapshot identifier is not a content hash")
    if not path.is_file():
        raise _corrupt(baseline, path, "snapshot file does not exist")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _corrupt(baseline, path, f"cannot read snapshot: {exc}") from exc
    try:
        wrapper = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _corrupt(baseline, path, f"truncated or invalid JSON: {exc}") from exc
    if not isinstance(wrapper, dict):
        raise _corrupt(baseline, path, "snapshot root is not an object")
    if wrapper.get("format") != SNAPSHOT_FORMAT:
        raise _corrupt(baseline, path, "unsupported snapshot format")
    payload = wrapper.get("payload")
    if not isinstance(payload, dict):
        raise _corrupt(baseline, path, "missing snapshot payload")
    payload_hash = sha256_text(canonical_json(payload))
    if wrapper.get("payload_sha256") != payload_hash:
        raise _corrupt(baseline, path, "payload SHA-256 mismatch")
    if wrapper.get("snapshot_id") != payload_hash:
        raise _corrupt(baseline, path, "snapshot ID does not match payload")
    if requested_id is not None and requested_id != payload_hash:
        raise _corrupt(baseline, path, "requested snapshot ID does not match content")
    if payload.get("rules_version") != RULES_VERSION:
        raise _corrupt(baseline, path, "unsupported rules version")
    return payload, path


def list_snapshots(state_dir: Path) -> list[dict[str, Any]]:
    snapshots_dir = Path(state_dir) / "snapshots"
    if not snapshots_dir.is_dir():
        return []
    result: list[dict[str, Any]] = []
    for path in sorted(snapshots_dir.iterdir()):
        if path.suffix != ".json" or not SNAPSHOT_ID_RE.fullmatch(path.stem):
            continue
        try:
            payload, _ = load_snapshot(state_dir, path.stem)
        except WorkbenchError:
            continue
        result.append(
            {
                "snapshot_id": path.stem,
                "rules_version": payload.get("rules_version"),
                "roots": payload.get("roots", []),
                "configs": payload.get("configs", []),
                "summary": payload.get("summary", {}),
            }
        )
    return result


def _root_hashes(payload: dict[str, Any]) -> dict[str, str]:
    return {
        item["name"]: item["root_hash"]
        for item in payload.get("roots", [])
    }


def _config_hashes(payload: dict[str, Any]) -> dict[str, str]:
    return {
        item["name"]: item["config_hash"]
        for item in payload.get("configs", [])
    }


STATE_FIELDS = ("reachability", "taint", "high_risk", "rule")


def diff_payloads(
    baseline: dict[str, Any],
    current: dict[str, Any],
    baseline_id: str,
) -> dict[str, Any]:
    baseline_calls = {
        (
            call["root"],
            call["config"],
            call["id"],
            call.get("fingerprint", ""),
        ): call
        for call in baseline.get("calls", [])
    }
    current_calls = {
        (
            call["root"],
            call["config"],
            call["id"],
            call.get("fingerprint", ""),
        ): call
        for call in current.get("calls", [])
    }
    added_keys = sorted(set(current_calls) - set(baseline_calls))
    removed_keys = sorted(set(baseline_calls) - set(current_calls))
    common_keys = sorted(set(current_calls) & set(baseline_calls))

    def brief(call: dict[str, Any]) -> dict[str, Any]:
        return {
            "root": call["root"],
            "config": call["config"],
            "id": call["id"],
            "callee": call["callee"],
            "location": call["location"],
            "reachability": call["reachability"],
            "taint": call["taint"],
            "high_risk": call["high_risk"],
            "rule": call["rule"],
        }

    added = [brief(current_calls[key]) for key in added_keys]
    disappeared = [brief(baseline_calls[key]) for key in removed_keys]
    transitions: list[dict[str, Any]] = []
    for key in common_keys:
        old = baseline_calls[key]
        new = current_calls[key]
        changed = [field for field in STATE_FIELDS if old.get(field) != new.get(field)]
        if not changed:
            continue
        transitions.append(
            {
                "root": new["root"],
                "config": new["config"],
                "id": new["id"],
                "callee": new["callee"],
                "location": new["location"],
                "changed_fields": changed,
                "from": {field: old.get(field) for field in changed},
                "to": {field: new.get(field) for field in changed},
                "reachability": new["reachability"],
                "taint": new["taint"],
                "high_risk": new["high_risk"],
                "rule": new["rule"],
            }
        )

    baseline_high = {
        (call["root"], call["config"], call["id"], call.get("fingerprint", ""))
        for call in baseline.get("calls", [])
        if call.get("high_risk")
    }
    new_high = [
        brief(current_calls[key])
        for key in sorted(current_calls)
        if current_calls[key].get("high_risk") and key not in baseline_high
    ]

    baseline_roots = _root_hashes(baseline)
    current_roots = _root_hashes(current)
    baseline_configs = _config_hashes(baseline)
    current_configs = _config_hashes(current)
    root_match = baseline_roots == current_roots
    config_match = baseline_configs == current_configs
    return {
        "baseline_snapshot": baseline_id,
        "added": added,
        "disappeared": disappeared,
        "transitions": transitions,
        "new_high": new_high,
        "new_high_count": len(new_high),
        "content_match": root_match and config_match,
        "root_content_match": root_match,
        "config_content_match": config_match,
        "version_changed": not (root_match and config_match),
        "changed_roots": sorted(
            name
            for name in set(baseline_roots) | set(current_roots)
            if baseline_roots.get(name) != current_roots.get(name)
        ),
        "changed_configs": sorted(
            name
            for name in set(baseline_configs) | set(current_configs)
            if baseline_configs.get(name) != current_configs.get(name)
        ),
    }

